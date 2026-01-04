from data.multi_task_sample import AutoTask, TaskCollator, MultiTaskDataLoader
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, HfArgumentParser, Trainer
from transformers import LlamaConfig, LlamaTokenizer, LlamaForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from typing import Optional, List, Union
from dataclasses import dataclass, field
from trainer import MyTrainer
import torch
import random
import numpy as np
import os
import datasets
from rank import *
from metrics import accuracy, pearson_corrcoef, matthews_corrcoef


@dataclass
class MyArguments:
    model_name_or_path: Optional[str] = field(default=None)
    tasks: Optional[List[str]] = field(default_factory=lambda: ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2'])
    max_length: Optional[int] = field(default=128)
    use_lora: bool = field(default=True)
    lora_rank: Optional[int] = field(default=16)
    lora_alpha: Optional[int] = field(default=32)
    target_modules: Optional[List[str]] = field(default_factory=lambda: ["q", "k", "v", "o", "wi", "wo"])
    epochs: Optional[int] = field(default=10)
    smooth_distribution: bool = field(default=True)
    sample_by_loss: bool = field(default=False)
    use_dyrank: bool = field(default=False)
    use_share_module: bool = field(default=False)
    expert_kernel_sizes: Optional[List[int]] = field(default=None)
    moe_top_k: Optional[int] = field(default=1)
    model_type: Optional[str] = field(default=None)  # 't5' 或 'llama'
    zero_init_expert_kernels: bool = field(default=False)


def main():
    parser = HfArgumentParser((TrainingArguments, MyArguments))
    training_args, data_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False

    torch.manual_seed(training_args.seed)
    torch.cuda.manual_seed_all(training_args.seed)
    np.random.seed(training_args.seed)
    random.seed(training_args.seed)

    #tasks = ['stsb']
    tasks = data_args.tasks
    print(tasks)
    #['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2']
    dataset_class = AutoTask
    train_datasets = [dataset_class.get(task).get_dataset(
        split="train") for task in tasks]#1238 1079
    eval_datasets = ({task: dataset_class.get(task, seed=1189).get_dataset(
                    split="validation") for task in tasks})

    dataset_sizes = [len(train_dataset) for train_dataset in train_datasets]
    print(train_datasets)
    print({tasks[i]:dataset_sizes[i] for i in range(len(tasks))})
    # train_datasets = datasets.concatenate_datasets(train_datasets)
    # print(train_datasets)

    # 根据路径或显式参数选择模型类型
    is_llama = False
    if data_args.model_type is not None:
        is_llama = (data_args.model_type.lower() == 'llama')
    else:
        mp = str(data_args.model_name_or_path or '').lower()
        is_llama = ('llama' in mp)

    if is_llama:
        config = LlamaConfig.from_pretrained(data_args.model_name_or_path)
        setattr(config, 'num_beams', getattr(config, 'num_beams', 1))
        tokenizer = LlamaTokenizer.from_pretrained(data_args.model_name_or_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = LlamaForCausalLM.from_pretrained(data_args.model_name_or_path)
        model.config.pad_token_id = tokenizer.pad_token_id
    else:
        config = T5Config.from_pretrained(data_args.model_name_or_path)
        setattr(config, 'num_beams', getattr(config, 'num_beams', 1))
        tokenizer = T5Tokenizer.from_pretrained(data_args.model_name_or_path)
        model = T5ForConditionalGeneration.from_pretrained(data_args.model_name_or_path)

    if data_args.use_lora:
        default_targets = data_args.target_modules
        if is_llama and (default_targets == ["q", "k", "v", "o", "wi", "wo"]):
            default_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]
        lora_config = LoraConfig(
            r=data_args.lora_rank,
            lora_alpha=data_args.lora_alpha,
            target_modules=default_targets,
            lora_dropout=0.1,
            bias="none",
            task_type='CAUSAL_LM' if is_llama else "SEQ_CLS"
        )
        model = get_peft_model(model, lora_config)
    print(data_args.target_modules)
    def _print_trainable_parameters(m):
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        pct = 100 * trainable / total if total > 0 else 0
        print(f"trainable params: {trainable} || all params: {total} || trainable%: {pct}")
    if data_args.expert_kernel_sizes is not None:
        sizes = data_args.expert_kernel_sizes
        if len(sizes) == 1:
            sizes = sizes * 8
        for m in model.modules():
            if hasattr(m, "set_expert_kernel_sizes"):
                m.set_expert_kernel_sizes("default", sizes)
    if getattr(data_args, "zero_init_expert_kernels", False):
        for m in model.modules():
            if hasattr(m, "lora_expert_kernels"):
                for key in m.lora_expert_kernels.keys():
                    for p in m.lora_expert_kernels[key]:
                        p.data.zero_()
    if data_args.moe_top_k is not None:
        for m in model.modules():
            if hasattr(m, "set_moe_top_k"):
                m.set_moe_top_k(data_args.moe_top_k)
    _print_trainable_parameters(model)
        #print(model)
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    def lmap(f, x) -> List:
        """list(map(f, x))"""
        return list(map(f, x))

    def compute_metrics(eval_prediction):
        predictions = eval_prediction.predictions
        label_ids = eval_prediction.label_ids
        pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        label_ids[label_ids == -100] = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        pred_str = lmap(lambda s: s.strip().lower().rstrip(".!,?:;"), pred_str)
        label_str = lmap(lambda s: s.strip().lower().rstrip(".!,?:;"), label_str)

        # task-aware normalization to valid label space
        try:
            task_name = tasks[get_task_id()]
            ds = dataset_class.get(task_name)
            valid = set(getattr(ds, 'label_list', []) or [])
            tm = getattr(ds, 'target_map', {}) or {}
            valid |= set(tm.values())
        except Exception as e:
            valid = set()
            tm = {}

        def norm_bool(s):
            s = s.lower()
            if s in {"yes", "y", "true", "t", "1"}:
                return "yes"
            if s in {"no", "n", "false", "f", "0"}:
                return "no"
            return s

        def norm_abcd(s):
            # Check for the first occurrence of A, B, C, D
            for ch in ("a", "b", "c", "d"):
                if ch in s.lower():
                    return ch.upper()
            return s
            
        def map_to_target(s, tm):
            # If s is a key in target_map (e.g., '0'), map to value (e.g., 'A')
            if s in tm:
                return tm[s]
            # Also check if s is a value in target_map, return as is
            if s in tm.values():
                return s
            return s

        def norm_entail(s):
            s = s.lower()
            if s in {"entailment", "entailed"}:
                return "entailment"
            if s in {"contradiction", "contradict"}:
                return "contradiction"
            if s in {"neutral"}:
                return "neutral"
            return s

        # apply heuristics based on common tasks
        if valid:
            # Apply target_map first if available (e.g. 0->A for piqa)
            if tm:
                label_str = [map_to_target(l, tm) for l in label_str]
                # We might want to map preds too if they output '0' instead of 'A'
                pred_str = [map_to_target(p, tm) for p in pred_str]

            if {"yes", "no"}.issubset(valid):
                pred_str = [norm_bool(p) for p in pred_str]
                label_str = [norm_bool(l) for l in label_str]
            elif {"A", "B"}.issubset(valid): # Relaxed from A,B,C,D to support PIQA (A,B)
                pred_str = [norm_abcd(p) for p in pred_str]
                label_str = [norm_abcd(l) for l in label_str]
            elif {"entailment", "contradiction", "neutral"}.issubset(valid):
                pred_str = [norm_entail(p) for p in pred_str]
                label_str = [norm_entail(l) for l in label_str]

        # metrics
        # LLaMA specific task handling
        if is_llama:
             # CoLA (Task ID 0) -> MCC
            if get_task_id() == 0 and 'cola' in tasks[0]: 
                 return matthews_corrcoef(pred_str, label_str)
            # STS-B (Task ID 7) -> Pearson
            elif get_task_id() == 7 and 'stsb' in tasks[7]: 
                pred_f = [float(pred) if pred.replace('.', '', 1).isdigit() else 0.0 for pred in pred_str]
                label_f = [float(label) for label in label_str]
                return pearson_corrcoef(pred_f, label_f)
            else:
                return accuracy(pred_str, label_str)

        # T5 handling (legacy logic)
        if not is_llama and get_task_id() == 0:
            return matthews_corrcoef(pred_str, label_str)
        elif not is_llama and get_task_id() == 7:
            pred_f = [float(pred) if pred.replace('.', '', 1).isdigit() else 0.0 for pred in pred_str]
            label_f = [float(label) for label in label_str]
            return pearson_corrcoef(pred_f, label_f)
        else:
            return accuracy(pred_str, label_str)

    my_trainer = MyTrainer(model=model, 
                           config=config,
                           args=training_args,
                           train_dataset=train_datasets,
                           eval_dataset=eval_datasets,
                           data_collator=TaskCollator(tokenizer, data_args=data_args),
                           #data_collator=data_collator,
                           compute_metrics=compute_metrics,
                           tokenizer=tokenizer,
                           )

    my_trainer.train()


if __name__ == "__main__":
    main()
