import sys, os
sys.path.insert(0, os.path.abspath('.'))
import argparse
import torch
from transformers import T5Config, T5ForConditionalGeneration
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import LoraLayer

def count_trainable(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def strict_breakdown(model, adapter_name="default"):
    parts = {
        "lora_core": 0,
        "lora_gate": 0,
        "lora_task_embedding": 0,
        "lora_expert_kernels": 0,
    }
    module_counts = {"q":0, "k":0, "v":0, "o":0, "wi_0":0, "wi_1":0, "wo":0}
    for key, mod in model.named_modules():
        if isinstance(mod, LoraLayer):
            # core
            if adapter_name in mod.lora_A and adapter_name in mod.lora_B:
                parts["lora_core"] += mod.lora_A[adapter_name].weight.numel()
                parts["lora_core"] += mod.lora_B[adapter_name].weight.numel()
            # gate
            if adapter_name in mod.lora_gate:
                gate = mod.lora_gate[adapter_name][0]
                parts["lora_gate"] += gate.weight.numel()
                if gate.bias is not None:
                    parts["lora_gate"] += gate.bias.numel()
            # task embedding
            if adapter_name in mod.lora_task_embedding:
                parts["lora_task_embedding"] += mod.lora_task_embedding[adapter_name].weight.numel()
            # expert kernels
            if adapter_name in mod.lora_expert_kernels:
                for p in mod.lora_expert_kernels[adapter_name]:
                    parts["lora_expert_kernels"] += p.numel()
            # type tally
            if key.endswith(".q"): module_counts["q"] += 1
            elif key.endswith(".k"): module_counts["k"] += 1
            elif key.endswith(".v"): module_counts["v"] += 1
            elif key.endswith(".o"): module_counts["o"] += 1
            elif key.endswith(".wi_0"): module_counts["wi_0"] += 1
            elif key.endswith(".wi_1"): module_counts["wi_1"] += 1
            elif key.endswith(".wo"): module_counts["wo"] += 1
    return parts, module_counts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name_or_path', type=str, default=None)
    ap.add_argument('--lora_rank', type=int, default=8)
    ap.add_argument('--lora_alpha', type=int, default=32)
    ap.add_argument('--target_modules', nargs='+', default=["q","k","v","o","wi","wo"])
    ap.add_argument('--expert_kernel_sizes', nargs='+', type=int, default=[2,2,4,4,8,8,16,16])
    ap.add_argument('--probe_train_steps', type=int, default=0)
    args = ap.parse_args()

    if args.model_name_or_path:
        base = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    else:
        cfg = T5Config(
            d_model=768,
            d_ff=2048,
            num_layers=12,
            num_decoder_layers=12,
            num_heads=12,
            decoder_start_token_id=0,
        )
        base = T5ForConditionalGeneration(cfg)

    peft_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(base, peft_cfg)
    sizes = args.expert_kernel_sizes
    if len(sizes) == 1:
        sizes = sizes * 8
    for m in model.modules():
        if hasattr(m, "set_expert_kernel_sizes"):
            m.set_expert_kernel_sizes("default", sizes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if args.probe_train_steps > 0:
        model.train()
        seq_len = 16
        batch = 4
        vocab = model.config.vocab_size
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        for _ in range(args.probe_train_steps):
            input_ids = torch.randint(low=0, high=vocab, size=(batch, seq_len)).to(device)
            attention_mask = torch.ones_like(input_ids).to(device)
            labels = input_ids.clone()
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            opt.zero_grad()
            loss.backward()
            opt.step()

    total, trainable = count_trainable(model)
    parts, mcounts = strict_breakdown(model)
    print({"all_params": total, "trainable_params": trainable, "trainable_pct": 100*trainable/total})
    print(parts)
    print(mcounts)

if __name__ == "__main__":
    main()
