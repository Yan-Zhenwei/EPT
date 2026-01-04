import sys, os
sys.path.insert(0, os.path.abspath('.'))
import torch
from transformers import T5Config, T5ForConditionalGeneration
from peft import LoraConfig, get_peft_model

def count_params(model, filt=None):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if filt is None:
        return total, trainable
    part = sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and filt(n))
    return part

def grad_mean_norm(model, filt=None):
    norms = []
    for n,p in model.named_parameters():
        if p.grad is None or not p.requires_grad:
            continue
        if filt is None or filt(n):
            norms.append(p.grad.detach().norm().item())
    if len(norms) == 0:
        return 0.0, 0
    return sum(norms)/len(norms), len(norms)

def f_core(n): return ("lora_A" in n) or ("lora_B" in n)
def f_gate(n): return "lora_gate" in n
def f_task(n): return "lora_task_embedding" in n
def f_kernels(n): return "lora_expert_kernels" in n

config = T5Config(
    d_model=768,
    d_ff=2048,
    num_layers=12,
    num_decoder_layers=12,
    num_heads=12,
    decoder_start_token_id=0,
)
base = T5ForConditionalGeneration(config)

peft_cfg = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q","k","v","o","wi","wo"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS",
)
model = get_peft_model(base, peft_cfg)

sizes = [2,2,4,4,8,8,16,16]
for m in model.modules():
    if hasattr(m, "set_expert_kernel_sizes"):
        m.set_expert_kernel_sizes("default", sizes)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.train()

seq_len = 16
batch = 2
vocab = model.config.vocab_size
input_ids = torch.randint(low=0, high=vocab, size=(batch, seq_len)).to(device)
attention_mask = torch.ones_like(input_ids).to(device)
labels = input_ids.clone()

out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
loss = out.loss
loss.backward()

total, trainable = count_params(model)
pct = 100 * trainable / total if total > 0 else 0
print(f"trainable params: {trainable} || all params: {total} || trainable%: {pct}")

core = count_params(model, f_core)
gate = count_params(model, f_gate)
task = count_params(model, f_task)
kern = count_params(model, f_kernels)
print({"lora_core": core, "lora_gate": gate, "lora_task_embedding": task, "lora_expert_kernels": kern})

mn_core, c_core = grad_mean_norm(model, f_core)
mn_gate, c_gate = grad_mean_norm(model, f_gate)
mn_task, c_task = grad_mean_norm(model, f_task)
mn_kern, c_kern = grad_mean_norm(model, f_kernels)
print({"grad_mean_norm": {"core": mn_core, "gate": mn_gate, "task": mn_task, "kernels": mn_kern},
       "grad_counts": {"core": c_core, "gate": c_gate, "task": c_task, "kernels": c_kern}})
