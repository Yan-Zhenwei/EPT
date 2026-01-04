# EPT

## Title

Expert Pyramid Tuning: Efficient Parameter Fine-Tuning for Task Allocation Based on Professional Specialization

## Framework

![framework](framework.png)

## Datasets

- **GLUE**: [GLUE Benchmark](https://gluebenchmark.com/tasks)
- **BoolQ, CB**: [SuperGLUE Benchmark](https://super.gluebenchmark.com/tasks)
- **SciTail**: [SciTail Dataset on Hugging Face](https://huggingface.co/datasets/allenai/scitail)

## Train

```bash
cd script
sh train_ept.sh
```

## Results

| Method | Params/Task | MNLI | QQP | QNLI | SST-2 | STS-B | MRPC | RTE | CoLA | AVG |
|--------|-------------|------|-----|------|-------|-------|------|-----|------|-----|
| Finetuning | 28M | 85.7 | 91.1 | 92.0 | 92.5 | 88.8 | 90.2 | 75.4 | 54.9 | 83.8 |
| Adapters | 1.8M | 86.3 | 90.5 | 93.2 | 93.0 | 89.9 | 90.2 | 70.3 | 61.5 | 84.4 |
| PT | 9.6K | 85.6 | 90.6 | 93.2 | 93.9 | 89.9 | 86.3 | 67.6 | 55.3 | 82.8 |
| LoRA (r=8) | 0.39M | 85.8 | 89.2 | 93.1 | 93.2 | 90.4 | 89.9 | 76.3 | 62.8 | 85.1 |
| LoRA (r=16) | 0.78M | 84.9 | 89.6 | 93.0 | 93.7 | 90.4 | 88.7 | 80.6 | 63.9 | 85.6 |
| HyperFormer | 638K | 85.7 | 90.0 | 93.0 | 94.0 | 89.7 | 87.2 | 75.4 | 63.7 | 84.8 |
| MPT | 10.5K | 84.3 | 90.0 | 93.0 | 93.3 | 90.4 | 89.2 | 82.7 | 63.5 | 85.8 |
| MultiLoRA | 1.56M | 85.9 | 89.7 | 92.8 | 94.5 | 89.8 | 88.2 | 80.6 | 66.9 | 86.0 |
| MixLoRA | 1.49M | 85.8 | 90.0 | 92.9 | 93.7 | 90.3 | 89.2 | 78.4 | 67.2 | 85.9 |
| MOELoRA | 0.81M | 86.3 | 90.4 | 93.2 | 94.2 | 89.8 | 90.7 | 79.9 | 65.3 | 86.2 |
| MoRE | 0.81M | 85.6 | 90.2 | 93.1 | 93.9 | 89.9 | 90.7 | 77.7 | 68.7 | 86.2 |
| **EPT** | **0.81M** | **86.4** | **90.2** | **93.6** | **94.5** | **90.0** | **90.7** | **82.0** | **68.9** | **87.0** |
