# md-eval-cot

# Multi-Dimensional Evaluation of Auto-Generated Chain-of-Thought Traces in Reasoning Models

**Authors:**  
Luis F. Becerra¹, Germán Sánchez-Torres², and John W. Branch-Bedoya¹  
¹ Facultad de Minas, Universidad Nacional de Colombia – Medellín  
² Facultad de Ingeniería, Universidad del Magdalena – Santa Marta  
Correspondence: gsanchez@unimagdalena.edu.co

---

## Overview

This repository accompanies the paper:

> **“Multi-Dimensional Evaluation of Auto-Generated Chain-of-Thought Traces in Reasoning Models” (2025)**  

It provides **datasets**, **metrics implementations**, and **reproducible analysis pipelines** used to evaluate the explanatory value of automatically generated reasoning traces (gCoT) beyond fidelity.  
The framework quantifies four complementary axes:

1. **Structural Coherence** – organization and semantic flow.  
2. **Logical–Factual Consistency** – self-consistency and entailment strength.  
3. **Linguistic Clarity** – readability, explicitness, and syntactic simplicity.  
4. **Coverage/Informativeness** – content density and non-redundancy.

All results in the paper can be reproduced from this repository.



## Repository Structure

md-eval-cot/
│
├── datasets/ # gCoT samples from GSM8K (~1.3k–1.4k per model)
│ ├── DS_LL/
│ ├── DS_QW/
│ ├── GPT_120B/
│ ├── GPT_20B/
│ ├── QWEN3_THK/
│ └── metadata.json
│
├── metrics/ # Automatic metric implementations
│ ├── coherence/
│ ├── consistency/
│ ├── clarity/
│ ├── informativeness/
│ └── logic_verification/
│
├── results/ # Output tables and figures
│ ├── automatic_metrics.csv
│ ├── human_eval_results.csv
│ ├── figures/
│ └── tables/
│
├── scripts/ # Main analysis and plotting pipelines
│ ├── generate_metrics.py
│ ├── evaluate_logic_z3.py
│ ├── plot_results.py
│ └── utils/
│
├── paper/ # PDF and supplementary materials
│ └── Multi-Dimensional_Evaluation.pdf
│
├── environment.yml # Conda environment for reproducibility
├── CITATION.cff # Citation metadata
└── LICENSE # CC BY 4.0

yaml
Copiar código

---

## Datasets

The datasets contain automatically generated **gCoT traces** (Chain-of-Thought) for the **GSM8K** benchmark, produced by five reasoning-oriented models:

| Model | Description |
|--------|--------------|
| **DeepSeek-R1-Distill-Llama-8B** | 8B dense model distilled from DeepSeek-R1 on Llama-3.1-8B-Base. |
| **DeepSeek-R1-Distill-Qwen-7B** | 7B model fine-tuned on Qwen-2.5-Math-7B for reasoning. |
| **GPT-OSS-20B / 120B** | Open-weight Mixture-of-Experts models under Apache 2.0. |
| **Qwen3-4B-Thinking-2507** | 4B causal LM with enhanced “thinking” mode. |

Each `.jsonl` record includes:

```json
{
  "id": "gsm8k_001",
  "question": "...",
  "gcot": "...",
  "answer": "...",
  "model": "DeepSeek-R1-Distill-Qwen-7B"
}
All datasets are distributed under the CC-BY 4.0 license for research and educational purposes.
