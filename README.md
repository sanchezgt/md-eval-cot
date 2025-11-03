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
```

All datasets are distributed under the **CC-BY 4.0 license** for research and educational purposes.

---

## Metrics Implemented

### **Coherence**
- **Coherence-Momentum** (probabilistic degradation test)  
- **Cross-Encoder coherence** (MPNet)  
- **Embedding-based local cohesion**  
- **Perplexity** (GPT-2 baseline)

### **Logical–Factual Consistency**
- **NLI-based contradiction detection** (DeBERTa-v3-MNLI-FEVER)  
- **AlignScore** (support / implication)  
- **FactCC** (factual consistency)

### **Clarity**
- **Flesch Reading Ease (RD)**  
- **Step Explicitness (SE)**  
- **Definition Coverage (DC)**  
- **Syntactic Simplicity (SSₗᵢₜₑ)**  
- **Aggregate Clarity Index (CLX)**

### **Logical Verification (Z3)**
Implements:
- **ACS** – Aggregate Consistency Score  
- **FAS₀** – Final Answer Soundness  
- **JSS** – Justification Set Size  
- **RCR** – Redundant Constraint Ratio

---

## Human Evaluation

A complementary human study was conducted using **Label Studio**, with four expert raters evaluating 400 items (20 problems × 5 models) along four dimensions:  
**coherence, cohesion, clarity, and informativeness.**

---

## Citation

If you use this repository or the datasets in your work, please cite:

```bibtex
@article{BecerraSanchezBranch2025_gCoT_Eval,
  title   = {Multi-Dimensional Evaluation of Auto-Generated Chain-of-Thought Traces in Reasoning Models},
  author  = {Luis F. Becerra and Germán Sánchez-Torres and John W. Branch-Bedoya},
  year    = {2025},
  journal = {AI},
  publisher = {MDPI},
  doi     = {10.3390/xxxxx}
}
```

## License

All code and datasets are released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.  
You are free to use, modify, and redistribute the materials provided that appropriate credit is given.
