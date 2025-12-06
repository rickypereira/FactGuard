# FactGuard

<p align="center">
  <strong>Scalable Automated Claim Verification via Knowledge Distillation</strong>
</p>

<p align="center">
  <em>W266 NLP Final Project — Fall 2025</em><br>
  <strong>Rick Pereira & Karan Patel</strong>
</p>

---

## Overview

**FactGuard** addresses the computational bottleneck of deploying large language models for automated fact-checking. By distilling knowledge from **Gemini 2.5 Flash** (teacher model) into smaller, efficient student architectures, FactGuard delivers production-ready claim verification without sacrificing accuracy.

### Key Results

| Model | Dataset | Accuracy | F1 Score |
|-------|---------|----------|----------|
| T5-Gemma + RAG | FEVER | 85% | 89% |
| Gemma-2B + RAG | FEVER | 85% | 89.27% |
| Gemma-2B + RAG | BoolQ | 70.30% | 75.23% |

## Architecture

FactGuard employs a **Teacher-Student** paradigm using knowledge distillation:

```
┌─────────────────────────────────────────────────────────────────┐
│                     TEACHER MODEL                                │
│                   Gemini 2.5 Flash                               │
│    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│    │   FEVER     │    │   SQuAD     │    │  Rationale  │        │
│    │   Claims    │ +  │   Q&A       │ →  │  Generation │        │
│    └─────────────┘    └─────────────┘    └─────────────┘        │
└─────────────────────────┬───────────────────────────────────────┘
                          │ Knowledge Distillation
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STUDENT MODELS                                │
│  ┌────────────────────┐      ┌────────────────────┐             │
│  │     T5-Gemma       │      │     Gemma-2B       │             │
│  │  (Encoder-Decoder) │      │   (Decoder-Only)   │             │
│  │      ~4B params    │      │      ~2B params    │             │
│  └────────────────────┘      └────────────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   RAG PIPELINE (Optional)                        │
│         DuckDuckGo Search → Context Augmentation                 │
└─────────────────────────────────────────────────────────────────┘
```

### Student Models

| Model | Architecture | Strengths |
|-------|--------------|-----------|
| **T5-Gemma** | Encoder-Decoder | Dense contextual representations, superior evidence synthesis |
| **Gemma-2B** | Decoder-Only | Lightweight, low latency, efficient autoregressive inference |

## Datasets

### Training (Distillation)
- **FEVER** — Fact Extraction and VERification: Claims labeled as SUPPORTS/REFUTES with Wikipedia evidence
- **SQuAD** — Stanford Question Answering Dataset: Converted to true/false claims with teacher-generated rationales

### Evaluation
- **FEVER** — Structured, evidence-driven claims
- **BoolQ** — Yes/no questions requiring passage reasoning
- **LIAR** — Real-world political statements (most challenging)

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Google Colab or local environment with GPU

### Setup

```bash
# Install dependencies
pip install duckduckgo-search langchain-community
pip install -U ddgs
pip install datasets==3.6.0
pip install -U bitsandbytes accelerate
pip install -U transformers
pip install trl peft
```

### Environment Variables

Set the following API keys (if using Google Colab, store in Colab secrets):

```python
import os
os.environ['GEMINI_API_KEY'] = 'your-gemini-api-key'
os.environ['HF_TOKEN'] = 'your-huggingface-token'
```

## Quick Start

### Load a Pre-trained FactGuard Model

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# Load T5-Gemma distilled model
base_model = "google/t5gemma-2b-2b-ul2-it"
finetuned_model = "rickpereira/FactGuard-Distilled-T5"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForSeq2SeqLM.from_pretrained(finetuned_model, device_map='auto')
```

### Verify a Claim

```python
from langchain_community.tools import DuckDuckGoSearchRun

def verify_claim(claim: str, use_rag: bool = True):
    # Optional: Retrieve context via web search
    context = ""
    if use_rag:
        search = DuckDuckGoSearchRun()
        context = search.invoke(claim)
    
    # Format prompt
    prompt = f"""**Fact-Check and Evidence Verification**

Determine the final verdict:
* **Yes:** If the claim is fully supported by the Context or external knowledge.
* **No:** If the claim is false, contradicted, or insufficient evidence.

Output Requirement: Output the final verdict ('Yes' or 'No') and nothing else.

--- Context ---
{context if context else 'No specific context provided.'}

--- Claim ---
{claim}
--- Verdict ---"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=10)
    verdict = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return verdict

# Example usage
result = verify_claim("The Eiffel Tower is located in Paris, France.")
print(f"Verdict: {result}")  # Output: Yes
```

## Training Your Own Model

### 1. Load Distillation Dataset

```python
from datasets import load_dataset

# Pre-generated distillation datasets with teacher rationales
fever_distilled = load_dataset("rickpereira/factguard_fever_distilled_datasets")
squad_distilled = load_dataset("rickpereira/factguard_squad_distilled_datasets")
```

### 2. Fine-tune with LoRA

```python
from peft import LoraConfig
from trl import SFTTrainer

peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    # ... additional training arguments
)
trainer.train()
```

### Hyperparameter Configuration

**Best T5-Gemma Configuration:**
| Parameter | Value |
|-----------|-------|
| Epochs | 1 |
| Batch Size | 8 |
| Learning Rate | 5e-05 |
| LoRA Rank | 16 |
| LoRA Alpha | 16 |
| Dropout | 0.05 |

**Best Gemma-2B Configuration:**
| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Batch Size | 4 |
| Learning Rate | 5e-04 |
| LoRA Rank | 64 |
| LoRA Alpha | 32 |
| Dropout | 0.05 |

## Results

### Performance Comparison

#### T5-Gemma Results

| Dataset | Evaluation | AU-PRC | Accuracy | F1 |
|---------|------------|--------|----------|-----|
| FEVER | Baseline | 0.72 | 31.70% | 25.84% |
| FEVER | Fine-tuned | 0.98 | 72.10% | 77.84% |
| FEVER | + RAG | 0.99 | **85%** | **89%** |
| BoolQ | Baseline | 0.60 | 60.10% | 75.08% |
| BoolQ | + RAG | 0.83 | 66.80% | 66.80% |
| LIAR | + RAG | 0.53 | 55% | 56.10% |

#### Gemma-2B Results

| Dataset | Evaluation | AU-PRC | Accuracy | F1 |
|---------|------------|--------|----------|-----|
| FEVER | Baseline | 0.94 | 67.40% | 73.92% |
| FEVER | Fine-tuned | 0.97 | 76.30% | 81.81% |
| FEVER | + RAG | 0.98 | **85%** | **89.27%** |
| BoolQ | + RAG | 0.78 | **70.30%** | **75.23%** |
| LIAR | Fine-tuned | 0.52 | 55.80% | 47.94% |

### Key Findings

1. **Structured datasets benefit most** — FEVER saw up to 53.3% accuracy improvement with fine-tuning
2. **RAG consistently improves performance** — External evidence retrieval provides measurable gains
3. **LIAR remains challenging** — Real-world political claims require more sophisticated approaches
4. **Efficient models can compete** — 2B parameter models achieve ~85% of teacher performance

## Evaluation Pipeline

Three evaluation configurations are available:

| Configuration | Description |
|---------------|-------------|
| **Baseline** | Pre-trained model without fine-tuning |
| **Fine-tuned LLM** | Distilled model using only parametric knowledge |
| **RAG** | Fine-tuned model + DuckDuckGo web retrieval |

## Model Availability

Pre-trained FactGuard models are available on Hugging Face:

- 🤗 [FactGuard-Distilled-T5](https://huggingface.co/rickpereira/FactGuard-Distilled-T5) — T5-Gemma encoder-decoder
- 🤗 [FactGuard-Distilled-Decoder](https://huggingface.co/rickpereira/FactGuard-Distilled-Decoder) — Gemma-2B decoder-only

Distillation datasets:
- 🤗 [factguard_fever_distilled_datasets](https://huggingface.co/datasets/rickpereira/factguard_fever_distilled_datasets)
- 🤗 [factguard_squad_distilled_datasets](https://huggingface.co/datasets/rickpereira/factguard_squad_distilled_datasets)

## Citation

```bibtex
@misc{factguard2025,
  title={FactGuard: Veridicity of Claims},
  author={Pereira, Rick and Patel, Karan},
  year={2025},
  institution={UC Berkeley},
  note={W266 NLP Final Project}
}
```

## Future Work

- **Model Scaling** — Experiment with Gemma-7B for improved multi-class verification
- **Enhanced RAG** — Replace DuckDuckGo with specialized retrieval systems
- **Multi-label Output** — Extend beyond binary (True/False) to include "Unverified"
- **Direct Preference Optimization** — Post-SFT alignment using TruthfulQA

## License

This project is released for academic and research purposes.

---

<p align="center">
  <sub>Built with 🔍 for scalable, efficient fact-checking</sub>
</p>
