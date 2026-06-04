# 🧠 CDPO: Causal Direct Preference Optimization with Backdoor Adjustment

This repository contains the official implementation of Causal Direct Preference Optimization (CDPO), proposed in the paper: 
> **[Causal Direct Preference Optimization for Language Model Alignment](https://aclanthology.org/2026.findings-eacl.58/)**  

CDPO extends Direct Preference Optimization (DPO) with causal backdoor adjustment to mitigate confounding effects in preference-based language model alignment.

The repository provides code for:

* Supervised Fine-Tuning (SFT) with backdoor adjustment
* Causal Direct Preference Optimization (CDPO)
* Training and evaluation on preference-learning benchmarks such as HH and SHP
* Reproducing the experiments reported in the paper

---

## 🚀 Key Idea

Standard DPO assumes that preference labels are unbiased given the prompt and candidate responses.

In practice, however, preference datasets (e.g., HH, SHP) are often influenced by **observed confounders**, such as:

* task category
* helpfulness vs. harmlessness emphasis
* domain or topic source
* annotation context or style

Observed confounders are explicitly introduced as additional conditioning variables and averaged over during training. By marginalizing across multiple confounder values rather than fixing a single one, CDPO approximates an interventional objective inspired by the causal backdoor criterion and reduces dependence on spurious correlations.

This allows preference learning to account for multiple data-generating contexts while preserving the simplicity and efficiency of DPO.

---

## 🧩 Training Pipeline

Training CDPO consists of two mandatory stages:

1. Supervised Fine-Tuning (SFT)
2. Causal Direct Preference Optimization (CDPO)

Both stages apply the same backdoor adjustment to ensure distributional consistency.

### 🔹 Stage 1: Supervised Fine-Tuning

Before preference optimization, the policy is fine-tuned using a modified SFT objective that incorporates additional conditioning information and marginalization during training.

This step aligns the SFT objective with the setup used in the subsequent preference optimization stage, ensuring stable and in-distribution training.

**Example (HH dataset):**

```bash
python -u train.py \
  model=qwen05b \
  datasets=[hh] \
  loss=sft \
  backdoor.enabled=true \
  exp_name=hh_sft \
  gradient_accumulation_steps=2 \
  batch_size=4 \
  eval_batch_size=4 \
  trainer=BasicTrainer \
  sample_during_eval=false
```

### 🔹 Stage 2: CDPO — Causal Direct Preference Optimization

Starting from the marginalized SFT checkpoint, CDPO performs preference learning while marginalizing over observed confounders.

Compared to standard DPO, CDPO differs only in how log-probabilities are computed; the optimization procedure and training infrastructure remain unchanged.

#### ▶️ Running CDPO

```bash
python -u train.py \
  model=qwen05b \
  datasets=[hh] \
  loss=dpo \
  loss.beta=0.1 \
  backdoor.enabled=true \
  model.archive=/path/to/sft/LATEST/policy.pt \
  exp_name=hh_cdpo \
  gradient_accumulation_steps=2 \
  batch_size=4 \
  eval_batch_size=4 \
  trainer=BasicTrainer \
  sample_during_eval=false
```

For further experimental details, please refer to our paper.

## Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{Le_etal_26Causal,
  title={Causal Direct Preference Optimization for Language Model Alignment},
  author={Le, Uyen and Nguyen, Thin and Nguyen, Toan and Doan, Toan and Le, Trung and Le, Bac},
  booktitle={Findings of the Association for Computational Linguistics: EACL 2026},
  pages={1098--1113},
  year={2026}
}
```

## Acknowledgements

We thank the authors and contributors of the Direct Preference Optimization (DPO) repository for their valuable contributions to the RLHF community. This work builds upon and extends their open-source implementation.
