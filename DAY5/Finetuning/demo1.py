#!/usr/bin/env python3
"""
LLM Fine‑Tuning with a Financial Dataset

This script shows:
1. Loading a **financial Q&A dataset (JSONL)**
2. Converting into HuggingFace Dataset
3. Tokenizing
4. Fine‑tuning a small LLaMA model using PEFT + LoRA
5. Saving the fine‑tuned model

NOTE:
- Replace file paths with your dataset path & model path.
- Use CUDA GPU for training.

"""

import os
import json
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# ------------------------------------------------------------
# 1. Load Financial Dataset (JSONL Format)
# ------------------------------------------------------------
# Expected dataset format (financial_data.jsonl):
# { "instruction": "What is ROCE?", "response": "Return on Capital Employed ..." }

DATA_PATH = "financial_data.jsonl"  # change this to your dataset file

# Create example dataset if file doesn't exist
if not os.path.exists(DATA_PATH):
    example_rows = [
        {
            "instruction": "What is EBITDA?",
            "response": "EBITDA stands for Earnings Before Interest, Taxes, Depreciation, and Amortization. It measures operating profitability."
        },
        {
            "instruction": "Explain what is Balance Sheet?",
            "response": "A balance sheet is a financial statement that reports a company's assets, liabilities, and equity at a specific point in time."
        },
        {
            "instruction": "What is P/E Ratio?",
            "response": "The Price-to-Earnings Ratio shows how much investors are willing to pay per rupee of earnings."
        }
    ]
    with open(DATA_PATH, "w") as f:
        for row in example_rows:
            f.write(json.dumps(row) + "\n") # same json.dump(example_rows,f)

# Load JSONL file using HuggingFace
raw_dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# ------------------------------------------------------------
# 2. Format Input → Prompt + Response
# ------------------------------------------------------------

def format_record(example):
    prompt = f"Instruction: {example['instruction']}\nAnswer: "
    label = example["response"]
    return {"prompt": prompt, "label": label}

formatted_dataset = raw_dataset.map(format_record)

# ------------------------------------------------------------
# 3. Load Tokenizer + Base Model (small model for demo)
# ------------------------------------------------------------

MODEL_NAME = "meta-llama/Llama-3.2-1B"  # You can replace with a small model

print("Loading model & tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# ------------------------------------------------------------
# 4. Apply LoRA Fine‑Tuning
# ------------------------------------------------------------

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# ------------------------------------------------------------
# 5. Tokenization Function
# ------------------------------------------------------------

def tokenize(batch):
    texts = [p + l for p, l in zip(batch["prompt"], batch["label"])]
    tokens = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=256
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

print("Tokenizing...")
tokenized_dataset = formatted_dataset.map(tokenize, batched=True)

# ------------------------------------------------------------
# 6. Training Setup
# ------------------------------------------------------------

training_args = TrainingArguments(
    output_dir="finetuned-financial-llm",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    report_to="none",
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# ------------------------------------------------------------
# 7. Start Training
# ------------------------------------------------------------
if __name__ == "__main__":
    print("Starting training...")
    trainer.train()

    # Save fine‑tuned model
    model.save_pretrained("financial_llm_lora")
    tokenizer.save_pretrained("financial_llm_lora")

    print("\nTraining Completed! Fine‑tuned model saved to financial_llm_lora/")
