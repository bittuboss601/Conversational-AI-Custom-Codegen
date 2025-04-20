# !pip install transformers datasets peft accelerate huggingface_hub -q
# use cuda

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import torch

def format_mbpp(example):
    return {
        "prompt": example["text"] or "",
        "code": example["code"] or ""
    }

def format_csn(example):
    return {
        "prompt": example["docstring"] or "",
        "code": example["code"] or ""
    }

def tokenize(example):
    inputs = tokenizer(example["prompt"], truncation=True, max_length=MAX_LENGTH)
    targets = tokenizer(example["code"], truncation=True, max_length=MAX_LENGTH)
    inputs["labels"] = targets["input_ids"]
    return inputs

# Load MBPP (subset for speed)
mbpp = load_dataset("mbpp")
mbpp = mbpp["train"].select(range(500))

# Load CodeSearchNet Python subset
csn = load_dataset("code_search_net", "python")
csn = csn["train"].select(range(500))

mbpp_formatted = mbpp.map(format_mbpp)
csn_formatted = csn.map(format_csn)

# Merge datasets
combined = mbpp_formatted + csn_formatted

model_name = "Salesforce/codegen-350M-mono"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenization
MAX_LENGTH = 512

train_data = Dataset.from_list(combined)
tokenized_data = train_data.map(tokenize, batched=True, remove_columns=["prompt", "code"])

training_args = TrainingArguments(
    output_dir="codegen-finetuned",
    per_device_train_batch_size=2,
    num_train_epochs=2,
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,
    fp16=torch.cuda.is_available(),
    push_to_hub=True,
    hub_model_id="tiwaripurnendu121/codegen-350M-mbpp-csn",
    hub_token="hf_xxx"  # Replace with your token
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
    tokenizer=tokenizer
)

trainer.train()

trainer.push_to_hub()



