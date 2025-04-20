from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()


# Load a HuggingFace model (you can change this to CodeT5, CodeGen, StarCoder etc.)
model_name = "tiwaripurnendu121/codegen-350M-mbpp-csn"#"Salesforce/codegen-350M-mono"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


class PromptRequest(BaseModel):
    prompt: str

def generate_response(prompt: str, max_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = model.generate(
        **inputs,
        max_length=max_tokens,
        pad_token_id=tokenizer.eos_token_id
    )
    # outputs = model.generate(
    #     inputs["input_ids"],
    #     max_new_tokens=150,  # tokens to generate after input
    #     do_sample=True,      # optional: enables randomness
    #     temperature=0.7,     # optional: controls creativity
    #     top_p=0.95,          # optional: nucleus sampling
    #     pad_token_id=tokenizer.eos_token_id  # prevent warning for no padding
    # )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Tokenize with truncation and attention mask
    # encoded = tokenizer(
    #     prompt,
    #     return_tensors="pt",
    #     truncation=True,
    #     max_length=512,
    #     padding=True
    # )
    # input_ids = encoded["input_ids"].to(device)
    # attention_mask = encoded["attention_mask"].to(device)

    # # Generate response
    # outputs = model.generate(
    #     input_ids=input_ids,
    #     attention_mask=attention_mask,
    #     max_new_tokens=max_tokens,
    #     do_sample=True,
    #     temperature=0.7,
    #     top_p=0.95,
    #     pad_token_id=tokenizer.eos_token_id
    # )
    
    # return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.post("/generate")
def generate_code(req: PromptRequest):
    result = generate_response(f"Write code for: {req.prompt}")
    return {"response": result}

@app.post("/debug")
def debug_code(req: PromptRequest):
    result = generate_response(f"Debug and explain: {req.prompt}")
    return {"response": result}
