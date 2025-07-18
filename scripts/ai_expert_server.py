# AI_expert_Server.py
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from transformers.utils import logging
    import torch
    import sys
except ImportError:
    print(
        "Optional AI expert dependencies were not found.\n"
        "If you want an AI expert feature, install:\n"
        "    pip install -r requirements_AI_expert.txt"
    )
def confirm_download(question: str) -> bool:
    """Ask the user a yes/no question. Return True for yes, False for no."""
    while True:
        ans = input(f"{question} (yes/no): ").strip().lower()
        if ans  == "yes":
            return True
        if ans == "no":
            return False
        print("Please enter 'yes' or 'no'.")



def model_exists_locally(model_name: str) -> bool:
    """Check if a model is already cached by the Transformers library."""
    # Hugging Face stores models in ~/.cache/huggingface/transformers/
    # We check if the config file is already present
    try:
        from pathlib import Path
        cache_dir = Path.home() / ".cache" / "huggingface" / "transformers"
        # Look for config.json for the given model
        return any(model_name.replace("/", "_") in f.stem for f in cache_dir.glob("*.json"))
    except Exception:
        return False

model_name = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"

# ❓ Ask for confirmation only if model not cached
if not model_exists_locally(model_name):
    if not confirm_download(f"Model '{model_name}' not found locally. Download 70 GB deepseek-ai model?"):
        print("Model download declined. Exiting.")
        sys.exit(0)
else:
    print(f"Model '{model_name}' already cached; skipping prompt.")

app = FastAPI(title="DeepSeek Coder V2 Lite-Instruct API")

quant_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Enable 4-bit quantization
            bnb_4bit_quant_type="nf4",  # Use normalized float 4
            bnb_4bit_compute_dtype=torch.bfloat16,  # Compute dtype for efficiency
            bnb_4bit_use_double_quant=True,  # Enable nested quantization for extra savings
        )

# Load tokenizer and quantized model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",  # Automatically offload to GPU/CPU
    trust_remote_code=True,
    torch_dtype=torch.bfloat16  # Match compute dtype
)

# === Request Schema ===
class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 1024
    do_sample: bool = False
    top_k: int = 50
    top_p: float = 0.95

@app.post("/generate")
def generate_code(req: PromptRequest):
    prompt_str = req.prompt
    try:
        # Sample prompt
        messages = [{'role': 'user', 'content': f"{prompt_str}"}]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(
            model.device)

        outputs = model.generate(inputs, max_new_tokens=req.max_new_tokens, do_sample=req.do_sample, top_k=req.top_k, top_p=req.top_p,
                                 num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
        return {"generated_code": tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, log_level="info")
