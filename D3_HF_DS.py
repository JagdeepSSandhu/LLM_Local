# Load model directly
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

LLM = "deepseek-ai/DeepSeek-V3.1-Base"
#LLM = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
#LLM = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"


tokenizer = AutoTokenizer.from_pretrained(LLM, trust_remote_code=True)

# To speed up inference, we can make several optimizations:
# 1. torch_dtype=torch.bfloat16: Use bfloat16 for faster computation and less memory,
#    if your GPU supports it (Ampere or newer). Use torch.float16 for older GPUs.
# 2. attn_implementation="flash_attention_2": Use Flash Attention for a significant speed boost
#    on compatible hardware (Ampere or newer). Requires `flash-attn` to be installed.
# 3. device_map="auto" is good, but for single-GPU, device_map="cuda" is also fine.
model = AutoModelForCausalLM.from_pretrained(
    LLM,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    # attn_implementation="flash_attention_2" # Uncomment if flash-attn is installed and GPU is supported
)

# For PyTorch 2.0+, torch.compile can provide a speedup after a one-time compilation cost on the first run.
#model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

messages = [
    #{"role": "user", "content": "Who are you?"},
    {"role": "user", "content": "<think> \n Solve: what is 54*24? Show steps, final answer in \\boxed {}."},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=500)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))