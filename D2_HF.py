from transformers import pipeline
import torch

# Check if GPU is available
gpu_available = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if gpu_available else "No GPU found"

print(f"GPU Available: {gpu_available}")
print(f"GPU Name: {device_name}")

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline1 = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    #{"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "system", "content": "You are the wise professor who always uses big words and explain things with great detail."},
    {"role": "user", "content": "How many planets in the solar system?"},
]


outputs = pipeline1(
    messages,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.1,
    top_p=0.95
)
question = messages[-1]["content"]
print(f"\nQuestion to professor:{question}\n")

print(outputs[0]["generated_text"][-1]['content'])