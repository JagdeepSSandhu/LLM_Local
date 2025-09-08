import torch
from transformers import pipeline

# 1. Initialize the pipeline
# We use the "text-generation" task because Gemma is a generative model.
# "google/gemma-2b-it" is an instruction-tuned version, great for Q&A.
pipe = pipeline(
    "text-generation",
    #model="google/gemma-2b-it",
    model="google/gemma-3-4b-it",
    ##model="unsloth/gemma-3-27b-it-qat",
    model_kwargs={"torch_dtype": torch.bfloat16}, # Use bfloat16 for less memory
    device="cuda", # Use "cuda" for GPU or "cpu" for CPU
)

# 2. Define your context and question
context = """
The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. 
It is named after the engineer Gustave Eiffel, whose company designed and built the tower. 
Constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair, it was initially 
criticized by some of France's leading artists and intellectuals for its design, but it has 
become a global cultural icon of France and one of the most recognizable structures in the world.
"""

print(context)

questions = ["Who is the president of France?",
             "Who was the Eiffel Tower named after?"]

# 3. Create a prompt for the model
# Gemma models respond well to a specific chat-like format. We create a prompt
# that instructs the model to answer the question based *only* on the provided text.
for question in questions:
    messages = [
        {"role": "user", "content": f"Answer the following question based only on the provided context.\n\nContext:\n{context}\n\nQuestion:\n{question}"},
    ]
    print(f"Question: {question}")


    # The apply_chat_template function formats the input correctly for the model.
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 4. Generate the answer
    outputs = pipe(
        prompt,
        max_new_tokens=256, # The maximum number of tokens (words/sub-words) to generate
        do_sample=True,
        temperature=0.1, # Lower temperature = more predictable, less "creative" output
        top_p=0.95,
    )

    # 5. Print the result
    # The output contains the original prompt, so we extract only the generated text.
    generated_text = outputs[0]["generated_text"]
    answer = generated_text[len(prompt):] # Get the text that comes after the prompt
    print(answer)
    print("-"*50)