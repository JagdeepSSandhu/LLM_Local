import ollama

model1 = 'gemma3'
model2 = 'llama3.1'
model3 = 'gpt-oss'
model = model1

client = ollama.Client()
try:
    while True:
        prompt = input(f'\nAsk {model}?\n')
        if prompt == '/bye':
            break
        
        stream = client.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            stream=True
        )
        
        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
finally:
    # Explicitly close the client and its underlying connections
    client._client.close()
    print("\nOllama client connection closed.")
