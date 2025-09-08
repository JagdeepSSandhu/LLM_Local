from gemma import gm
#import jax.random as jrandom

# Model and parameters
model = gm.nn.Gemma3_1B    
params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_1B_IT)

# Get the tokenizer for the model.
# The `gemma` library provides a path to the default sentencepiece model.
tokenizer = gm.tokenizer.Tokenizer(gm.ckpts.SPM_PATH)

# Define the sampler for generation
sampler = gm.sampler.Sampler(
    model=model,
    params=params,
    tokenizer=tokenizer,
)

# --- Interactive Chat Loop ---
print("--- Gemma 3 1B Interactive Chat ---")
print("Type '/bye' to exit.")

# The Gemma instruction-tuned models expect a specific format for conversation.
# We will build a list of turns, where each turn has a role ('user' or 'model')
# and content.
conversation = []

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == '/bye':
        print("Goodbye!")
        break

    # Add user's message to the conversation
    conversation.append({'role': 'user', 'content': user_input})

    # Generate a response from the model
    response = sampler(conversation, jrandom.PRNGKey(0))

    print(f"Gemma: {response.text}")

    # Add model's response to the conversation for context in the next turn
    conversation.append({'role': 'model', 'content': response.text})
