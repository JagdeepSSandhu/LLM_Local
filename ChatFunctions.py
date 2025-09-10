# This script demonstrates a chat interface that uses a local LLM (gemma3:1b via Ollama)
# for function calling with Pydantic validation and conversational argument gathering.

import json
from typing import Any, Dict, List
import ollama
from pydantic import BaseModel, ValidationError

# --- 1. Function and Pydantic Model Definitions ---
# Pydantic models are used to define the required arguments and validate them.

class GetWeatherRequest(BaseModel):
    """
    Get the current weather for a specific city.
    """
    city: str
    
    # You can add more validation here, e'g., regex for city names
    # @field_validator('city')
    # def validate_city(cls, v):
    #     if not v.isalpha():
    #         raise ValueError('City name must contain only letters')
    #     return v

class SendEmailRequest(BaseModel):
    """
    Send an email to a recipient with a specified subject and body.
    """
    recipient: str
    subject: str
    body: str

# Define the functions that the LLM can "call."
# The arguments for these functions must match the Pydantic models.

def get_weather(city: str) -> str:
    """
    Fetches mock weather data. In a real application, this would
    call a weather API.
    """
    # This is a mock implementation.
    return f"The weather in {city} is sunny with a chance of clouds."

def send_email(recipient: str, subject: str, body: str) -> str:
    """
    Sends a mock email.
    """
    # This is a mock implementation.
    return f"Successfully sent email to '{recipient}' with subject '{subject}'."

# --- 2. Tool Registry and LLM Prompting ---
# This dictionary acts as a registry for the available functions and their schemas.

AVAILABLE_TOOLS = {
    "get_weather": {
        "function": get_weather,
        "schema": GetWeatherRequest
    },
    "send_email": {
        "function": send_email,
        "schema": SendEmailRequest
    }
}

# The system prompt instructs the LLM on its role and how to use the tools.
# It provides the function schemas in a format the model can parse.
# The 'response_format' parameter will ensure the model returns JSON.
SYSTEM_PROMPT = """
You are a helpful assistant with access to the following tools.
Use the tools only when a user's request requires it.

When you need to use a tool, respond with a JSON object in the following format:
{{
  "tool_name": "name_of_the_tool",
  "tool_input": {{ "arg1": "value1", "arg2": "value2" }}
}}

If the user asks for something you cannot do with the available tools, or if a tool call is not necessary, respond naturally.

Available tools and their schemas:
- get_weather: {{
    "name": "get_weather",
    "description": "Get the current weather for a specific city.",
    "parameters": GetWeatherRequest.model_json_schema()
}}
- send_email: {{
    "name": "send_email",
    "description": "Send an email to a recipient with a specified subject and body.",
    "parameters": SendEmailRequest.model_json_schema()
}}
"""

# --- 3. Chat Loop and Logic ---
def main():
    """
    Main function to run the chat interface.
    """
    client = ollama.Client()
    model_name = "gemma3"
    
    # Check if the model is available
    try:
        client.show(model_name)
    except ollama.RequestError as e:
        print(f"Error: The model '{model_name}' could not be found. Please ensure Ollama is running and you have pulled the model with 'ollama pull {model_name}'.")
        return
        
    print(f"Chatbot powered by {model_name} is now active. Type 'exit' to quit.")
    print("Example prompts:")
    print("- What's the weather in London?")
    print("- Can you send an email to alice@example.com with the subject 'Meeting' and body 'Hello, let's meet tomorrow.'?")
    print("-" * 50)
    
    chat_history = []
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'bye' or user_input.lower() == 'exit':
            break

        if user_input.lower() == 'reset':
            chat_history = []
            print("Chat history has been reset.")
            continue
        
        chat_history.append({"role": "user", "content": user_input})
        
        try:
            # First, try to get a tool call from the model
            response = client.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *chat_history
                ],
                stream=False,
                format="json" # This is crucial for getting structured output
            )

            # Check if the model's response is a tool call
            try:
                model_response_str = response['message']['content']
                print(f"\nModel response: {model_response_str}\n")
                tool_call = json.loads(model_response_str)
            except json.JSONDecodeError:
                # If the response isn't valid JSON, it's likely a natural language response
                print(f"Bot: {response['message']['content']}")
                continue

            tool_name = tool_call.get("tool_name")
            tool_input = tool_call.get("tool_input")
            print(f"\ntool_name: {tool_name}, tool_input: {tool_input}\n")
            
            if tool_name and tool_input:
                if tool_name in AVAILABLE_TOOLS:
                    tool_info = AVAILABLE_TOOLS[tool_name]
                    pydantic_schema = tool_info["schema"]
                    
                    try:
                        # Validate the arguments using Pydantic
                        validated_input = pydantic_schema(**tool_input)
                        
                        # Call the function with the validated arguments
                        function_result = tool_info["function"](**validated_input.model_dump())
                        print(f"Bot: Calling '{tool_name}' with args {tool_input}")
                        
                        # Pass the function result back to the model for a natural language response
                        chat_history.append({
                            "role": "tool",
                            "content": function_result,
                            "name": tool_name
                        })

                        # Get the final response from the model
                        final_response = client.chat(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                *chat_history
                            ],
                            stream=False
                        )
                        print(f"Bot: {final_response['message']['content']}")
                        
                    except ValidationError as e:
                        # If validation fails, tell the user what's missing
                        missing_args = [f for f in e.errors()]
                        error_message = f"Validation failed for function '{tool_name}': The following arguments are missing or invalid: {missing_args}. Please provide the required information."
                        print(f"Bot: {error_message}")
                        
                        # Add the error message to the history so the model knows what to ask for
                        chat_history.append({
                            "role": "assistant",
                            "content": error_message
                        })

                else:
                    print(f"Bot: I was asked to use a tool I don't know: '{tool_name}'.")
                    
            else:
                # The response was JSON but not a tool call, handle it as a natural response.
                print(f"Bot: {model_response_str}")
        
        except ollama.RequestError as e:
            print(f"Bot: An error occurred with the Ollama API: {e}")
        except Exception as e:
            print(f"Bot: An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()