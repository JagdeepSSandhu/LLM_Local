import json
#import pyodbc
import ollama

# --- 1. Define SQL Operations as Tools ---
# These remain the same as the previous implementation.
SQL_OPERATIONS = {
    "get_customer_by_id": "SELECT * FROM Customers WHERE CustomerID = ?",
    "add_new_customer": "INSERT INTO Customers (FirstName, LastName, City) VALUES (?, ?, ?)",
    "update_customer_city": "UPDATE Customers SET City = ? WHERE CustomerID = ?",
    "delete_customer": "DELETE FROM Customers WHERE CustomerID = ?"
}

# The tool definitions for the model
tools = [
    {
        "name": "get_customer_by_id",
        "description": "Retrieves a customer's information from the database using their ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "integer",
                    "description": "The unique ID of the customer."
                }
            },
            "required": ["customer_id"]
        }
    },
    {
        "name": "add_new_customer",
        "description": "Adds a new customer to the database.",
        "parameters": {
            "type": "object",
            "properties": {
                "first_name": {"type": "string", "description": "The first name of the customer."},
                "last_name": {"type": "string", "description": "The last name of the customer."},
                "city": {"type": "string", "description": "The city of the customer."}
            },
            "required": ["first_name", "last_name", "city"]
        }
    },
    {
        "name": "update_customer_city",
        "description": "Updates the city of an existing customer.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "integer", "description": "The ID of the customer."},
                "new_city": {"type": "string", "description": "The new city for the customer."}
            },
            "required": ["customer_id", "new_city"]
        }
    },
    {
        "name": "delete_customer",
        "description": "Deletes a customer from the database using their ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "integer", "description": "The unique ID of the customer."}
            },
            "required": ["customer_id"]
        }
    }
]

# --- 2. The Main Refactored Function ---
def chat_to_sql_with_tools(user_query):
    """
    Handles the entire chat-to-SQL workflow using Ollama for tool calling.
    """
    # 1. Generate the prompt for tool use
    tool_descriptions = "\n".join([
        f"Tool Name: {t['name']}\nDescription: {t['description']}" for t in tools
    ])
    prompt = f"""
You are an AI assistant that can call a set of tools to interact with a SQL database.
Based on the user's request, identify the appropriate tool to call and the parameters for that tool.
Respond with a JSON object containing the tool call. Do not generate any other text.
Available Tools:
{tool_descriptions}

User Request: {user_query}
"""

    # 2. Get the model's response using Ollama
    # Note: Ensure the Ollama server is running and the model is downloaded.
    try:
        response = ollama.chat(
            model='gemma3:latest', # Use the model name you pulled
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.1} # Lower temperature for more predictable JSON output
        )
        
        # 3. Extract and parse the JSON response from the model
        raw_response = response['message']['content']
        print("--------------------------------------")
        print(raw_response)
        start = raw_response.find('{')
        end = raw_response.rfind('}') + 1
        raw_response = raw_response[start:end]
        print(raw_response)
        print("--------------------------------------")
        tool_call = json.loads(raw_response.replace('json',''))
        print(tool_call)
        tool_name = tool_call.get("tool_name")
        params = tool_call.get("parameters", {})
        
    except (json.JSONDecodeError, KeyError) as e:
        return f"Error: Could not parse the model's response. The model may not have returned a valid JSON object. Details: {e}"
    except Exception as e:
        return f"Ollama or network error: {e}"

    # 4. Find the SQL statement and execute it
    if tool_name in SQL_OPERATIONS:
        sql_query = SQL_OPERATIONS[tool_name]
        param_values = list(params.values()) if params else []
        return sql_query, param_values
        
        # Connection string for your SQL Server
        # IMPORTANT: Replace these with your actual database credentials
        conn_str = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER=YourServer;DATABASE=YourDatabase;UID=YourUser;PWD=YourPassword'
        
        # try:
        #     conn = pyodbc.connect(conn_str)
        #     cursor = conn.cursor()
            
        #     # The order of parameters must match the order in the SQL statement
        #     # This handles cases where a function has no parameters
        #     param_values = list(params.values()) if params else []
        #     cursor.execute(sql_query, param_values)
            
        #     # Check if it's a SELECT query to fetch results
        #     if sql_query.upper().strip().startswith("SELECT"):
        #         results = cursor.fetchall()
        #         return results
        #     else:
        #         conn.commit()
        #         return "Operation successful."
                
        # except pyodbc.Error as ex:
        #     return f"Database error: {ex}"
        # finally:
        #     if 'conn' in locals():
        #         conn.close()
    else:
        return f"Error: The model requested an unsupported tool: {tool_name}"

# --- 3. Example Usage Loop ---
if __name__ == "__main__":
    print("Welcome to the SQL Chatbot. Type 'exit' to quit.")
    while True:
        user_message = input("You: ")
        if user_message.lower() == "exit":
            break
        
        response = chat_to_sql_with_tools(user_message)
        print("Bot:", response)
