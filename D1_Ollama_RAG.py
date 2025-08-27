import ollama
import requests
from bs4 import BeautifulSoup

def get_page_content(url):
    """Downloads the content of a webpage from the given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error downloading page from {url}: {e}")
        return None

def extract_text_from_html(html_content):
    """Extracts readable text from HTML content."""
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, 'html.parser')
    # Remove script and style elements
    for script_or_style in soup(['script', 'style']):
        script_or_style.extract()
    text = soup.get_text()
    # Break into lines and remove leading/trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # Break multi-hyphenated words and remove empty lines
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # Drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text

model = 'gemma3' # You can change this to any Ollama model you have
client = ollama.Client()

while True:
    url = input("\nEnter a URL to analyze or type '/bye' to exit: ")
    if url == '/bye':
        break

    print(f"\nDownloading content from: {url}")
    html_content = get_page_content(url)

    if html_content:
        print("Extracting text from HTML...")
        page_text = extract_text_from_html(html_content)
        
        if not page_text.strip():
            print("No readable text found on the page.")
            continue

        # Limit the text length to avoid exceeding model context window
        max_text_length = 124000  # Adjust based on your model's context window
        if len(page_text) > max_text_length:
            print(f"Trimming text from {len(page_text)} to {max_text_length} characters.")
            page_text = page_text[:max_text_length]
        
        user_prompt = input("\nWhat would you like to ask about this page? (e.g., 'Summarize this page')\n> ")
        if not user_prompt.strip():
            print("No prompt provided. Please enter a URL and a prompt.")
            continue

        prompt = f"Based on the following text, please answer the question.\n\n--- TEXT ---\n{page_text}\n\n--- QUESTION ---\n{user_prompt}"

        print(f"\n--- Answering with {model} ---")
        try:
            # Using stream=True to get a streaming response
            stream = client.chat(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                stream=True,
            )

            # Print the streaming response
            for chunk in stream:
                print(chunk['message']['content'], end='', flush=True)
            print("\n\n--- End of Answer ---")
        except ollama.ResponseError as e:
            print(f"\nError from Ollama: {e.error}")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")