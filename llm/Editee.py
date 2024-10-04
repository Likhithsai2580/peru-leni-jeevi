import requests
from typing import Union, Dict, Tuple, Optional
import re

def real_time(prompt: str, system_prompt: str = "Don't Write Code unless Mentioned", web_access: bool = True, stream: bool = True) -> str:
    """
    Generates a response for the given prompt using the Blackbox.ai API.

    Parameters:
    - prompt (str): The prompt to generate a response for.
    - system_prompt (str): The system prompt to be used in the conversation. Defaults to "Don't Write Code unless Mentioned".
    - web_access (bool): A flag indicating whether to access web resources during the conversation. Defaults to True.
    - stream (bool): A flag indicating whether to print the conversation messages. Defaults to True.

    Returns:
    - str: The complete response generated.
    """

    chat_endpoint = "https://www.blackbox.ai/api/chat"

    payload = {
        "messages": [{"content": system_prompt, "role": "system"}, {"content": prompt, "role": "user"}],
        "agentMode": {},
        "trendingAgentMode": {},
    }

    if web_access:
        payload["codeModelMode"] = web_access

    response = requests.post(chat_endpoint, json=payload, stream=True)
    print(response)

    full_response = ""

    for line in response.iter_lines(decode_unicode=True):
        if line:
            # Remove the prefix and suffix
            cleaned_line = re.sub(r'\$@\$.*?\$@\$', '', line)
            cleaned_line = re.sub(r'\$~~~\$.*?\$~~~\$', '', cleaned_line)
            cleaned_line = cleaned_line.strip()
            if cleaned_line:
                if stream:
                    print(cleaned_line, end='', flush=True)
                full_response += cleaned_line + "\n"

    return full_response.strip()

def generate(
    prompt: str,
    model: str = "mistrallarge",
    timeout: int = 30,
    proxies: Dict[str, str] = {},
    stream: bool = True,
    system_prompt: str = "Don't Write Code unless Mentioned",
    history: Optional[str] = None
) -> Union[str, None]:
    """
    Generates text based on the given prompt and model.

    Args:
    - prompt (str): The input prompt to generate text from.
    - model (str): The model to use for text generation. Defaults to "mistrallarge".
    - timeout (int): The timeout in seconds for the API request. Defaults to 30.
    - proxies (Dict[str, str]): A dictionary of proxies to use for the API request. Defaults to an empty dictionary.
    - stream (bool): Whether to stream the response or not. Defaults to True.
    - system_prompt (str): The system prompt to be used in the conversation. Defaults to "Don't Write Code unless Mentioned".
    - history (Optional[str]): The conversation history to be used for context. Defaults to None.

    Returns:
    - Union[str, None]: The generated text or None if an error occurs.
    """

    # Define the available models
    available_models = [
        "gemini",  # Gemini 1.5pro
        "claude",  # Claude 3.5
        "gpt4",  # GPT4o
        "mistrallarge"  # Mistral Large2
    ]

    # Check if the model is valid
    if model not in available_models:
        raise ValueError(f"Invalid model: {model}. Choose from: {available_models}")

    if model == "gemini":
        return real_time(prompt, system_prompt, True, stream)
    else:
        # Define the API endpoint and headers
        api_endpoint = "https://editee.com/submit/chatgptfree"
        headers = {
            "Authority": "editee.com",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Content-Type": "application/json",
            "Origin": "https://editee.com",
            "Referer": "https://editee.com/chat-gpt",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
            "X-Requested-With": "XMLHttpRequest"
        }

        # Update the payload to include the system_prompt
        payload = {
            "context": system_prompt,
            "selected_model": model,
            "template_id": "",
            "user_input": prompt
        }

        if history is not None:
            payload["history"] = history

        try:
            # Make the API request
            response = requests.post(
                api_endpoint,
                json=payload,
                headers=headers,
                timeout=timeout,
                proxies=proxies
            )

            # Check if the response was successful
            response.raise_for_status()

            # Get the response JSON
            resp = response.json()

            # Get the full response text
            full_response = resp.get('text', '')

            # Clean up the response
            cleaned_response = re.sub(r'\$@\$.*?\$@\$', '', full_response)
            cleaned_response = re.sub(r'\$~~~\$.*?\$~~~\$', '', cleaned_response)
            cleaned_response = cleaned_response.strip()

            # If streaming is enabled, print the response
            if stream:
                print(cleaned_response)

            # Return the cleaned response text
            return cleaned_response

        except requests.RequestException as e:
            # Print the error message
            print(f"Error occurred during API request: {e}")

            # If the response is not None, print the response content
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response content: {e.response.text}")

            # Return None
            return None

# Example usage:
if __name__ == '__main__':
    while True:
        user_input = input("Enter a prompt: ")
        response = real_time(prompt=user_input, stream=True)
        print("\n\nGenerated response:", response)