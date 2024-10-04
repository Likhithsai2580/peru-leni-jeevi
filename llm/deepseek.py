import requests
import json
import re
import os
from typing import Optional
from dotenv import load_dotenv; load_dotenv()

class DeepSeekAPI:
    """
    A class to interact with the DeepSeek API for initiating chat sessions.
    """

    def __init__(self, api_token: Optional[str] = os.environ.get("DEEPSEEK")):
        """
        Initializes the DeepSeekAPI with necessary authorization headers.

        Args:
            api_token (str): The Bearer token for API authorization.
        """
        self.auth_headers = {
            'Authorization': f'Bearer {api_token}'
        }
        self.api_base_url = 'https://chat.deepseek.com/api/v0/chat'
        self.api_session = requests.Session()
        self.api_session.headers.update(self.auth_headers)

    def clear_chat(self) -> None:
        """
        Clears the chat context by making a POST request to the clear_context endpoint.
        """
        clear_payload = {"model_class": "deepseek_chat", "append_welcome_message": False}
        clear_response = self.api_session.post(f'{self.api_base_url}/clear_context', json=clear_payload)
        clear_response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code

    def generate(self, user_message: str, response_temperature: float = 1.0, model_type: Optional[str] = "deepseek_chat", verbose: bool = False, system_prompt: Optional[str] = "Be Short & Concise") -> str:
        """
        Generates a response from the DeepSeek API based on the provided message.

        Args:
            user_message (str): The message to send to the chat API.
            response_temperature (float, optional): The creativity level of the response. Defaults to 1.0.
            model_type (str, optional): The model class to be used for the chat session.
            verbose (bool, optional): Whether to print the response content. Defaults to False.
            system_prompt (str, optional): The system prompt to be used. Defaults to "Be Short & Concise".

        Returns:
            str: The concatenated response content received from the API.

        Available models:
            - deepseek_chat
            - deepseek_code
        """
        request_payload = {
            "message": f"[Instructions: {system_prompt}]\n\nUser Query:{user_message}",
            "stream": True,
            "model_preference": None,
            "model_class": model_type,
            "temperature": response_temperature
        }
        api_response = self.api_session.post(f'{self.api_base_url}/completions', json=request_payload, stream=True)
        api_response.raise_for_status()

        combined_response = ""
        for chunk in api_response.iter_lines():
            if chunk:
                chunk_data = json.loads(chunk.decode('utf-8').split('data: ')[1])
                if 'choices' in chunk_data and chunk_data['choices']:
                    delta = chunk_data['choices'][0].get('delta', {})
                    if 'content' in delta:
                        content = delta['content']
                        combined_response += content
                        if not verbose:
                            print(content, end='', flush=True)
                    elif 'finish_reason' in delta:
                        break
        
        print()  # Add a newline after the response
        return combined_response.strip()

# Example usage
if __name__ == "__main__":

    api = DeepSeekAPI()
    while True:
        print("\nYou: ", end="", flush=True)
        user_query = input()

        if user_query == "/bye":
            api.clear_chat()
            break

        print("AI: ", end="", flush=True)
        api_response_content = api.generate(user_message=user_query, model_type='deepseek_code', verbose=False)
        # print("\n\n" + api_response_content)
