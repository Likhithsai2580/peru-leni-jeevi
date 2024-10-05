import aiohttp
import json
import os
from typing import Optional
from dotenv import load_dotenv; load_dotenv()
import asyncio
import logging

logger = logging.getLogger(__name__)

class DeepSeekAPI:
    """
    A class to interact with the DeepSeek API for initiating chat sessions asynchronously.
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
        self.session = None

    async def initialize(self):
        """
        Initialize the aiohttp session.
        """
        self.session = aiohttp.ClientSession(headers=self.auth_headers)

    async def close(self):
        """
        Close the aiohttp session.
        """
        if self.session:
            await self.session.close()

    async def clear_chat(self) -> None:
        """
        Clears the chat context by making a POST request to the clear_context endpoint.
        """
        clear_payload = {"model_class": "deepseek_chat", "append_welcome_message": False}
        async with self.session.post(f'{self.api_base_url}/clear_context', json=clear_payload) as response:
            response.raise_for_status()

    async def generate(self, user_message: str, response_temperature: float = 1.0, model_type: Optional[str] = "deepseek_chat", verbose: bool = False, system_prompt: Optional[str] = "Be Short & Concise") -> str:
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

        combined_response = ""
        async with self.session.post(f'{self.api_base_url}/completions', json=request_payload) as response:
            response.raise_for_status()
            async for chunk in response.content:
                if chunk:
                    try:
                        decoded_chunk = chunk.decode('utf-8').strip()
                        if not decoded_chunk.startswith('data: '):
                            if verbose:
                                logger.debug(f"Unexpected chunk format: {decoded_chunk}")
                            continue  # Skip chunks that do not start with 'data: '

                        chunk_content = decoded_chunk.split('data: ', 1)[1]
                        if chunk_content == '[DONE]':
                            break

                        chunk_data = json.loads(chunk_content)
                        if 'choices' in chunk_data and chunk_data['choices']:
                            delta = chunk_data['choices'][0].get('delta', {})
                            if 'content' in delta:
                                content = delta['content']
                                combined_response += content
                                if not verbose:
                                    print(content, end='', flush=True)
                            elif 'finish_reason' in delta:
                                break
                    except (IndexError, json.JSONDecodeError) as e:
                        if verbose:
                            logger.error(f"Error parsing chunk: {e}")
                        continue  # Skip malformed chunks

        if verbose:
            print()  # Add a newline after the response
        return combined_response.strip()

# Example usage
async def main():
    api = DeepSeekAPI()
    await api.initialize()

    try:
        while True:
            print("\nYou: ", end="", flush=True)
            user_query = input()

            if user_query == "/bye":
                await api.clear_chat()
                break

            print("AI: ", end="", flush=True)
            api_response_content = await api.generate(user_message=user_query, model_type='deepseek_code', verbose=False)
            # print("\n\n" + api_response_content)
    finally:
        await api.close()

if __name__ == "__main__":
    asyncio.run(main())
