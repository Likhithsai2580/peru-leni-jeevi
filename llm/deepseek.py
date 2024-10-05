import aiohttp
import json
import os
from typing import Optional
from dotenv import load_dotenv
import asyncio
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class DeepSeekAPI:
    """A class to interact with the DeepSeek API for asynchronous chat sessions."""

    def __init__(self, api_token: Optional[str] = os.environ.get("DEEPSEEK")):
        self.auth_headers = {'Authorization': f'Bearer {api_token}'}
        self.api_base_url = 'https://chat.deepseek.com/api/v0/chat'
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def initialize(self):
        """Initialize the aiohttp session."""
        self.session = aiohttp.ClientSession(headers=self.auth_headers)

    async def close(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()

    async def clear_chat(self) -> None:
        """Clear the chat context."""
        clear_payload = {"model_class": "deepseek_chat", "append_welcome_message": False}
        async with self.session.post(f'{self.api_base_url}/clear_context', json=clear_payload) as response:
            response.raise_for_status()

    async def generate(self, user_message: str, response_temperature: float = 1.0, 
                       model_type: str = "deepseek_chat", verbose: bool = False, 
                       system_prompt: str = "Be Short & Concise") -> str:
        """Generate a response from the DeepSeek API."""
        request_payload = {
            "message": f"[Instructions: {system_prompt}]\n\nUser Query:{user_message}",
            "stream": True,
            "model_preference": None,
            "model_class": model_type,
            "temperature": response_temperature
        }

        combined_response = []
        try:
            async with self.session.post(f'{self.api_base_url}/completions', json=request_payload) as response:
                response.raise_for_status()
                async for chunk in response.content:
                    if chunk:
                        decoded_chunk = chunk.decode('utf-8').strip()
                        if not decoded_chunk.startswith('data: '):
                            if verbose:
                                logger.debug(f"Unexpected chunk format: {decoded_chunk}")
                            continue

                        chunk_content = decoded_chunk.removeprefix('data: ')
                        if chunk_content == '[DONE]':
                            break

                        try:
                            chunk_data = json.loads(chunk_content)
                            delta = chunk_data['choices'][0].get('delta', {})
                            if 'content' in delta:
                                content = delta['content']
                                combined_response.append(content)
                                if verbose:
                                    print(content, end='', flush=True)
                            elif 'finish_reason' in delta:
                                break
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to decode JSON from chunk: {chunk}")
                        except Exception as e:
                            logger.error(f"Error processing chunk: {e}")
        except aiohttp.ClientResponseError as e:
            logger.error(f"HTTP error occurred: {e.status} {e.message}")
            raise
        except aiohttp.ClientError as e:
            logger.error(f"Network error occurred: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}")
            raise

        if verbose:
            print()  # Add a newline after the response
        return ''.join(combined_response).strip()

async def main():
    async with DeepSeekAPI() as api:
        while True:
            user_query = input("\nYou: ")
            if user_query.lower() == "/bye":
                await api.clear_chat()
                break
            print("AI: ", end="", flush=True)
            api_response_content = await api.generate(user_message=user_query, model_type='deepseek_code', verbose=False)
            print(api_response_content)

if __name__ == "__main__":
    asyncio.run(main())
