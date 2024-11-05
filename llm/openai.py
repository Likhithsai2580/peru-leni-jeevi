import json
import base64
from typing import List, Dict, Union, Optional
import aiohttp
import asyncio
import logging
from dotenv import load_dotenv
from llm.blackbox import blackbox_api
load_dotenv()
logger = logging.getLogger(__name__)

class OpenAI:
    """
    A client for interacting with the unofficial OpenAI-compatible API.
    
    Provides methods for chat completions, audio generation, and model listing.
    Uses aiohttp for async HTTP requests with optimized connection settings.
    """

    def __init__(self):
        """Initialize the OpenAI client with optimized connection settings."""
        self.base_url = "https://openai-devsdocode.vercel.app"
        self.session: Optional[aiohttp.ClientSession] = None
        self.timeout = aiohttp.ClientTimeout(
            total=60,
            connect=10,
            sock_read=30
        )
        self.connector = aiohttp.TCPConnector(
            limit=50,
            ttl_dns_cache=300,
            force_close=False,
            enable_cleanup_closed=True
        )
        
    async def __aenter__(self):
        """Async context manager entry point. Initializes the session."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit point. Closes the session."""
        await self.close()

    async def initialize(self):
        """Initialize the aiohttp session with optimized settings."""
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=self.timeout,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Connection": "keep-alive"
            },
            raise_for_status=False
        )

    async def close(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()
            
    async def list_models(self) -> Dict:
        """
        Get available models from the API.
        
        Returns:
            Dict: JSON response containing available models and their details
        """
        async with self.session.get(f"{self.base_url}/models") as response:
            return await response.json()
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4o",
        temperature: float = 0.5,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        top_p: float = 1,
        stream: bool = False,
        retry_attempts: int = 5,
        retry_delay: float = 2.0,
        chat_history_id: str = None,
        session_id: str = None,
        csrf_token: str = None
    ) -> Union[Dict, str]:
        """
        Generate chat completions with retries and error handling.
        
        Available models:
        - gpt-4o-mini-2024-07-18 (Created: 1715367049, Owner: DevsDoCode & Mr Leader)
        - gpt-4o-audio-preview-2024-10-01 (Created: 1715368132, Owner: DevsDoCode & Mr Leader)
        - gpt-4o-mini (Created: 1715367049, Owner: DevsDoCode & Mr Leader)
        - text-embedding-3-large (Created: 1705953180, Owner: DevsDoCode & Mr Leader)
        - gpt-4o-2024-05-13 (Created: 1715368132, Owner: DevsDoCode & Mr Leader)
        - text-embedding-3-small (Created: 1705948997, Owner: DevsDoCode & Mr Leader)
        - gpt-4o-audio-preview (Created: 1715368132, Owner: DevsDoCode & Mr Leader)
        - gpt-3.5-turbo-1106 (Created: 1698959748, Owner: DevsDoCode & Mr Leader)
        - gpt-3.5-turbo-0613 (Created: 1686587434, Owner: openai)
        - text-embedding-ada-002 (Created: 1671217299, Owner: openai-internal)
        - gpt-4 (Created: 1678604602, Owner: openai)
        - gpt-3.5-turbo (Created: 1677610602, Owner: openai)
        - gpt-3.5-turbo-0125 (Created: 1706048358, Owner: DevsDoCode & Mr Leader)
        - gpt-3.5-turbo-0301 (Created: 1677649963, Owner: openai)
        - gpt-4o (Created: 1715367049, Owner: DevsDoCode & Mr Leader)

        Args:
            messages: List of message dictionaries with role and content
            model: Model ID to use (default: gpt-4o-mini-2024-07-18)
            temperature: Controls randomness (0-1)
            presence_penalty: Penalty for new topics
            frequency_penalty: Penalty for repetition  
            top_p: Nuclear sampling parameter
            stream: Enable streaming responses
            retry_attempts: Number of retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            Union[Dict, str]: JSON response for non-streaming or generator for streaming
        """
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "top_p": top_p,
            "stream": stream
        }
        
        for attempt in range(retry_attempts):
            try:
                async with self.session.post(url, json=payload) as response:
                    if response.status == 200:
                        if stream:
                            return self._handle_streaming_response(response)
                        return await response.json()
                    
                    if response.status == 500:
                        error_text = await response.text()
                        logger.error(f"Server error: {error_text}")

                        return await blackbox_api(messages, "gpt-4o", chat_history_id, session_id, csrf_token)
                    elif response.status == 429:  # Rate limit
                        if attempt < retry_attempts - 1:
                            return await blackbox_api(messages, "gpt-4o", chat_history_id, session_id, csrf_token)
                    elif response.status == 404:
                        return await blackbox_api(messages, "gpt-4o", chat_history_id, session_id, csrf_token)
                    
                    return {
                        "error": f"Request failed with status {response.status}",
                        "details": await response.text()
                    }
                    
            except aiohttp.ClientError as e:
                logger.error(f"Request attempt {attempt + 1} failed: {str(e)}")
                if attempt < retry_attempts - 1:
                    delay = retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue
                return {"error": f"Network error: {str(e)}"}
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                return {"error": f"Unexpected error: {str(e)}"}
    
    async def audio_chat(
        self,
        messages: List[Dict[str, str]],
        voice: str = "fable",
        audio_format: str = "wav",
        temperature: float = 0.9
    ) -> Dict:
        """
        Generate both text and audio responses.
        
        Args:
            messages: List of message dictionaries with role and content
            voice: Voice ID to use (default: fable)
            audio_format: Audio format to generate (default: wav)
            temperature: Controls randomness (0-1)
            
        Returns:
            Dict: JSON response containing text and audio data
        """
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "messages": messages,
            "model": "gpt-4o-audio-preview",
            "modalities": ["text", "audio"],
            "audio": {
                "voice": voice,
                "format": audio_format
            },
            "temperature": temperature
        }
        
        try:
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_msg = f"Request failed with status {response.status}"
                    logger.error(error_msg)
                    return {"error": error_msg}
                    
        except aiohttp.ClientError as e:
            error_msg = f"Error making request: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    async def _handle_streaming_response(self, response):
        """
        Handle streaming responses with optimized buffering.
        
        Args:
            response: aiohttp response object
            
        Yields:
            str: Content chunks from the streaming response
        """
        try:
            buffer = ""
            async for line in response.content:
                if line:
                    try:
                        buffer += line.decode('utf-8')
                        if buffer.endswith('\n'):
                            lines = buffer.split('\n')
                            for line in lines[:-1]:
                                if line.startswith("data: "):
                                    try:
                                        content = json.loads(line[6:])
                                        if "choices" in content:
                                            yield content["choices"][0]["delta"].get("content", "")
                                    except json.JSONDecodeError:
                                        continue
                            buffer = lines[-1]
                    except UnicodeDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            yield f"Error during streaming: {str(e)}"
    
    async def save_audio_response(self, response_data: Dict, filename: str = "response.wav"):
        """
        Save audio response to a file.
        
        Args:
            response_data: JSON response containing audio data
            filename: Output filename (default: response.wav)
            
        Returns:
            bool: True if audio was saved successfully, False otherwise
        """
        if "choices" in response_data and response_data["choices"]:
            message = response_data["choices"][0].get("message", {})
            if "audio" in message and "data" in message["audio"]:
                wav_bytes = base64.b64decode(message["audio"]["data"])
                with open(filename, "wb") as f:
                    f.write(wav_bytes)
                return True
        return False

async def openai_api(messages: List[Dict[str, str]], stream: bool = False, chat_history_id: str = None, session_id: str = None, csrf_token: str = None):
    """
    High-level function to interact with the OpenAI API.
    
    Args:
        messages: List of message dictionaries with role and content
        stream: Enable streaming responses
        
    Returns:
        Union[Dict, str]: API response or streamed content
    """
    async with OpenAI() as client:
        response = await client.chat_completion(messages, stream=stream, chat_history_id=chat_history_id, session_id=session_id, csrf_token=csrf_token)
        if stream:
            full_response = ""
            async for chunk in response:
                full_response += chunk
                print(chunk, end="", flush=True)
            return full_response
        return response

async def openai_chat(prompt: str, session_id: str = None, csrf_token: str = None) -> str:
    try:
        # Convert prompt string to proper messages format
        messages = [{"role": "user", "content": prompt}]
        
        # Make API call with correct message format
        response = await openai_api(messages, chat_history_id=None, session_id=session_id, csrf_token=csrf_token)
        
        # Handle the response
        if isinstance(response, dict):
            if 'choices' in response and len(response['choices']) > 0:
                content = response['choices'][0]['message']['content']
                if content:  # Check if content is not None or empty
                    return content
            # Fall back to blackbox_api for any other case
            from llm.blackbox import blackbox_api
            return await blackbox_api(prompt, model="gpt-4o", chat_history_id=None, session_id=session_id, csrf_token=csrf_token)
        
        # If response is not a dict or is None, fall back to blackbox_api
        from llm.blackbox import blackbox_api
        return await blackbox_api(prompt, model="gpt-4o", chat_history_id=None, session_id=session_id, csrf_token=csrf_token)

    except Exception as e:
        logger.error(f"Error in openai_chat: {e}")
        # On any error, fall back to blackbox_api
        from llm.blackbox import blackbox_api
        return await blackbox_api(prompt, model="gpt-4o", chat_history_id=None, session_id=session_id, csrf_token=csrf_token)

if __name__ == "__main__":
    async def main():
        while True:
            prompt = input("You: ")
            if prompt.lower() in ['exit', 'quit']:
                break
            response = await openai_chat(prompt)
            print(f"Assistant: {response}\n")
            
    asyncio.run(main())