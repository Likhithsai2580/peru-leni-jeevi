import requests
import json
import os
from datetime import datetime
from typing import Optional, Tuple
import aiohttp
import asyncio
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class Deepseek:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://chat.deepseek.com/api/v0"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "x-app-version": "20241018.0"
        }
        self.history_dir = "chat_history"
        self.session: Optional[aiohttp.ClientSession] = None
        os.makedirs(self.history_dir, exist_ok=True)

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def initialize(self):
        """Initialize the aiohttp session."""
        self.session = aiohttp.ClientSession(headers=self.headers)

    async def close(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()

    async def _create_chat_session(self) -> str:
        create_session_url = f"{self.base_url}/chat_session/create"
        create_session_payload = {
            "code": 0,
            "msg": "",
            "data": {
                "biz_code": 0,
                "biz_msg": "",
                "biz_data": {
                    "id": "f3377b5b-de79-40de-809c-5892828f3b81",
                    "seq_id": 19,
                    "agent": "chat",
                    "title": None,
                    "version": 0,
                    "current_message_id": None,
                    "inserted_at": 1730696356.725656,
                    "updated_at": 1730696356.725656
                }
            }
        }

        async with self.session.post(create_session_url, json=create_session_payload) as response:
            session_data = await response.json()
            return session_data['data']['biz_data']['id']

    def _load_chat_history(self, chat_history_id: str) -> Tuple[Optional[str], Optional[str]]:
        history_file = os.path.join(self.history_dir, f"{chat_history_id}.json")
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
                messages = history.get('messages', [])
                # Return session_id and last message_id if available
                session_id = history.get('session_id')
                parent_message_id = messages[-1]['message_id'] if messages else None
                return session_id, parent_message_id
        return None, None

    def _save_chat_history(self, chat_history_id: str, prompt: str, response: str):
        history_file = os.path.join(self.history_dir, f"{chat_history_id}.json")
        
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = {
                'created_at': datetime.now().isoformat(),
                'session_id': None,  # Will be updated when session is created
                'messages': []
            }

        history['messages'].append({
            'timestamp': datetime.now().isoformat(),
            'message_id': None,  # This should be updated with the actual message_id
            'prompt': prompt,
            'response': response
        })

        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)

    async def chat(self, prompt: str, chat_history_id: Optional[str] = None) -> str:
        # If no chat_history_id provided, create a new one based on timestamp
        if not chat_history_id:
            chat_history_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Try to load existing session or create new one
        session_id, parent_message_id = self._load_chat_history(chat_history_id)
        if not session_id:
            session_id = await self._create_chat_session()

        # Send message
        chat_completion_url = f"{self.base_url}/chat/completion"
        chat_completion_payload = {
            "chat_session_id": session_id,
            "parent_message_id": parent_message_id,
            "prompt": prompt,
            "ref_file_ids": []
        }

        full_response = ""
        message_id = None

        async with self.session.post(chat_completion_url, json=chat_completion_payload) as response:
            async for line in response.content:
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        try:
                            json_data = json.loads(decoded_line[6:])
                            if 'message_id' in json_data:
                                message_id = json_data['message_id']
                            if 'choices' in json_data and json_data['choices']:
                                delta = json_data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    content = delta['content']
                                    full_response += content
                        except json.JSONDecodeError:
                            pass

        # Save chat history
        self._save_chat_history(chat_history_id, prompt, full_response)
        
        return full_response

async def deepseek_api(prompt: str, chat_history_id: str = None, api_key: str = None):
    if not api_key:
        raise ValueError("API key is required")
        
    async with Deepseek(api_key) as deepseek:
        response = await deepseek.chat(prompt, chat_history_id)
        return response

if __name__ == "__main__":
    # Example usage
    api_key = "" # TODO: Add API key
    chat_history_id = "test_conversation"
    
    async def main():
        while True:
            prompt = input("You: ")
            response = await deepseek_api(prompt, chat_history_id, api_key)
            print(f"Assistant: {response}\n")
            
    asyncio.run(main())