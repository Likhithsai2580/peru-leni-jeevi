import aiohttp
import asyncio
import json
import time
import os
from datetime import datetime
from typing import Optional
import traceback

class BlackboxAI:
    def __init__(self, session_id=None, csrf_token=None):
        self.base_url = "https://www.blackbox.ai"
        self.session_id = session_id
        self.csrf_token = csrf_token
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
        self.session = aiohttp.ClientSession(headers=self._get_headers())

    async def close(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()

    def _get_headers(self):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.5',
            'Content-Type': 'application/json',
            'Origin': 'https://www.blackbox.ai',
            'Referer': 'https://www.blackbox.ai/',
            'DNT': '1',
            'Sec-GPC': '1',
            'Connection': 'keep-alive'
        }
        
        if self.session_id and self.csrf_token:
            headers['Cookie'] = f'sessionId={self.session_id}; __Host-authjs.csrf-token={self.csrf_token}'
            
        return headers

    def _load_chat_history(self, chat_history_id: str) -> Optional[list]:
        history_file = os.path.join(self.history_dir, f"{chat_history_id}.json")
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
                return history.get('messages', [])
        return []

    def _save_chat_history(self, chat_history_id: str, prompt: str, response: str):
        history_file = os.path.join(self.history_dir, f"{chat_history_id}.json")
        
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = {
                'created_at': datetime.now().isoformat(),
                'messages': []
            }

        history['messages'].append({
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt,
            'response': response
        })

        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)

    async def chat(self, message: str, chat_history_id: Optional[str] = None, model="claude-sonnet-3.5"):
        # MODELS: blackboxai, claude-sonnet-3.5, gpt-4o
        if not chat_history_id:
            chat_history_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        endpoint = f"{self.base_url}/api/chat"
        message_id = f"msg_{int(time.time())}"
        
        payload = {
            "messages": [{
                "id": message_id,
                "content": message,
                "role": "user"
            }],
            "id": message_id,
            "previewToken": None,
            "userId": None,
            "codeModelMode": True,
            "agentMode": {},
            "trendingAgentMode": {},
            "isMicMode": False,
            "userSystemPrompt": None,
            "maxTokens": 1024,
            "playgroundTopP": 0.9,
            "playgroundTemperature": 0.5,
            "isChromeExt": False,
            "githubToken": None,
            "clickedAnswer2": False,
            "clickedAnswer3": False,
            "clickedForceWebSearch": False,
            "visitFromDelta": False,
            "mobileClient": False,
            "userSelectedModel": model,
            "validated": "69783381-2ce4-4dbd-ac78-35e9063feabc"
        }

        try:
            async with self.session.post(endpoint, json=payload) as response:
                if response.status == 200:
                    full_response = ""
                    async for line in response.content:
                        if line:
                            decoded = line.decode('utf-8')
                            # Skip the search results section
                            if decoded.startswith('$~~~$'):
                                continue
                            try:
                                # Try to parse as JSON first
                                json_response = json.loads(decoded)
                                if isinstance(json_response, dict):
                                    full_response += json_response.get('content', '')
                                else:
                                    full_response += str(json_response)
                            except json.JSONDecodeError:
                                # If not JSON, treat as plain text
                                full_response += decoded
                    
                    self._save_chat_history(chat_history_id, message, full_response)
                    return full_response.strip()
                else:
                    print(f"Request failed with status code: {response.status}")
                    print(f"Response text: {await response.text()}")
                    return None
                    
        except aiohttp.ClientTimeout:
            traceback.print_exc()
            print("Request timed out")
            return None
        except aiohttp.ClientError as e:
            traceback.print_exc()
            print(f"Error making request: {e}")
            return None

# Example usage
async def blackbox_api(message: str,model: str, chat_history_id: str = None, session_id: str = None, csrf_token: str = None):
    if not session_id or not csrf_token:
        raise ValueError("Session ID and CSRF token are required")
        
    async with BlackboxAI(session_id, csrf_token) as blackbox:
        response = await blackbox.chat(message, chat_history_id, model)
        return response

if __name__ == "__main__":
    async def main():
        bot = BlackboxAI(
            session_id="",# TODO: Add session ID
            csrf_token=""# TODO: Add CSRF token
        )
        async with bot:
            response = await bot.chat("how are you?","20241104_165000", model="claude-sonnet-3.5")
            response = response.strip()
            print(response)
    
    asyncio.run(main())
