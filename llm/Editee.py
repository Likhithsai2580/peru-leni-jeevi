from typing import Union, Dict, Tuple, Optional
import re
import asyncio
import aiohttp
from discord.ext import tasks

class Editee:
    def __init__(self):
        self.session = None

    async def initialize(self):
        self.session = aiohttp.ClientSession()

    async def close(self):
        if self.session:
            await self.session.close()

    async def edit_code(self, code):
        await asyncio.sleep(0.1)  # Reduced sleep time
        return f"# Edited code\n{code}"

async def real_time(prompt: str, system_prompt: str = "Don't Write Code unless Mentioned", web_access: bool = True, stream: bool = True) -> str:
    chat_endpoint = "https://www.blackbox.ai/api/chat"
    payload = {
        "messages": [{"content": system_prompt, "role": "system"}, {"content": prompt, "role": "user"}],
        "agentMode": {},
        "trendingAgentMode": {},
    }
    if web_access:
        payload["codeModelMode"] = web_access

    async with aiohttp.ClientSession() as session:
        async with session.post(chat_endpoint, json=payload) as response:
            full_response = ""
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if line:
                    cleaned_line = re.sub(r'\$@\$.*?\$@\$', '', line)
                    cleaned_line = re.sub(r'\$~~~\$.*?\$~~~\$', '', cleaned_line)
                    cleaned_line = cleaned_line.strip()
                    if cleaned_line:
                        if stream:
                            print(cleaned_line, end='', flush=True)
                        full_response += cleaned_line + "\n"
            return full_response.strip()

async def generate(
    prompt: str,
    model: str = "mistrallarge",
    timeout: int = 30,
    proxies: Dict[str, str] = {},
    stream: bool = True,
    system_prompt: str = "Don't Write Code unless Mentioned",
    history: Optional[str] = None
) -> Union[str, None]:
    available_models = ["gemini", "claude", "gpt4", "mistrallarge"]
    if model not in available_models:
        raise ValueError(f"Invalid model: {model}. Choose from: {available_models}")

    if model == "gemini":
        return await real_time(prompt, system_prompt, True, stream)
    else:
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
        payload = {
            "context": system_prompt,
            "selected_model": model,
            "template_id": "",
            "user_input": prompt
        }
        if history is not None:
            payload["history"] = history

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(api_endpoint, json=payload, headers=headers, timeout=timeout, proxy=proxies.get('http')) as response:
                    response.raise_for_status()
                    resp = await response.json()
                    full_response = resp.get('text', '')
                    cleaned_response = re.sub(r'\$@\$.*?\$@\$', '', full_response)
                    cleaned_response = re.sub(r'\$~~~\$.*?\$~~~\$', '', cleaned_response)
                    cleaned_response = cleaned_response.strip()
                    if stream:
                        print(cleaned_response)
                    return cleaned_response
        except aiohttp.ClientError as e:
            print(f"Error occurred during API request: {e}")
            return None

@tasks.loop(seconds=5.0)
async def maintain_connection(bot):
    try:
        await bot.change_presence(activity=bot.activity)
    except Exception as e:
        print(f"Error in maintain_connection: {e}")

# Example usage:
if __name__ == '__main__':
    async def main():
        editee = Editee()
        await editee.initialize()
        while True:
            user_input = input("Enter a prompt: ")
            response = await real_time(prompt=user_input, stream=True)
            print("\n\nGenerated response:", response)
        await editee.close()

    asyncio.run(main())
