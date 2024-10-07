import httpx
import asyncio
import os
import json
headers = json.load(os.environ.get("HEADERS"))

# Define your async function
async def response(message_content, message_id="flTSP0c", enhance_prompt=False, use_functions=False):
    params = {
        "messages": [{"id": message_id, "content": message_content, "role": "user"}],
        "id": message_id,
        "enhancePrompt": enhance_prompt,
        "useFunctions": use_functions
    }
    url = "https://www.whiterabbitneo.com/api/chat"
    async with httpx.AsyncClient(timeout=10.0) as client:  # Set timeout
        for attempt in range(3):  # Retry mechanism
            try:
                response = await client.post(url, headers=headers, json=params)
                response.raise_for_status()  # Raise an error for bad responses
                return response.json()  # Return the parsed JSON response
            except httpx.ReadTimeout:
                print(f"Timeout occurred, retrying... (Attempt {attempt + 1})")
                await asyncio.sleep(1)  # Wait before retrying
            except ValueError:
                return response.text
            except httpx.HTTPStatusError as e:
                return f"HTTP error: {e.response.status_code}, {e.response.text}"
    return "Failed to get a valid response after retries."

async def main(prompt):
    result = await response(prompt)
    print(result)

if __name__ == "__main__":
    asyncio.run(main("make an interesting game in python"))

