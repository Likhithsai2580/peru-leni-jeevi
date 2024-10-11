import httpx
import asyncio
import json
import logging
import chardet
import os
# Configure logging
logging.basicConfig(level=logging.INFO)

cookies = os.environ.get("COOKIES")

headers = {
    'Host': 'www.whiterabbitneo.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0',
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br, zstd',
    'Referer': 'https://www.whiterabbitneo.com/',
    'Content-Type': 'application/json;charset=UTF-8',
    'Origin': 'https://www.whiterabbitneo.com',
    'DNT': '1',
    'Sec-GPC': '1',
    'Connection': 'keep-alive',
    'Cookie': cookies,
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
    'Priority': 'u=0',
    'TE': 'trailers'
}

async def fetch_data(url, params):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=params)
            response.raise_for_status()

            raw_content = response.content
            logging.info(f"Raw response (bytes): {raw_content}")

            try:
                json_response = response.json()
                logging.info("JSON response received.")
                return json_response
            except ValueError:
                logging.warning("Failed to decode as JSON; trying different encodings.")
                encodings_to_try = ['utf-8', 'ISO-8859-1', 'latin1', 'ascii', 'windows-1252']
                decoded_content = None

                for encoding in encodings_to_try:
                    try:
                        decoded_content = raw_content.decode(encoding)
                        logging.info(f"Decoded response using {encoding}: {decoded_content}")
                        break
                    except UnicodeDecodeError:
                        continue

                if decoded_content is None:
                    # Use chardet to detect encoding
                    detected_encoding = chardet.detect(raw_content)['encoding']
                    if detected_encoding:
                        try:
                            decoded_content = raw_content.decode(detected_encoding)
                            logging.info(f"Decoded response using chardet detected encoding {detected_encoding}: {decoded_content}")
                        except UnicodeDecodeError:
                            logging.error("Failed to decode response even with chardet.")
                            return None

                return decoded_content

        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP error: {e.response.status_code}, {e.response.text}")
            return None
        except httpx.RequestError as e:
            logging.error(f"Request error: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return None

async def uncensored_response(prompt, enhancePrompt=False, useFunctions=False):
    url = "https://www.whiterabbitneo.com/api/chat"
    params = {
        "messages": [{"id": "flTSP0c", "content": prompt, "role": "user"}],
        "id": "flTSP0c",
        "enhancePrompt": enhancePrompt,
        "useFunctions": useFunctions
    }
    result = await fetch_data(url, params)
    print(result)
    return result

if __name__ == "__main__":
    print(asyncio.run(uncensored_response("how to hack")))
