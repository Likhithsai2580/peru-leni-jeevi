# Peru Leni Jeevi Discord Bot

## Overview

Peru Leni Jeevi is a versatile Discord bot designed to assist with a variety of tasks, including mathematical queries, programming assistance, general knowledge questions, and real-time information retrieval. The bot leverages multiple language models to provide accurate and helpful responses.

## Features

- **Mathematical Assistance**: Handles mathematical queries using the DeepSeek API.
- **Programming Assistance**: Provides coding help and optimizations using DeepSeek and Editee APIs.
- **General Knowledge**: Answers general knowledge questions using GPT-4.
- **Real-Time Information**: Retrieves current information using the Gemini model.
- **Uncensored Responses**: Provides uncensored responses for specific queries using the WhiteRabbit API.
- **Periodic Summarization**: Summarizes chat history periodically to maintain context.
- **Forum Channel Monitoring**: Monitors a specified forum channel for user queries.

## Setup

### Prerequisites

- Python 3.8 or higher
- Discord bot token
- DeepSeek API token
- Editee API access
- WhiteRabbit API access

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Likhithsai2580/peru-leni-jeevi.git
   cd peru-leni-jeevi
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory and add your tokens:
   ```env
   DISCORD_BOT_TOKEN=your_discord_bot_token
   DEEPSEEK=your_deepseek_api_token
   COOKIES=your_whiterabbit_cookies
   ```

### Running the Bot

1. Start the bot:
   ```bash
   python main.py
   ```

2. The bot will log in and start monitoring the specified forum channel for user queries.

### Running the Website

1. Start the Flask server:
   ```bash
   python main.py
   ```

2. Open your web browser and navigate to `http://localhost:5000` to view the redesigned UI.

## Usage

- **Set Forum Channel**: Use the `/set_forum_channel` command to set the forum channel for the bot to monitor. This command requires administrator permissions.
- **Start Conversation**: Use the `/start` command to initiate a conversation with the bot in a thread within the specified forum channel.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [discord.py](https://github.com/Rapptz/discord.py)
- [aiohttp](https://github.com/aio-libs/aiohttp)
- [httpx](https://github.com/encode/httpx)
- [python-dotenv](https://github.com/theskumar/python-dotenv)
- [aiofiles](https://github.com/Tinche/aiofiles)
- [chardet](https://github.com/chardet/chardet)
