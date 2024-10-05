import os
import json
import asyncio
from typing import Optional, List, Tuple
from dotenv import load_dotenv
import discord
from discord.ext import commands
from llm.Editee import generate as editee_generate, real_time
from llm.deepseek import DeepSeekAPI
import aiofiles
import logging
from discord.ext import tasks
load_dotenv()

CHAT_HISTORY_FOLDER = "chat_history"
CHAT_SUMMARY_FOLDER = "chat_summaries"
ASSISTANT_NAME = "Peru Leni Jeevi"
DEVELOPER_NAME = "Likhith Sai (likhithsai2580 on GitHub)"
FORUM_CHANNEL_ID_FILE = "forum_channel_id.json"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('discord')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')
handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(handler)

# Ensure the chat history folder exists
os.makedirs(CHAT_HISTORY_FOLDER, exist_ok=True)
os.makedirs(CHAT_SUMMARY_FOLDER, exist_ok=True)

# Create a singleton instance of DeepSeekAPI
deepseek_api = DeepSeekAPI()

# Summarization task
async def summarize_chat_history(filepath: str) -> str:
    history = await load_chat_history(filepath)
    if not history:
        return ""
    
    # Construct the conversation to summarize
    conversation = "\n".join([f"User: {user}\n{ASSISTANT_NAME}: {ai}" for user, ai in history])
    
    summarization_prompt = f"""As {ASSISTANT_NAME}, please provide a concise summary of the following conversation:

{conversation}

Summary:"""
    
    summary = await editee_generate(summarization_prompt, model="gpt4", stream=False)
    summary = summary.strip()
    logger.info(f"Summarization result for {filepath}: {summary}")
    return summary

async def load_chat_history(filepath: str) -> List[Tuple[str, str]]:
    if not os.path.exists(filepath):
        return []

    async with aiofiles.open(filepath, "r") as f:
        lines = await f.readlines()

    history = []
    for i in range(0, len(lines), 2):
        if i + 1 < len(lines):
            user_line = lines[i].strip()
            ai_line = lines[i + 1].strip()
            if user_line.startswith("User: ") and ai_line.startswith(f"{ASSISTANT_NAME}: "):
                user = user_line.replace("User: ", "")
                ai = ai_line.replace(f"{ASSISTANT_NAME}: ", "")
                history.append((user, ai))
            else:
                logger.warning(f"Unexpected line format at lines {i+1} and {i+2}")
        else:
            logger.warning(f"Incomplete entry found in chat history at line {i+1}")
    return history


async def save_chat_history(history: List[Tuple[str, str]], filepath: str):
    async with aiofiles.open(filepath, "w") as f:
        for user, ai in history:
            if user and ai:
                await f.write(f"User: {user}\n")
                await f.write(f"{ASSISTANT_NAME}: {ai}\n")
            else:
                logger.warning(f"Skipping incomplete entry: User: {user}, AI: {ai}")

async def append_summary_to_history(filepath: str, summary: str):
    summary_file = os.path.join(CHAT_SUMMARY_FOLDER, f"summary_{filepath}")
    async with aiofiles.open(summary_file, "w") as f:
        await f.write(f"Summary: {summary}\n")
    logger.info(f"Summary appended to {summary_file}")

@tasks.loop(hours=1)
async def periodic_summarization():
    for filename in os.listdir(CHAT_HISTORY_FOLDER):
        if filename.endswith(".txt"):
            filepath = os.path.join(CHAT_HISTORY_FOLDER, filename)
            summary = await summarize_chat_history(filepath)
            if summary:
                await append_summary_to_history(filepath, summary)
        
def handle_error(e: Exception) -> str:
    logger.error(f"Error occurred: {str(e)}", exc_info=True)
    return f"An error occurred: {str(e)}. Please try again."

async def select_llm(query: str) -> str:
    classification_prompt = f"""As {ASSISTANT_NAME}, assess the user query and determine the most appropriate category from the following options:

    Mathematical: Questions involving math, calculations, or equations.
    Programming: Questions related to coding, algorithms, or debugging.
    General: Queries about general knowledge or conversational topics.
    Realtime: Questions needing current information or when the LLM may not have knowledge on the topic.
    User Query: "{query}"

    Respond with only the category name, e.g., "Mathematical" or "Programming"."""

    response = await editee_generate(classification_prompt, model="gpt4", stream=False)
    response = response.strip().lower()
    logger.info(f"Classification response: {response}")
    if "mathematical" in response:
        return "deepseek chat"
    elif "programming" in response:
        return "coder"
    elif "realtime" in response:
        return "gemini"
    else:
        return "gpt4"


def is_mathematical_question(query: str) -> bool:
    math_keywords = [
        "calculate", "solve", "equation", "math", "arithmetic", "algebra", "geometry", "calculus",
        "trigonometry", "statistics", "probability", "derivative", "integral", "matrix", "vector"
    ]
    math_symbols = set("+-*/^√∫∑∏≈≠≤≥angle∠πεδ")
    return any(keyword in query.lower() for keyword in math_keywords) or any(symbol in query for symbol in math_symbols)

async def get_llm_response(query: str, llm: str, chat_history: List[Tuple[str, str]]) -> Tuple[str, str]:
    valid_chat_history = [(user, ai) for entry in chat_history if isinstance(entry, tuple) and len(entry) == 2 for user, ai in [entry]]
    # Load full or last N entries from history
    last_n = 10  # You can adjust this number based on your needs
    history_prompt = "\n".join([f"Human: {user}\n{ASSISTANT_NAME}: {ai}" for user, ai in valid_chat_history[-last_n:]])

    system_prompt = f"""
    You are {ASSISTANT_NAME}, developed by {DEVELOPER_NAME}, an advanced AI assistant designed to handle diverse topics.
    Below is the conversation history:
    {history_prompt}
    Respond to the user's current query with accurate and helpful information.
    """

    try:
        if llm == "deepseek chat":
            logger.info(f"Generating response for mathematical question: {is_mathematical_question(query)}")
            response = await deepseek_api.generate(user_message=system_prompt + f"\nUser: {query}\n{ASSISTANT_NAME}:", model_type="deepseek_chat")
        
        elif llm == "coder":
            logger.info("Handling coding response.")
            response = await handle_coder_response(query, chat_history)

        elif llm == "gemini":
            logger.info("Handling real-time response.")
            response = await real_time(query)

        else:
            logger.info("Generating general GPT-4 response.")
            response = await editee_generate(system_prompt + f"\nUser: {query}\n{ASSISTANT_NAME}:", model="gpt4", stream=False)

        return query, response

    except Exception as e:
        return query, await handle_error(e)

async def handle_coder_response(query: str, system_prompt: str) -> Tuple[str, str]:
    last_query = query
    last_code = ""
    max_iterations = 5

    for _ in range(max_iterations):
        try:
            deepseek_prompt = f"{ASSISTANT_NAME}, generate or optimize the code.\nHuman Query: {last_query}\nPrevious Code (if any): {last_code}\nResponse:"
            deepseek_response = await deepseek_api.generate(user_message=deepseek_prompt, model_type="deepseek_code")
            logger.info(f"DeepSeek response: {deepseek_response}")
            claude_prompt = f"Analyze the code and suggest improvements. Respond with 'COMPLETE' if optimal.\nUser Query: {last_query}\nCode: {deepseek_response}\nResponse:"
            claude_response = await editee_generate(claude_prompt, model="claude", stream=False)
            logger.info(f"Claude response: {claude_response}")
            if "complete" in claude_response.lower():
                return query, deepseek_response
            last_query = claude_response
            last_code = deepseek_response
        except Exception as e:
            logger.error(f"Error in handle_coder_response: {str(e)}")
            return query, f"An error occurred while processing your request: {str(e)}"

    try:
        final_response = await deepseek_api.generate(user_message=f"Generate the final, optimized code.\nQuery: {last_query}\nCode: {last_code}", model_type="deepseek_code")
        return query, final_response
    except Exception as e:
        logger.error(f"Error in handle_coder_response final generation: {str(e)}")
        return query, f"An error occurred while generating the final response: {str(e)}"

# Initialize Discord bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)
tree = bot.tree

def save_forum_channel_id(channel_id):
    data = {"forum_channel_id": channel_id}
    with open(FORUM_CHANNEL_ID_FILE, "w") as f:
        json.dump(data, f)

def load_forum_channel_id() -> Optional[int]:
    if not os.path.exists(FORUM_CHANNEL_ID_FILE):
        return None
    with open(FORUM_CHANNEL_ID_FILE, "r") as f:
        data = json.load(f)
    return data.get("forum_channel_id")

@bot.event
async def on_ready():
    await deepseek_api.initialize()
    await tree.sync()
    logger.info(f"Logged in as {bot.user.name} and synced slash commands.")
    keep_alive.start()
    periodic_summarization.start()

@tasks.loop(minutes=5)
async def keep_alive():
    logger.info("Keeping the bot alive...")


@tree.command(name="set_forum_channel", description="Set the forum channel for Peru Leni Jeevi to monitor (Admin only)")
@commands.has_permissions(administrator=True)
async def set_forum_channel(interaction: discord.Interaction, channel: discord.ForumChannel):
    save_forum_channel_id(channel.id)
    await interaction.response.send_message(f"Forum channel set to: {channel.name}", ephemeral=True)

@tree.command(name="start", description="Initiate conversation with Peru Leni Jeevi in a thread.")
async def start_convo(interaction: discord.Interaction):
    forum_channel_id = load_forum_channel_id()
    if forum_channel_id is None:
        await interaction.response.send_message("Forum channel is not configured yet.", ephemeral=True)
    else:
        await interaction.response.send_message(f"Hello! I am {ASSISTANT_NAME}. Please create a thread to start.", ephemeral=True)

@bot.event
async def on_message(message: discord.Message):
    await bot.process_commands(message)

    if isinstance(message.channel, discord.Thread):
        forum_channel_id = load_forum_channel_id()
        if forum_channel_id is None:
            logger.warning("No forum channel configured.")
            return

        if message.channel.parent_id == int(forum_channel_id) and message.author != bot.user:
            history_file_name = f"thread_{message.channel.id}.txt"
            user_input = message.content

            try:
                async with message.channel.typing():
                    bot_response = await main(user_input, history_file_name)

                if len(bot_response) > 1900:
                    file_name = f"response_{message.id}.txt"
                    async with aiofiles.open(file_name, "w", encoding="utf-8") as file:
                        await file.write(bot_response)
                    await message.channel.send(f"{ASSISTANT_NAME}: The response is too long. Find it in the attached file.", file=discord.File(file_name))
                    os.remove(file_name)
                else:
                    await message.channel.send(bot_response)
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}", exc_info=True)
                await message.channel.send(f"An error occurred: {str(e)}. Please try again.")

async def main(user_query: str, history_file_name: str) -> str:
    session_file = os.path.join(CHAT_HISTORY_FOLDER, history_file_name)
    chat_history = await load_chat_history(session_file)

    selected_llm = await select_llm(user_query)
    logger.info(f"{ASSISTANT_NAME} is thinking...")

    user_query, response = await get_llm_response(user_query, selected_llm, chat_history)
    chat_history.append((user_query, response))
    await save_chat_history(chat_history, session_file)
    return f"{ASSISTANT_NAME}: {response}"

@bot.event
async def on_disconnect():
    await deepseek_api.close()
    logger.info("DeepSeekAPI session closed.")

# Run the bot
bot.run(os.environ.get("DISCORD_BOT_TOKEN"))
