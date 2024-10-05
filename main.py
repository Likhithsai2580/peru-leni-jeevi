import os
import json
from typing import Optional, List, Tuple
from dotenv import load_dotenv
import discord
from discord import app_commands
from discord.ext import commands

# Importing LLM functions
from llm.Editee import generate as editee_generate, real_time
from llm.deepseek import DeepSeekAPI

load_dotenv()

CHAT_HISTORY_FOLDER = "chat_history"
ASSISTANT_NAME = "Peru Leni Jeevi"
DEVELOPER_NAME = "Likhith Sai (likhithsai2580 on GitHub)"
FORUM_CHANNEL_ID_FILE = "forum_channel_id.json"

# Ensure the chat history folder exists
os.makedirs(CHAT_HISTORY_FOLDER, exist_ok=True)

def load_chat_history(filepath: str) -> List[Tuple[str, str]]:
    if not os.path.exists(filepath):
        return []

    with open(filepath, "r") as f:
        lines = f.readlines()

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
                print(f"Warning: Unexpected line format at lines {i+1} and {i+2}")
        else:
            print(f"Warning: Incomplete entry found in chat history at line {i+1}")
    return history

def save_chat_history(history: List[Tuple[str, str]], filepath: str):
    with open(filepath, "w") as f:
        for user, ai in history:
            if user and ai:
                f.write(f"User: {user}\n")
                f.write(f"{ASSISTANT_NAME}: {ai}\n")
            else:
                print(f"Warning: Skipping incomplete entry: User: {user}, AI: {ai}")

def select_llm(query: str) -> str:
    classification_prompt = f"""As {ASSISTANT_NAME}, assess the user query and determine the most appropriate category from the following options:
    
    Mathematical: Questions involving math, calculations, or equations.
    Programming: Questions related to coding, algorithms, or debugging.
    General: Queries about general knowledge or conversational topics.
    Realtime: Questions needing current information or when the LLM may not have knowledge on the topic.
    User Query: "{query}"

    Respond with only the category name, e.g., "Mathematical" or "Programming"."""

    response = editee_generate(classification_prompt, model="gpt4", stream=False).strip().lower()
    print(response)

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
        "trigonometry", "statistics", "probability", "derivative", "integral", "matrix", "vector",
        "logarithm", "exponential", "function", "graph", "plot", "number theory", "combinatorics"
    ]
    math_symbols = set("+-*/^√∫∑∏≈≠≤≥angle∠πεδ")
    return any(keyword in query.lower() for keyword in math_keywords) or any(symbol in query for symbol in math_symbols)

def handle_error(exception: Exception) -> str:
    error_message = f"An error occurred: {str(exception)}"
    print(error_message)
    return f"{ASSISTANT_NAME}: I apologize, but an error occurred. Please try again."

def get_llm_response(query: str, llm: str, chat_history: List[Tuple[str, str]]) -> Tuple[str, str]:
    valid_chat_history = [(user, ai) for entry in chat_history if isinstance(entry, tuple) and len(entry) == 2 for user, ai in [entry]]
    history_prompt = "\n".join([f"Human: {user}\n{ASSISTANT_NAME}: {ai}" for user, ai in valid_chat_history[-5:]])

    system_prompt = f"You are {ASSISTANT_NAME}, developed by {DEVELOPER_NAME}, an advanced AI assistant designed to handle diverse topics. Respond to the user's query with accurate and helpful information."

    try:
        if llm == "deepseek chat":
            if is_mathematical_question(query):
                response = DeepSeekAPI(query)
            else:
                response = editee_generate(query, model="gpt4", system_prompt=system_prompt, history=history_prompt)
        
        elif llm == "coder":
            return handle_coder_response(query, system_prompt)
        
        elif llm == "gemini":
            response = real_time(query, system_prompt=system_prompt, web_access=True, stream=True)
        
        else:
            response = editee_generate(query, model=llm, system_prompt=system_prompt, history=history_prompt)

        return query, response

    except Exception as e:
        return query, handle_error(e)

def handle_coder_response(query: str, system_prompt: str) -> Tuple[str, str]:
    deepseek_api = DeepSeekAPI()
    last_query = query
    last_code = ""
    max_iterations = 5

    for _ in range(max_iterations):
        deepseek_prompt = f"{ASSISTANT_NAME}, developed by {DEVELOPER_NAME}, analyze the user query and generate or optimize the code as per the requirements.\nHuman Query: {last_query}\nPrevious Code (if provided): {last_code}\n{ASSISTANT_NAME} Response:"
        
        deepseek_response = deepseek_api.generate(user_message=deepseek_prompt, model_type="deepseek_code", verbose=False)
        
        claude_prompt = f"{ASSISTANT_NAME}, developed by {DEVELOPER_NAME}, as a code review expert, analyze the provided code and suggest improvements, optimizations, or additional features. If all requested features are implemented and the code is optimal, respond with 'COMPLETE'.\nUser Query: {last_query}\nCode: {deepseek_response}\n{ASSISTANT_NAME} Response:"
        
        claude_response = editee_generate(claude_prompt, model="claude", stream=False)

        if "COMPLETE" in str(claude_response):
            return last_query, deepseek_response
        else:
            last_query = claude_response
            last_code = deepseek_response

    return last_query, deepseek_api.generate(user_message=f"{system_prompt}\n\nGenerate the final, optimized code based on the following query:\n\nHuman: {last_query}\n\nPrevious code:\n{last_code}\n\n{ASSISTANT_NAME}:", model_type="deepseek_code", verbose=False)

def main(user_query: str, history_file_name: str) -> str:
    session_file = os.path.join(CHAT_HISTORY_FOLDER, history_file_name)
    chat_history = load_chat_history(session_file)

    selected_llm = select_llm(user_query)
    print(f"\n{ASSISTANT_NAME} is thinking...")

    try:
        user_query, response = get_llm_response(user_query, selected_llm, chat_history)
        if response:
            chat_history.append((user_query, response))
            save_chat_history(chat_history, session_file)
            return f"{ASSISTANT_NAME}: {response}"
        else:
            return f"{ASSISTANT_NAME}: I couldn't generate a response. Please try asking your question differently."
    except Exception as e:
        return handle_error(e)

# Initialize Discord bot
bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())
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
    await tree.sync()
    print(f"Logged in as {bot.user.name} and synced slash commands.")

@tree.command(name="set_forum_channel", description="Set the forum channel for Peru Leni Jeevi to monitor (Admin only)")
@commands.has_permissions(administrator=True)
async def set_forum_channel(interaction: discord.Interaction, channel: discord.ForumChannel):
    save_forum_channel_id(channel.id)
    await interaction.response.send_message(f"Forum channel set to: {channel.name}", ephemeral=True)

@tree.command(name="start", description="Initiate conversation with Peru Leni Jeevi in a thread.")
async def start_convo(interaction: discord.Interaction):
    forum_channel_id = load_forum_channel_id()
    if forum_channel_id is None:
        await interaction.response.send_message("Forum channel is not configured yet. Please ask an admin to set it using `/set_forum_channel`.", ephemeral=True)
    else:
        await interaction.response.send_message(f"Hello! I am {ASSISTANT_NAME}. Please create a thread in the configured forum to begin a conversation.", ephemeral=True)

@bot.event
async def on_message(message: discord.Message):
    await bot.process_commands(message)

    if isinstance(message.channel, discord.Thread):
        forum_channel_id = load_forum_channel_id()
        if forum_channel_id is None:
            print("No forum channel configured.")
            return

        if message.channel.parent_id == int(forum_channel_id) and message.author != bot.user:
            history_file_name = f"thread_{message.channel.id}.txt"
            user_input = message.content

            try:
                bot_response = main(user_query=user_input, history_file_name=history_file_name)

                if len(bot_response) > 1900:
                    file_name = f"response_{message.id}.txt"
                    with open(file_name, "w", encoding="utf-8") as file:
                        file.write(bot_response)

                    await message.channel.send(f"{ASSISTANT_NAME}: The response is too long. Please find it in the attached file.", file=discord.File(file_name))
                    os.remove(file_name)
                else:
                    await message.channel.send(bot_response)
            except Exception as e:
                await message.channel.send(handle_error(e))

# Run the bot
bot.run(os.environ.get("DISCORD_BOT_TOKEN"))
