import os
from typing import Optional, List, Tuple
from dotenv import load_dotenv
from llm.Editee import generate as editee_generate, real_time
from llm.deepseek import DeepSeekAPI
import re
from datetime import datetime
import discord
from discord import app_commands
from discord.ext import commands
import json
from llm.Editee import real_time
load_dotenv()

CHAT_HISTORY_FOLDER = "chat_history"
ASSISTANT_NAME = "Peru Leni Jeevi"

# Ensure the chat history folder exists
if not os.path.exists(CHAT_HISTORY_FOLDER):
    os.makedirs(CHAT_HISTORY_FOLDER)

def load_chat_history(filepath: str) -> List[Tuple[str, str]]:
    if not os.path.exists(filepath):
        return []

    with open(filepath, "r") as f:
        lines = f.readlines()

    history = []
    for i in range(0, len(lines), 2):
        # Check if both user and AI lines are available
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
        for entry in history:
            if isinstance(entry, tuple) and len(entry) == 2:
                user, ai = entry
                if user and ai:  # Only save if both user and AI responses exist
                    f.write(f"User: {user}\n")
                    f.write(f"{ASSISTANT_NAME}: {ai}\n")
                else:
                    print(f"Warning: Skipping incomplete entry: User: {user}, AI: {ai}")
            else:
                print(f"Warning: Invalid entry in chat history: {entry}")

def select_llm(query: str) -> str:
    classification_prompt = f"""As {ASSISTANT_NAME}, assess the user query and determine the most appropriate category from the following options:

    Mathematical: Questions involving math, calculations, or equations.
    Programming: Questions related to coding, algorithms, or debugging.
    General: Queries about general knowledge or conversational topics.
    Realtime: Questions needing current information or when the LLM may not have knowledge on the topic.Including dates, time, or any other real-time information.
    User Query: "{query}"

    Respond with only the category name, e.g., "Mathematical" or "Programming"."""

    response = editee_generate(classification_prompt, model="gpt4", stream=False)
    category = str(response).strip().lower()
    print(category)

    if "mathematical" in category:
        return "deepseek chat"
    elif "programming" in category:
        return "coder"
    elif "realtime" in category:
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

def get_llm_response(query: str, llm: str, chat_history: List[Tuple[str, str]]) -> Tuple[str, str]:
    # Filter out invalid entries
    valid_chat_history = [
        (user, ai) for entry in chat_history if isinstance(entry, tuple) and len(entry) == 2
        for user, ai in [entry]
    ]

    history_prompt = "\n".join(
        [f"Human: {user}\n{ASSISTANT_NAME}: {ai}" for user, ai in valid_chat_history[-5:]]
    )

    system_prompt = f"You are {ASSISTANT_NAME}, an advanced AI assistant designed to handle diverse topics. Respond to the user's query with accurate and helpful information."

    if llm == "deepseek chat":
        if is_mathematical_question(query):
            response = DeepSeekAPI(query)
            return query, response
        else:
            response = editee_generate(
                query, model="gpt4", system_prompt=system_prompt, history=history_prompt
            )
            return query, response
    elif llm == "coder":
        deepseek_api = DeepSeekAPI()
        claude_api = lambda prompt: editee_generate(prompt, model="claude", stream=False)

        last_query = query
        last_code = ""
        max_iterations = 5

        for _ in range(max_iterations):
            deepseek_prompt = f"""{ASSISTANT_NAME}, analyze the user query and generate or optimize the code as per the requirements.
            \nHuman Query: {last_query}
            \nPrevious Code (if provided):{last_code}
            \n{ASSISTANT_NAME} Response:"""
            try:
                deepseek_response = deepseek_api.generate(user_message=deepseek_prompt, model_type="deepseek_code", verbose=False)

                claude_prompt = f"""{ASSISTANT_NAME}, as a code review expert, analyze the provided code and suggest improvements, optimizations, or additional features. If all requested features are implemented and the code is optimal, respond with "COMPLETE."
                \nUser Query: {last_query}
                \nCode:{deepseek_response}
                \n{ASSISTANT_NAME} Response:"""
                claude_response = claude_api(claude_prompt)

                if "COMPLETE" in str(claude_response):
                    response = deepseek_response
                    break
                else:
                    last_query = claude_response
                    last_code = deepseek_response
            except Exception as e:
                print(f"Ran into error {e}")
                break

        if 'response' not in locals():
            try:
                response = deepseek_api.generate(user_message=f"{system_prompt}\n\nGenerate the final, optimized code based on the following query:\n\nHuman: {last_query}\n\nPrevious code:\n{last_code}\n\n{ASSISTANT_NAME}:", model_type="deepseek_code", verbose=False)
            except Exception as e:
                response = f"I apologize, but an error occurred while generating the code. Error: {str(e)}"
    else:
        max_chunk_size = 7500  # Leave some room for the system prompt
        if len(full_prompt) > max_chunk_size:
            chunks = [full_prompt[i:i+max_chunk_size] for i in range(0, len(full_prompt), max_chunk_size)]
            response = ""
            for chunk in chunks:
                try:
                    chunk_response = editee_generate(chunk, model=llm, stream=False)
                    response += str(chunk_response) + " "
                except Exception as e:
                    print(f"Error occurred while processing chunk: {str(e)}. Skipping...")
        else:
            try:
                response = editee_generate(full_prompt, model=llm, stream=False)
            except Exception as e:
                response = f"I apologize, but an error occurred while generating the response. Error: {str(e)}"

    chat_history.append((query, str(response)))
    return query, response
    elif llm == "gemini":
        response = real_time(query, system_prompt=system_prompt, web_access=True, stream=True)
        return query, response
    else:
        response = editee_generate(
            query, model="gpt4", system_prompt=system_prompt, history=history_prompt
        )
        return query, response

def main(user_query, history_file_name):
    session_file = os.path.join(CHAT_HISTORY_FOLDER, history_file_name)

    # Load existing chat history
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
            return f"{ASSISTANT_NAME}: I apologize, but I couldn't generate a response. Please try asking your question in a different way."
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        return f"{ASSISTANT_NAME}: I apologize, but an error occurred while processing your request. Please try again. Error: {str(e)}"

# Initialize Discord bot
bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())
tree = bot.tree

FORUM_CHANNEL_ID_FILE = "forum_channel_id.json"

def save_forum_channel_id(channel_id):
    data = {"forum_channel_id": channel_id}
    with open(FORUM_CHANNEL_ID_FILE, "w") as f:
        json.dump(data, f)

def load_forum_channel_id():
    if not os.path.exists(FORUM_CHANNEL_ID_FILE):
        return None
    with open(FORUM_CHANNEL_ID_FILE, "r") as f:
        data = json.load(f)
    return data.get("forum_channel_id")

# Sync slash commands and handle bot startup
@bot.event
async def on_ready():
    await tree.sync()
    print(f"Logged in as {bot.user.name} and synced slash commands.")

# Slash command for admins to configure the forum channel
@tree.command(name="set_forum_channel", description="Set the forum channel for Peru Leni Jeevi to monitor (Admin only)")
@commands.has_permissions(administrator=True)
async def set_forum_channel(interaction: discord.Interaction, channel: discord.ForumChannel):
    save_forum_channel_id(channel.id)  # Save the forum channel ID
    await interaction.response.send_message(f"Forum channel set to: {channel.name}", ephemeral=True)

# Slash command to start conversation
@tree.command(name="start", description="Initiate conversation with Peru Leni Jeevi in a thread.")
async def start_convo(interaction: discord.Interaction):
    forum_channel_id = load_forum_channel_id()
    if forum_channel_id is None:
        await interaction.response.send_message("Forum channel is not configured yet. Please ask an admin to set it using `/set_forum_channel`.", ephemeral=True)
    else:
        await interaction.response.send_message(f"Hello! I am {ASSISTANT_NAME}. Please create a thread in the configured forum to begin a conversation.", ephemeral=True)

# Event: Monitor messages in the forum threads
@bot.event
async def on_message(message):
    await bot.process_commands(message)  # Ensure commands are processed

    if isinstance(message.channel, discord.Thread):
        forum_channel_id = load_forum_channel_id()
        if forum_channel_id is None:
            print("No forum channel configured.")
            return

        if message.channel.parent_id == int(forum_channel_id) and message.author != bot.user:
            # Use the thread's ID as the history file name
            history_file_name = f"thread_{message.channel.id}.txt"
            user_input = message.content

            try:
                bot_response = main(user_query=user_input, history_file_name=history_file_name)

                if len(bot_response) > 1900:
                    # Create a text file with the response
                    file_name = f"response_{message.id}.txt"
                    with open(file_name, "w", encoding="utf-8") as file:
                        file.write(bot_response)

                    # Send the response as a file attachment
                    await message.channel.send(f"{ASSISTANT_NAME}: The response is too long. Please find it in the attached file.", file=discord.File(file_name))

                    # Delete the temporary file
                    os.remove(file_name)
                else:
                    await message.channel.send(bot_response)
            except Exception as e:
                error_message = f"An error occurred while processing the message: {str(e)}"
                print(error_message)
                await message.channel.send(f"{ASSISTANT_NAME}: I apologize, but an error occurred. Please try again.")

# Run the bot
bot.run(os.environ.get("DISCORD_BOT_TOKEN"))
