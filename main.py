import os
import json
import asyncio
from typing import Optional, List, Tuple
from dotenv import load_dotenv
import discord
from dotenv import load_dotenv
from discord.ext import commands
import aiofiles
import logging
from discord.ext import tasks
from flask import Flask, render_template, request, send_from_directory, jsonify
import threading
import zipfile
import shutil
import subprocess
import tensorflow as tf
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, pipeline
from datasets import Dataset
import nltk
from nltk.corpus import wordnet
import nlpaug.augmenter.word as naw
from googletrans import Translator
import torch
from datetime import datetime

# LLM APIs
from llm.deepseek import deepseek_api
from llm.openai import openai_chat
from llm.blackbox import blackbox_api
from llm.pentestgpt import pentestgpt_api
load_dotenv()

nltk.download('wordnet')

CHAT_HISTORY_FOLDER = "chat_history"
CHAT_SUMMARY_FOLDER = "chat_summaries"
ASSISTANT_NAME = "Peru Leni Jeevi"
DEVELOPER_NAME = "Likhith Sai (likhithsai2580 on GitHub)"
FORUM_CHANNEL_ID_FILE = "forum_channel_id.json"
FRONTEND_DIR = "frontend"
ZIP_FILE_NAME = "peru_leni_jeevi.zip"

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
os.makedirs(FRONTEND_DIR, exist_ok=True)


DEEPSEEK_API_KEY = os.getenv("DEEPSEEK")
PENTESTGPT_API_KEY = os.getenv("PENTESTGPT")
BLACKBOX_SESSION_ID = os.getenv("SESSION_ID")
BLACKBOX_CSRF_TOKEN = os.getenv("CSRF_TOKEN")
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

    summary = await openai_chat(summarization_prompt)
    summary = summary.strip()
    
    # Save summary in JSON format
    summary_file = os.path.join(CHAT_SUMMARY_FOLDER, f"summary_{filepath}.json")
    summary_data = {
        'timestamp': datetime.now().isoformat(),
        'summary': summary
    }
    async with aiofiles.open(summary_file, 'w') as f:
        await f.write(json.dumps(summary_data, indent=2))
        
    logger.info(f"Summarization result for {filepath}: {summary}")
    return summary

async def load_chat_history(filepath: str) -> List[Tuple[str, str]]:
    history_file = os.path.join(CHAT_HISTORY_FOLDER, f"{filepath}.json")
    if os.path.exists(history_file):
        async with aiofiles.open(history_file, 'r') as f:
            data = json.loads(await f.read())
            messages = data.get('messages', [])
            history = []
            for message in messages:
                if 'prompt' in message and 'response' in message:
                    history.append((message['prompt'], message['response']))
            return history
    return []

async def save_chat_history(history: List[Tuple[str, str]], filepath: str):
    history_file = os.path.join(CHAT_HISTORY_FOLDER, f"{filepath}.json")
    
    if os.path.exists(history_file):
        async with aiofiles.open(history_file, 'r') as f:
            data = json.loads(await f.read())
    else:
        data = {
            'created_at': datetime.now().isoformat(),
            'model': 'peru_leni_jeevi',
            'messages': []
        }

    for user, ai in history:
        if user and ai:
            data['messages'].append({
                'timestamp': datetime.now().isoformat(),
                'prompt': user,
                'response': ai
            })

    async with aiofiles.open(history_file, 'w') as f:
        await f.write(json.dumps(data, indent=2))

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

async def handle_error(e: Exception) -> str:
    logger.error(f"Error occurred: {str(e)}", exc_info=True)
    return f"An error occurred: {str(e)}. Please try again."

async def select_llm(query: str) -> str:
    classification_prompt = f"""Classify this query into exactly one category:
    1. Mathematical: Math, calculations, equations, numerical problems
    2. Programming: Code, debugging, algorithms, development
    3. Realtime: Current events, time, weather, live data
    4. Censored: Illegal activities, hacks, exploits
    5. General: Everything else

    Query: "{query}"
    Category:"""

    response = await openai_chat(classification_prompt)
    response = response.strip().lower()
    logger.info(f"Classification response: {response}")
    
    if "mathematical" in response:
        return "deepseek chat"
    elif "programming" in response:
        return "coder"
    elif "realtime" in response:
        return "gemini"
    elif "censored" in response:
        return "uncensored"
    else:
        return "gpt4"


def is_mathematical_question(query: str) -> bool:
    math_keywords = [
        "calculate", "solve", "equation", "math", "arithmetic", "algebra", "geometry", "calculus",
        "trigonometry", "statistics", "probability", "derivative", "integral", "matrix", "vector"
    ]
    math_symbols = set("+-*/^√∫∑∏≈≠≤≥angle∠πεδ")
    return any(keyword in query.lower() for keyword in math_keywords) or any(symbol in query for symbol in math_symbols)

def augment_data(text):
    aug = naw.SynonymAug(aug_src='wordnet')
    augmented_text = aug.augment(text)
    return augmented_text

def back_translate(text):
    try:
        from googletrans import Translator
        translator = Translator()
        # Translate to French and back to English
        fr_text = translator.translate(text, dest='fr').text
        back_translated = translator.translate(fr_text, dest='en').text
        return back_translated
    except Exception as e:
        logger.error(f"Error during back-translation: {e}")
        return text  # Return original text if translation fails

async def train_model(training_data: List[str]):
    logger.info("Training model with provided data...")

    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    def preprocess_function(examples):
        inputs = [f"train: {example}" for example in examples["text"]]
        model_inputs = tokenizer(inputs, max_length=128, padding="max_length", truncation=True)
        return model_inputs

    dataset = Dataset.from_dict({"text": training_data})
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    augmented_data = []
    for text in training_data:
        augmented_data.append(augment_data(text))
        augmented_data.append(back_translate(text))

    augmented_dataset = Dataset.from_dict({"text": augmented_data})
    augmented_tokenized_datasets = augmented_dataset.map(preprocess_function, batched=True)

    tokenized_datasets = tokenized_datasets.concatenate(augmented_tokenized_datasets)

    model = T5ForConditionalGeneration.from_pretrained('t5-base')

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=32,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        fp16=True, # Enable mixed precision training if your GPU supports it
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        eval_dataset=tokenized_datasets, # Placeholder for evaluation dataset
    )

    trainer.train()
    trainer.save_model("./trained_model")
    logger.info("Model training completed.")

def load_llm():
    try:
        logger.info("Loading pre-trained model...")
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        model = T5ForConditionalGeneration.from_pretrained("./trained_model")
        generator = pipeline('text2text-generation', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        return generator
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

async def get_llm_response(query: str, llm: str, chat_history: List[Tuple[str, str]], thread_id: str = None) -> Tuple[str, str]:
    try:
        system_prompt = f"""You are {ASSISTANT_NAME}, a helpful AI assistant created by {DEVELOPER_NAME}. 
        You provide accurate, concise responses while maintaining a friendly tone.
        For real-time queries, acknowledge the time-sensitive nature and suggest reliable sources.
        For programming questions, include code examples when relevant.
        For mathematical problems, show your work step-by-step.
        Never claim to be created by OpenAI or any other company."""

        full_prompt = f"{system_prompt}\n\nUser: {query}"
        
        response = None
        if llm == "deepseek chat":
            response = await deepseek_api(full_prompt, thread_id, DEEPSEEK_API_KEY)
        elif llm == "coder":
            deepseek_response = await deepseek_api(full_prompt, None, DEEPSEEK_API_KEY)
            for _ in range(5):
                claude_response = await blackbox_api(deepseek_response, "claude-sonnet-3.5", None, BLACKBOX_SESSION_ID, BLACKBOX_CSRF_TOKEN)
                deepseek_response = await deepseek_api(claude_response, None, DEEPSEEK_API_KEY)
            response = await deepseek_api(deepseek_response, thread_id, DEEPSEEK_API_KEY)
        elif llm == "gemini":
            response = await blackbox_api(full_prompt, "blackboxai", thread_id, BLACKBOX_SESSION_ID, BLACKBOX_CSRF_TOKEN)
        elif llm == "uncensored":
            response = await pentestgpt_api(full_prompt, PENTESTGPT_API_KEY)
        else:
            response = await openai_chat(full_prompt)
        
        # Handle different response types
        if isinstance(response, dict):
            if 'content' in response:
                response = response['content']
            elif 'choices' in response and len(response['choices']) > 0:
                response = response['choices'][0].get('message', {}).get('content', '')
            else:
                response = str(response)
                
        response = str(response).strip()
        return query, response
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        try:
            fallback_response = await openai_chat(f"{system_prompt}\n\nUser: {query}")
            if isinstance(fallback_response, dict):
                fallback_response = fallback_response.get('content', str(fallback_response))
            return query, str(fallback_response).strip()
        except Exception as e2:
            logger.error(f"Fallback also failed: {e2}")
            return query, f"I apologize, but I'm experiencing technical difficulties. Please try again later. Error: {str(e)}"

# Initialize Discord bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)
tree = bot.tree

async def save_forum_channel_id(guild_id: int, channel_id: int):
    config_file = os.path.join(CHAT_HISTORY_FOLDER, "forum_channels.json")
    
    if os.path.exists(config_file):
        async with aiofiles.open(config_file, 'r') as f:
            data = json.loads(await f.read())
    else:
        data = {}
    
    data[str(guild_id)] = channel_id
    
    async with aiofiles.open(config_file, 'w') as f:
        await f.write(json.dumps(data, indent=2))

def load_forum_channel_id(guild_id: int) -> Optional[int]:
    config_file = os.path.join(CHAT_HISTORY_FOLDER, "forum_channels.json")
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            data = json.load(f)
            return data.get(str(guild_id))
    return None

@bot.event
async def on_ready():
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
    try:
        await save_forum_channel_id(interaction.guild_id, channel.id)
        await interaction.response.send_message(
            f"Forum channel set to: {channel.name}",
            ephemeral=True
        )
    except Exception as e:
        logger.error(f"Error setting forum channel: {e}")
        await interaction.response.send_message(
            "Failed to set forum channel. Please try again.",
            ephemeral=True
        )

@tree.command(name="start", description="Initiate conversation with Peru Leni Jeevi in a thread.")
async def start_convo(interaction: discord.Interaction):
    try:
        forum_channel_id = load_forum_channel_id(interaction.guild_id)
        if forum_channel_id is None:
            await interaction.response.send_message(
                "Forum channel is not configured for this server. Ask an admin to use /set_forum_channel",
                ephemeral=True
            )
        else:
            forum_channel = interaction.guild.get_channel(forum_channel_id)
            if forum_channel is None:
                await interaction.response.send_message(
                    "Configured forum channel no longer exists. Ask an admin to use /set_forum_channel",
                    ephemeral=True
                )
            else:
                await interaction.response.send_message(
                    f"Hello! I am {ASSISTANT_NAME}. Please create a thread in {forum_channel.mention} to start.",
                    ephemeral=True
                )
    except Exception as e:
        logger.error(f"Error in start command: {e}")
        await interaction.response.send_message(
            "An error occurred. Please try again later.",
            ephemeral=True
        )

@tree.command(name="train", description="Train the LLM (Admin only)")
@commands.has_permissions(administrator=True)
async def train_llm(interaction: discord.Interaction):
    await interaction.response.send_message("Training the LLM... (This may take some time)", ephemeral=True)
    training_data = await load_training_data()
    await train_model(training_data)
    await interaction.followup.send("LLM training completed.", ephemeral=True)
    await zip_files_and_share(interaction)

async def load_training_data() -> List[str]:
    # Load training data from chat history
    training_data = []
    for filename in os.listdir(CHAT_HISTORY_FOLDER):
        if filename.endswith(".txt"):
            filepath = os.path.join(CHAT_HISTORY_FOLDER, filename)
            async with aiofiles.open(filepath, "r") as f:
                data = await f.read()
                training_data.append(data)
    return training_data

async def zip_files_and_share(interaction: discord.Interaction):
    try:
        with zipfile.ZipFile(ZIP_FILE_NAME, 'w') as zipf:
            zipf.write('./trained_model', 'trained_model')
            zipf.write('./results', 'results')
            zipf.write('./logs', 'logs')
        
        await interaction.followup.send(
            "Here's the trained model and associated files:",
            file=discord.File(ZIP_FILE_NAME)
        )
    finally:
        if os.path.exists(ZIP_FILE_NAME):
            os.remove(ZIP_FILE_NAME)

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    await bot.process_commands(message)

    if isinstance(message.channel, discord.Thread):
        try:
            forum_channel_id = load_forum_channel_id(message.guild.id)
            if forum_channel_id is None:
                return

            if message.channel.parent_id == forum_channel_id:
                history_file_name = f"guild_{message.guild.id}_thread_{message.channel.id}"
                user_input = message.content

                async with message.channel.typing():
                    bot_response = await process_query(user_input, history_file_name, message.guild.id)

                if len(bot_response) > 1900:
                    file_name = f"response_{message.id}.txt"
                    async with aiofiles.open(file_name, "w", encoding="utf-8") as file:
                        await file.write(bot_response)
                    await message.channel.send(
                        f"{ASSISTANT_NAME}: The response is too long. Find it in the attached file.",
                        file=discord.File(file_name)
                    )
                    os.remove(file_name)
                else:
                    await message.channel.send(bot_response)
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await message.channel.send(f"An error occurred: {str(e)}. Please try again.")

async def process_query(user_query: str, history_file_name: str, guild_id: int) -> str:
    try:
        # Include guild_id in the file name
        base_name = f"guild_{guild_id}_{history_file_name.replace('.txt', '')}"
        session_file = f"{base_name}.json"
        
        chat_history = await load_chat_history(session_file)
        thread_id = base_name.split('_')[3]  # Adjusted index for new naming format
        
        selected_llm = await select_llm(user_query)
        logger.info(f"{ASSISTANT_NAME} is thinking...")

        user_query, response = await get_llm_response(user_query, selected_llm, chat_history, thread_id)
        chat_history.append((user_query, response))
        await save_chat_history(chat_history, session_file)
        return f"{ASSISTANT_NAME}: {response}"
    except Exception as e:
        return await handle_error(e)

@bot.event
async def on_disconnect():
    await deepseek_api.close()
    logger.info("DeepSeekAPI session closed.")

# Flask app for website
app = Flask(__name__)

@app.route('/')
def index():
    if os.listdir(CHAT_HISTORY_FOLDER):
        return render_template('index.html')
    else:
        return render_template('showcase.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if user_input:
        response = asyncio.run(process_query(user_input, "web_chat.txt", 0))
        return jsonify({'response': response})
    return jsonify({'error': 'No message provided'})

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy'})

def run_flask():
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)

async def main():
    try:
        # Initialize bot and APIs
        load_llm()
        
        # Start Flask in a separate thread
        flask_thread = threading.Thread(target=run_flask)
        flask_thread.daemon = True
        flask_thread.start()
        
        # Run the bot with a timeout
        await asyncio.wait_for(
            bot.start(os.environ.get("DISCORD_BOT_TOKEN")),
            timeout=21300  # 5 hours 55 minutes (just under GitHub's 6-hour limit)
        )
    except asyncio.TimeoutError:
        logger.info("Bot session timed out - this is normal for GitHub Actions")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        await bot.close()
        logger.info("Bot shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())
