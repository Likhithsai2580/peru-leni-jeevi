import os
import json
import asyncio
from typing import Optional, List, Tuple
from dotenv import load_dotenv
import discord
from discord.ext import commands
from llm.whiterabbit import uncensored_response
from llm.Editee import generate as editee_generate, real_time
from llm.deepseek import DeepSeekAPI
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

async def handle_error(e: Exception) -> str:
    logger.error(f"Error occurred: {str(e)}", exc_info=True)
    return f"An error occurred: {str(e)}. Please try again."

async def select_llm(query: str) -> str:
    classification_prompt = f"""As {ASSISTANT_NAME}, assess the user query and determine the most appropriate category from the following options:

    Mathematical: Questions involving math, calculations, or equations.
    Programming: Questions related to coding, algorithms, or debugging.
    General: Queries about general knowledge or conversational topics.
    Realtime: Questions needing current information or when the LLM may not have knowledge on the topic.
    Censored: If question is illegal return this classification including game hacks etc
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
    translator = Translator()
    try:
        translated = translator.translate(text, dest='fr')
        back_translated = translator.translate(translated.text, dest='en')
        return back_translated.text
    except Exception as e:
        logger.error(f"Error during back-translation: {e}")
        return text

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

async def get_llm_response(query: str, llm: str, chat_history: List[Tuple[str, str]]) -> Tuple[str, str]:
    generator = load_llm()
    if generator is None:
        return query, "Error loading LLM"

    try:
        response = generator(query, max_length=128, num_return_sequences=1)[0]['generated_text']
        return query, response
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return query, f"An error occurred: {str(e)}"

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
        response = asyncio.run(main(user_input, "web_chat.txt"))
        return jsonify({'response': response})
    return jsonify({'error': 'No message provided'})

# Run the bot and Flask app in separate threads
if __name__ == "__main__":
    # Load the LLM before starting the bot
    load_llm()
    flask_thread = threading.Thread(target=lambda: app.run(debug=True, port=5000))
    flask_thread.start()
    bot.run(os.environ.get("DISCORD_BOT_TOKEN"))
