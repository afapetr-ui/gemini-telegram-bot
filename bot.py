"""Telegram-бот, который отвечает через Gemini."""

import asyncio
import logging
import os
from typing import Dict, List, Set

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import types
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash").strip()
# -1 отдаёт объём «размышлений» на усмотрение модели, 0 отключает их полностью.
THINKING_BUDGET = int(os.environ.get("THINKING_BUDGET", "-1"))
SYSTEM_INSTRUCTION = os.environ.get(
    "SYSTEM_INSTRUCTION",
    "Ты полезный ассистент в Telegram. Отвечай по-русски, коротко и по делу, "
    "без вводных вроде «конечно» и «отличный вопрос». Не используй Markdown-разметку.",
).strip()

# Сколько последних реплик держим в памяти на чат (пары «вопрос-ответ» = 2 реплики).
HISTORY_LIMIT = 20
# Лимит Telegram на одно сообщение — 4096 символов.
TELEGRAM_MESSAGE_LIMIT = 4000

# Render подставляет публичный адрес сервиса и порт сам. Если их нет — работаем локально
# через long polling.
PUBLIC_URL = os.environ.get("RENDER_EXTERNAL_URL", "").strip().rstrip("/")
PORT = int(os.environ.get("PORT", "0"))
WEBHOOK_PATH = "telegram"
# Telegram шлёт этот токен в заголовке, чтобы никто чужой не подделал апдейт.
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "").strip()
# Бесплатный сервис Render засыпает после 15 минут без входящих запросов, поэтому будим
# его сами.
KEEPALIVE_INTERVAL = 600

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("gemini-bot")

client = genai.Client(api_key=GEMINI_API_KEY)
generation_config = types.GenerateContentConfig(
    system_instruction=SYSTEM_INSTRUCTION,
    thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
)
histories: Dict[int, List[types.Content]] = {}


def split_message(text: str) -> List[str]:
    chunks = []
    while len(text) > TELEGRAM_MESSAGE_LIMIT:
        split_at = text.rfind("\n", 0, TELEGRAM_MESSAGE_LIMIT)
        if split_at <= 0:
            split_at = TELEGRAM_MESSAGE_LIMIT
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip()
    chunks.append(text)
    return chunks


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Я отвечаю через Gemini. Просто напиши сообщение.\n"
        "/reset — забыть контекст разговора."
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    histories.pop(update.effective_chat.id, None)
    await update.message.reply_text("Контекст очищен.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_text = update.message.text

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    history = histories.setdefault(chat_id, [])
    history.append(types.Content(role="user", parts=[types.Part(text=user_text)]))

    try:
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=history,
            config=generation_config,
        )
        answer = (response.text or "").strip()
    except Exception:
        log.exception("Ошибка запроса к Gemini")
        history.pop()
        await update.message.reply_text("Не получилось получить ответ от Gemini. Попробуй ещё раз.")
        return

    if not answer:
        history.pop()
        await update.message.reply_text("Gemini вернул пустой ответ. Попробуй переформулировать.")
        return

    history.append(types.Content(role="model", parts=[types.Part(text=answer)]))
    del history[:-HISTORY_LIMIT]

    for chunk in split_message(answer):
        await update.message.reply_text(chunk)


async def ping_self() -> None:
    async with httpx.AsyncClient(timeout=30) as http:
        while True:
            await asyncio.sleep(KEEPALIVE_INTERVAL)
            try:
                await http.get(PUBLIC_URL)
            except Exception as error:
                log.warning("Не получилось разбудить себя: %s", error)


async def start_keepalive(app: Application) -> None:
    # Ссылку на задачу держим в множестве, иначе сборщик мусора убьёт её на первом же цикле.
    background: Set[asyncio.Task] = app.bot_data.setdefault("background", set())
    task = asyncio.create_task(ping_self())
    background.add(task)
    task.add_done_callback(background.discard)


def main() -> None:
    if not TELEGRAM_TOKEN:
        raise SystemExit("Нет TELEGRAM_BOT_TOKEN. Заполни .env по образцу .env.example")
    if not GEMINI_API_KEY:
        raise SystemExit("Нет GEMINI_API_KEY. Заполни .env по образцу .env.example")

    builder = Application.builder().token(TELEGRAM_TOKEN)
    if PUBLIC_URL and PORT:
        builder = builder.post_init(start_keepalive)

    app = builder.build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    log.info("Бот запущен, модель %s", GEMINI_MODEL)
    if PUBLIC_URL and PORT:
        app.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=WEBHOOK_PATH,
            webhook_url=f"{PUBLIC_URL}/{WEBHOOK_PATH}",
            secret_token=WEBHOOK_SECRET or None,
            allowed_updates=Update.ALL_TYPES,
        )
    else:
        app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
