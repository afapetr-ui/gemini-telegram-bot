"""Telegram-бот, который отвечает через Gemini."""

from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import types
from telegram import Message, Update
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
    "без вводных вроде «конечно» и «отличный вопрос». Не используй Markdown-разметку. "
    "Если спрашивают новости, цены, погоду, события или факты, которые могли измениться, "
    "ищи в интернете. Не говори, что у тебя нет доступа в сеть: поиск у тебя есть. "
    "Источники указывай обычным текстом, без ссылок в разметке. "
    "Тебе могут прислать фото, документы, таблицы, голос и видео. Читай вложения, "
    "считай по ним, извлекай факты и выполняй задание из подписи. "
    "Если файла нет в этом сообщении, но он был раньше в разговоре — опирайся на него.",
).strip()

# Сколько последних реплик держим в памяти на чат (пары «вопрос-ответ» = 2 реплики).
HISTORY_LIMIT = 20
# Лимит Telegram на одно сообщение — 4096 символов.
TELEGRAM_MESSAGE_LIMIT = 4000
# Бесплатный инстанс Render — 512 МБ, Telegram отдаёт боту файлы до 20 МБ.
MAX_FILE_BYTES = 15 * 1024 * 1024
MAX_STORED_FILES = 5
MAX_EXTRACT_CHARS = 120_000
DEFAULT_FILE_PROMPT = (
    "Посмотри вложение. Если это задача — реши. Если документ или таблица — "
    "разбери и сделай нужные расчёты. Если фото — опиши и ответь, если видна задача."
)

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

SSML = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"

NATIVE_MIME_PREFIXES = ("image/", "audio/", "video/", "text/")
NATIVE_MIMES = {
    "application/pdf",
    "application/json",
    "application/xml",
    "text/csv",
    "text/plain",
    "text/html",
    "text/markdown",
    "text/xml",
    "text/rtf",
}
MIME_BY_EXT = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".heic": "image/heic",
    ".txt": "text/plain",
    ".md": "text/plain",
    ".csv": "text/csv",
    ".json": "application/json",
    ".html": "text/html",
    ".htm": "text/html",
    ".xml": "application/xml",
    ".rtf": "text/rtf",
    ".mp3": "audio/mp3",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".mp4": "video/mp4",
    ".mov": "video/mov",
    ".docx": DOCX_MIME,
    ".xlsx": XLSX_MIME,
    ".pptx": PPTX_MIME,
}

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("gemini-bot")

client = genai.Client(api_key=GEMINI_API_KEY)
generation_config = types.GenerateContentConfig(
    system_instruction=SYSTEM_INSTRUCTION,
    thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
    tools=[types.Tool(google_search=types.GoogleSearch())],
)
histories: Dict[int, List[types.Content]] = {}
# Последние вложения чата — чтобы уточнения без нового файла всё ещё видели документ.
stored_files: Dict[int, List["AttachedFile"]] = {}
album_messages: Dict[str, List[Message]] = {}
album_tasks: Dict[str, asyncio.Task] = {}
album_lock = asyncio.Lock()


class AttachedFile(NamedTuple):
    name: str
    mime: str
    data: bytes


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


def guess_mime(name: str, mime: str) -> str:
    if mime and mime != "application/octet-stream":
        return mime
    ext = os.path.splitext(name.lower())[1]
    return MIME_BY_EXT.get(ext, mime or "application/octet-stream")


def clip_text(text: str) -> str:
    if len(text) <= MAX_EXTRACT_CHARS:
        return text
    return text[:MAX_EXTRACT_CHARS] + "\n\n[текст обрезан, файл слишком длинный]"


def extract_docx_text(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        xml = archive.read("word/document.xml")
    xml = re.sub(rb"</w:p>", b"\n", xml)
    xml = re.sub(rb"<[^>]+>", b"", xml)
    return clip_text(xml.decode("utf-8", errors="replace").strip()) or "(пустой документ)"


def extract_xlsx_text(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        shared: List[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall(f"{SSML}si"):
                shared.append("".join(node.text or "" for node in item.iter(f"{SSML}t")))
        sheets = sorted(
            name
            for name in archive.namelist()
            if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
        )
        chunks = []
        for index, sheet_name in enumerate(sheets, start=1):
            root = ET.fromstring(archive.read(sheet_name))
            rows = []
            for row in root.iter(f"{SSML}row"):
                values = []
                for cell in row.findall(f"{SSML}c"):
                    value_el = cell.find(f"{SSML}v")
                    if value_el is None or value_el.text is None:
                        values.append("")
                        continue
                    raw = value_el.text
                    if cell.get("t") == "s":
                        try:
                            values.append(shared[int(raw)])
                        except (ValueError, IndexError):
                            values.append(raw)
                    else:
                        values.append(raw)
                if any(values):
                    rows.append("\t".join(values))
            if rows:
                chunks.append(f"Лист {index}\n" + "\n".join(rows))
    return clip_text("\n\n".join(chunks)) or "(пустая таблица)"


def extract_pptx_text(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        slides = sorted(
            name
            for name in archive.namelist()
            if name.startswith("ppt/slides/slide") and name.endswith(".xml")
        )
        chunks = []
        for index, slide_name in enumerate(slides, start=1):
            xml = archive.read(slide_name)
            xml = re.sub(rb"</a:p>", b"\n", xml)
            xml = re.sub(rb"<[^>]+>", b"", xml)
            text = xml.decode("utf-8", errors="replace").strip()
            if text:
                chunks.append(f"Слайд {index}\n{text}")
    return clip_text("\n\n".join(chunks)) or "(пустая презентация)"


def parts_for_file(attached: AttachedFile) -> Optional[List[types.Part]]:
    mime = guess_mime(attached.name, attached.mime)
    try:
        if mime == DOCX_MIME or attached.name.lower().endswith(".docx"):
            return [
                types.Part(text=f"Файл {attached.name} (Word):\n{extract_docx_text(attached.data)}")
            ]
        if mime == XLSX_MIME or attached.name.lower().endswith(".xlsx"):
            return [
                types.Part(text=f"Файл {attached.name} (таблица):\n{extract_xlsx_text(attached.data)}")
            ]
        if mime == PPTX_MIME or attached.name.lower().endswith(".pptx"):
            return [
                types.Part(
                    text=f"Файл {attached.name} (презентация):\n{extract_pptx_text(attached.data)}"
                )
            ]
    except Exception:
        log.exception("Не разобрал файл %s", attached.name)
        return None
    if mime in {"image/heic", "image/heif"}:
        return None
    if mime in NATIVE_MIMES or mime.startswith(NATIVE_MIME_PREFIXES):
        return [types.Part.from_bytes(data=attached.data, mime_type=mime)]
    return None


def remember_files(chat_id: int, new_files: List[AttachedFile]) -> None:
    stored = stored_files.setdefault(chat_id, [])
    stored.extend(new_files)
    del stored[:-MAX_STORED_FILES]
    total = 0
    kept: List[AttachedFile] = []
    for item in reversed(stored):
        if total + len(item.data) > MAX_FILE_BYTES:
            continue
        kept.append(item)
        total += len(item.data)
    stored_files[chat_id] = list(reversed(kept))


async def download_attachment(message: Message) -> Tuple[Optional[AttachedFile], Optional[str]]:
    file_id_source = None
    name = "file"
    mime = "application/octet-stream"
    size = 0

    if message.photo:
        file_id_source = message.photo[-1]
        name = f"photo_{file_id_source.file_unique_id}.jpg"
        mime = "image/jpeg"
        size = file_id_source.file_size or 0
    elif message.document:
        file_id_source = message.document
        name = file_id_source.file_name or f"document_{file_id_source.file_unique_id}"
        mime = file_id_source.mime_type or "application/octet-stream"
        size = file_id_source.file_size or 0
    elif message.voice:
        file_id_source = message.voice
        name = f"voice_{file_id_source.file_unique_id}.ogg"
        mime = file_id_source.mime_type or "audio/ogg"
        size = file_id_source.file_size or 0
    elif message.audio:
        file_id_source = message.audio
        name = file_id_source.file_name or f"audio_{file_id_source.file_unique_id}.mp3"
        mime = file_id_source.mime_type or "audio/mpeg"
        size = file_id_source.file_size or 0
    elif message.video:
        file_id_source = message.video
        name = file_id_source.file_name or f"video_{file_id_source.file_unique_id}.mp4"
        mime = file_id_source.mime_type or "video/mp4"
        size = file_id_source.file_size or 0
    elif message.video_note:
        file_id_source = message.video_note
        name = f"videonote_{file_id_source.file_unique_id}.mp4"
        mime = "video/mp4"
        size = file_id_source.file_size or 0

    if file_id_source is None:
        return None, None
    if size and size > MAX_FILE_BYTES:
        return None, "Файл слишком большой. Пришлите до 15 МБ."

    telegram_file = await file_id_source.get_file()
    data = bytes(await telegram_file.download_as_bytearray())
    if len(data) > MAX_FILE_BYTES:
        return None, "Файл слишком большой. Пришлите до 15 МБ."
    return AttachedFile(name=name, mime=guess_mime(name, mime), data=data), None


async def keep_typing(bot, chat_id: int) -> None:
    try:
        while True:
            await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        return


async def reply_with_gemini(
    message: Message,
    context: ContextTypes.DEFAULT_TYPE,
    user_text: str,
    new_files: List[AttachedFile],
) -> None:
    chat_id = message.chat_id
    if new_files:
        remember_files(chat_id, new_files)
    files_for_model = new_files or stored_files.get(chat_id, [])

    parts: List[types.Part] = []
    skipped: List[str] = []
    for attached in files_for_model:
        converted = parts_for_file(attached)
        if converted is None:
            skipped.append(attached.name)
            continue
        parts.extend(converted)

    prompt = user_text.strip() or (DEFAULT_FILE_PROMPT if files_for_model else "")
    if skipped:
        names = ", ".join(skipped)
        await message.reply_text(
            f"Не читаю этот тип файла: {names}. "
            "Подойдут фото, PDF, текст, CSV, Word, Excel, голос и видео."
        )
        if not parts and not prompt:
            return
    if not prompt and not parts:
        await message.reply_text("Напишите задание или прикрепите файл.")
        return
    if prompt:
        if files_for_model and not new_files:
            names = ", ".join(item.name for item in files_for_model)
            prompt = f"Вложения из прошлого сообщения ({names}).\n{prompt}"
        parts.append(types.Part(text=prompt))

    history = histories.setdefault(chat_id, [])
    history.append(types.Content(role="user", parts=parts))

    typing_task = asyncio.create_task(keep_typing(context.bot, chat_id))
    try:
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=history,
            config=generation_config,
        )
        answer = (response.text or "").strip()
    except Exception as error:
        log.exception("Ошибка запроса к Gemini")
        history.pop()
        if "RESOURCE_EXHAUSTED" in str(error) or "429" in str(error):
            await message.reply_text(
                "У Gemini закончилась бесплатная квота на сегодня. Завтра заработает снова."
            )
        else:
            await message.reply_text("Не получилось получить ответ от Gemini. Попробуй ещё раз.")
        return
    finally:
        typing_task.cancel()

    if not answer:
        history.pop()
        await message.reply_text("Gemini вернул пустой ответ. Попробуй переформулировать.")
        return

    # В историю кладём только текст: байты файлов и так лежат в stored_files.
    file_note = ""
    if files_for_model:
        file_note = "Файлы: " + ", ".join(item.name for item in files_for_model) + "\n"
    history[-1] = types.Content(
        role="user",
        parts=[types.Part(text=f"{file_note}{prompt}".strip())],
    )
    history.append(types.Content(role="model", parts=[types.Part(text=answer)]))
    del history[:-HISTORY_LIMIT]

    for chunk in split_message(answer):
        await message.reply_text(chunk)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Я отвечаю через Gemini. Напишите сообщение или прикрепите файл "
        "с заданием в подписи: фото, PDF, Word, Excel, текст, голос, видео.\n"
        "/reset — забыть разговор и вложения."
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    histories.pop(chat_id, None)
    stored_files.pop(chat_id, None)
    await update.message.reply_text("Контекст очищен.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await reply_with_gemini(update.message, context, update.message.text or "", [])


async def process_media_messages(
    messages: List[Message], context: ContextTypes.DEFAULT_TYPE
) -> None:
    first = messages[0]
    user_text = next((item.caption or item.text or "" for item in messages if item.caption or item.text), "")
    downloaded: List[AttachedFile] = []
    for item in messages:
        attached, error = await download_attachment(item)
        if error:
            await first.reply_text(error)
            return
        if attached is not None:
            downloaded.append(attached)
    await reply_with_gemini(first, context, user_text, downloaded)


async def flush_album(key: str, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        return
    async with album_lock:
        messages = album_messages.pop(key, [])
        album_tasks.pop(key, None)
    if messages:
        try:
            await process_media_messages(messages, context)
        except Exception:
            log.exception("Ошибка обработки вложения")
            await messages[0].reply_text("Не получилось прочитать файл. Попробуй ещё раз.")


async def handle_attachment(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    group_id = message.media_group_id
    if not group_id:
        try:
            await process_media_messages([message], context)
        except Exception:
            log.exception("Ошибка обработки вложения")
            await message.reply_text("Не получилось прочитать файл. Попробуй ещё раз.")
        return
    key = f"{message.chat_id}:{group_id}"
    async with album_lock:
        album_messages.setdefault(key, []).append(message)
        previous = album_tasks.get(key)
        if previous is not None:
            previous.cancel()
        album_tasks[key] = asyncio.create_task(flush_album(key, context))


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

    media = (
        filters.PHOTO
        | filters.Document.ALL
        | filters.AUDIO
        | filters.VOICE
        | filters.VIDEO
        | filters.VIDEO_NOTE
    )
    app = builder.build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(media, handle_attachment))
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
