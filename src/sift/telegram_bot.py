from __future__ import annotations

import logging
from collections import defaultdict, deque
from html import escape

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from sift.config import Preferences, Settings
from sift.digest import (
    get_backlog_count,
    run_digest,
    run_more,
)
from sift.llm import LLM
from sift.storage import connect, recent_articles, record_feedback

DigestItem = tuple[int, str]  # (item number in the digest, article id)
FEEDBACK_PREFIX = "fb"  # callback_data form: fb:<article_id>:<+1|-1>

log = logging.getLogger(__name__)

TG_MSG_LIMIT = 4096

CHAT_SYSTEM = """You are a personal news agent answering follow-up questions over
recent articles the user has been sent. Be concise; cite article titles inline."""


class Bot:
    def __init__(self, *, settings: Settings, prefs: Preferences, llm: LLM) -> None:
        self.settings = settings
        self.prefs = prefs
        self.llm = llm
        self._paused = False
        self._chat_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=10))
        self.app: Application = Application.builder().token(settings.telegram_bot_token).build()
        self._wire_handlers()

    def _wire_handlers(self) -> None:
        chat_ids = self.settings.chat_ids
        auth = filters.User(user_id=chat_ids)

        # Group -1 runs first: anyone *not* on the allowlist gets a polite reject
        # with their numeric chat id so the owner can whitelist them.
        self.app.add_handler(MessageHandler(~auth, self._unauthorized), group=-1)

        # Group 0 (default): authorized handlers.
        self.app.add_handler(CommandHandler("start", self._start, filters=auth))
        self.app.add_handler(CommandHandler("prefs", self._prefs, filters=auth))
        self.app.add_handler(CommandHandler("pause", self._pause, filters=auth))
        self.app.add_handler(CommandHandler("resume", self._resume, filters=auth))
        self.app.add_handler(CommandHandler("digest", self._digest, filters=auth))
        self.app.add_handler(CommandHandler("more", self._more, filters=auth))
        self.app.add_handler(CommandHandler("backlog", self._backlog, filters=auth))
        self.app.add_handler(CommandHandler("recent", self._recent, filters=auth))
        self.app.add_handler(MessageHandler(auth & filters.TEXT & ~filters.COMMAND, self._on_text))
        # Inline-keyboard callbacks from the per-article thumbs buttons attached
        # to digest messages. CallbackQuery has no `from_user` filter form, so we
        # gate inside the handler.
        self.app.add_handler(
            CallbackQueryHandler(self._on_feedback, pattern=rf"^{FEEDBACK_PREFIX}:")
        )

    async def send_message_safe(self, text: str, *, items: list[DigestItem] | None = None) -> None:
        """Broadcast text to every authorized chat. HTML first, plain fallback per chat.

        If `items` is provided, attach a thumbs keyboard to the final chunk and
        disable URL previews — digests carry many URLs and Telegram would
        otherwise render a huge preview card per chunk, breaking the visual flow
        with embedded Reddit/YouTube/GitHub panels mid-message. Earlier chunks
        get no keyboard so the buttons stay anchored to the bottom.
        """
        if self._paused:
            log.info("paused, dropping broadcast")
            return
        chunks = _chunk(text, TG_MSG_LIMIT)
        is_digest = items is not None
        markup = _build_thumbs_keyboard(items) if items else None
        for chat_id in self.settings.chat_ids:
            await self._send_to_chat(chat_id, chunks, markup, disable_preview=is_digest)

    async def _send_to_chat(
        self,
        chat_id: int,
        chunks: list[str],
        markup: InlineKeyboardMarkup | None = None,
        *,
        disable_preview: bool = False,
    ) -> None:
        try:
            for i, chunk in enumerate(chunks):
                await self.app.bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=disable_preview,
                    reply_markup=markup if i == len(chunks) - 1 else None,
                )
        except Exception:
            log.exception("HTML send to %d failed; retrying as plain text", chat_id)
            try:
                for i, chunk in enumerate(chunks):
                    await self.app.bot.send_message(
                        chat_id=chat_id,
                        text=_strip_html(chunk),
                        disable_web_page_preview=disable_preview,
                        reply_markup=markup if i == len(chunks) - 1 else None,
                    )
            except Exception:
                log.exception("plain-text fallback to %d also failed", chat_id)

    async def _unauthorized(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        chat_id = update.effective_chat.id if update.effective_chat else None
        log.info(
            "unauthorized request from user=%s chat_id=%s",
            getattr(user, "username", None) or getattr(user, "id", "?"),
            chat_id,
        )
        if update.message:
            await update.message.reply_text(
                f"You're not authorised to use this bot.\n\n"
                f"Send your chat id `{chat_id}` to the owner to be added.",
                parse_mode=ParseMode.MARKDOWN,
            )

    async def _start(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(
            f"News stream active in *digest mode*. Daily digest at {self.prefs.digest_time}.\n\n"
            "Commands:\n"
            "/digest — send today's digest now\n"
            "/more [N] — next N from backlog (default "
            f"{self.prefs.more_size})\n"
            "/backlog — count of articles waiting\n"
            "/prefs — current settings\n"
            "/pause /resume — toggle outbound messages\n"
            "/recent — last 10 sent\n\n"
            "Send any text to chat about recent articles.",
            parse_mode=ParseMode.MARKDOWN,
        )

    async def _prefs(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        enabled = [s.id for s in self.prefs.sources if s.enabled]
        backlog = get_backlog_count(self.settings, self.prefs)
        chat_ids = self.settings.chat_ids
        await update.message.reply_text(
            f"Threshold: {self.prefs.relevance_threshold}/10\n"
            f"Digest: {self.prefs.digest_size} items @ {self.prefs.digest_time}\n"
            f"Max scoring per source per poll: {self.prefs.max_per_cycle}\n"
            f"Backlog (≥ threshold, unsent): {backlog}\n"
            f"Sources ({len(enabled)}): {', '.join(enabled)}\n"
            f"Authorised chats ({len(chat_ids)}): {', '.join(str(c) for c in chat_ids)}\n\n"
            f"Edit preferences.yaml or AUTHORIZED_CHAT_IDS in .env, then restart."
        )

    async def _pause(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        self._paused = True
        await update.message.reply_text(
            "Paused — no outbound messages until /resume. Affects all users."
        )

    async def _resume(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        self._paused = False
        await update.message.reply_text("Resumed. Affects all users.")

    async def _digest(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text("Building digest…")
        n = await run_digest(self.settings, self.prefs, self.send_message_safe)
        if n == 0:
            return  # run_digest already sent the "nothing to send" notice
        log.info("/digest: sent %d articles", n)

    async def _more(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        import contextlib

        n = self.prefs.more_size
        parts = (update.message.text or "").split()
        if len(parts) > 1:
            with contextlib.suppress(ValueError):
                n = max(1, min(20, int(parts[1])))
        sent = await run_more(self.settings, self.prefs, self.send_message_safe, n)
        log.info("/more: sent %d articles", sent)

    async def _backlog(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        n = get_backlog_count(self.settings, self.prefs)
        await update.message.reply_text(
            f"{n} articles in backlog above threshold {self.prefs.relevance_threshold}/10."
        )

    async def _recent(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        with connect(self.settings.db_path) as conn:
            rows = recent_articles(conn, limit=10)
        if not rows:
            await update.message.reply_text("Nothing sent yet.")
            return
        lines = [
            f'• <a href="{escape(r["url"], quote=True)}">{escape(r["title"])}</a> '
            f"({r['source_id']}, {r['relevance_score']}/10)"
            for r in rows
        ]
        await update.message.reply_text(
            "\n".join(lines), parse_mode=ParseMode.HTML, disable_web_page_preview=True
        )

    async def _on_feedback(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle a thumbs button press from a digest message.

        callback_data shape: "fb:<article_id>:<+1|-1>". Authorisation is by user id
        because callback queries don't go through the message filters; an attacker
        replaying a callback they snooped from another chat would otherwise bypass
        the chat allowlist entirely."""
        query = update.callback_query
        if query is None:
            return
        user_id = query.from_user.id if query.from_user else None
        if user_id not in self.settings.chat_ids:
            log.info("rejecting feedback callback from non-allowlisted user %s", user_id)
            await query.answer("Not authorised.", show_alert=False)
            return
        parsed = _parse_feedback_callback(query.data or "")
        if parsed is None:
            await query.answer("Bad button data.", show_alert=False)
            return
        article_id, rating = parsed
        with connect(self.settings.db_path) as conn:
            record_feedback(conn, article_id, rating, None)
        # Acknowledge in-place. Telegram requires answering every callback query
        # within 15s or the loading spinner stays on the user's screen indefinitely.
        emoji = "👍" if rating > 0 else "👎"
        await query.answer(f"{emoji} recorded")

    async def _on_text(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        history = self._chat_history[chat_id]
        with connect(self.settings.db_path) as conn:
            rows = recent_articles(conn, limit=20)
        context_blob = "\n".join(
            f"- {r['title']} ({r['source_id']}): {r['summary'] or ''}" for r in rows
        )
        system = CHAT_SYSTEM + "\n\nRecent articles:\n" + context_blob
        reply = await self.llm.chat(system, list(history), update.message.text)
        history.append({"role": "user", "content": update.message.text})
        history.append({"role": "assistant", "content": reply})
        await update.message.reply_text(reply)


def _chunk(text: str, limit: int) -> list[str]:
    """Split text under `limit` chars per chunk, preferring blank-line breaks
    so HTML tags from a digest entry stay within one chunk."""
    if len(text) <= limit:
        return [text]
    out: list[str] = []
    while len(text) > limit:
        # Prefer a paragraph break (blank line) before the limit; fall back to
        # any newline; finally a hard cut.
        cut = text.rfind("\n\n", 0, limit)
        if cut <= 0:
            cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        out.append(text[:cut].rstrip())
        text = text[cut:].lstrip("\n")
    if text:
        out.append(text)
    return out


def _strip_html(text: str) -> str:
    """Crude HTML-to-text for the plain fallback path. Keeps href targets visible."""
    import re

    text = re.sub(r"<a[^>]*href=\"([^\"]*)\"[^>]*>(.*?)</a>", r"\2 (\1)", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text


def _build_thumbs_keyboard(items: list[DigestItem]) -> InlineKeyboardMarkup:
    """Compact 4-per-row layout: two items per row, each as [👍 N][👎 N]. Halves
    keyboard height vs one-item-per-row while keeping per-item granularity.
    Number labels match the digest's visible numbering. callback_data is bounded
    to ~64 bytes by Telegram; article_ids are 16-char hex so we're well under."""
    rows: list[list[InlineKeyboardButton]] = []
    for i in range(0, len(items), 2):
        row: list[InlineKeyboardButton] = []
        for n, aid in items[i : i + 2]:
            row.append(InlineKeyboardButton(f"👍 {n}", callback_data=f"{FEEDBACK_PREFIX}:{aid}:+1"))
            row.append(InlineKeyboardButton(f"👎 {n}", callback_data=f"{FEEDBACK_PREFIX}:{aid}:-1"))
        rows.append(row)
    return InlineKeyboardMarkup(rows)


def _parse_feedback_callback(data: str) -> tuple[str, int] | None:
    """Decode 'fb:<article_id>:<+1|-1>' → (article_id, rating). None on malformed input."""
    parts = data.split(":")
    if len(parts) != 3 or parts[0] != FEEDBACK_PREFIX:
        return None
    article_id = parts[1]
    if not article_id:
        return None
    if parts[2] == "+1":
        return article_id, 1
    if parts[2] == "-1":
        return article_id, -1
    return None
