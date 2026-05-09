from __future__ import annotations

import logging
from collections import defaultdict, deque
from html import escape

from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
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
    SuggestionFooter,
    get_backlog_count,
    run_digest,
    run_more,
)
from sift.llm import LLM
from sift.storage import (
    connect,
    mark_suggestion_surfaced,
    recent_articles,
    record_feedback,
    respond_to_suggestion,
)

DigestItem = tuple[int, str]  # (item number in the digest, article id)
FEEDBACK_PREFIX = "fb"  # callback_data forms: fb:<article_id>:<+1|-1> | fb:expand
EXPAND_CALLBACK = f"{FEEDBACK_PREFIX}:expand"
SUGGESTION_PREFIX = "sg"  # callback_data form: sg:<suggestion_id>:<add|decline|mute>
# Cap on stored digest→items state to prevent unbounded memory growth across
# many days of digests. Tapping Rate on an older digest just returns "expired".
DIGEST_MEMORY_CAP = 50

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
        # Maps Telegram message_id → digest items, so the "Rate items" expand
        # button can recover which articles to thumbs-button. Lives in memory
        # only — restart drops state and old Rate buttons answer "expired."
        self._digest_items: dict[int, list[DigestItem]] = {}
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
        # gate inside each handler.
        self.app.add_handler(
            CallbackQueryHandler(self._on_feedback, pattern=rf"^{FEEDBACK_PREFIX}:")
        )
        self.app.add_handler(
            CallbackQueryHandler(self._on_suggestion, pattern=rf"^{SUGGESTION_PREFIX}:")
        )

    async def register_commands(self) -> None:
        """Register the slash-command list with Telegram so clients show the
        inline `/` autocomplete popup. Without this, users have to type a
        command and hit send before the client surfaces the help list."""
        await self.app.bot.set_my_commands(
            [
                BotCommand("digest", "Send today's digest now"),
                BotCommand("more", "Next batch from backlog"),
                BotCommand("backlog", "Show unsent count"),
                BotCommand("recent", "Show last 10 sent"),
                BotCommand("prefs", "Show current settings"),
                BotCommand("pause", "Stop outbound messages"),
                BotCommand("resume", "Resume outbound messages"),
                BotCommand("start", "Help and welcome"),
            ]
        )

    async def send_message_safe(
        self,
        text: str,
        *,
        items: list[DigestItem] | None = None,
        per_chat_suggestions: dict[int, SuggestionFooter] | None = None,
    ) -> None:
        """Broadcast text to every authorized chat. HTML first, plain fallback per chat.

        If `items` is provided, the message is treated as a digest:
          - URL previews are disabled so Telegram doesn't embed a giant
            Reddit/YouTube/GitHub card per chunk
          - A single 'Rate items' button is attached to the final chunk;
            tapping it expands into a per-item thumbs keyboard
          - The items list is stashed against each broadcast's message_id so
            the expand callback can recover what to render
        Earlier chunks get no keyboard so the button stays anchored to the bottom.

        If `per_chat_suggestions` is provided, after the broadcast each chat with
        an entry gets a small follow-up message with the suggestion text and
        Add/Decline/Mute buttons. Chats without a queued suggestion stay
        single-message.
        """
        if self._paused:
            log.info("paused, dropping broadcast")
            return
        chunks = _chunk(text, TG_MSG_LIMIT)
        is_digest = items is not None
        markup = _build_collapsed_keyboard() if items else None
        for chat_id in self.settings.chat_ids:
            message_id = await self._send_to_chat(
                chat_id, chunks, markup, disable_preview=is_digest
            )
            if items and message_id is not None:
                self._digest_items[message_id] = items
        self._prune_digest_memory()
        for chat_id in self.settings.chat_ids:
            suggestion = (per_chat_suggestions or {}).get(chat_id)
            if suggestion is None:
                continue
            await self._send_suggestion_followup(chat_id, suggestion)

    async def _send_to_chat(
        self,
        chat_id: int,
        chunks: list[str],
        markup: InlineKeyboardMarkup | None = None,
        *,
        disable_preview: bool = False,
    ) -> int | None:
        """Send all chunks to one chat and return the message_id of the final
        chunk (the one that carries the keyboard). None if every send failed."""
        last_id: int | None = None
        try:
            for i, chunk in enumerate(chunks):
                sent = await self.app.bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=disable_preview,
                    reply_markup=markup if i == len(chunks) - 1 else None,
                )
                last_id = sent.message_id
            return last_id
        except Exception:
            log.exception("HTML send to %d failed; retrying as plain text", chat_id)
            try:
                last_id = None
                for i, chunk in enumerate(chunks):
                    sent = await self.app.bot.send_message(
                        chat_id=chat_id,
                        text=_strip_html(chunk),
                        disable_web_page_preview=disable_preview,
                        reply_markup=markup if i == len(chunks) - 1 else None,
                    )
                    last_id = sent.message_id
                return last_id
            except Exception:
                log.exception("plain-text fallback to %d also failed", chat_id)
                return None

    def _prune_digest_memory(self) -> None:
        """Cap _digest_items at DIGEST_MEMORY_CAP entries — drop oldest by
        message_id (Telegram message_ids are monotonic per chat but globally
        ordered enough for a coarse FIFO eviction)."""
        if len(self._digest_items) <= DIGEST_MEMORY_CAP:
            return
        for stale in sorted(self._digest_items)[:-DIGEST_MEMORY_CAP]:
            del self._digest_items[stale]

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
        """Handle the two callback shapes attached to digest messages:
          - 'fb:expand'                — swap the placeholder button for the
                                         per-item thumbs keyboard
          - 'fb:<article_id>:<+1|-1>' — record a thumbs vote for one article

        Authorisation is by user id because callback queries don't go through
        the message filters; an attacker replaying a callback they snooped
        from another chat would otherwise bypass the chat allowlist entirely.
        Telegram requires answering every callback query within 15s or the
        loading spinner stays on the user's screen indefinitely — so always
        answer, even on rejection."""
        query = update.callback_query
        if query is None:
            return
        user_id = query.from_user.id if query.from_user else None
        if user_id not in self.settings.chat_ids:
            log.info("rejecting feedback callback from non-allowlisted user %s", user_id)
            await query.answer("Not authorised.", show_alert=False)
            return
        if query.data == EXPAND_CALLBACK:
            await self._handle_expand(query)
            return
        parsed = _parse_feedback_callback(query.data or "")
        if parsed is None:
            await query.answer("Bad button data.", show_alert=False)
            return
        article_id, rating = parsed
        with connect(self.settings.db_path) as conn:
            record_feedback(conn, article_id, rating, None)
        emoji = "👍" if rating > 0 else "👎"
        await query.answer(f"{emoji} recorded")

    async def _send_suggestion_followup(self, chat_id: int, suggestion: SuggestionFooter) -> None:
        """Per-chat follow-up after the digest: 'we noticed you engage with X
        — want to add it to your interests?' with three buttons. Marks the
        suggestion as surfaced once Telegram accepts the send so we don't
        re-surface a suggestion that failed to deliver."""
        plural = "articles" if suggestion.evidence_count != 1 else "article"
        text = (
            f"💡 Suggested topic: <b>{escape(suggestion.topic)}</b>\n"
            f"You've engaged with {suggestion.evidence_count} related {plural} recently."
        )
        keyboard = _build_suggestion_keyboard(suggestion.suggestion_id)
        try:
            await self.app.bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
                reply_markup=keyboard,
            )
        except Exception:
            log.exception("suggestion follow-up to %d failed", chat_id)
            return
        with connect(self.settings.db_path) as conn:
            mark_suggestion_surfaced(conn, suggestion.suggestion_id)

    async def _on_suggestion(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle Add/Decline/Mute on a suggestion follow-up message.

        callback_data shape: 'sg:<suggestion_id>:<add|decline|mute>'.
        On Add: record 'added', append the topic to in-memory prefs.topics so
        the LLM uses it in the very next scoring call. The user is shown a
        copy-pasteable snippet for their preferences.yaml so the change
        survives a restart — we deliberately don't rewrite the file (it would
        clobber hand-written comments) and the recommender's 'added' state
        prevents re-suggesting the same topic next digest.
        On Decline: 30-day cooldown (enforced by the recommender query).
        On Mute: permanent block (enforced by last_response_for_topic check)."""
        query = update.callback_query
        if query is None:
            return
        user_id = query.from_user.id if query.from_user else None
        if user_id not in self.settings.chat_ids:
            log.info("rejecting suggestion callback from non-allowlisted user %s", user_id)
            await query.answer("Not authorised.", show_alert=False)
            return
        parsed = _parse_suggestion_callback(query.data or "")
        if parsed is None:
            await query.answer("Bad button data.", show_alert=False)
            return
        sid, action = parsed
        response = {"add": "added", "decline": "declined", "mute": "muted"}[action]
        with connect(self.settings.db_path) as conn:
            respond_to_suggestion(conn, sid, response)
            row = conn.execute(
                "SELECT topic FROM suggestion_candidates WHERE id = ?", (sid,)
            ).fetchone()
        topic = row["topic"] if row else "topic"
        if action == "add":
            # Mutate in-memory so the next score_relevance call sees it. The bullet
            # format mirrors the topics block style users hand-write in prefs.
            self.prefs.topics = f"{self.prefs.topics.rstrip()}\n  - {topic}"
            ack = (
                f"✓ Added <b>{escape(topic)}</b> to your interests for this session.\n\n"
                f"To make permanent, add this line under <code>topics:</code> "
                f"in <code>preferences.yaml</code>:\n"
                f"<code>  - {escape(topic)}</code>"
            )
        elif action == "decline":
            ack = f"OK — won't suggest <b>{escape(topic)}</b> again for ~30 days."
        else:  # mute
            ack = f"🔇 Muted — won't suggest <b>{escape(topic)}</b> again."
        try:
            await query.edit_message_text(text=ack, parse_mode=ParseMode.HTML)
        except Exception:
            log.exception("editing suggestion message after %s failed", action)
        await query.answer()

    async def _handle_expand(self, query) -> None:  # type: ignore[no-untyped-def]
        """Swap the 'Rate items' button for the full per-item keyboard. If we
        no longer have items for this message (bot restart, very old digest
        evicted by the memory cap), tell the user politely."""
        message_id = query.message.message_id if query.message else None
        items = self._digest_items.get(message_id) if message_id is not None else None
        if items is None:
            await query.answer(
                "This digest is no longer rateable — bot may have restarted.",
                show_alert=False,
            )
            return
        await query.edit_message_reply_markup(reply_markup=_build_thumbs_keyboard(items))
        await query.answer()

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


def _build_collapsed_keyboard() -> InlineKeyboardMarkup:
    """Single-button 'Rate items' placeholder shown by default under a digest.
    Tapping it fires fb:expand and the bot swaps in the full thumbs keyboard.
    Keeps the digest's resting state clean rather than dumping a wall of
    buttons under every message users may never engage with."""
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("👍 / 👎  Rate items", callback_data=EXPAND_CALLBACK)]]
    )


def _build_thumbs_keyboard(items: list[DigestItem]) -> InlineKeyboardMarkup:
    """Per-item rating layout shown after the user taps 'Rate items'.
    One row per article: [👍 N][👎 N]. Tall but pairs each button unambiguously
    to its article number — clearer than packing items 2-per-row."""
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(f"👍 {n}", callback_data=f"{FEEDBACK_PREFIX}:{aid}:+1"),
                InlineKeyboardButton(f"👎 {n}", callback_data=f"{FEEDBACK_PREFIX}:{aid}:-1"),
            ]
            for n, aid in items
        ]
    )


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


def _build_suggestion_keyboard(suggestion_id: int) -> InlineKeyboardMarkup:
    """Three-button row under a suggestion follow-up: Add / Decline / Mute.
    Compact single row — the message text already explains what's being suggested."""
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "➕ Add", callback_data=f"{SUGGESTION_PREFIX}:{suggestion_id}:add"
                ),
                InlineKeyboardButton(
                    "👎 No", callback_data=f"{SUGGESTION_PREFIX}:{suggestion_id}:decline"
                ),
                InlineKeyboardButton(
                    "🔇 Mute", callback_data=f"{SUGGESTION_PREFIX}:{suggestion_id}:mute"
                ),
            ]
        ]
    )


def _parse_suggestion_callback(data: str) -> tuple[int, str] | None:
    """Decode 'sg:<suggestion_id>:<add|decline|mute>' → (id, action). None on malformed."""
    parts = data.split(":")
    if len(parts) != 3 or parts[0] != SUGGESTION_PREFIX:
        return None
    try:
        sid = int(parts[1])
    except ValueError:
        return None
    if parts[2] not in ("add", "decline", "mute"):
        return None
    return sid, parts[2]
