from __future__ import annotations

import asyncio
import logging

from sift.config import Settings, load_preferences
from sift.digest import digest_loop
from sift.llm import LLM
from sift.scheduler import run_scheduler
from sift.sources import build_sources
from sift.storage import init_db
from sift.telegram_bot import Bot

log = logging.getLogger(__name__)


async def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    settings = Settings()  # reads .env
    prefs = load_preferences(settings.preferences_path)
    init_db(settings.db_path)

    llm = LLM(
        base_url=settings.llm_base_url, api_key=settings.llm_api_key, model=settings.llm_model
    )
    bot = Bot(settings=settings, prefs=prefs, llm=llm)
    sources = build_sources(prefs)
    log.info("starting with %d source(s): %s", len(sources), [s.id for s in sources])

    # The PTB Application has its own startup/shutdown lifecycle. Run it manually
    # so the scheduler shares this process's event loop.
    await bot.app.initialize()
    await bot.app.start()
    await bot.app.updater.start_polling()
    try:
        await asyncio.gather(
            run_scheduler(sources, settings, prefs, llm),
            digest_loop(settings, prefs, llm, bot.send_message_safe),
        )
    finally:
        await bot.app.updater.stop()
        await bot.app.stop()
        await bot.app.shutdown()


def run() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    run()
