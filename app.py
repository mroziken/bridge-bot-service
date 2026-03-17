import os

from bridge_bot_service.logging_config import configure_logging, get_logger
from bridge_bot_service.api import app

configure_logging()
logger = get_logger("bridge_bot_service.app")


# ------------------------------------------------------------------------------
# Local entrypoint
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    logger.info(
        "Starting uvicorn application",
        extra={"request_id": "-", "port": port},
    )
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
