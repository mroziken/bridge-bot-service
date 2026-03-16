import os
from bridge_bot_service.api import app


# ------------------------------------------------------------------------------
# Local entrypoint
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
