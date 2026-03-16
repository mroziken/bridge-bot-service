import os

from dotenv import load_dotenv
from google.adk.agents.llm_agent import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

load_dotenv()

MODEL_NAME = os.getenv("ADK_MODEL", "gemini-2.5-flash")
APP_NAME = "bridge_bot_service"

root_agent = Agent(
    model=MODEL_NAME,
    name="bridge_bot_agent",
    description="Chooses one legal bridge action from a provided game state.",
    instruction=(
        "You are a bridge bot. "
        "You receive the full request in the user message. "
        "Return valid JSON only. "
        "Choose exactly one action from legalActions."
    ),
)

session_service = InMemorySessionService()
runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service,
)
