import os

from collections.abc import Iterable

from dotenv import load_dotenv
from google.adk.agents.llm_agent import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from .logging_config import get_logger

load_dotenv()
logger = get_logger(__name__)

MODEL_NAME = os.getenv("ADK_MODEL", "gemini-2.5-flash")
APP_NAME = "bridge_bot_service"

HCP_VALUES = {
    "A": 4,
    "K": 3,
    "Q": 2,
    "J": 1,
}


def calculate_hcp(hand: list[str]) -> dict[str, object]:
    """Calculate bridge high-card points for a hand."""
    if not isinstance(hand, Iterable) or isinstance(hand, (str, bytes)):
        raise ValueError("hand must be a list of card strings like ['SA', 'HK']")

    normalized_hand: list[str] = []
    card_points: list[dict[str, int]] = []
    total_hcp = 0

    for card in hand:
        if not isinstance(card, str) or len(card) < 2:
            raise ValueError("each card must be a string like 'SA' or 'C10'")

        normalized_card = card.strip().upper()
        rank = normalized_card[1:]
        points = HCP_VALUES.get(rank, 0)
        total_hcp += points
        normalized_hand.append(normalized_card)
        card_points.append({"card": normalized_card, "hcp": points})

    return {
        "hand": normalized_hand,
        "hcp": total_hcp,
        "card_points": card_points,
    }

root_agent = Agent(
    model=MODEL_NAME,
    name="bridge_bot_agent",
    description="Chooses one legal bridge action from a provided game state.",
    instruction=(
        "You are a bridge bot. "
        "You receive the full request in the user message. "
        "If the request already includes HCP for the acting hand, use that value. "
        "If the request does not include HCP, call the calculate_hcp tool with state.private.hand before deciding. "
        "Use the tool result as part of your bridge evaluation. "
        "Return valid JSON only. "
        "Choose exactly one action from legalActions."
    ),
    tools=[calculate_hcp],
)

session_service = InMemorySessionService()
runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service,
)

logger.info(
    "ADK agent configured",
    extra={"request_id": "-", "model_name": MODEL_NAME, "app_name": APP_NAME},
)
