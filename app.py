import json
import os
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from google.adk.agents.llm_agent import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv()

# ------------------------------------------------------------------------------
# Enums / DTOs
# ------------------------------------------------------------------------------

class Seat(str, Enum):
    N = "N"
    E = "E"
    S = "S"
    W = "W"


class Partnership(str, Enum):
    NS = "NS"
    EW = "EW"


class Vulnerability(str, Enum):
    NONE = "NONE"
    NS = "NS"
    EW = "EW"
    BOTH = "BOTH"


class Phase(str, Enum):
    BIDDING = "BIDDING"
    PLAY = "PLAY"


class DecisionRequestType(str, Enum):
    BID_REQUEST = "BID_REQUEST"
    PLAY_REQUEST = "PLAY_REQUEST"


class ActionType(str, Enum):
    BID = "BID"
    PLAY = "PLAY"


class Strain(str, Enum):
    C = "C"
    D = "D"
    H = "H"
    S = "S"
    NT = "NT"


BidCall = str
CardCode = str


class BidAction(BaseModel):
    type: Literal["BID"]
    call: BidCall


class PlayAction(BaseModel):
    type: Literal["PLAY"]
    card: CardCode
    fromSeat: Optional[Seat] = None


LegalAction = Union[BidAction, PlayAction]


class GameInfo(BaseModel):
    gameId: str
    tableId: str
    boardId: str
    format: str
    scoring: str
    dealer: Seat
    vulnerability: Vulnerability


class ActorInfo(BaseModel):
    agentInstanceId: str
    playerId: str
    seat: Seat
    partnership: Partnership
    handContextId: Optional[str] = None


class ContractInfo(BaseModel):
    level: int = Field(ge=1, le=7)
    strain: Strain
    declarer: Seat
    doubled: bool = False
    redoubled: bool = False


class TrickCard(BaseModel):
    seat: Seat
    card: Optional[CardCode] = None


class CompletedTrick(BaseModel):
    trickNumber: int
    leader: Seat
    cards: List[TrickCard]
    winner: Seat


class DummyInfo(BaseModel):
    seat: Seat
    cards: List[CardCode]


class PublicState(BaseModel):
    auction: List[Dict[str, Any]] = Field(default_factory=list)
    contract: Optional[ContractInfo] = None
    dummy: Optional[DummyInfo] = None
    currentTrick: List[TrickCard] = Field(default_factory=list)
    completedTricks: List[CompletedTrick] = Field(default_factory=list)
    tricksWon: Dict[str, int] = Field(default_factory=lambda: {"NS": 0, "EW": 0})


class PrivateState(BaseModel):
    hand: List[CardCode]


class StateEnvelope(BaseModel):
    private: PrivateState
    public: PublicState


class DecisionSpec(BaseModel):
    type: DecisionRequestType
    turnSeat: Seat
    actingSeat: Seat
    legalActions: List[LegalAction]
    timeBudgetMs: Optional[int] = 3000


class BotProfile(BaseModel):
    botType: str = "NATURAL_V1"
    serviceVersion: str = "1.0.0"


class RequestOptions(BaseModel):
    includeExplanation: bool = False


class BridgeDecisionRequest(BaseModel):
    schemaVersion: str = "1.0"
    requestId: str
    timestamp: Optional[str] = None
    game: GameInfo
    actor: ActorInfo
    decision: DecisionSpec
    state: StateEnvelope
    botProfile: Optional[BotProfile] = None
    options: Optional[RequestOptions] = None

    @model_validator(mode="after")
    def validate_phase_specific_actions(self) -> "BridgeDecisionRequest":
        if self.decision.type == DecisionRequestType.BID_REQUEST:
            if not all(a.type == "BID" for a in self.decision.legalActions):
                raise ValueError("BID_REQUEST must contain only BID legalActions")
        elif self.decision.type == DecisionRequestType.PLAY_REQUEST:
            if not all(a.type == "PLAY" for a in self.decision.legalActions):
                raise ValueError("PLAY_REQUEST must contain only PLAY legalActions")
        return self


class DecisionResponsePayload(BaseModel):
    type: ActionType
    call: Optional[BidCall] = None
    card: Optional[CardCode] = None
    fromSeat: Optional[Seat] = None


class ResponseMeta(BaseModel):
    decisionTimeMs: Optional[int] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    model: Optional[str] = None


class ErrorPayload(BaseModel):
    code: str
    message: str


class BridgeDecisionResponse(BaseModel):
    schemaVersion: str = "1.0"
    requestId: str
    decision: Optional[DecisionResponsePayload] = None
    meta: Optional[ResponseMeta] = None
    error: Optional[ErrorPayload] = None


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def legal_action_to_dict(action: LegalAction) -> Dict[str, Any]:
    if isinstance(action, BidAction):
        return {"type": "BID", "call": action.call}
    return {"type": "PLAY", "card": action.card, "fromSeat": action.fromSeat}


def is_legal_decision(
    decision: DecisionResponsePayload,
    legal_actions: List[LegalAction],
) -> bool:
    legal_dicts = [legal_action_to_dict(a) for a in legal_actions]

    if decision.type == ActionType.BID:
        candidate = {"type": "BID", "call": decision.call}
        return candidate in legal_dicts

    candidate = {"type": "PLAY", "card": decision.card, "fromSeat": decision.fromSeat}
    # Allow model to omit fromSeat when it isn't needed.
    if candidate in legal_dicts:
        return True

    candidate_without_from = {"type": "PLAY", "card": decision.card, "fromSeat": None}
    if candidate_without_from in legal_dicts:
        return True

    # Also accept if legal action omitted fromSeat entirely
    legal_simplified = [
        {"type": d["type"], "card": d.get("card")}
        for d in legal_dicts
        if d["type"] == "PLAY"
    ]
    return {"type": "PLAY", "card": decision.card} in legal_simplified


def fallback_action(legal_actions: List[LegalAction]) -> DecisionResponsePayload:
    """Deterministic fallback: choose first legal action."""
    first = legal_actions[0]
    if isinstance(first, BidAction):
        return DecisionResponsePayload(type=ActionType.BID, call=first.call)
    return DecisionResponsePayload(type=ActionType.PLAY, card=first.card, fromSeat=first.fromSeat)


def build_agent_prompt(req: BridgeDecisionRequest) -> str:
    """
    Convert the structured request into a strict prompt.
    The model must return JSON only.
    """
    legal_actions = [legal_action_to_dict(a) for a in req.decision.legalActions]

    # Optional phase hint
    phase_hint = "BIDDING" if req.decision.type == DecisionRequestType.BID_REQUEST else "PLAY"

    include_expl = bool(req.options and req.options.includeExplanation)

    prompt_payload = {
        "requestId": req.requestId,
        "schemaVersion": req.schemaVersion,
        "phase": phase_hint,
        "actor": req.actor.model_dump(),
        "game": req.game.model_dump(),
        "decision": {
            "type": req.decision.type.value,
            "turnSeat": req.decision.turnSeat.value,
            "actingSeat": req.decision.actingSeat.value,
            "timeBudgetMs": req.decision.timeBudgetMs,
            "legalActions": legal_actions,
        },
        "state": req.state.model_dump(),
        "botProfile": req.botProfile.model_dump() if req.botProfile else None,
        "options": {"includeExplanation": include_expl},
    }

    response_contract = {
        "decision": {
            "type": "BID or PLAY",
            "call": "required only for BID",
            "card": "required only for PLAY",
            "fromSeat": "optional, mainly when declarer is choosing dummy card"
        },
        "meta": {
            "confidence": "0.0 to 1.0",
            "reasoning": "very short string, optional"
        }
    }

    return f"""
You are a contract bridge decision engine.

Your job:
- Read the request payload.
- Choose exactly one legal action.
- Never invent cards or bids.
- Never return anything outside the provided legalActions.
- Use only the information visible in the payload.
- Do not assume hidden cards beyond normal bridge inference.
- If this is a PLAY decision and the legal action list contains fromSeat, preserve it correctly.
- Return STRICT JSON ONLY.
- No markdown.
- No code fence.
- No commentary before or after JSON.

Decision policy:
- Prefer bridge-sound, convention-consistent play.
- In bidding, respect the bot profile if supplied.
- In play, follow the contract, trick history, dummy visibility, and legalActions.

INPUT PAYLOAD:
{json.dumps(prompt_payload, ensure_ascii=False)}

RESPONSE JSON SHAPE:
{json.dumps(response_contract, ensure_ascii=False)}

Return one object only.
""".strip()


def extract_final_text(event: Any) -> Optional[str]:
    """
    Best-effort extraction of final text from ADK event.
    """
    try:
        if hasattr(event, "is_final_response") and event.is_final_response():
            content = getattr(event, "content", None)
            if content and getattr(content, "parts", None):
                texts = []
                for part in content.parts:
                    text = getattr(part, "text", None)
                    if text:
                        texts.append(text)
                if texts:
                    return "".join(texts).strip()
    except Exception:
        return None
    return None


def parse_model_json(text: str) -> Dict[str, Any]:
    """
    Handle occasional model output quirks while still expecting strict JSON.
    """
    text = text.strip()

    # Common cleanup if the model still adds fences
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()

    return json.loads(text)


# ------------------------------------------------------------------------------
# ADK Agent setup
# ------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------------------

app = FastAPI(title="Bridge Bot Service", version="1.0.0")


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "service": "bridge-bot-service"}


@app.post("/bot/decision", response_model=BridgeDecisionResponse)
async def bot_decision(req: BridgeDecisionRequest) -> BridgeDecisionResponse:
    started = time.perf_counter()

    # Strong validation before calling the model
    if req.decision.actingSeat != req.actor.seat:
        raise HTTPException(
            status_code=400,
            detail="actor.seat must match decision.actingSeat"
        )

    if not req.decision.legalActions:
        raise HTTPException(
            status_code=400,
            detail="legalActions cannot be empty"
        )

    # Stateless isolation:
    # create a fresh session for each request, keyed by requestId
    session_id = req.requestId
    user_id = req.actor.agentInstanceId

    try:
        await session_service.create_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
            state={},
        )
    except Exception:
        # If create_session is not idempotent in your runtime, ignore duplicate create.
        pass

    prompt = build_agent_prompt(req)
    content = types.Content(
        role="user",
        parts=[types.Part(text=prompt)],
    )

    final_text: Optional[str] = None

    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=content,
        ):
            maybe_text = extract_final_text(event)
            if maybe_text:
                final_text = maybe_text
    except Exception as exc:
        return BridgeDecisionResponse(
            schemaVersion=req.schemaVersion,
            requestId=req.requestId,
            error=ErrorPayload(
                code="AGENT_RUNTIME_ERROR",
                message=str(exc),
            ),
        )

    if not final_text:
        return BridgeDecisionResponse(
            schemaVersion=req.schemaVersion,
            requestId=req.requestId,
            error=ErrorPayload(
                code="EMPTY_AGENT_RESPONSE",
                message="Agent produced no final response text.",
            ),
        )

    try:
        raw = parse_model_json(final_text)
    except Exception as exc:
        return BridgeDecisionResponse(
            schemaVersion=req.schemaVersion,
            requestId=req.requestId,
            error=ErrorPayload(
                code="INVALID_JSON_FROM_AGENT",
                message=f"Agent response was not valid JSON: {exc}",
            ),
            meta=ResponseMeta(
                model=MODEL_NAME,
                reasoning=final_text[:500],
            ),
        )

    if "decision" not in raw:
        return BridgeDecisionResponse(
            schemaVersion=req.schemaVersion,
            requestId=req.requestId,
            error=ErrorPayload(
                code="MISSING_DECISION",
                message="Agent JSON did not contain 'decision'.",
            ),
            meta=ResponseMeta(model=MODEL_NAME),
        )

    try:
        decision = DecisionResponsePayload(**raw["decision"])
    except Exception as exc:
        return BridgeDecisionResponse(
            schemaVersion=req.schemaVersion,
            requestId=req.requestId,
            error=ErrorPayload(
                code="INVALID_DECISION_SHAPE",
                message=str(exc),
            ),
            meta=ResponseMeta(model=MODEL_NAME),
        )

    if not is_legal_decision(decision, req.decision.legalActions):
        # Safer operational behavior:
        # return fallback, but also surface the model issue in meta.
        safe = fallback_action(req.decision.legalActions)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        return BridgeDecisionResponse(
            schemaVersion=req.schemaVersion,
            requestId=req.requestId,
            decision=safe,
            meta=ResponseMeta(
                decisionTimeMs=elapsed_ms,
                confidence=0.0,
                reasoning="Model returned illegal action; deterministic fallback used.",
                model=MODEL_NAME,
            ),
        )

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    raw_meta = raw.get("meta", {}) if isinstance(raw.get("meta"), dict) else {}

    return BridgeDecisionResponse(
        schemaVersion=req.schemaVersion,
        requestId=req.requestId,
        decision=decision,
        meta=ResponseMeta(
            decisionTimeMs=elapsed_ms,
            confidence=raw_meta.get("confidence"),
            reasoning=raw_meta.get("reasoning"),
            model=MODEL_NAME,
        ),
    )


# ------------------------------------------------------------------------------
# Local entrypoint
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)