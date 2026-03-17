import logging
import json
from typing import Any, Dict, List, Optional

from .enums import ActionType, DecisionRequestType
from .logging_config import get_logger, payload_logging_enabled, truncate_for_log
from .schemas import (
    BidAction,
    BridgeDecisionRequest,
    DecisionResponsePayload,
    LegalAction,
    PlayAction,
)

logger = get_logger(__name__)


def legal_action_to_dict(action: LegalAction) -> Dict[str, Any]:
    if isinstance(action, BidAction):
        return {"type": "BID", "call": action.call}
    return {"type": "PLAY", "card": action.card, "fromSeat": action.fromSeat}


def is_legal_decision(
    decision: DecisionResponsePayload,
    legal_actions: List[LegalAction],
) -> bool:
    legal_dicts = [legal_action_to_dict(action) for action in legal_actions]

    if decision.type == ActionType.BID:
        candidate = {"type": "BID", "call": decision.call}
        return candidate in legal_dicts

    candidate = {"type": "PLAY", "card": decision.card, "fromSeat": decision.fromSeat}
    if candidate in legal_dicts:
        return True

    candidate_without_from = {"type": "PLAY", "card": decision.card, "fromSeat": None}
    if candidate_without_from in legal_dicts:
        return True

    legal_simplified = [
        {"type": action["type"], "card": action.get("card")}
        for action in legal_dicts
        if action["type"] == "PLAY"
    ]
    return {"type": "PLAY", "card": decision.card} in legal_simplified


def fallback_action(legal_actions: List[LegalAction]) -> DecisionResponsePayload:
    first = legal_actions[0]
    if isinstance(first, BidAction):
        return DecisionResponsePayload(type=ActionType.BID, call=first.call)
    if isinstance(first, PlayAction):
        return DecisionResponsePayload(
            type=ActionType.PLAY,
            card=first.card,
            fromSeat=first.fromSeat,
        )
    raise TypeError("Unsupported legal action type")


def build_agent_prompt(req: BridgeDecisionRequest) -> str:
    legal_actions = [legal_action_to_dict(action) for action in req.decision.legalActions]
    phase_hint = (
        "BIDDING"
        if req.decision.type == DecisionRequestType.BID_REQUEST
        else "PLAY"
    )
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
            "fromSeat": "optional, mainly when declarer is choosing dummy card",
        },
        "meta": {
            "confidence": "0.0 to 1.0",
            "reasoning": "very short string, optional",
        },
    }

    prompt = f"""
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

    logger.debug(
        "Built agent prompt",
        extra={
            "request_id": req.requestId,
            "phase": phase_hint,
            "legal_action_count": len(legal_actions),
            "prompt_length": len(prompt),
        },
    )
    if payload_logging_enabled():
        logger.debug(
            "Prompt payload details: %s",
            truncate_for_log(prompt_payload, limit=2000),
            extra={"request_id": req.requestId},
        )

    return prompt


def extract_final_text(event: Any) -> Optional[str]:
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
        logger.exception("Failed to extract final text from agent event")
        return None
    return None


def log_agent_event_debug(event: Any, request_id: str = "-") -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return

    try:
        if hasattr(event, "get_function_calls"):
            for function_call in event.get_function_calls():
                logger.debug(
                    "Agent called tool '%s' with args=%s",
                    function_call.name,
                    truncate_for_log(function_call.args or {}, limit=1000),
                    extra={"request_id": request_id},
                )

        if hasattr(event, "get_function_responses"):
            for function_response in event.get_function_responses():
                logger.debug(
                    "Tool '%s' returned %s",
                    function_response.name,
                    truncate_for_log(function_response.response or {}, limit=1000),
                    extra={"request_id": request_id},
                )
    except Exception:
        logger.exception(
            "Failed to log agent tool event",
            extra={"request_id": request_id},
        )


def parse_model_json(text: str, request_id: str = "-") -> Dict[str, Any]:
    logger.debug(
        "Parsing model JSON response",
        extra={"request_id": request_id},
    )
    text = text.strip()

    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()

    if payload_logging_enabled():
        logger.debug(
            "Raw model text: %s",
            truncate_for_log(text),
            extra={"request_id": request_id},
        )

    return json.loads(text)
