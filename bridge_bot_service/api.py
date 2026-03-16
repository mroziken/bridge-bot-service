import time
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from google.genai import types

from .agents import APP_NAME, MODEL_NAME, runner, session_service
from .helpers import (
    build_agent_prompt,
    extract_final_text,
    fallback_action,
    is_legal_decision,
    parse_model_json,
)
from .schemas import (
    BridgeDecisionRequest,
    BridgeDecisionResponse,
    DecisionResponsePayload,
    ErrorPayload,
    ResponseMeta,
)

app = FastAPI(title="Bridge Bot Service", version="1.0.0")


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "service": "bridge-bot-service"}


@app.post("/bot/decision", response_model=BridgeDecisionResponse)
async def bot_decision(req: BridgeDecisionRequest) -> BridgeDecisionResponse:
    started = time.perf_counter()

    if req.decision.actingSeat != req.actor.seat:
        raise HTTPException(
            status_code=400,
            detail="actor.seat must match decision.actingSeat",
        )

    if not req.decision.legalActions:
        raise HTTPException(
            status_code=400,
            detail="legalActions cannot be empty",
        )

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
        pass

    prompt = build_agent_prompt(req)
    content = types.Content(role="user", parts=[types.Part(text=prompt)])
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
