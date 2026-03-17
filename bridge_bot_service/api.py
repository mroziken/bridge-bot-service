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
    log_agent_event_debug,
    parse_model_json,
)
from .logging_config import (
    configure_logging,
    get_log_level,
    get_logger,
    payload_logging_enabled,
    truncate_for_log,
)
from .schemas import (
    BridgeDecisionRequest,
    BridgeDecisionResponse,
    DecisionResponsePayload,
    ErrorPayload,
    ResponseMeta,
)

configure_logging()
logger = get_logger(__name__)
app = FastAPI(title="Bridge Bot Service", version="1.0.0")
logger.info(
    "API module initialized",
    extra={
        "request_id": "-",
        "log_level": get_log_level(),
        "payload_logging": payload_logging_enabled(),
    },
)


@app.get("/health")
async def health() -> Dict[str, str]:
    logger.debug("Health check requested", extra={"request_id": "-"})
    return {"status": "ok", "service": "bridge-bot-service"}


@app.post("/bot/decision", response_model=BridgeDecisionResponse)
async def bot_decision(req: BridgeDecisionRequest) -> BridgeDecisionResponse:
    started = time.perf_counter()
    log_extra = {
        "request_id": req.requestId,
        "acting_seat": req.decision.actingSeat.value,
        "decision_type": req.decision.type.value,
        "legal_action_count": len(req.decision.legalActions),
    }
    logger.info("Decision request received", extra=log_extra)
    if payload_logging_enabled():
        logger.debug(
            "Decision request payload: %s",
            truncate_for_log(req.model_dump(), limit=4000),
            extra={"request_id": req.requestId},
        )

    if req.decision.actingSeat != req.actor.seat:
        logger.warning(
            "Rejecting request due to seat mismatch",
            extra={"request_id": req.requestId},
        )
        raise HTTPException(
            status_code=400,
            detail="actor.seat must match decision.actingSeat",
        )

    if not req.decision.legalActions:
        logger.warning(
            "Rejecting request because no legal actions were provided",
            extra={"request_id": req.requestId},
        )
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
        logger.debug(
            "Session already exists or could not be created idempotently",
            extra={"request_id": req.requestId},
            exc_info=True,
        )

    prompt = build_agent_prompt(req)
    content = types.Content(role="user", parts=[types.Part(text=prompt)])
    final_text: Optional[str] = None

    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=content,
        ):
            log_agent_event_debug(event, request_id=req.requestId)
            maybe_text = extract_final_text(event)
            if maybe_text:
                final_text = maybe_text
    except Exception as exc:
        logger.exception(
            "Agent runtime error while processing request",
            extra={"request_id": req.requestId},
        )
        return BridgeDecisionResponse(
            schemaVersion=req.schemaVersion,
            requestId=req.requestId,
            error=ErrorPayload(
                code="AGENT_RUNTIME_ERROR",
                message=str(exc),
            ),
        )

    if not final_text:
        logger.error(
            "Agent produced no final response text",
            extra={"request_id": req.requestId},
        )
        return BridgeDecisionResponse(
            schemaVersion=req.schemaVersion,
            requestId=req.requestId,
            error=ErrorPayload(
                code="EMPTY_AGENT_RESPONSE",
                message="Agent produced no final response text.",
            ),
        )

    try:
        raw = parse_model_json(final_text, request_id=req.requestId)
    except Exception as exc:
        logger.warning(
            "Agent returned invalid JSON",
            extra={"request_id": req.requestId},
        )
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
        logger.warning(
            "Agent JSON did not contain decision",
            extra={"request_id": req.requestId},
        )
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
        logger.warning(
            "Agent decision payload had invalid shape",
            extra={"request_id": req.requestId},
        )
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
        logger.warning(
            "Model returned illegal action; using fallback",
            extra={
                "request_id": req.requestId,
                "elapsed_ms": elapsed_ms,
                "fallback_decision": truncate_for_log(safe.model_dump(), limit=200),
            },
        )

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
    logger.info(
        "Decision request completed",
        extra={
            "request_id": req.requestId,
            "elapsed_ms": elapsed_ms,
            "decision_type": decision.type.value,
            "decision": truncate_for_log(decision.model_dump(), limit=200),
        },
    )

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
