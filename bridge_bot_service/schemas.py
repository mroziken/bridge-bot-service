from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from .enums import (
    ActionType,
    DecisionRequestType,
    Partnership,
    Seat,
    Strain,
    Vulnerability,
)

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
            if not all(action.type == "BID" for action in self.decision.legalActions):
                raise ValueError("BID_REQUEST must contain only BID legalActions")
        elif self.decision.type == DecisionRequestType.PLAY_REQUEST:
            if not all(action.type == "PLAY" for action in self.decision.legalActions):
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
