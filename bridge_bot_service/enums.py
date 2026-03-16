from enum import Enum


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
