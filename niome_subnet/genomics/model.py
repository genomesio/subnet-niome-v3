from pydantic import BaseModel
from typing import Generic, TypeVar


class Task(BaseModel):
    """Data model for a genomics simulation task."""
    id: str
    contract_url: str   # presigned URL to fetch the contract
    hbb_ref_url: str    # presigned URL to fetch the HBB reference data


class ValidationContext:
    """Container for all validation metadata."""
    miner_uid: int
    miner_hotkey: str
    validator_uid: int
    validator_hotkey: str

    def __init__(self, miner_uid: int = 0, miner_hotkey: str = "", validator_uid: int = 0, validator_hotkey: str = ""):
        self.miner_uid = miner_uid
        self.miner_hotkey = miner_hotkey
        self.validator_uid = validator_uid
        self.validator_hotkey = validator_hotkey


class TaskPayload(BaseModel):
    """Payload structure for task generation requests."""
    timestamp: float
    hotkey: str
    uuid: str
    netuid: str


PayloadType = TypeVar('PayloadType', bound=BaseModel)

class SignedRequest(BaseModel, Generic[PayloadType]):
    """Generic signed request structure."""
    payload: PayloadType
    payload_raw: str
    signature: str


class Stage1Result(BaseModel):
    min: float
    max: float
    mean: float
    valid: float


class Stage2Result(BaseModel):
    min: float
    max: float
    mean: float


class Stage3Result(BaseModel):
    n: int
    cut_rate: float
    mean_indel_length: float
    mean_energy: float
    outcomes: dict[str, int]


class Stage4Result(BaseModel):
    consistency_score: float


class MinerScore(BaseModel):
    uid: int
    stage1: Stage1Result | None
    stage2: Stage2Result | None
    stage3: Stage3Result | None
    stage4: Stage4Result | None
    final_score: float
    log: str


class MinerScoreDto(BaseModel):
    """Data transfer object for miner score submission."""
    task_id: str
    uid: int
    hotkey: str
    stage1: float
    stage2: float
    stage3: float
    stage4: float
    final_score: float
    log: str
    weight: float
