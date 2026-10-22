"""Quote-only Torch source choices; locations and authority remain program-owned."""
from typing import Literal

from schemas.partition_analysis import CandidatePaperDefinition, CandidateSourceDefinition
from schemas.runtime_science import TorchInferenceProposalBase


class TorchInferenceCandidate(TorchInferenceProposalBase[
    Literal["torch-cpu-inference-candidate-v2"], CandidatePaperDefinition, CandidateSourceDefinition,
]):
    pass
