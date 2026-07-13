"""
Protocol definitions for the Drug Response Prediction Subnet.

This module defines the communication protocols between validators and miners
for drug response prediction tasks using synthetic genomic data.
"""
import bittensor as bt

from typing import Optional
from niome_subnet.genomics.model import Task

class GenomicsTaskSynapse(bt.Synapse):
    """Protocol for genomics simulation tasks."""

    # Input fields
    task: Optional[Task] = None
    presigned_url: str = ""
    timeout: Optional[float] = None  # Timeout window for submission

    def deserialize(self) -> bt.Synapse:
        """Deserialize the GenomicsTaskSynapse Object."""
        return self