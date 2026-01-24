"""
Pipeline execution state management.
Tracks the progress of running ML pipeline stages in real-time.
"""

from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum
import threading
from common.constants import *
from common.utils import setup_logging

logger = setup_logging(__name__, PATHS["app_log_file"])


class PipelineStatus(str, Enum):
    """Pipeline execution status."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    IDLE = "idle"


class StageStatus(str, Enum):
    """Individual stage status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineStateManager:
    """
    Manages global pipeline execution state.
    Allows progress updates from background tasks and status queries from API endpoints.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.current_pipeline_id: Optional[str] = None
        self.overall_status = PipelineStatus.IDLE
        self.start_time: Optional[datetime] = None
        
        # Stage progress: stage_name -> {"status": StageStatus, "progress_percent": int, "elapsed_seconds": int}
        self.stages = {
            "stage_1_preprocessing": {
                "status": "pending",
                "progress_percent": 0,
                "elapsed_seconds": 0,
            },
            "stage_2_embeddings": {
                "status": "pending",
                "progress_percent": 0,
                "elapsed_seconds": 0,
            },
            "stage_3_matrices": {
                "status": "pending",
                "progress_percent": 0,
                "elapsed_seconds": 0,
            },
            "stage_4_training": {
                "status": "pending",
                "progress_percent": 0,
                "elapsed_seconds": 0,
            },
            "stage_5_evaluation": {
                "status": "pending",
                "progress_percent": 0,
                "elapsed_seconds": 0,
            },
        }
        self.artifacts_generated = []
        self.error_message: Optional[str] = None

    def start_pipeline(self, pipeline_id: str) -> None:
        """Initialize a new pipeline execution."""
        with self.lock:
            self.current_pipeline_id = pipeline_id
            self.overall_status = PipelineStatus.RUNNING
            self.start_time = datetime.now()
            
            # Reset all stages to pending
            for stage_name in self.stages:
                self.stages[stage_name] = {
                    "status": "pending",
                    "progress_percent": 0,
                    "elapsed_seconds": 0,
                }
            self.artifacts_generated = []
            self.error_message = None
            
            logger.info(f"Pipeline {pipeline_id} started")

    def update_stage_status(
        self,
        stage_name: str,
        status: str,
        progress_percent: int = None,
    ) -> None:
        """Update progress for a specific stage."""
        with self.lock:
            if stage_name not in self.stages:
                logger.warning(f"Unknown stage: {stage_name}")
                return

            elapsed = 0
            if self.start_time:
                elapsed = int((datetime.now() - self.start_time).total_seconds())

            self.stages[stage_name] = {
                "status": status,
                "progress_percent": progress_percent or self.stages[stage_name]["progress_percent"],
                "elapsed_seconds": elapsed,
            }
            
            logger.debug(f"Updated {stage_name}: {status}, {progress_percent}%")

    def start_stage(self, stage_name: str) -> None:
        """Mark a stage as running."""
        self.update_stage_status(stage_name, "running", 0)

    def complete_stage(self, stage_name: str) -> None:
        """Mark a stage as completed."""
        self.update_stage_status(stage_name, "completed", 100)

    def fail_stage(self, stage_name: str, error_msg: str) -> None:
        """Mark a stage as failed."""
        with self.lock:
            self.stages[stage_name]["status"] = "failed"
            self.overall_status = PipelineStatus.FAILED
            self.error_message = error_msg
            logger.error(f"Stage {stage_name} failed: {error_msg}")

    def complete_pipeline(self, artifacts: list = None) -> None:
        """Mark pipeline as completed."""
        with self.lock:
            self.overall_status = PipelineStatus.COMPLETED
            self.artifacts_generated = artifacts or []
            logger.info(f"Pipeline {self.current_pipeline_id} completed with {len(self.artifacts_generated)} artifacts")

    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status as dictionary."""
        with self.lock:
            overall_progress = 0
            stage_count = len(self.stages)
            
            # Calculate overall progress as average of stage progress
            total_progress = sum(
                stage["progress_percent"]
                for stage in self.stages.values()
            )
            overall_progress = int(total_progress / stage_count) if stage_count > 0 else 0

            total_duration = 0
            if self.start_time and self.overall_status in [PipelineStatus.COMPLETED, PipelineStatus.FAILED]:
                total_duration = int((datetime.now() - self.start_time).total_seconds())
            elif self.start_time:
                total_duration = int((datetime.now() - self.start_time).total_seconds())

            return {
                "pipeline_id": self.current_pipeline_id,
                "status": self.overall_status.value,
                "current_stage": self._get_current_running_stage(),
                "progress": {
                    stage_name: {
                        "status": stage_data["status"],
                        "progress_percent": stage_data["progress_percent"],
                        "elapsed_seconds": stage_data["elapsed_seconds"],
                    }
                    for stage_name, stage_data in self.stages.items()
                },
                "overall_progress": overall_progress,
                "total_duration_seconds": total_duration,
                "artifacts_generated": self.artifacts_generated,
                "error_message": self.error_message,
            }

    def _get_current_running_stage(self) -> Optional[str]:
        """Return the name of the currently running stage, if any."""
        for stage_name, stage_data in self.stages.items():
            if stage_data["status"] == "running":
                return stage_name
        return None

    def is_running(self) -> bool:
        """Check if any pipeline is currently running."""
        with self.lock:
            return self.overall_status == PipelineStatus.RUNNING

    def reset(self) -> None:
        """Reset pipeline state (when no pipeline is running)."""
        with self.lock:
            self.current_pipeline_id = None
            self.overall_status = PipelineStatus.IDLE
            self.start_time = None
            self.error_message = None
            self.artifacts_generated = []
            
            for stage_name in self.stages:
                self.stages[stage_name] = {
                    "status": "pending",
                    "progress_percent": 0,
                    "elapsed_seconds": 0,
                }
