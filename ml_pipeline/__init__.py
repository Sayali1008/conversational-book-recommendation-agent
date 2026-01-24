"""
ML Pipeline - Functional ML workflow orchestration.

Simple functions for each pipeline stage.
"""

from ml_pipeline.handler import (
    run_stage_1_preprocessing,
    run_stage_2_embeddings,
    run_stage_3_matrices,
    run_stage_4_training,
    run_stage_5_evaluation,
    STAGES,
)

__all__ = [
    "run_stage_1_preprocessing",
    "run_stage_2_embeddings",
    "run_stage_3_matrices",
    "run_stage_4_training",
    "run_stage_5_evaluation",
    "STAGES",
]
