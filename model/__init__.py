"""
ML Pipeline - Functional ML workflow orchestration.

Simple functions for each pipeline stage.
"""

from model.handler import (  # run_stage_1_preprocessing,; run_stage_2_migration,; run_stage_2_embeddings,; run_stage_3_matrices,; run_stage_4_training,; run_stage_5_evaluation,
    run_data_pipeline,
    run_model_pipeline,
)

__all__ = [
    "run_data_pipeline",
    "run_model_pipeline",
    # "run_stage_1_preprocessing",
    # "run_stage_2_migration",
    # "run_stage_2_embeddings",
    # "run_stage_3_matrices",
    # "run_stage_4_training",
    # "run_stage_5_evaluation",
]
