"""
Standalone ML pipeline training and evaluation script.

This script runs the ML pipeline for model training and evaluation, allowing for:
- Model retraining with different configurations
- Hyperparameter optimization and experimentation
- Quick evaluation cycles
- Reproducibility with existing cleaned data

Usage:
    python scripts/model_training.py                    # Run full pipeline (data + model)
    python scripts/model_training.py --stage data       # Run data pipeline only
    python scripts/model_training.py --stage model      # Run model pipeline only
"""

import argparse

from common.constants import PATHS
from common.utils import setup_logging
from model import handler as model_handler

logger = setup_logging(__name__, PATHS["eval_log_file"])


def run_stage_data():
    """Run data preparation pipeline (cleaning, embeddings, migration)."""
    try:
        logger.info("=" * 80)
        logger.info("STAGE 1: DATA PIPELINE")
        logger.info("=" * 80)
        model_handler.run_data_pipeline()
        logger.info("✓ Data pipeline completed")
        return True
    except Exception as e:
        logger.error(f"Data pipeline failed: {str(e)}")
        raise


def run_stage_model():
    """Run model training and CV evaluation pipeline."""
    try:
        logger.info("=" * 80)
        logger.info("STAGE 2: MODEL PIPELINE")
        logger.info("=" * 80)
        model_handler.run_model_pipeline()
        logger.info("✓ Model pipeline completed")
        return True
    except Exception as e:
        logger.error(f"Model pipeline failed: {str(e)}")
        raise


def run_all_stages():
    """Run full pipeline: data + model."""
    try:
        run_stage_data()
        run_stage_model()
        logger.info("=" * 80)
        logger.info("✓ FULL PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        return True
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Standalone ML pipeline for training and evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python scripts/model_training.py                # Run full pipeline (data + model)
            python scripts/model_training.py --stage data   # Run data pipeline only
            python scripts/model_training.py --stage model  # Run model pipeline only
        """,
    )

    parser.add_argument(
        "--stage",
        choices=["data", "model"],
        help="Run specific stage only (data=data pipeline, model=model training/eval). Omit to run all.",
    )

    args = parser.parse_args()

    try:
        if args.stage == "data":
            return run_stage_data()
        elif args.stage == "model":
            return run_stage_model()
        else:
            return run_all_stages()
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
