import time

import numpy as np
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

from common.constants import PATHS, CF_MODEL_PARAMS
from common.utils import setup_logging

logger = setup_logging(__name__, PATHS["eval_log_file"])


def model_initialization(factors=64, regularization=0.1, iterations=20, alpha=20):
    """
    Params are the hyperparameters required for model initialization

    :param factors: Number of latent factors
    :param regularization: L2 regularization
    :param iterations: Number of ALS iterations
    :param alpha: Confidence scaling factor (used as: 1 + alpha * rating)
    """

    # Initialize model
    model = AlternatingLeastSquares(
        factors=CF_MODEL_PARAMS["factors"],
        regularization=CF_MODEL_PARAMS["regularization"],
        iterations=CF_MODEL_PARAMS["iterations"],
        alpha=CF_MODEL_PARAMS["alpha"],
        random_state=CF_MODEL_PARAMS["random_state"],
    )

    logger.info(f"ALS Model Initialized:")
    logger.info(f"  Factors: {factors}")
    logger.info(f"  Regularization: {regularization}")
    logger.info(f"  Iterations: {iterations}")
    logger.info(f"  Alpha: {alpha}")

    return model


def model_training(model, train_matrix):
    # train_matrix = (user x book) sparse matrix
    logger.info(f"Training matrix shape (user x book): {train_matrix.shape}")
    logger.info("Starting training...")
    start_time = time.time()

    # Fit the model
    model.fit(train_matrix, show_progress=True)

    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

    return model


def test_predictions(user_factors, book_factors, user_cf_idx=0, k=10):
    try:
        # Get this user's factor vector
        user_vec = user_factors[user_cf_idx]  # (64,)

        # Compute scores for all books: user_vec · book_factors^T
        scores = book_factors.dot(user_vec)  # (11591,)

        # Get top K books
        top_k_indices = np.argsort(scores)[::-1][:k]
        top_k_scores = scores[top_k_indices]

        logger.info(f"Top {k} predictions for user {user_cf_idx} (manual calculation):")
        logger.info(f"Book indices: {top_k_indices}")
        logger.info(f"Scores: {top_k_scores}")
        logger.info(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        logger.info(f"Score mean: {scores.mean():.4f}")
        logger.info(f"Score std: {scores.std():.4f}")

        # Check if scores are reasonable
        if top_k_scores.max() < 0.01:
            logger.warning("⚠️ Prediction scores are very small! Model may not be learning well.")
        else:
            logger.info("✓ Prediction scores look reasonable")

        return top_k_indices, top_k_scores

    except Exception as e:
        logger.error(f"Prediction test failed: {e}")
        return None, None
