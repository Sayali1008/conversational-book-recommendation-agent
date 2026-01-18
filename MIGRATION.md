# RecommendationService Refactoring - Migration Guide

## What Changed

`server/recommendation_service.py` has been refactored to use the new `recommenders/` module architecture.

### Before (Old Architecture)

```python
# OLD: Parameter explosion
indices, scores, sources = _hybrid_recommender.recommend_with_cold_start(
    user_idx=user_cf_idx,
    is_warm_user=warm,
    user_factors=self.user_factors,
    book_factors=self.book_factors,
    train_matrix=self.train_matrix,
    catalog_embeddings=self.catalog_embeddings,
    cf_idx_to_catalog_id_map=self.cf_idx_to_catalog_id_map,
    cf_idx_to_book=self.cf_idx_to_book,
    k=k,
    lambda_weight=HYBRID_LAMBDA_WEIGHT,
    candidate_pool_size=HYBRID_CANDIDATE_POOL_SIZE,
    filter_rated=HYBRID_FILTER_RATED,
    seed_catalog_indices=seed_catalog_indices,
    norm=HYBRID_NORM,
    norm_metadata=HYBRID_NORM_METADATA
)
```

### After (New Architecture)

```python
# NEW: Clean 2-object interface
indices, scores, sources = self.orchestrator.recommend(
    user_idx=user_cf_idx,
    is_warm_user=warm,
    k=k,
    seed_catalog_indices=seed_catalog_array,
)
```

## Key Improvements

### 1. **Immutable Context**
```python
self.context = RecommendationContext(
    user_factors=...,
    book_factors=...,
    train_matrix=...,
    catalog_embeddings=...,
    index_mappings=...,
    catalog_df=...,
)
# Loaded once, reused for all recommendations
```

### 2. **Centralized Configuration**
```python
self.config = EvalConfig(
    norm=HYBRID_NORM,
    norm_metadata=HYBRID_NORM_METADATA,
    lambda_weight=HYBRID_LAMBDA_WEIGHT,
    candidate_pool_size=HYBRID_CANDIDATE_POOL_SIZE,
    filter_rated=HYBRID_FILTER_RATED,
)
# All settings in one place
```

### 3. **Orchestrator Pattern**
```python
self.orchestrator = RecommendationOrchestrator(self.context, self.config)
# Routes to CollaborativeRecommender or ContentBasedRecommender
# Implements warm/cold fallback logic (OPTION 2)
```

## Interface Compatibility

### Public API Unchanged ✅

The `recommend()` method signature is **mostly the same**:

```python
# OLD
recs, strategy = service.recommend(user_cf_idx, k=10, seed_catalog_indices=None, lambda_weight=0.65)

# NEW
recs, strategy = service.recommend(user_cf_idx, k=10, seed_catalog_indices=None)
```

**Breaking change**: `lambda_weight` parameter removed (now comes from `EvalConfig`)
- **Why**: Lambda is a constant hyperparameter, shouldn't vary per request
- **Solution**: Change `EvalConfig.lambda_weight` in `config.py` if you need to adjust

### Result Format Unchanged ✅

```python
recs = [
    {
        "book_id": int,
        "title": str,
        "score": float,  # [0, 1]
        "source": str,   # "hybrid" or "embedding_only"
    },
    ...
]

strategy = "warm_hybrid" | "cold_embed"
```

## Testing the Refactored Service

### Quick Test

```python
from server.recommendation_service import RecommendationService

service = RecommendationService()  # Loads all artifacts

# Test warm user
recs, strategy = service.recommend(user_cf_idx=160, k=10)
print(f"Got {len(recs)} recommendations via {strategy}")

# Test cold user
recs, strategy = service.recommend(user_cf_idx=None, k=10)
print(f"Got {len(recs)} recommendations via {strategy}")

# Test with seeds
seed_ids = [100, 200, 300]
seed_indices = service.book_ids_to_catalog_indices(seed_ids)
recs, strategy = service.recommend(user_cf_idx=None, k=10, seed_catalog_indices=seed_indices)
```

## Under the Hood

### Request Flow (Warm User)

```
recommend(user_cf_idx=160, is_warm=True, k=10)
    ↓
RecommendationOrchestrator.recommend()
    ↓
CollaborativeRecommender.recommend()  ← Hybrid CF + embeddings
    ├─ Compute CF scores
    ├─ Select candidate pool
    ├─ Compute embedding scores
    └─ Blend: λ*CF + (1-λ)*embeddings
    ↓
[Returns 10 hybrid recommendations or fewer if all returned]
    ↓
[If <10 results, use ContentBasedRecommender as fallback]
    ├─ Compute embedding-only scores
    └─ Fill gap to k=10
    ↓
Merge, rank, return
```

### Request Flow (Cold User)

```
recommend(user_cf_idx=None, is_warm=False, k=10)
    ↓
RecommendationOrchestrator.recommend()
    ↓
ContentBasedRecommender.recommend()  ← Embedding-only
    ├─ Build profile from seeds or catalog mean
    ├─ Compute embedding scores
    └─ Normalize & select top-k
    ↓
Return 10 embedding-only recommendations
```

## Configuration Adjustments

All hyperparameters are in [config.py](../config.py):

```python
# Hybrid blending
HYBRID_LAMBDA_WEIGHT = 0.65         # Increase for more CF, decrease for more content
HYBRID_CANDIDATE_POOL_SIZE = 300    # Larger = slower but potentially better
HYBRID_NORM = "minmax"              # Or "softmax" or "zscore"
HYBRID_NORM_METADATA = None         # Temperature if using softmax (e.g., 0.01-0.9)
HYBRID_FILTER_RATED = True          # Always exclude already-rated books
HYBRID_BOOK_FACTOR_SCALE = 10.0     # Scale to address tiny factor values
```

Change these values and they automatically apply to all recommendations (no code changes needed).

## Logging

Service now logs via Python's standard `logging` module:

```python
logger = logging.getLogger(__name__)
logger.info("✓ RecommendationService initialized successfully")
logger.info(f"Generating {k} recommendations for user_idx={user_cf_idx}")
logger.info(f"Returned {len(indices)} recommendations using strategy: {strategy}")
```

Configure logging level in your application setup.

## Original Code Preserved

The original `_hybrid_recommender.py` remains **completely unchanged** for backward compatibility. If needed, you can still:

```python
import _hybrid_recommender
indices, scores, sources = _hybrid_recommender.recommend_with_cold_start(...)
```

## Next Steps

1. **Test the service** with your existing API calls
2. **Monitor logs** to confirm recommendations are working
3. **Adjust hyperparameters** in `config.py` if needed
4. **Consider migrating pipeline.py** later (currently unchanged)

## Rollback Plan

If issues arise:
1. Keep git branch with old code
2. Revert `server/recommendation_service.py` to use old `_hybrid_recommender` functions
3. The data models and classes in `recommenders/` won't interfere

## Questions?

See [recommenders/README.md](../recommenders/README.md) for detailed architecture documentation.
