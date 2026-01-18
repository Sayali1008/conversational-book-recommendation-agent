# Refactored Recommendation System Architecture

## Overview

The original `_hybrid_recommender.py` has been refactored into a clean, class-based architecture in the `recommenders/` directory. The original code remains untouched for backward compatibility.

## Directory Structure

```
recommenders/
├── __init__.py                 # Module exports
├── data_models.py             # Immutable data structures
├── base.py                    # Abstract base recommender
├── collaborative.py           # Hybrid CF + content-based recommender
├── content_based.py           # Pure content-based (embedding) recommender
└── orchestrator.py            # Coordinates both recommenders
```

## Data Models

### RecommendationContext
**Immutable world state** - all pre-loaded data needed for recommendations.

```python
@dataclass(frozen=True)
class RecommendationContext:
    user_factors: np.ndarray                    # (n_users, n_factors)
    book_factors: np.ndarray                    # (n_books, n_factors)
    train_matrix: sp.spmatrix                   # (n_users, n_books) sparse interactions
    catalog_embeddings: np.ndarray              # (n_catalog_books, embedding_dim)
    index_mappings: IndexMappings              # ID conversion mappings
    catalog_df: pd.DataFrame                   # Book metadata
```

### ModelConfig
**Training/learning hyperparameters** - how the models were built.

```python
@dataclass
class ModelConfig:
    factors: int = 64
    regularization: float = 0.10
    iterations: int = 20
    alpha: int = 20
    book_factor_scale: float = 10.0
```

### EvalConfig
**Runtime behavior parameters** - how recommendations are generated and scored.

```python
@dataclass
class EvalConfig:
    norm: str = "minmax"                        # Normalization: minmax|softmax|zscore
    norm_metadata: Optional[float] = None       # Softmax temperature
    lambda_weight: float = 0.65                 # CF vs embedding blend
    k: int = 10                                 # Number of recommendations
    candidate_pool_size: int = 300              # Candidate selection threshold
    filter_rated: bool = True                   # Exclude rated items
    k_values: list = [5, 10]                    # Evaluation K values
    lambda_values: list = [0.0, ..., 1.0]      # Evaluation lambda values
    min_validation_items: int = 2
    min_confidence: int = 1
```

### IndexMappings
**Index conversion structures** - convert between different ID spaces.

```python
@dataclass(frozen=True)
class IndexMappings:
    cf_idx_to_catalog_id: Dict[int, int]       # CF matrix book idx → catalog row idx
    user_to_cf_idx: Dict[Any, int]             # User ID → CF matrix user idx
    cf_idx_to_user: Dict[int, Any]             # CF matrix user idx → User ID
    cf_idx_to_book: Dict[int, int]             # CF matrix book idx → Book ID
    book_id_to_catalog_idx: Dict[int, int]     # Book ID → catalog row idx
```

## Recommender Classes

### BaseRecommender
Abstract base class defining the recommender interface.

```python
class BaseRecommender(ABC):
    def __init__(self, context: RecommendationContext, config: EvalConfig):
        self.context = context
        self.config = config
    
    @abstractmethod
    def recommend(
        self,
        user_idx: int,
        k: int = 10,
        seed_catalog_indices: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (catalog_indices, scores, sources)"""
```

### CollaborativeRecommender
**Hybrid CF + Content-Based** recommender.
- For warm users (those with interaction history)
- Blends collaborative filtering with semantic embeddings
- Returns `source="hybrid"`

**Method**: 
1. Compute CF scores for all unrated books
2. Select top-N candidates by CF score (two-stage ranking)
3. Compute embedding scores on candidates
4. Normalize both scores
5. Blend: `score = lambda_weight * cf_norm + (1 - lambda_weight) * embedding_norm`

### ContentBasedRecommender
**Content-Based (Embedding-Only)** recommender.
- For cold users (no interaction history)
- Also used as fallback for warm users when hybrid returns <k results
- Returns `source="embedding_only"`

**Method**:
1. Build user profile from interaction history OR seed items OR catalog mean
2. Compute cosine similarity to all candidate items
3. Select top-k

### RecommendationOrchestrator
**Orchestrates both recommenders** implementing OPTION 2 strategy.

**For Warm Users:**
1. Get hybrid recommendations (CollaborativeRecommender)
2. If `len(warm) >= k`: return only warm recommendations
3. If `len(warm) < k`: fill gap with cold recommendations (ContentBasedRecommender)

**For Cold Users:**
1. Get content-based recommendations only (ContentBasedRecommender)

## Comparison: Old vs New

| Aspect | Old (Functions) | New (Classes) |
|--------|---|---|
| **Parameters to functions** | 12+ | 2 (context, config) |
| **State management** | Parameters passed everywhere | Encapsulated in instance |
| **Testing** | Mock 12+ parameters | Mock 2 objects |
| **Adding features** | Change function signature | Add methods |
| **Code organization** | Single file (391 lines) | Separate modules by concern |
| **Type safety** | Loose | Tight (dataclasses) |
| **Reusability** | Load artifacts repeatedly | Load once, reuse |

## Usage Examples

### Initialize Context and Config

```python
from recommenders import (
    RecommendationContext, EvalConfig, RecommendationOrchestrator
)

context = RecommendationContext(
    user_factors=np.load(...),
    book_factors=np.load(...),
    train_matrix=sp.load_npz(...),
    catalog_embeddings=np.load(...),
    index_mappings=IndexMappings(...),
    catalog_df=pd.read_feather(...),
)

config = EvalConfig(
    norm="minmax",
    lambda_weight=0.65,
    candidate_pool_size=300,
)
```

### Generate Recommendations

```python
orchestrator = RecommendationOrchestrator(context, config)

indices, scores, sources = orchestrator.recommend(
    user_idx=160,
    is_warm_user=True,
    k=10,
)

# Returns:
# indices: [5234, 1023, 8901, ...] catalog indices
# scores:  [0.87, 0.85, 0.83, ...] normalized scores [0,1]
# sources: ['hybrid', 'hybrid', 'hybrid', ...] recommendation source
```

### Use Individual Recommenders

```python
cf_recommender = CollaborativeRecommender(context, config)
indices, scores, sources = cf_recommender.recommend(user_idx=160, k=10)

cb_recommender = ContentBasedRecommender(context, config)
indices, scores, sources = cb_recommender.recommend(
    user_idx=160,
    user_profile=my_profile,
)
```

## Benefits

1. **Cleaner Code**: Separated concerns, no parameter explosion
2. **Easier Testing**: Mock simple objects instead of massive parameter lists
3. **Extensible**: Add new recommenders by subclassing `BaseRecommender`
4. **Type Safe**: Dataclasses provide structure and validation
5. **Reusable**: Load artifacts once, use multiple times
6. **Documented**: Clear docstrings and type hints
7. **Maintainable**: Logic organized logically, not scattered

## Next Steps

1. Update `server/recommendation_service.py` to use new classes
2. Update `pipeline.py` evaluation to use new classes
3. Add comprehensive logging to orchestrator
4. Consider caching strategies for repeated recommendations
5. Add monitoring/metrics for recommendation quality

## Backward Compatibility

**Original code (`_hybrid_recommender.py`) remains unchanged.**
The new code coexists, allowing gradual migration.
