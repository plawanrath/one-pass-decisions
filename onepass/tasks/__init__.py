"""Task definitions: each module turns a public corpus into DecisionRecords with a fixed question layout."""
from .pools import PoolSpec, lock_pools, load_pool, pool_manifest  # noqa: F401
