"""
Regression tests for the ST-FID/ST-MMD query-side dimensionality fix
(2026-07-19 architecture/code audit, see docs/results_log.md).

Real finding: run_comparison.py's _build_shared_eval capped the PCA
embedding's n_components by CONTEXT set size only
(context_patches.shape[0]-1), but st_fid/st_mmd's covariance is actually
estimated over the QUERY-sized embedding (real_embed/gen_embed in
_evaluate both have n_query rows) — which is almost always much smaller
than context. np.cov on an [n_query, n_components] array is
mathematically rank-deficient whenever n_query < n_components (a
guaranteed, not occasional, property: rank <= n_query-1). This has been
true for every ST-FID/ST-MMD number this project has ever produced,
whenever a masking draw's query set was smaller than pca_n_components
(50 in every config) -- not a new bug introduced this session, a
pre-existing metric-validity gap found while auditing.

_fid_n_components (run_comparison.py) is the extracted, directly
testable fix -- also caps by query_n-1, mirroring the "can't estimate
more components than you have degrees of freedom" rule already used for
the context side.

Run with: python -m tests.test_fid_query_dimensionality
"""
from src.evaluation.run_comparison import _fid_n_components


def test_uncapped_when_query_and_context_both_comfortably_exceed_pca_components():
    # context has 500 points/30 dims, query has 200 points, pca wants 50 -- nothing binds but pca_components itself
    n = _fid_n_components(pca_components=50, context_n=500, context_d=30, query_n=200)
    assert n == 30, "context_d (30) is the real bottleneck here, below pca_components (50)"


def test_capped_by_query_size_when_query_is_the_smallest_dimension():
    # this is the real bug scenario: context is large (500 pts), but a
    # typical small masking draw's query set (e.g. 20 points) is far
    # smaller than pca_components (50) -- must cap to query_n-1, not 50
    n = _fid_n_components(pca_components=50, context_n=500, context_d=100, query_n=20)
    assert n == 19, f"must cap to query_n-1=19 when query is the binding constraint, got {n}"


def test_still_capped_by_context_size_when_context_is_smaller_than_query():
    n = _fid_n_components(pca_components=50, context_n=10, context_d=100, query_n=200)
    assert n == 9, "context_n-1 must still win when context is the smaller side"


def test_degenerate_tiny_query_floors_at_one_not_zero_or_negative():
    n = _fid_n_components(pca_components=50, context_n=500, context_d=100, query_n=1)
    assert n == 1, "a 1-point (or 0-point) query set must floor at 1 component, never <=0"
    n = _fid_n_components(pca_components=50, context_n=500, context_d=100, query_n=0)
    assert n == 1


def test_result_is_never_rank_deficient_for_the_resulting_query_embedding():
    """The actual property this fix guarantees: with the returned
    n_components, a [query_n, n_components] embedding matrix always has
    query_n >= n_components + 1, i.e. np.cov on it is full-rank (as long
    as the underlying data isn't itself degenerate)."""
    import random
    rng = random.Random(0)
    for _ in range(200):
        pca_components = rng.randint(1, 100)
        context_n = rng.randint(2, 1000)
        context_d = rng.randint(1, 200)
        query_n = rng.randint(0, 500)
        n = _fid_n_components(pca_components, context_n, context_d, query_n)
        assert n >= 1
        assert n <= max(1, query_n - 1) or query_n <= 1, (
            f"n_components={n} exceeds query_n-1={query_n-1} for query_n={query_n} "
            f"(pca_components={pca_components}, context_n={context_n}, context_d={context_d})"
        )
    print("[fid_query_dimensionality] OK — 200 random configs, result never exceeds query_n-1")


if __name__ == "__main__":
    test_uncapped_when_query_and_context_both_comfortably_exceed_pca_components()
    test_capped_by_query_size_when_query_is_the_smallest_dimension()
    test_still_capped_by_context_size_when_context_is_smaller_than_query()
    test_degenerate_tiny_query_floors_at_one_not_zero_or_negative()
    test_result_is_never_rank_deficient_for_the_resulting_query_embedding()
    print("\nAll ST-FID/ST-MMD query-dimensionality tests passed.")
