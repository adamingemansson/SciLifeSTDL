from gen2_architectures.evaluation.audit_evaluation import _fid_n_components


def test_requires_query_n_at_least_5x_components():
    # query_n=200 -> at most 40 components from the query-side constraint,
    # regardless of how many were requested or how big context is.
    assert _fid_n_components(pca_components=50, context_n=1000, context_d=17000, query_n=200) == 40


def test_small_query_n_forces_few_components_not_query_n_minus_1():
    # 2026-07-27 bugfix regression check: query_n=6 used to allow 5
    # components (query_n - 1) -- a near-singular covariance case that
    # produced garbage ST-FID values on a real run. Now capped much lower.
    assert _fid_n_components(pca_components=50, context_n=1000, context_d=17000, query_n=6) == 1


def test_floor_is_always_at_least_one_component():
    assert _fid_n_components(pca_components=50, context_n=1000, context_d=17000, query_n=0) == 1
    assert _fid_n_components(pca_components=50, context_n=1000, context_d=17000, query_n=1) == 1


def test_requested_components_still_caps_the_result_when_smallest():
    assert _fid_n_components(pca_components=5, context_n=1000, context_d=17000, query_n=1000) == 5
