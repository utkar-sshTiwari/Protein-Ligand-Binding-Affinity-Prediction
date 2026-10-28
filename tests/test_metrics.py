import numpy as np

from deepdta.metrics import concordance_index, regression_metrics


def brute_force_ci(y, p):
    num = den = 0.0
    for i in range(len(y)):
        for j in range(len(y)):
            if y[i] > y[j]:
                den += 1
                num += 1.0 if p[i] > p[j] else 0.5 if p[i] == p[j] else 0.0
    return num / den


def test_perfect_ranking_is_one():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert concordance_index(y, y * 10) == 1.0


def test_reversed_ranking_is_zero():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert concordance_index(y, -y) == 0.0


def test_constant_prediction_is_half():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert concordance_index(y, np.zeros(4)) == 0.5


def test_matches_brute_force_on_random_data():
    rng = np.random.default_rng(0)
    y, p = rng.normal(size=60), rng.normal(size=60)
    assert abs(concordance_index(y, p) - brute_force_ci(y, p)) < 1e-12


def test_regression_metrics_known_values():
    m = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 5.0])
    assert abs(m["mse"] - 4 / 3) < 1e-9
    assert abs(m["rmse"] - (4 / 3) ** 0.5) < 1e-9
