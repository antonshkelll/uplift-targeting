"""Проверяем метрики на данных, где правильный ответ известен заранее."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models import ClassTransformation, TLearner  # noqa: E402
from uplift_metrics import qini_auc, qini_curve, uplift_at_k  # noqa: E402


def make_data(n=20_000, seed=0):
    """Воздействие работает только при x0 > 0: там P(y) растёт с 10% до 40%."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    t = rng.integers(0, 2, n)
    p = 0.1 + 0.3 * t * (X[:, 0] > 0)
    y = (rng.random(n) < p).astype(int)
    return X, y, t


def test_oracle_beats_random():
    X, y, t = make_data()
    oracle = (X[:, 0] > 0).astype(float)
    rnd = np.random.default_rng(1).random(len(y))
    assert qini_auc(y, t, oracle) > 0.01
    assert abs(qini_auc(y, t, rnd)) < 0.004


def test_uplift_at_k_recovers_true_effect():
    X, y, t = make_data()
    oracle = X[:, 0]
    # в топ-30% все клиенты с x0 > 0, истинный uplift = 0.3
    assert abs(uplift_at_k(y, t, oracle, 0.3) - 0.3) < 0.03


def test_qini_curve_ends_at_total_effect():
    X, y, t = make_data()
    x, q = qini_curve(y, t, X[:, 0])
    total = y[t == 1].sum() - y[t == 0].sum() * t.sum() / (1 - t).sum()
    assert x[0] == 0 and abs(x[-1] - 1) < 1e-9
    assert abs(q[-1] - total) < 1e-6


def test_models_find_heterogeneity():
    X, y, t = make_data()
    for model in (TLearner(), ClassTransformation()):
        s = model.fit(X[:15_000], y[:15_000], t[:15_000]).predict(X[15_000:])
        assert np.corrcoef(s, X[15_000:, 0] > 0)[0, 1] > 0.5
