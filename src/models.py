"""Три классических подхода к uplift-моделированию.

S-learner  — одна модель, флаг воздействия как признак; uplift = f(x, 1) − f(x, 0).
T-learner  — две модели, на тесте и на контроле; uplift = f_t(x) − f_c(x).
Class transformation (Jaskowski & Jaroszewicz, 2012) — новая цель
    Z = Y·T + (1−Y)·(1−T); при доле теста 50% uplift = 2·P(Z=1|x) − 1.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


def _gbm(seed: int = 0) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=200, l2_regularization=1.0, random_state=seed)


class SLearner:
    def fit(self, X, y, t):
        self.m = _gbm().fit(np.c_[X, t], y)
        return self

    def predict(self, X):
        ones, zeros = np.ones(len(X)), np.zeros(len(X))
        return self.m.predict_proba(np.c_[X, ones])[:, 1] - self.m.predict_proba(np.c_[X, zeros])[:, 1]


class TLearner:
    def fit(self, X, y, t):
        self.mt = _gbm(1).fit(X[t == 1], y[t == 1])
        self.mc = _gbm(2).fit(X[t == 0], y[t == 0])
        return self

    def predict(self, X):
        return self.mt.predict_proba(X)[:, 1] - self.mc.predict_proba(X)[:, 1]


class ClassTransformation:
    def fit(self, X, y, t):
        assert abs(t.mean() - 0.5) < 0.02, "трансформация классов корректна при доле теста ≈ 50%"
        z = (y * t + (1 - y) * (1 - t)).astype(int)
        self.m = _gbm(3).fit(X, z)
        return self

    def predict(self, X):
        return 2 * self.m.predict_proba(X)[:, 1] - 1


class TLearnerRegressor:
    """T-learner для денежной цели (траты): две регрессии, uplift = E[Y|x,T=1] − E[Y|x,T=0]."""

    def fit(self, X, y, t):
        kw = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0)
        self.mt = HistGradientBoostingRegressor(random_state=1, **kw).fit(X[t == 1], y[t == 1])
        self.mc = HistGradientBoostingRegressor(random_state=2, **kw).fit(X[t == 0], y[t == 0])
        return self

    def predict(self, X):
        return self.mt.predict(X) - self.mc.predict(X)
