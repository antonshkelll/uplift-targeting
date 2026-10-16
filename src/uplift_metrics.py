"""Метрики качества uplift-моделей.

У каждого клиента мы видим исход только в одной группе, поэтому «правильного ответа»
для конкретного человека нет. Качество оцениваем на уровне групп: сортируем клиентов
по предсказанному uplift и смотрим, сколько дополнительного эффекта набирается в топе.
"""
from __future__ import annotations

import numpy as np


def qini_curve(y: np.ndarray, t: np.ndarray, score: np.ndarray, n_points: int = 101):
    """Qini-кривая: накопленный инкрементальный эффект при таргетинге топ-доли клиентов.

    Q(k) = Y_t(k) − Y_c(k) · N_t(k) / N_c(k), где Y_t/Y_c — сумма исхода в тесте/контроле
    среди топ-k по скору. Возвращает (доли, значения кривой).
    """
    order = np.argsort(-score, kind="stable")
    y, t = y[order].astype(float), t[order].astype(int)
    yt, yc = np.cumsum(y * t), np.cumsum(y * (1 - t))
    nt, nc = np.cumsum(t), np.cumsum(1 - t)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = yt - yc * np.where(nc > 0, nt / nc, 0.0)
    idx = np.unique(np.linspace(0, len(y) - 1, n_points).astype(int))
    frac = (idx + 1) / len(y)
    return np.r_[0.0, frac], np.r_[0.0, q[idx]]


def qini_auc(y, t, score) -> float:
    """Площадь между Qini-кривой модели и диагональю случайного таргетинга (нормированная на N)."""
    x, q = qini_curve(y, t, score)
    random_line = x * q[-1]
    return float(np.trapezoid(q - random_line, x) / len(y))


def uplift_at_k(y, t, score, k: float) -> float:
    """Разница средних исходов тест − контроль среди топ-k доли клиентов по скору."""
    n = max(1, int(len(y) * k))
    top = np.argsort(-score, kind="stable")[:n]
    yt, tt = y[top], t[top]
    if tt.sum() == 0 or (1 - tt).sum() == 0:
        return float("nan")
    return float(yt[tt == 1].mean() - yt[tt == 0].mean())
