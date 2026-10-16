"""Полный анализ: проверка эксперимента → средние эффекты → неоднородность →
uplift-модели → экономика таргетинга → политика «лучшее предложение каждому».

Запуск: python src/run.py   (≈1 минута на ноутбуке)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

from data import FEATURES, load
from models import ClassTransformation, SLearner, TLearner, TLearnerRegressor
from uplift_metrics import qini_auc, qini_curve, uplift_at_k

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

MARGIN = 0.30          # маржинальность выручки
CONTACT_COST = 0.15    # $ на контакт: стоимость канала + скидка в предложении
SEEDS = range(5)

plt.rcParams.update({"figure.dpi": 140, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.2})
INK, BLUE, RED, GREY = "#1d1d1f", "#0071e3", "#e3383a", "#a1a1a6"


def mean_diff(a: pd.Series, b: pd.Series) -> dict:
    d = a.mean() - b.mean()
    se = np.sqrt(a.var() / len(a) + b.var() / len(b))
    return dict(effect=round(d, 5), ci95=[round(d - 1.96 * se, 5), round(d + 1.96 * se, 5)],
                rel=round(d / b.mean(), 3), p=float(f"{2 * stats.norm.sf(abs(d / se)):.2e}"))


def check_experiment(df: pd.DataFrame) -> dict:
    counts = df.arm.value_counts()
    srm_p = stats.chisquare(counts.values).pvalue
    # стандартизированная разница средних: |SMD| < 0.1 считается хорошим балансом
    smd = {}
    ctrl = df[df.arm == "control"]
    for arm in ("mens", "womens"):
        g = df[df.arm == arm]
        smd[arm] = {f: round(float((g[f].mean() - ctrl[f].mean()) / np.sqrt((g[f].var() + ctrl[f].var()) / 2)), 4)
                    for f in FEATURES}
    worst = max(abs(v) for a in smd.values() for v in a.values())
    return dict(sizes=counts.to_dict(), srm_p=round(float(srm_p), 3), max_abs_smd=round(worst, 4))


def average_effects(df: pd.DataFrame) -> dict:
    ctrl = df[df.arm == "control"]
    return {arm: {y: mean_diff(df[df.arm == arm][y], ctrl[y]) for y in ("visit", "conversion", "spend")}
            for arm in ("mens", "womens")}


def heterogeneity(df: pd.DataFrame) -> pd.DataFrame:
    """Эффект мужского письма на визиты в разных сегментах."""
    df = df.assign(buyer=np.select([(df.mens == 1) & (df.womens == 0), (df.mens == 0) & (df.womens == 1)],
                                   ["только мужское", "только женское"], "оба"))
    rows = []
    for col, label in [("buyer", "что покупал раньше"), ("newbie", "новый клиент"),
                       ("channel", "канал покупок"), ("history_segment", "сумма покупок за год")]:
        src = df if col != "channel" else df.assign(channel=df[[c for c in df if c.startswith("channel_")]].idxmax(axis=1)
                                                    .str.replace("channel_", ""))
        for val, g in src.groupby(col):
            t, c = g[g.arm == "mens"].visit, g[g.arm == "control"].visit
            r = mean_diff(t, c)
            rows.append(dict(признак=label, значение=str(val), клиентов=len(g) // 3 * 2,
                             uplift_пп=round(r["effect"] * 100, 1),
                             ci95=f"[{r['ci95'][0] * 100:.1f}; {r['ci95'][1] * 100:.1f}]"))
    return pd.DataFrame(rows)


def compare_models(df: pd.DataFrame, target: str) -> tuple[dict, dict]:
    """Мужское письмо vs без письма. 5 случайных разбиений 70/30 — чтобы не зависеть от удачного сплита."""
    d = df[df.arm.isin(["mens", "control"])]
    X, y, t = d[FEATURES].to_numpy(float), d[target].to_numpy(), (d.arm == "mens").to_numpy(int)
    models = {"S-learner": SLearner, "T-learner": TLearner, "Class transformation": ClassTransformation}
    res = {m: dict(qini=[], up10=[], up30=[]) for m in models}
    curves = {}
    for seed in SEEDS:
        Xtr, Xte, ytr, yte, ttr, tte = train_test_split(X, y, t, test_size=0.3, random_state=seed,
                                                        stratify=t * 2 + y)
        for name, cls in models.items():
            s = cls().fit(Xtr, ytr, ttr).predict(Xte)
            res[name]["qini"].append(qini_auc(yte, tte, s))
            res[name]["up10"].append(uplift_at_k(yte, tte, s, 0.1))
            res[name]["up30"].append(uplift_at_k(yte, tte, s, 0.3))
            if seed == 0:
                curves[name] = qini_curve(yte, tte, s)
        if seed == 0:
            curves["_random"] = qini_curve(yte, tte, np.random.default_rng(0).random(len(yte)))
            curves["_avg_uplift"] = float(yte[tte == 1].mean() - yte[tte == 0].mean())
    summary = {m: {k: [round(float(np.mean(v)), 5), round(float(np.std(v)), 5)] for k, v in r.items()}
               for m, r in res.items()}
    return summary, curves


def targeting_economics(df: pd.DataFrame) -> dict:
    """Прибыль от рассылки топ-k% клиентов по скору T-learner. Усредняем по 5 разбиениям:
    траты редкие и шумные, одно разбиение может обмануть."""
    d = df[df.arm.isin(["mens", "control"])]
    X, t = d[FEATURES].to_numpy(float), (d.arm == "mens").to_numpy(int)
    visit, spend_all = d.visit.to_numpy(), d.spend.to_numpy()
    ks = np.round(np.linspace(0.1, 1.0, 10), 2)
    curves = {"по uplift визитов": [], "по uplift трат": [], "случайно": []}
    for seed in SEEDS:
        tr, te = train_test_split(np.arange(len(d)), test_size=0.3, random_state=seed, stratify=t)
        by_visit = TLearner().fit(X[tr], visit[tr], t[tr]).predict(X[te])
        by_spend = TLearnerRegressor().fit(X[tr], spend_all[tr], t[tr]).predict(X[te])
        spend, tt, n = spend_all[te], t[te], len(te)
        for policy, s in (("по uplift визитов", by_visit), ("по uplift трат", by_spend),
                          ("случайно", np.random.default_rng(seed).random(n))):
            row = []
            for k in ks:
                top = np.argsort(-s)[: int(n * k)]
                inc = spend[top][tt[top] == 1].mean() - spend[top][tt[top] == 0].mean()  # $ на контакт
                row.append(1000 * k * (inc * MARGIN - CONTACT_COST))                    # $ на 1000 клиентов
            curves[policy].append(row)
    m = {p: np.mean(v, 0) for p, v in curves.items()}
    sd = {p: np.std(v, 0) for p, v in curves.items()}

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for policy, color in (("по uplift визитов", BLUE), ("по uplift трат", RED), ("случайно", GREY)):
        ax.plot(ks * 100, m[policy], color=color, lw=2, label=policy)
        ax.fill_between(ks * 100, m[policy] - sd[policy], m[policy] + sd[policy], color=color, alpha=0.15, lw=0)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set(xlabel="кому отправляем, % базы (по убыванию прогноза uplift)", ylabel="прибыль, $ на 1000 клиентов",
           title="Экономика рассылки (среднее ± разброс по 5 разбиениям)")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "profit.png"); plt.close(fig)
    table = {f"top_{int(k * 100)}%": {p: [round(float(m[p][i]), 1), round(float(sd[p][i]), 1)] for p in m}
             for i, k in enumerate(ks)}
    return dict(table=table, margin=MARGIN, contact_cost=CONTACT_COST)


def capture_rates(df: pd.DataFrame) -> dict:
    """Какую долю всех дополнительных визитов даёт рассылка только топ-k% базы (T-learner, 5 разбиений)."""
    d = df[df.arm.isin(["mens", "control"])]
    X, y, t = d[FEATURES].to_numpy(float), d.visit.to_numpy(), (d.arm == "mens").to_numpy(int)
    ks = (0.1, 0.2, 0.3, 0.5, 0.7)
    res = {k: [] for k in ks}
    for seed in SEEDS:
        Xtr, Xte, ytr, yte, ttr, tte = train_test_split(X, y, t, test_size=0.3, random_state=seed, stratify=t * 2 + y)
        x, q = qini_curve(yte, tte, TLearner().fit(Xtr, ytr, ttr).predict(Xte), n_points=1001)
        for k in ks:
            res[k].append(np.interp(k, x, q) / q[-1])
    return {f"top_{int(k * 100)}%": [round(float(np.mean(v)), 3), round(float(np.std(v)), 3)] for k, v in res.items()}


def next_best_offer(df: pd.DataFrame) -> dict:
    """Политика «каждому своё»: мужское письмо, женское или ничего — по прогнозу uplift визита.

    Ценность политики оцениваем без нового эксперимента: группы назначены случайно с вероятностью 1/3,
    поэтому V(π) = среднее Y · 1[T = π(x)] / (1/3) — несмещённая IPW-оценка на отложенной выборке.
    """
    X = df[FEATURES].to_numpy(float)
    arm = df.arm.to_numpy()
    runs = [_nbo_once(df, X, arm, seed) for seed in SEEDS]
    out = {}
    for name in runs[0]:
        if name.startswith("_"):
            continue
        out[name] = {k: [round(float(np.mean([r[name][k] for r in runs])), 1),
                         round(float(np.std([r[name][k] for r in runs])), 1)] for k in runs[0][name]}
    out["_policy_mix"] = runs[0]["_policy_mix"]
    return out


def _nbo_once(df, X, arm, seed) -> dict:
    tr, te = train_test_split(np.arange(len(df)), test_size=0.3, random_state=seed, stratify=df.arm)
    preds = {}
    for a in ("mens", "womens"):
        idx = tr[np.isin(arm[tr], [a, "control"])]
        m = TLearner().fit(X[idx], df.visit.to_numpy()[idx], (arm[idx] == a).astype(int))
        preds[a] = m.predict(X[te])
    # ожидаемая выгода письма = uplift визита × средний чек визита × маржа − стоимость контакта
    spend_per_visit = df.loc[df.visit == 1, "spend"].mean()
    gain = {a: preds[a] * spend_per_visit * MARGIN - CONTACT_COST for a in preds}
    choice = np.where(np.maximum(gain["mens"], gain["womens"]) <= 0, "control",
                      np.where(gain["mens"] >= gain["womens"], "mens", "womens"))

    y_visit, y_spend, a_te = df.visit.to_numpy()[te], df.spend.to_numpy()[te], arm[te]

    def value(policy: np.ndarray) -> dict:
        w = (a_te == policy) * 3.0
        contacted = (policy != "control").mean()
        profit = (y_spend * w).mean() * MARGIN - contacted * CONTACT_COST
        return dict(visits_per_1000=1000 * (y_visit * w).mean(),
                    profit_per_1000=1000 * profit, contacted_pct=100 * float(contacted))

    rng = np.random.default_rng(seed)
    out = {name: value(p) for name, p in {
        "никому": np.full(len(te), "control"),
        "всем мужское": np.full(len(te), "mens"),
        "всем женское": np.full(len(te), "womens"),
        "случайно": rng.choice(["mens", "womens", "control"], len(te)),
        "модель: лучшее каждому": choice,
    }.items()}
    out["_policy_mix"] = pd.Series(choice).value_counts(normalize=True).round(3).to_dict()
    return out


def plot_qini(curves: dict, target: str) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    colors = {"S-learner": GREY, "T-learner": BLUE, "Class transformation": RED}
    for name, xq in curves.items():
        if name.startswith("_"):
            continue
        x, q = xq
        ax.plot(x * 100, q, color=colors[name], lw=2, label=name)
    x, q = curves["_random"]
    ax.plot([0, 100], [0, q[-1]], color=INK, ls="--", lw=1, label="случайный порядок")
    ax.set(xlabel="доля клиентов, которым отправили письмо, %", ylabel="дополнительные визиты (накопленно)",
           title=f"Qini-кривые, цель: {target}")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / f"qini_{target}.png"); plt.close(fig)


def main() -> None:
    df = load()
    out = dict(n=len(df), experiment=check_experiment(df), average_effects=average_effects(df))
    sp = out["average_effects"]["mens"]["spend"]
    to_profit = lambda u: round(1000 * (u * MARGIN - CONTACT_COST), 1)   # $ на 1000 клиентов
    out["send_all_mens_profit_per_1000"] = dict(estimate=to_profit(sp["effect"]),
                                                ci95=[to_profit(sp["ci95"][0]), to_profit(sp["ci95"][1])])
    het = heterogeneity(df)
    het.to_csv(ROOT / "reports" / "heterogeneity.csv", index=False)
    out["heterogeneity"] = het.to_dict("records")
    out["models"] = {}
    for target in ("visit", "conversion"):
        summary, curves = compare_models(df, target)
        out["models"][target] = summary
        plot_qini(curves, target)
    out["capture_rates_visits"] = capture_rates(df)
    out["economics"] = targeting_economics(df)
    out["next_best_offer"] = next_best_offer(df)
    (ROOT / "reports" / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
