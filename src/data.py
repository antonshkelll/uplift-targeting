"""Загрузка и подготовка датасета Hillstrom (MineThatData E-Mail Analytics Challenge, 2008).

64 000 покупателей интернет-магазина случайно разделены на три группы:
письмо с мужской коллекцией, письмо с женской коллекцией, без письма.
Через две недели измерены визит на сайт, покупка и сумма покупки.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "hillstrom.csv.gz"
URL = "https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz"

ARMS = {"No E-Mail": "control", "Mens E-Mail": "mens", "Womens E-Mail": "womens"}
FEATURES = ["recency", "history", "mens", "womens", "newbie",
            "zip_Rural", "zip_Surburban", "zip_Urban",
            "channel_Multichannel", "channel_Phone", "channel_Web"]


def load() -> pd.DataFrame:
    if not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(URL).to_csv(RAW, index=False, compression="gzip")
    df = pd.read_csv(RAW)
    df["arm"] = df["segment"].map(ARMS)
    df = pd.get_dummies(df, columns=["zip_code", "channel"], prefix=["zip", "channel"], dtype=int)
    return df
