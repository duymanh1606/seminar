"""
utils.py — Các lớp dữ liệu dùng chung, ghi nhật ký, và các tiện ích tải dữ liệu.
====================================================================
Mọi module khác đều import từ đây. Tệp này sở hữu các kiểu dữ liệu
miền chuẩn để các lớp duy trì sự tách biệt (Đảo ngược sự phụ thuộc).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# ── Ghi nhật ký ─────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("financial_ai")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
DECISIONS_LOG = PROJECT_ROOT / "decisions.jsonl"

for _d in (DATA_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ── Các lớp dữ liệu miền ─────────────────────────────────────────

@dataclass
class SentimentResult:
    """Đầu ra của Layer 1 — RAG + FinBERT."""
    text: str
    label: str                          # "positive" | "negative" | "neutral"
    score: float                        # −1.0 … +1.0
    source: str = ""                    # trích dẫn / nguồn gốc
    faithfulness: float = 1.0           # điểm số độ trung thực của RAG


@dataclass
class PredictionResult:
    """Đầu ra của Layer 2 — LSTM/GRU + MC Dropout."""
    mean: float                         # μ  — giá dự đoán
    variance: float                     # σ² — độ bất định nhận thức
    std: float                          # σ
    lower_95: float                     # μ − 2σ
    upper_95: float                     # μ + 2σ
    reject_flag: bool                   # True → quá bất định
    threshold: float                    # ngưỡng từ chối được sử dụng


@dataclass
class ExplanationResult:
    """Đầu ra của Layer 3 — SHAP XAI."""
    top_features: List[Tuple[str, float]]   # [(tên, |shap|), ...]
    shap_values: Optional[np.ndarray] = None


@dataclass
class TradeDecision:
    """Quyết định tổng hợp được ghi vào decisions.jsonl."""
    timestamp: str
    dataset: str
    sample_index: int
    sentiment: Optional[SentimentResult] = None
    prediction: Optional[PredictionResult] = None
    explanation: Optional[ExplanationResult] = None


# ── Trình ghi nhật ký JSON ─────────────────────────────────────────────────

class DecisionLogger:
    """Trình ghi nhật ký dạng JSONL chỉ nối thêm cho mọi quyết định giao dịch."""

    def __init__(self, path: Path = DECISIONS_LOG) -> None:
        self.path = path

    def log(self, decision: TradeDecision) -> None:
        record: Dict[str, Any] = {
            "timestamp": decision.timestamp,
            "dataset": decision.dataset,
            "sample_index": decision.sample_index,
        }
        if decision.prediction:
            record["prediction"] = {
                "mean": round(decision.prediction.mean, 4),
                "variance": round(decision.prediction.variance, 6),
                "std": round(decision.prediction.std, 4),
                "reject_flag": decision.prediction.reject_flag,
            }
        if decision.sentiment:
            record["sentiment"] = {
                "label": decision.sentiment.label,
                "score": round(decision.sentiment.score, 4),
                "faithfulness": round(decision.sentiment.faithfulness, 4),
            }
        if decision.explanation:
            record["top_features"] = [
                {"name": n, "importance": round(v, 4)}
                for n, v in decision.explanation.top_features
            ]

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Tiện ích tải dữ liệu ──────────────────────────────────────

FEATURE_NAMES: List[str] = [
    "interest_rate",
    "inflation",
    "trading_volume",
    "sentiment_score",
]


def fetch_yfinance_data(
    ticker: str = "^GSPC",
    start: str = "2018-01-01",
    end: str = "2026-05-31",
) -> pd.DataFrame:
    """
    Tải OHLCV từ Yahoo Finance.
    Chuyển về dữ liệu tổng hợp nếu yfinance không khả dụng hoặc ngoại tuyến.
    """
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            raise ValueError("DataFrame from yfinance is empty")
        df = df.reset_index()
        logger.info("Loaded %d rows for %s via yfinance.", len(df), ticker)
        return df
    except Exception as e:
        logger.warning("yfinance is unavailable (%s). Generating synthetic data.", e)
        return generate_synthetic_ohlcv(n_days=1500, label=ticker)


def generate_synthetic_ohlcv(
    n_days: int = 1000,
    label: str = "SYNTHETIC",
    crash_start: int = -1,
    crash_end: int = -1,
    noise_scale: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Trình tạo OHLCV tổng hợp tất định.

    Tham số
    ----------
    crash_start, crash_end : int
        Phạm vi chỉ mục để đưa vào nhiễu cực đoan (×8). Đặt cả hai thành -1 để bỏ qua.
    noise_scale : float
        Hệ số nhân nhiễu toàn cục (1.0 = giống S&P, 3.0 = giống crypto).
    """
    dates = pd.bdate_range(start="2018-01-02", end="2026-05-31")
    n_days = len(dates)

    rng = np.random.RandomState(seed)
    t = np.arange(n_days, dtype=np.float32)

    base_price = 100 + 0.02 * t + 15 * np.sin(2 * np.pi * t / 252)
    noise = rng.normal(0, noise_scale, n_days)

    if crash_start >= 0 and crash_end > crash_start:
        noise[crash_start:crash_end] *= 8.0

    close = (base_price + noise).astype(np.float32)
    high = close + np.abs(rng.normal(0, 0.5 * noise_scale, n_days))
    low = close - np.abs(rng.normal(0, 0.5 * noise_scale, n_days))
    open_ = close + rng.normal(0, 0.3 * noise_scale, n_days)
    volume = (1e6 + rng.normal(0, 1e5, n_days)).clip(1e4).astype(np.int64)

    df = pd.DataFrame({
        "Date": dates[:n_days],
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    })
    logger.info("Generated %d synthetic rows [%s].", n_days, label)
    return df


def generate_mock_sentiment(
    dates: pd.DatetimeIndex,
    crash_start: int = -1,
    crash_end: int = -1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    CSV mô phỏng cảm tính trên mạng xã hội (đại diện cho Twitter/Reddit).
    Trả về DataFrame với các cột: Date, headline, sentiment_score.
    """
    rng = np.random.RandomState(seed)
    n = len(dates)

    base_sent = 0.6 + 0.2 * np.sin(
        2 * np.pi * np.arange(n) / 90
    ) + rng.normal(0, 0.08, n)

    if crash_start >= 0 and crash_end > crash_start:
        base_sent[crash_start:crash_end] -= 0.9

    base_sent = np.clip(base_sent, -1.0, 1.0)

    headlines = [
        "Market rallies on strong earnings" if s > 0.3
        else "Growing concerns about economic recession" if s < -0.1
        else "Market trades mixed amid uncertainty"
        for s in base_sent
    ]

    return pd.DataFrame({
        "Date": dates,
        "headline": headlines,
        "sentiment_score": base_sent.astype(np.float32),
    })


def build_feature_matrix(
    ohlcv: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    seq_length: int = 14,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Hợp nhất OHLCV + cảm tính → ma trận đặc trưng + mục tiêu.

    Trả về
    -------
    X : ndarray (N, num_features)
    y : ndarray (N,)
    feature_names : list[str]
    """
    df = ohlcv.copy()

    # Các chỉ báo kỹ thuật
    df["return_1d"] = df["Close"].pct_change()
    df["volatility_5d"] = df["return_1d"].rolling(5).std()
    df["sma_10"] = df["Close"].rolling(10).mean()
    df["volume_norm"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # Hợp nhất cảm tính (dịch chuyển 1 để ngăn ngừa rò rỉ dữ liệu tương lai)
    sent = sentiment_df[["Date", "sentiment_score"]].copy()
    sent["sentiment_score"] = sent["sentiment_score"].shift(1)
    df = df.merge(sent, on="Date", how="left")
    df["sentiment_score"] = df["sentiment_score"].fillna(0)

    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        "return_1d", "volatility_5d", "sma_10",
        "volume_norm", "sentiment_score",
    ]
    X = df[feature_cols].values.astype(np.float32)
    y = df["Close"].values.astype(np.float32)

    return X, y, feature_cols


class TimeSeriesScaler:
    """Chỉ khớp trên tập huấn luyện, biến đổi cả tập huấn luyện và kiểm tra (không rò rỉ)."""

    def __init__(self) -> None:
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_sc = self.scaler_X.fit_transform(X)
        y_sc = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        return X_sc.astype(np.float32), y_sc.astype(np.float32)

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        return self.scaler_X.transform(X).astype(np.float32)

    def inverse_y(self, y_scaled: np.ndarray) -> np.ndarray:
        return self.scaler_y.inverse_transform(
            y_scaled.reshape(-1, 1)
        ).flatten()

    @property
    def y_scale(self) -> float:
        return float(self.scaler_y.scale_[0])
