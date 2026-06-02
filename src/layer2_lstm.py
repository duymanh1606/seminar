"""
layer2_lstm.py — Layer 2: Hybrid LSTM-GRU với MC Dropout Uncertainty.
======================================================================
Cung cấp pipeline dự đoán chuỗi thời gian với việc định lượng độ không chắc chắn
*epistemic* (nhận thức) thông qua Monte Carlo Dropout. Kiến trúc xếp chồng một bộ
mã hóa (encoder) LSTM với một lớp tinh chỉnh GRU, theo sau là dropout và một
đầu chiếu tuyến tính (linear projection head).

Public API
----------
- ``create_sequences``      — hàm hỗ trợ tạo cửa sổ trượt cho dữ liệu chuỗi thời gian.
- ``HybridLSTMGRU``         — nn.Module xương sống (backbone).
- ``MCDropoutPredictor``    — wrapper bậc cao để huấn luyện, hiệu chuẩn,
                              và suy luận nhận biết độ không chắc chắn.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils import PredictionResult, logger


# ── Xây Dựng Chuỗi ──────────────────────────────────────────────


def create_sequences(
    data_X: np.ndarray,
    data_y: np.ndarray,
    seq_length: int = 14,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tạo các chuỗi cửa sổ trượt chồng chéo nhau cho mô hình hóa chuỗi thời gian.

    Parameters
    ----------
    data_X : np.ndarray
        Ma trận đặc trưng có kích thước ``(T, num_features)``.
    data_y : np.ndarray
        Vector mục tiêu có kích thước ``(T,)``.
    seq_length : int, optional
        Số bước thời gian nhìn lại cho mỗi mẫu (mặc định ``14``).

    Returns
    -------
    X_seq : np.ndarray
        Các chuỗi có kích thước ``(N, seq_length, num_features)`` với
        ``N = T - seq_length``.
    y_seq : np.ndarray
        Các mục tiêu tương ứng có kích thước ``(N,)`` — mỗi mục là
        giá trị mục tiêu tại bước *ngay sau* cửa sổ chuỗi.
    """
    X_seq: list[np.ndarray] = []
    y_seq: list[float] = []

    for i in range(len(data_X) - seq_length):
        X_seq.append(data_X[i : i + seq_length])
        y_seq.append(data_y[i + seq_length])

    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


# ── Backbone Hybrid LSTM-GRU ───────────────────────────────────


class HybridLSTMGRU(nn.Module):
    """Bộ mã hóa xếp chồng LSTM → GRU với dropout và một đầu tuyến tính.

    Architecture
    ------------
    1. Bộ mã hóa **LSTM** ánh xạ ``(batch, seq_length, input_dim)`` →
       trạng thái ẩn có kích thước ``hidden_dim``.
    2. **GRU** tinh chỉnh chuỗi đầu ra của LSTM và tạo ra biểu diễn ẩn
       của riêng nó.
    3. **Dropout** được áp dụng vào trạng thái ẩn GRU cuối cùng cho mục đích
       điều chuẩn (và MC Dropout tại thời điểm suy luận).
    4. Đầu **Linear** chiếu ``hidden_dim`` → ``1`` (dự đoán vô hướng).

    Parameters
    ----------
    input_dim : int
        Số lượng đặc trưng đầu vào cho mỗi bước thời gian.
    hidden_dim : int, optional
        Kích thước ẩn cho cả LSTM và GRU (mặc định ``64``).
    num_layers : int, optional
        Số lượng lớp hồi quy trong *mỗi* LSTM và GRU
        (mặc định ``1``).
    dropout : float, optional
        Xác suất dropout được áp dụng sau GRU (mặc định ``0.2``).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lượt truyền xuôi.

        Parameters
        ----------
        x : torch.Tensor
            Tensor đầu vào có kích thước ``(batch, seq_length, input_dim)``.

        Returns
        -------
        torch.Tensor
            Các dự đoán có kích thước ``(batch, 1)``.
        """
        # Bộ mã hóa LSTM — sử dụng toàn bộ chuỗi đầu ra làm đầu vào cho GRU
        lstm_out, _ = self.lstm(x)          # (batch, seq, hidden)

        # Tinh chỉnh GRU — lấy trạng thái ẩn cuối cùng
        _, gru_hidden = self.gru(lstm_out)  # gru_hidden: (layers, batch, hidden)

        # Sử dụng trạng thái ẩn từ lớp GRU cuối cùng
        out = gru_hidden[-1]                # (batch, hidden)
        out = self.dropout(out)             # điều chuẩn / MC Dropout
        out = self.fc(out)                  # (batch, 1)
        return out


# ── Bộ Dự Đoán MC Dropout ───────────────────────────────────────


class MCDropoutPredictor:
    """Wrapper bậc cao bao quanh :class:`HybridLSTMGRU` bổ sung tính năng
    huấn luyện, tự động hiệu chuẩn ngưỡng từ chối, và ước lượng độ không chắc chắn
    MC Dropout.

    Parameters
    ----------
    input_dim : int
        Số lượng đặc trưng đầu vào cho mỗi bước thời gian.
    hidden_dim : int, optional
        Số chiều ẩn (mặc định ``64``).
    num_layers : int, optional
        Độ sâu hồi quy (mặc định ``1``).
    dropout : float, optional
        Tỷ lệ dropout (mặc định ``0.2``).
    device : str, optional
        Chuỗi thiết bị Torch (mặc định ``"cpu"``).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = HybridLSTMGRU(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

    # ── Huấn Luyện ──────────────────────────────────────────────────

    def train_model(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 64,
    ) -> List[float]:
        """Huấn luyện mô hình sử dụng hàm mất mát MSE và bộ tối ưu hóa Adam.

        Parameters
        ----------
        X_train : torch.Tensor
            Các chuỗi huấn luyện ``(N, seq_length, input_dim)``.
        y_train : torch.Tensor
            Các mục tiêu huấn luyện ``(N,)``.
        epochs : int, optional
            Số lần duyệt qua toàn bộ dữ liệu (mặc định ``30``).
        lr : float, optional
            Tốc độ học cho Adam (mặc định ``1e-3``).
        batch_size : int, optional
            Kích thước mini-batch (mặc định ``64``).

        Returns
        -------
        List[float]
            Mất mát huấn luyện trung bình mỗi epoch.
        """
        dataset = TensorDataset(
            X_train.to(self.device),
            y_train.to(self.device),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        loss_history: list[float] = []
        self.model.train()

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                preds = self.model(X_batch).squeeze(-1)  # (batch,)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * X_batch.size(0)

            avg_loss = epoch_loss / len(dataset)
            loss_history.append(avg_loss)

            if epoch % 10 == 0 or epoch == 1:
                logger.info("Epoch %3d/%d  —  loss: %.6f", epoch, epochs, avg_loss)

        logger.info("Training complete. Final loss: %.6f", loss_history[-1])
        return loss_history

    # ── Hiệu Chuẩn Ngưỡng ───────────────────────────────────────────

    def calibrate_threshold(
        self,
        X_train: torch.Tensor,
        T: int = 50,
        z: float = 2.5,
    ) -> float:
        """Tự động hiệu chuẩn ngưỡng từ chối từ độ không chắc chắn của tập huấn luyện.

        Chạy suy luận MC Dropout trên ``X_train`` để thu được một phân phối
        của các độ lệch chuẩn trên mỗi mẫu, sau đó thiết lập ngưỡng là:

            ``ngưỡng = mean(stds) + z * std(stds)``

        Parameters
        ----------
        X_train : torch.Tensor
            Các chuỗi huấn luyện ``(N, seq_length, input_dim)``.
        T : int, optional
            Số lượt truyền xuôi MC (mặc định ``50``).
        z : float, optional
            Số độ lệch chuẩn trên mức trung bình để đặt
            ngưỡng (mặc định ``2.5``).

        Returns
        -------
        float
            Ngưỡng từ chối được hiệu chuẩn.
        """
        _, stds, _ = self.predict_with_uncertainty(X_train, T=T, threshold=0.0)

        mu_std = float(np.mean(stds))
        sigma_std = float(np.std(stds))
        threshold = mu_std + z * sigma_std

        logger.info(
            "Calibrated rejection threshold: %.6f  (μ_σ=%.6f, σ_σ=%.6f, z=%.1f)",
            threshold, mu_std, sigma_std, z,
        )
        return threshold

    # ── Suy Luận MC Dropout ─────────────────────────────────────────

    def predict_with_uncertainty(
        self,
        X: torch.Tensor,
        T: int = 50,
        threshold: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Chạy suy luận MC Dropout để thu được các dự đoán trung bình,
        độ không chắc chắn nhận thức (độ lệch chuẩn std), và các cờ từ chối.

        Parameters
        ----------
        X : torch.Tensor
            Các chuỗi đầu vào ``(N, seq_length, input_dim)``.
        T : int, optional
            Số lượt truyền xuôi ngẫu nhiên (mặc định ``50``).
        threshold : float, optional
            Ngưỡng từ chối — các mẫu có std vượt quá giá trị này
            sẽ được gắn cờ (mặc định ``0.0``, tức là gắn cờ tất cả).

        Returns
        -------
        means : np.ndarray
            Dự đoán trung bình, kích thước ``(N,)``.
        stds : np.ndarray
            Độ lệch chuẩn (độ không chắc chắn nhận thức), kích thước ``(N,)``.
        reject_flags : np.ndarray
            Mảng boolean kích thước ``(N,)`` — ``True`` khi
            độ không chắc chắn vượt quá ``threshold``.

        Notes
        -----
        Chúng tôi chủ ý gọi ``model.train()`` thay vì ``model.eval()``
        trước các lượt truyền xuôi. Điều này giữ cho các lớp dropout **kích hoạt**
        trong quá trình suy luận, đó là ý tưởng cốt lõi đằng sau *MC Dropout*
        (Gal & Ghahramani, 2016). Bằng cách lấy mẫu ``T`` mặt nạ dropout
        khác nhau, chúng tôi thu được một xấp xỉ phân phối dự đoán hậu nghiệm
        mà độ phân tán của nó định lượng độ không chắc chắn nhận thức của mô hình.
        """
        # ── QUAN TRỌNG: giữ dropout kích hoạt để lấy mẫu MC ──
        # model.train() đảm bảo nn.Dropout vẫn mang tính ngẫu nhiên nên mỗi
        # lượt trong T lượt truyền xuôi tạo ra một dự đoán khác nhau,
        # xấp xỉ một hậu nghiệm Bayes trên các trọng số.
        self.model.train()

        X_dev = X.to(self.device)
        predictions: list[np.ndarray] = []

        with torch.no_grad():
            for _ in range(T):
                preds = self.model(X_dev).squeeze(-1).cpu().numpy()  # (N,)
                predictions.append(preds)

        # Xếp chồng → (T, N) sau đó tổng hợp qua các lượt truyền ngẫu nhiên
        stacked = np.stack(predictions, axis=0)  # (T, N)
        means: np.ndarray = stacked.mean(axis=0)  # (N,)
        stds: np.ndarray = stacked.std(axis=0)    # (N,)

        # Gắn cờ các mẫu có độ không chắc chắn vượt quá ngưỡng
        reject_flags: np.ndarray = stds > threshold  # (N,) bool

        n_rejected = int(reject_flags.sum())
        logger.info(
            "MC Dropout (T=%d): %d/%d samples rejected (threshold=%.6f)",
            T, n_rejected, len(means), threshold,
        )

        return means, stds, reject_flags
