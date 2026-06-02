"""
layer3_xai.py — Lớp 3: Giải thích & Phân tích nhân quả
=========================================================
Cung cấp các giải thích về tầm quan trọng của đặc trưng dựa trên SHAP và một mô-đun
đồ thị nhân quả giả lập nhẹ. Cả hai lớp đều tạo ra các thành phần đầu vào cho
lớp phức hợp :class:`TradeDecision` được ghi lại bởi *utils.DecisionLogger*.

Các lớp
-------
ShapExplainer       – bao bọc ``shap.GradientExplainer`` cho các mô hình PyTorch.
CausalGraphAnalyzer – xây dựng / vẽ một DAG kinh tế vĩ mô hợp lý và
                      kiểm tra chéo nó với đầu ra của SHAP.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from utils import ExplanationResult, logger

# ── Nhập SHAP một cách mượt mà ────────────────────────────────────────

try:
    import shap  # type: ignore[import-untyped]

    _SHAP_AVAILABLE: bool = True
except ImportError:
    shap = None  # type: ignore[assignment]
    _SHAP_AVAILABLE = False
    logger.warning(
        "shap is not installed — ShapExplainer will raise an error at runtime."
    )


# ── ShapExplainer ───────────────────────────────────────────────


class ShapExplainer:
    """Tính toán các giá trị SHAP cho từng đặc trưng cho một mô hình PyTorch.

    Sử dụng ``shap.GradientExplainer`` ở bên dưới. Mô hình được đặt ở
    chế độ ``eval()`` trong quá trình giải thích để vô hiệu hóa việc cập nhật
    dropout / batch-norm.

    Tham số
    ----------
    model : nn.Module
        Một mô hình PyTorch đã được huấn luyện có lượt truyền xuôi chấp nhận một tensor 2-D
        ``(batch, features)``.
    background_data : torch.Tensor
        Tập dữ liệu nền tiêu biểu (thường là tập huấn luyện hoặc một
        mẫu con). Kích thước ``(N, features)``.
    feature_names : list[str]
        Tên con người có thể đọc được căn chỉnh với trục đặc trưng.

    Ngoại lệ
    ------
    RuntimeError
        Nếu gói ``shap`` chưa được cài đặt.
    """

    def __init__(
        self,
        model: nn.Module,
        background_data: torch.Tensor,
        feature_names: List[str],
    ) -> None:
        if not _SHAP_AVAILABLE:
            raise RuntimeError(
                "shap được yêu cầu cho ShapExplainer nhưng chưa được cài đặt. "
                "Cài đặt nó với:  pip install shap"
            )

        self.model = model
        self.background_data = background_data
        self.feature_names = feature_names

        # Xây dựng GradientExplainer một lần (được sử dụng lại cho mỗi lệnh gọi explain).
        self.model.eval()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.explainer = shap.GradientExplainer(self.model, self.background_data)

        logger.info(
            "ShapExplainer initialized — %d background samples, %d features.",
            background_data.shape[0],
            len(feature_names),
        )

    # ── API công khai ──────────────────────────────────────────────

    def explain(
        self,
        sample: torch.Tensor,
        top_k: int = 3,
    ) -> ExplanationResult:
        """Tính toán các giá trị SHAP cho *sample* và trả về top-k đặc trưng.

        Tham số
        ----------
        sample : torch.Tensor
            Tensor đầu vào có kích thước ``(1, features)`` hoặc ``(features,)``.
            Một chiều batch được thêm tự động khi cần thiết.
        top_k : int, tùy chọn
            Số lượng các đặc trưng quan trọng nhất để đưa vào kết quả
            (mặc định ``3``).

        Trả về
        -------
        ExplanationResult
            ``top_features`` chứa các cặp ``(name, mean_|shap|)`` được sắp xếp
            theo thứ tự tầm quan trọng giảm dần. ``shap_values`` lưu trữ
            mảng SHAP thô.
        """
        if sample.ndim == 1:
            sample = sample.unsqueeze(0)

        self.model.eval()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_shap: np.ndarray = np.array(
                self.explainer.shap_values(sample)
            )

        # raw_shap có thể có kích thước (outputs, batch, features) hoặc
        # (batch, features) — chuẩn hóa thành (batch, features).
        if raw_shap.ndim == 3:
            raw_shap = raw_shap[0]

        # Giá trị tuyệt đối trung bình của SHAP trên toàn bộ batch cho mỗi đặc trưng.
        mean_abs: np.ndarray = np.mean(np.abs(raw_shap), axis=0)

        # Xếp hạng các đặc trưng theo tầm quan trọng (giảm dần).
        ranked_indices: np.ndarray = np.argsort(mean_abs)[::-1]
        top_features: List[Tuple[str, float]] = [
            (self.feature_names[i], float(mean_abs[i]))
            for i in ranked_indices[:top_k]
        ]

        logger.info(
            "SHAP explanation — top-%d: %s",
            top_k,
            ", ".join(f"{n}={v:.4f}" for n, v in top_features),
        )

        return ExplanationResult(
            top_features=top_features,
            shap_values=raw_shap,
        )

    # ── trực quan hóa ───────────────────────────────────────────

    def plot_importance(
        self,
        result: ExplanationResult,
        save_path: Optional[str] = None,
    ) -> None:
        """Vẽ biểu đồ thanh ngang về tầm quan trọng của đặc trưng SHAP.

        Biểu đồ tuân theo phong cách trực quan học thuật: họ phông chữ serif,
        các đường lưới tinh tế và DPI cao (300).

        Tham số
        ----------
        result : ExplanationResult
            Đầu ra của :meth:`explain`.
        save_path : str hoặc None, tùy chọn
            Nếu được cung cấp, hình ảnh được lưu vào đường dẫn này thay vì được
            hiển thị tương tác.
        """
        names = [t[0] for t in result.top_features]
        values = [t[1] for t in result.top_features]

        # Đảo ngược để đặc trưng quan trọng nhất xuất hiện ở trên cùng.
        names = names[::-1]
        values = values[::-1]

        # Cài đặt matplotlib theo phong cách học thuật.
        with plt.rc_context(
            {
                "font.family": "serif",
                "font.size": 11,
                "axes.grid": True,
                "grid.alpha": 0.35,
                "grid.linestyle": "--",
                "figure.dpi": 300,
            }
        ):
            fig, ax = plt.subplots(figsize=(7, max(3, 0.55 * len(names))))
            y_pos = np.arange(len(names))
            ax.barh(
                y_pos,
                values,
                color="#3b7dd8",
                edgecolor="#1f4e8c",
                linewidth=0.6,
                height=0.6,
            )
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names)
            ax.set_xlabel("Trung bình |Giá trị SHAP|")
            ax.set_title("Tầm quan trọng của đặc trưng (SHAP)", fontweight="bold")
            fig.tight_layout()

            if save_path is not None:
                fig.savefig(save_path, bbox_inches="tight")
                logger.info("SHAP importance plot saved → %s", save_path)
                plt.close(fig)
            else:
                plt.show()


# ── CausalGraphAnalyzer ────────────────────────────────────────


class CausalGraphAnalyzer:
    """Mô-đun suy luận nhân quả giả lập.

    Xây dựng một đồ thị có hướng không chu trình (DAG) hợp lý giữa các đặc trưng
    kinh tế vĩ mô và mục tiêu dự đoán, sau đó xác thực chéo các kết quả SHAP
    với cấu trúc nhân quả.

    Không yêu cầu phụ thuộc ``networkx`` — trực quan hóa DAG sử dụng
    các chú thích ``matplotlib`` đơn thuần với ``FancyArrowPatch``.
    """

    def __init__(self) -> None:
        self.edges: List[Tuple[str, str]] = []
        self.nodes: List[str] = []
        self.target: str = ""

    # ── Xây dựng DAG ────────────────────────────────────────

    def build_dag(
        self,
        feature_names: List[str],
        target: str = "price",
    ) -> Dict[str, List[str]]:
        """Trả về một từ điển mã hóa các cạnh của một DAG hợp lý về mặt kinh tế.

        DAG mã hóa các mối quan hệ nhân quả cách điệu chẳng hạn như:

        * ``interest_rate  → inflation``
        * ``inflation      → sentiment_score``
        * ``sentiment_score → trading_volume``
        * ``trading_volume → <target>``
        * ``interest_rate  → <target>``

        Chỉ các đặc trưng có trong *feature_names* mới được đưa vào.

        Tham số
        ----------
        feature_names : list[str]
            Các tên đặc trưng có sẵn (thứ tự không quan trọng).
        target : str, tùy chọn
            Tên của nút mục tiêu dự đoán (mặc định ``"price"``).

        Trả về
        -------
        dict[str, list[str]]
            Danh sách kề ánh xạ mỗi nút cha tới danh sách các
            nút con của nó.
        """
        self.target = target
        self.nodes = list(feature_names) + [target]

        # Các cạnh hợp lý về mặt kinh tế đã được định nghĩa trước.
        plausible_edges: List[Tuple[str, str]] = [
            ("interest_rate", "inflation"),
            ("interest_rate", target),
            ("inflation", "sentiment_score"),
            ("inflation", target),
            ("sentiment_score", "trading_volume"),
            ("sentiment_score", target),
            ("trading_volume", target),
        ]

        # Chỉ giữ lại các cạnh mà cả hai điểm cuối của nó đều tồn tại trong tập hợp nút.
        node_set = set(self.nodes)
        self.edges = [
            (u, v)
            for u, v in plausible_edges
            if u in node_set and v in node_set
        ]

        # Đồng thời thêm một cạnh trực tiếp vào mục tiêu cho bất kỳ đặc trưng nào chưa
        # có, để DAG vẫn được kết nối.
        children_of: Dict[str, List[str]] = {}
        for u, v in self.edges:
            children_of.setdefault(u, []).append(v)

        for feat in feature_names:
            if feat not in children_of or target not in children_of.get(feat, []):
                # Kiểm tra xem có *bất kỳ* đường dẫn nào đến mục tiêu chưa — bỏ qua nếu có
                # (kiểm tra 1 bước nhảy đơn giản là đủ cho bản giả lập).
                has_path = any(v == target for u, v in self.edges if u == feat)
                if not has_path:
                    self.edges.append((feat, target))
                    children_of.setdefault(feat, []).append(target)

        # Xây dựng lại danh sách kề cho giá trị trả về.
        adjacency: Dict[str, List[str]] = {}
        for u, v in self.edges:
            adjacency.setdefault(u, []).append(v)

        logger.info(
            "Causal DAG built — %d nodes, %d edges.",
            len(self.nodes),
            len(self.edges),
        )
        return adjacency

    # ── Trực quan hóa DAG ───────────────────────────────────────

    def plot_dag(self, save_path: Optional[str] = None) -> None:
        """Vẽ DAG sử dụng các chú thích ``matplotlib`` (không có networkx).

        Các nút được sắp xếp theo bố cục phân lớp và được kết nối thông qua các
        mũi tên chú thích cong.

        Tham số
        ----------
        save_path : str hoặc None, tùy chọn
            Nếu được cung cấp, hình ảnh được lưu vào đường dẫn này thay vì được
            hiển thị tương tác.

        Ngoại lệ
        ------
        RuntimeError
            Nếu :meth:`build_dag` chưa được gọi.
        """
        if not self.edges:
            raise RuntimeError(
                "Không có cạnh nào để vẽ — hãy gọi build_dag() trước."
            )

        # ── Bố cục phân lớp (định vị thủ công) ─────────────────
        # Đặt mục tiêu ở ngoài cùng bên phải; các đặc trưng ở bên trái,
        # được phân bổ theo chiều dọc.
        features = [n for n in self.nodes if n != self.target]
        n_feat = len(features)

        positions: Dict[str, Tuple[float, float]] = {}
        for idx, feat in enumerate(features):
            x = 0.15 + 0.35 * (idx % 2)  # sắp xếp các cột xen kẽ
            y = 1.0 - (idx + 0.5) / n_feat if n_feat > 0 else 0.5
            positions[feat] = (x, y)
        positions[self.target] = (0.85, 0.5)

        # ── Vẽ ────────────────────────────────────────────────
        with plt.rc_context(
            {
                "font.family": "serif",
                "font.size": 10,
                "figure.dpi": 300,
            }
        ):
            fig, ax = plt.subplots(figsize=(8, max(4, 1.2 * n_feat)))
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.axis("off")
            ax.set_title("DAG nhân quả (giả lập)", fontweight="bold", fontsize=13)

            # Vẽ các cạnh (mũi tên).
            for u, v in self.edges:
                x_start, y_start = positions[u]
                x_end, y_end = positions[v]
                ax.annotate(
                    "",
                    xy=(x_end, y_end),
                    xytext=(x_start, y_start),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="#555555",
                        lw=1.3,
                        connectionstyle="arc3,rad=0.12",
                    ),
                )

            # Vẽ các nút.
            for node, (x, y) in positions.items():
                bbox_style = (
                    "round,pad=0.35"
                    if node != self.target
                    else "round,pad=0.45"
                )
                face_color = "#d0e8ff" if node != self.target else "#ffd6d6"
                ax.text(
                    x,
                    y,
                    node,
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold" if node == self.target else "normal",
                    bbox=dict(
                        boxstyle=bbox_style,
                        facecolor=face_color,
                        edgecolor="#333333",
                        linewidth=0.8,
                    ),
                )

            fig.tight_layout()

            if save_path is not None:
                fig.savefig(save_path, bbox_inches="tight")
                logger.info("Causal DAG plot saved → %s", save_path)
                plt.close(fig)
            else:
                plt.show()

    # ── Kiểm tra tính nhất quán SHAP ↔ DAG ───────────────────────────

    def validate_shap_against_dag(
        self,
        shap_result: ExplanationResult,
    ) -> str:
        """Kiểm tra xem đặc trưng SHAP hàng đầu có đường dẫn nhân quả trực tiếp đến mục tiêu hay không.

        Đây là một xác thực *giả lập* — nó chỉ đơn giản tra cứu đặc trưng được
        xếp hạng cao nhất trong danh sách cạnh đã lưu và báo cáo xem có tồn tại
        một cạnh trực tiếp hay không.

        Tham số
        ----------
        shap_result : ExplanationResult
            Đầu ra của :meth:`ShapExplainer.explain`.

        Trả về
        -------
        str
            Chuỗi kết luận mà con người có thể đọc được.
        """
        if not shap_result.top_features:
            return "No SHAP features to validate."

        top_feature: str = shap_result.top_features[0][0]
        top_importance: float = shap_result.top_features[0][1]

        # Kiểm tra một cạnh trực tiếp → mục tiêu.
        direct: bool = any(
            u == top_feature and v == self.target for u, v in self.edges
        )

        if direct:
            verdict = (
                f"✓ CONSISTENT — Top SHAP feature '{top_feature}' "
                f"(|SHAP|={top_importance:.4f}) has a DIRECT causal edge "
                f"to '{self.target}' in the DAG."
            )
        else:
            # Kiểm tra đường dẫn gián tiếp (trung gian 1 bước nhảy).
            intermediaries = [
                v for u, v in self.edges if u == top_feature and v != self.target
            ]
            indirect_via = [
                m
                for m in intermediaries
                if any(u == m and v == self.target for u, v in self.edges)
            ]
            if indirect_via:
                verdict = (
                    f"~ INDIRECT — Top SHAP feature '{top_feature}' "
                    f"(|SHAP|={top_importance:.4f}) reaches '{self.target}' "
                    f"indirectly via {indirect_via}."
                )
            else:
                verdict = (
                    f"✗ NO CAUSAL PATH — Top SHAP feature '{top_feature}' "
                    f"(|SHAP|={top_importance:.4f}) has no direct or 1-hop "
                    f"causal path to '{self.target}' in the DAG."
                )

        logger.info("DAG validation: %s", verdict)
        return verdict
