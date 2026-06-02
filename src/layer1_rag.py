"""
layer1_rag.py — Lớp 1: Truy xuất RAG + Phân tích cảm xúc FinBERT
====================================================================
Kết hợp Retrieval-Augmented Generation (RAG) với chấm điểm cảm xúc
FinBERT để tạo ra phân tích cảm xúc *nhận thức dữ kiện* (fact-aware)
trên các tài liệu tài chính.

Các thành phần
----------
- **FinBERTAnalyzer**        : Lớp bọc (wrapper) HuggingFace FinBERT (dự phòng mock nhẹ nhàng).
- **RAGRetriever**           : Trình truy xuất ngữ nghĩa dựa trên FAISS (dự phòng bằng từ khóa).
- **FactAwareContextEngine** : Điều phối luồng xử lý từ truy xuất → phân tích cảm xúc.

Tất cả các phụ thuộc nặng (transformers, faiss, langchain, sentence-transformers)
được nạp trễ (lazy import) để module vẫn có thể được import trong các môi trường hạn chế.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from utils import SentimentResult, logger

# ════════════════════════════════════════════════════════════════════
#  Trình Phân Tích Cảm Xúc FinBERT
# ════════════════════════════════════════════════════════════════════

class FinBERTAnalyzer:
    """Lớp bọc mỏng quanh ``ProsusAI/finbert`` để chấm điểm cảm xúc tài chính.

    Nếu thiếu thư viện ``transformers`` hoặc mô hình không tải được
    lớp này sẽ tự động dự phòng bằng một mock dựa trên luật gọn nhẹ
    để mã ở phần sau không bao giờ bị hỏng.

    Thuộc tính
    ----------
    _pipeline : Optional[Pipeline]
        Luồng (pipeline) phân loại văn bản của HuggingFace (``None`` khi ở chế độ mock).
    _mock_mode : bool
        ``True`` khi mô hình thực sự không thể tải được.
    """

    _MODEL_NAME: str = "ProsusAI/finbert"

    def __init__(self) -> None:
        """Khởi tạo FinBERT; dự phòng bằng mock khi thất bại."""
        self._pipeline = None
        self._mock_mode: bool = False
        self._load_model()

    # ── hàm trợ giúp nội bộ ────────────────────────────────────────────

    def _load_model(self) -> None:
        """Cố gắng tải luồng phân tích (pipeline) FinBERT từ HuggingFace."""
        try:
            from transformers import pipeline as hf_pipeline  # type: ignore[import-untyped]

            self._pipeline = hf_pipeline(
                "text-classification",
                model=self._MODEL_NAME,
                top_k=None,           # trả về điểm số của tất cả các nhãn
                truncation=True,
                max_length=512,
            )
            logger.info("FinBERT model loaded successfully (%s).", self._MODEL_NAME)
        except Exception as exc:
            logger.warning(
                "Failed to load FinBERT (%s). Using mock scorer.", exc,
            )
            self._mock_mode = True

    def _mock_score(self, text: str) -> SentimentResult:
        """Dự phòng dựa trên luật: từ khóa tích cực / tiêu cực → điểm số.

        Điều này cố tình được làm đơn giản — mục đích duy nhất của nó là giữ cho
        luồng phân tích có thể kiểm thử được khi không có ``transformers``.
        """
        text_lower = text.lower()

        positive_keywords = {
            "rally", "gain", "profit", "surge", "bullish", "growth",
            "strong", "uptick", "rise", "recover", "optimistic", "earnings",
        }
        negative_keywords = {
            "loss", "crash", "decline", "bearish", "recession", "fear",
            "downturn", "drop", "concern", "slowdown", "risk", "plunge",
        }

        pos_count = sum(1 for kw in positive_keywords if kw in text_lower)
        neg_count = sum(1 for kw in negative_keywords if kw in text_lower)
        total = pos_count + neg_count

        if total == 0:
            label, score = "neutral", 0.0
        elif pos_count > neg_count:
            label, score = "positive", round(pos_count / total, 4)
        elif neg_count > pos_count:
            label, score = "negative", round(-neg_count / total, 4)
        else:
            label, score = "neutral", 0.0

        return SentimentResult(
            text=text,
            label=label,
            score=score,
            source="mock_finbert",
            faithfulness=0.5,          # độ tin cậy thấp hơn đối với kết quả mock
        )

    # ── API công khai ─────────────────────────────────────────────────

    def score(self, texts: List[str]) -> List[SentimentResult]:
        """Chấm điểm cảm xúc tài chính cho một loạt văn bản.

        Tham số
        ----------
        texts : List[str]
            Các đoạn văn bản thô (tiêu đề, đoạn bài báo, …).

        Trả về
        -------
        List[SentimentResult]
            Một kết quả cho mỗi văn bản đầu vào bao gồm nhãn, điểm số, nguồn, và
            siêu dữ liệu về độ trung thực (faithfulness).
        """
        if not texts:
            return []

        # ── nhánh xử lý mock ──────────────────────────────────────────────
        if self._mock_mode or self._pipeline is None:
            logger.debug("Scoring %d texts using mock FinBERT.", len(texts))
            return [self._mock_score(t) for t in texts]

        # ── nhánh mô hình thực ────────────────────────────────────────
        try:
            raw_outputs = self._pipeline(texts)
            results: list[SentimentResult] = []

            for text, output in zip(texts, raw_outputs):
                # output là một danh sách các dict: [{"label": ..., "score": ...}, ...]
                # Chọn nhãn có xác suất cao nhất.
                best = max(output, key=lambda d: d["score"])
                label = best["label"].lower()

                # Ánh xạ sang số thực có dấu: tích cực → +score, tiêu cực → −score
                signed_score: float
                if label == "positive":
                    signed_score = best["score"]
                elif label == "negative":
                    signed_score = -best["score"]
                else:
                    signed_score = 0.0

                results.append(SentimentResult(
                    text=text,
                    label=label,
                    score=round(signed_score, 4),
                    source=self._MODEL_NAME,
                    faithfulness=1.0,
                ))

            logger.info("FinBERT scored %d texts.", len(results))
            return results

        except Exception as exc:
            logger.error("FinBERT inference failed (%s). Falling back to mock.", exc)
            return [self._mock_score(t) for t in texts]


# ════════════════════════════════════════════════════════════════════
#  Trình truy xuất RAG (FAISS + sentence-transformers)
# ════════════════════════════════════════════════════════════════════

class RAGRetriever:
    """Trình truy xuất tài liệu ngữ nghĩa dựa trên FAISS.

    Các tài liệu được chia nhỏ (chunk) bằng ``RecursiveCharacterTextSplitter`` của LangChain,
    được nhúng (embed) thông qua một mô hình sentence-transformer, và được lập chỉ mục
    trong kho vector FAISS để truy xuất láng giềng gần nhất một cách nhanh chóng.

    Nếu FAISS, LangChain, hoặc sentence-transformers không khả dụng,
    trình truy xuất sẽ giảm cấp xuống thành đối sánh trùng lặp từ khóa đơn giản.

    Tham số
    ----------
    documents : List[str]
        Chuỗi tài liệu thô cần được lập chỉ mục.
    embedding_model : str
        Tên của mô hình ``sentence-transformers`` dùng để nhúng.
    chunk_size : int
        Số lượng ký tự tối đa cho mỗi đoạn (chunk).
    chunk_overlap : int
        Số lượng ký tự chồng chéo giữa các đoạn liên tiếp.
    """

    _DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    def __init__(
        self,
        documents: List[str],
        embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        self._raw_documents = documents
        self._embedding_model_name = embedding_model
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

        # Sẽ được thiết lập trong quá trình lập chỉ mục
        self._vectorstore = None
        self._chunks: List[str] = []
        self._fallback_mode: bool = False

        self._build_index()

    # ── hàm trợ giúp nội bộ ────────────────────────────────────────────

    def _build_index(self) -> None:
        """Chia nhỏ tài liệu, nhúng, và lưu trữ vào FAISS."""
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore[import-untyped]
            from langchain_community.vectorstores import FAISS  # type: ignore[import-untyped]
            from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore[import-untyped]

            # 1. Chia nhỏ (Chunk)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            self._chunks = splitter.create_documents(self._raw_documents)

            # 2. Nhúng + lập chỉ mục
            embeddings = HuggingFaceEmbeddings(model_name=self._embedding_model_name)
            self._vectorstore = FAISS.from_documents(self._chunks, embeddings)

            logger.info(
                "FAISS index built: %d chunks from %d documents.",
                len(self._chunks),
                len(self._raw_documents),
            )

        except Exception as exc:
            logger.warning(
                "FAISS/LangChain unavailable (%s). Using keyword fallback.", exc,
            )
            self._fallback_mode = True
            self._chunks = self._simple_chunk(self._raw_documents)

    def _simple_chunk(self, documents: List[str]) -> List[str]:
        """Chia nhỏ đơn giản cho chế độ dự phòng: tách tại ranh giới câu.

        Tạo ra các đoạn (chunk) với khoảng ``self._chunk_size`` ký tự.
        """
        chunks: list[str] = []
        for doc in documents:
            # Tách ở dấu câu kết thúc câu theo sau bởi khoảng trắng
            sentences = re.split(r"(?<=[.!?])\s+", doc)
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 > self._chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    # Giữ sự chồng chéo bằng cách giữ lại phần đuôi của đoạn trước đó
                    current_chunk = current_chunk[-self._chunk_overlap:] + " " + sentence
                else:
                    current_chunk = (current_chunk + " " + sentence).strip()
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

        logger.info("Fallback chunking generated %d chunks.", len(chunks))
        return chunks

    def _keyword_retrieve(self, query: str, top_k: int) -> List[str]:
        """Xếp hạng các đoạn dựa trên sự trùng lặp từ khóa với truy vấn (dự phòng TF).

        Mỗi đoạn được chấm điểm bằng cách đếm số lượng token truy vấn duy nhất
        xuất hiện trong đó (không phân biệt hoa thường).
        """
        query_tokens = set(query.lower().split())

        scored: list[tuple[float, str]] = []
        for chunk in self._chunks:
            chunk_lower = chunk.lower()
            overlap = sum(1 for token in query_tokens if token in chunk_lower)
            scored.append((overlap, chunk))

        # Sắp xếp giảm dần theo số lượng trùng lặp, chọn top_k
        scored.sort(key=lambda pair: pair[0], reverse=True)
        results = [text for _, text in scored[:top_k]]
        return results

    # ── API công khai ─────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Truy xuất các đoạn tài liệu phù hợp nhất cho một truy vấn.

        Tham số
        ----------
        query : str
            Truy vấn tìm kiếm bằng ngôn ngữ tự nhiên.
        top_k : int, tùy chọn
            Số lượng đoạn cần trả về (mặc định ``3``).

        Trả về
        -------
        List[str]
            Tối đa *top_k* đoạn tài liệu được xếp hạng theo độ liên quan.
        """
        if not self._chunks:
            logger.warning("No chunks available for retrieval.")
            return []

        # ── nhánh xử lý FAISS ─────────────────────────────────────────────
        if not self._fallback_mode and self._vectorstore is not None:
            try:
                docs = self._vectorstore.similarity_search(query, k=top_k)
                results = [doc.page_content for doc in docs]
                logger.debug(
                    "FAISS retrieved %d chunks for query: '%s'",
                    len(results),
                    query[:60],
                )
                return results
            except Exception as exc:
                logger.error("FAISS retrieval failed (%s). Using keyword fallback.", exc)

        # ── dự phòng bằng từ khóa ───────────────────────────────────────
        results = self._keyword_retrieve(query, top_k)
        logger.debug(
            "Keyword fallback retrieved %d chunks for query: '%s'",
            len(results),
            query[:60],
        )
        return results


# ════════════════════════════════════════════════════════════════════
#  Công cụ Ngữ cảnh Nhận thức Dữ kiện (RAG + FinBERT)
# ════════════════════════════════════════════════════════════════════

class FactAwareContextEngine:
    """Điều phối truy xuất và phân tích cảm xúc vào một lệnh gọi duy nhất.

    Luồng công việc
    --------
    1. Lập chỉ mục các tài liệu được cung cấp trong một ``RAGRetriever``.
    2. Với một truy vấn, truy xuất các đoạn phù hợp nhất.
    3. Chạy ``FinBERTAnalyzer`` trên các đoạn đã truy xuất.
    4. Tính toán điểm trung bình về *độ trung thực* (faithfulness) trên các kết quả.

    Tham số
    ----------
    finbert : Optional[FinBERTAnalyzer]
        Trình phân tích đã được khởi tạo trước. Tự động được tạo nếu
        là ``None`` (mặc định).
    embedding_model : str
        Mô hình sentence-transformer được chuyển tiếp tới ``RAGRetriever``.
    chunk_size : int
        Chuyển tiếp tới ``RAGRetriever``.
    chunk_overlap : int
        Chuyển tiếp tới ``RAGRetriever``.
    """

    def __init__(
        self,
        finbert: Optional[FinBERTAnalyzer] = None,
        embedding_model: str = RAGRetriever._DEFAULT_EMBEDDING_MODEL,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        self._finbert = finbert or FinBERTAnalyzer()
        self._embedding_model = embedding_model
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    # ── API công khai ─────────────────────────────────────────────────

    def analyze(
        self,
        query: str,
        documents: List[str],
        top_k: int = 3,
    ) -> Tuple[List[SentimentResult], float]:
        """Truy xuất ngữ cảnh liên quan, chấm điểm cảm xúc, trả về độ trung thực.

        Tham số
        ----------
        query : str
            Câu hỏi bằng ngôn ngữ tự nhiên của người dùng.
        documents : List[str]
            Kho tài liệu thô để tìm kiếm.
        top_k : int, tùy chọn
            Số lượng đoạn cần truy xuất (mặc định ``3``).

        Trả về
        -------
        Tuple[List[SentimentResult], float]
            ``(sentiment_results, avg_faithfulness)``
            - ``sentiment_results`` — một ``SentimentResult`` cho mỗi đoạn được truy xuất.
            - ``avg_faithfulness`` — trung bình độ trung thực trên tất cả các kết quả
              (1.0 = hoàn hảo, 0.0 = không thể xác minh).
        """
        if not documents:
            logger.warning("No documents provided to FactAwareContextEngine.analyze().")
            return [], 0.0

        # 1. Xây dựng trình truy xuất từ các tài liệu được cung cấp
        retriever = RAGRetriever(
            documents=documents,
            embedding_model=self._embedding_model,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )

        # 2. Truy xuất top-k đoạn phù hợp
        relevant_chunks: List[str] = retriever.retrieve(query, top_k=top_k)

        if not relevant_chunks:
            logger.warning("No relevant chunks found for query: '%s'", query[:80])
            return [], 0.0

        # 3. Chấm điểm cảm xúc trên các đoạn đã truy xuất
        sentiment_results: List[SentimentResult] = self._finbert.score(relevant_chunks)

        # 4. Tính điểm trung bình về độ trung thực
        avg_faithfulness: float = (
            sum(r.faithfulness for r in sentiment_results) / len(sentiment_results)
            if sentiment_results
            else 0.0
        )

        logger.info(
            "FactAwareContextEngine: query='%s' → %d results, "
            "avg faithfulness=%.3f",
            query[:50],
            len(sentiment_results),
            avg_faithfulness,
        )

        return sentiment_results, round(avg_faithfulness, 4)


# ════════════════════════════════════════════════════════════════════
#  Kiểm tra nhanh (python -m layer1_rag)
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    sample_docs = [
        "Apple reported record quarterly earnings, driven by strong iPhone "
        "and Services revenue. The company announced a $90 billion share "
        "buyback programme and raised its dividend by 4%.",
        "Federal Reserve officials signalled concern over persistent "
        "inflation, suggesting that interest-rate cuts may be delayed until "
        "late 2025. Bond markets reacted with a sharp sell-off.",
        "Tesla shares plunged 12% after the company missed delivery "
        "estimates for Q3. Analysts cited rising competition from BYD "
        "and softening demand in Europe.",
    ]

    engine = FactAwareContextEngine()
    results, faithfulness = engine.analyze(
        query="How are tech stocks performing?",
        documents=sample_docs,
    )

    print("\n" + "=" * 60)
    print("  Layer 1 — Fact-Aware Sentiment Analysis")
    print("=" * 60)
    for r in results:
        print(f"  [{r.label:>8}] score={r.score:+.4f}  faith={r.faithfulness:.2f}")
        print(f"           {r.text[:80]}…")
    print(f"\n  Average faithfulness: {faithfulness:.4f}")
    print("=" * 60)
