"""
test_layers.py — Unit tests cho các Layer 1, 2, và 3.
===================================================
Chạy bằng lệnh: python -m unittest test_layers.py
"""

import unittest
import numpy as np
import torch
import sys
from pathlib import Path

# Thêm thư mục src vào sys.path để có thể import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils import SentimentResult
from layer1_rag import FactAwareContextEngine
from layer2_lstm import HybridLSTMGRU, MCDropoutPredictor, create_sequences
from layer3_xai import CausalGraphAnalyzer

class TestLayer1RAG(unittest.TestCase):
    def test_fact_aware_engine_mock(self):
        """Kiểm tra chế độ mock fallback của FactAwareContextEngine."""
        engine = FactAwareContextEngine()
        docs = [
            "The market is crashing, huge losses reported.",
            "Apple reports strong earnings, stock rallies."
        ]
        results, faith = engine.analyze("What about Apple?", docs, top_k=1)
        self.assertIsInstance(results, list)
        self.assertTrue(0.0 <= faith <= 1.0)
        
        if results:
            self.assertIsInstance(results[0], SentimentResult)


class TestLayer2LSTM(unittest.TestCase):
    def test_create_sequences(self):
        """Kiểm tra logic tạo cửa sổ trượt (sliding window) cho chuỗi thời gian."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        seq_length = 10
        X_seq, y_seq = create_sequences(X, y, seq_length)
        
        self.assertEqual(X_seq.shape, (90, 10, 5))
        self.assertEqual(y_seq.shape, (90,))
        self.assertEqual(y_seq[0], y[10])

    def test_lstm_forward(self):
        """Kiểm tra quá trình lan truyền xuôi (forward pass) của mô hình Hybrid."""
        model = HybridLSTMGRU(input_dim=5, hidden_dim=16)
        x = torch.randn(32, 10, 5) # (batch, seq, features)
        out = model(x)
        self.assertEqual(out.shape, (32, 1))

    def test_mc_dropout_inference(self):
        """Kiểm tra ranh giới độ không chắc chắn trong MCDropout."""
        predictor = MCDropoutPredictor(input_dim=5, hidden_dim=16)
        x = torch.randn(20, 10, 5)
        means, stds, flags = predictor.predict_with_uncertainty(x, T=5, threshold=99.0)
        
        self.assertEqual(means.shape, (20,))
        self.assertEqual(stds.shape, (20,))
        self.assertEqual(flags.shape, (20,))
        self.assertFalse(flags.any()) # Với ngưỡng cao, không có sample nào bị cờ (flagged)


class TestLayer3XAI(unittest.TestCase):
    def test_causal_dag(self):
        """Kiểm tra logic xây dựng DAG giả lập."""
        causal = CausalGraphAnalyzer()
        feats = ["interest_rate", "inflation", "trading_volume"]
        dag = causal.build_dag(feats, target="price")
        
        self.assertIn("interest_rate", dag)
        self.assertIn("inflation", dag["interest_rate"])
        
        # Kiểm tra logic xác thực (validation)
        from utils import ExplanationResult
        mock_shap = ExplanationResult(top_features=[("interest_rate", 0.5)])
        verdict = causal.validate_shap_against_dag(mock_shap)
        self.assertIn("interest_rate", verdict)


if __name__ == "__main__":
    unittest.main()
