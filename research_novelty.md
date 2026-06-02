# Báo Cáo Nghiên Cứu: Nền Tảng Toán Học & Tính Đổi Mới Của Hệ Thống AI Tài Chính Đáng Tin Cậy

Báo cáo này phân tích chuyên sâu về khía cạnh học thuật, toán học và những đóng góp mới (novelties) của kiến trúc Trustworthy AI trong giao dịch tài chính định lượng.

---

## 1. Phương pháp Đổi mới (Innovative Method)
Thay vì đi theo lối mòn của AI tài chính truyền thống là **Point Estimation** (Dự báo điểm tĩnh), phương pháp đổi mới ở đây là chuyển dịch toàn bộ kiến trúc sang **Probabilistic Reasoning** (Suy luận xác suất) thông qua việc kết hợp 3 trụ cột (Triad of Trust):
1. **RAG (Retrieval-Augmented Generation):** Đóng vai trò là mỏ neo sự thật, ép AI ngôn ngữ phải đọc dữ kiện thực tế trước khi phân tích tâm lý.
2. **Mạng Nơ-ron Bayes (Bayesian Neural Networks - BNNs):** Chuyển hóa mạng LSTM thành một mô hình xác suất, có khả năng tự nhận thức giới hạn tri thức của chính nó.
3. **Đồ thị Nhân quả (Causal Graph):** Định chuẩn lại tính minh bạch của mô hình thông qua lăng kính kinh tế học, thay vì chỉ giải thích tương quan thống kê đơn thuần.

Sự đổi mới lớn nhất nằm ở triết lý thiết kế: Mô hình không bị ép phải "đoán mò" trong mọi hoàn cảnh. Nó được cấp đặc quyền **Reject Option** (Quyền từ chối dự đoán) khi phát hiện thị trường bước vào trạng thái Out-of-Distribution (Vượt ngoài phân phối học).

---

## 2. Đã nghiên cứu được gì? (Research Contributions)
Qua quá trình thực nghiệm khắt khe, nghiên cứu này đã đóng góp 3 luận điểm thực nghiệm quan trọng:
- **Định lượng Rủi ro là khả thi trong Học sâu:** Nghiên cứu đã chứng minh thành công kỹ thuật Monte Carlo Dropout có thể đóng vai trò như một bộ lọc xấp xỉ Bayes để đo lường *Độ bất định Epistemic* (Epistemic Uncertainty) đối với chuỗi thời gian nhiễu loạn của thị trường tài chính.
- **Sự Đánh đổi (Trade-off) để Chiến thắng:** Thông qua thực nghiệm trên tập Test chứa cú sốc thiên nga đen, việc tự động kích hoạt Cơ chế Từ chối (Rejection Rate = 16.47%) đã giúp cải thiện sai số RMSE của hệ thống lên **10.48%** trên những lệnh được thực thi (Execute Only).
- **Mối liên kết giữa XAI và Kinh tế vĩ mô:** Nghiên cứu đã thiết lập thành công phương pháp đối chiếu tự động giữa giá trị Shapley (SHAP) và Đồ thị Nhân quả (Causal DAG). Điều này chứng minh AI không vướng phải hiện tượng "học vẹt" (Spurious Correlation) mà thực sự tuân theo quy luật cung cầu.

---

## 3. Về mặt Toán học (Mathematical Foundations)
Kiến trúc của hệ thống được xây dựng trên nền tảng toán học khắt khe, xoay quanh các phương trình cốt lõi sau:

### A. Phương trình Ước lượng Bayes bằng MC Dropout
Trong kiến trúc này, Dropout được giữ ở trạng thái kích hoạt (Active) ngay cả trong pha Suy luận (Inference). Với cùng một vector đầu vào $x_t$, mô hình chạy xấp xỉ $T$ lần lấy mẫu (Monte Carlo Sampling), tạo ra một chuỗi các kết quả phân phối $\{\hat{y}_1, \hat{y}_2, ..., \hat{y}_T\}$.

> [!NOTE] 
> **Kỳ vọng (Mean - Dự báo giá trị):** 
> Trọng tâm của phân phối này được dùng làm giá trị dự đoán chính thức.
> $$\mu \approx \frac{1}{T} \sum_{t=1}^T \hat{y}_t$$

> [!IMPORTANT]
> **Phương sai (Variance - Độ bất định Epistemic):** 
> Đây là thước đo rủi ro. Phương sai càng lớn, độ tán xạ của $T$ lần dự báo càng cao, chứng tỏ mô hình đang "bối rối" trước dữ liệu.
> $$\sigma^2 \approx \frac{1}{T} \sum_{t=1}^T (\hat{y}_t - \mu)^2$$

### B. Toán học của Cơ chế Từ chối (Reject Option)
Hệ thống thiết lập một ngưỡng rủi ro an toàn $\theta$ (được tối ưu hóa qua hàm tổn thất trên tập Validation). Định lý quyết định của hệ thống là một hàm Indicator phi tuyến:

$$ \text{Action} = \begin{cases} 
\text{Execute (Thực thi giao dịch)} & \text{nếu } \sigma^2 \le \theta \\
\text{Reject (Đứng ngoài thị trường)} & \text{nếu } \sigma^2 > \theta
\end{cases} $$

### C. Giải thích Nhân quả (Causal XAI với Gradient SHAP)
Dựa trên Lý thuyết Trò chơi (Game Theory), dự báo $\hat{y}$ được phân rã thành tổng mức đóng góp $\phi_i$ của từng đặc trưng (feature) $x_i$:
$$ \hat{y} = \phi_0 + \sum_{i=1}^M \phi_i(x_i) $$

Vector tầm quan trọng $\Phi = [\phi_1, \phi_2, ..., \phi_M]$ sau đó được đối chiếu nhân vô hướng với Ma trận kề (Adjacency Matrix $A$) của Đồ thị Causal DAG $G(V,E)$ để đảm bảo các cạnh tương tác mạnh nhất trong mạng Nơ-ron đều tương ứng với các cạnh có thật trong lý thuyết tài chính.

---

## 4. Nó cải thiện vấn đề gì? (Problem Solving)
Về bản chất, kiến trúc toán học này giải quyết triệt để 3 tử huyệt lớn nhất của hệ thống AI tài chính hiện đại:

1. **Vấn đề "Overconfidence" (Sự tự tin thái quá của Black-box):**
   - *Vấn đề:* Các mô hình Deep Learning truyền thống luôn dự đoán sai với độ tự tin 99% khi gặp dữ liệu lạ.
   - *Cải thiện:* Việc tính toán $\sigma^2$ ép mô hình phải lượng hóa và khai báo trung thực mức độ "không biết" của bản thân, triệt tiêu sự kiêu ngạo vô căn cứ của thuật toán học sâu.

2. **Vấn đề "Out-of-Distribution" (Sập hầm khi trượt phân phối):**
   - *Vấn đề:* Thị trường tài chính liên tục thay đổi (non-stationary). Mô hình huấn luyện năm 2020 không thể sống sót qua cú sốc năm 2026.
   - *Cải thiện:* Cơ chế *Reject Option* đóng vai trò như một cầu chì an toàn. Khi phát hiện dữ liệu khác thường, AI sẽ phanh lại, bảo vệ toàn vẹn nguồn vốn thay vì lao vào giao dịch và chịu cảnh Drawdown nặng nề.

3. **Vấn đề Niềm tin Kiểm toán (The Trust Deficit):**
   - *Vấn đề:* Quản lý quỹ đầu tư không bao giờ cấp phép cho một thuật toán giao dịch bằng tiền thật nếu họ không hiểu tại sao thuật toán đó lại ra lệnh Mua/Bán.
   - *Cải thiện:* Việc sử dụng Causal DAG để kiểm toán điểm SHAP biến AI từ một "Hộp đen mờ ám" trở thành một "Cố vấn minh bạch". Mọi nguyên nhân ra quyết định đều được giải trình và có tính lưu vết cao, đáp ứng các tiêu chuẩn khắt khe nhất của Ủy ban chứng khoán và kiểm toán tài chính.
