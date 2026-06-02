import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np

def generate_pipeline_figures(
    dataset_name: str,
    loss_history: list,
    y_true: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    reject_flags: np.ndarray,
    threshold: float,
    rmse_baseline: float,
    rmse_proposed: float,
    shap_features: list,
    causal_edges: list
):
    """
    Sinh 4 biểu đồ báo cáo và 1 biểu đồ tổng hợp dựa trên kết quả chạy thực tế của pipeline.
    Các file hình ảnh sẽ được lưu dưới định dạng .png có gắn tên tập dữ liệu.
    """
    # Cấu hình giao diện chung
    sns.set_theme(style="whitegrid", context="talk")
    dataset_slug = dataset_name.replace(" ", "_").replace("&", "n").lower()
    
    import os
    save_dir = os.path.join("figures", dataset_slug)
    os.makedirs(save_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # Fig 1: Training Convergence
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(loss_history) + 1), loss_history, color="#2ecc71", linewidth=3)
    plt.title(f"Training Convergence - {dataset_name}", fontsize=18, fontweight="bold")
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("MSE Loss", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig1_training_convergence.png"), dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # Fig 2: Risk-Aware Prediction
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    x_axis = np.arange(len(y_true))
    plt.plot(x_axis, y_true, label="Ground Truth", color="black", alpha=0.7)
    plt.plot(x_axis, means, label="Prediction ($\\mu$)", color="#3498db", linewidth=2)
    plt.fill_between(x_axis, means - 2*stds, means + 2*stds, color="#3498db", alpha=0.2, label="95% CI ($2\\sigma$)")
    
    # Đánh dấu các vùng bị từ chối do rủi ro
    in_reject = False
    start_idx = 0
    added_label = False
    for i, flag in enumerate(reject_flags):
        if flag and not in_reject:
            in_reject = True
            start_idx = i
        elif not flag and in_reject:
            in_reject = False
            label = "Rejected Region (High Risk)" if not added_label else ""
            plt.axvspan(start_idx, i, color="#e74c3c", alpha=0.3, label=label)
            added_label = True
            
    if in_reject:
        label = "Rejected Region (High Risk)" if not added_label else ""
        plt.axvspan(start_idx, len(reject_flags)-1, color="#e74c3c", alpha=0.3, label=label)
        
    plt.title(f"Risk-Aware Prediction (MC Dropout) - {dataset_name}", fontsize=18, fontweight="bold")
    plt.xlabel("Time Steps (Test Set)", fontsize=14)
    plt.ylabel("Price", fontsize=14)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig2_risk_aware_prediction.png"), dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # Fig 3: RMSE Comparison
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    models = ["Baseline (No Reject)", "Proposed (Reject Option)"]
    rmses = [rmse_baseline, rmse_proposed]
    sns.barplot(x=models, y=rmses, hue=models, palette=["#95a5a6", "#2ecc71"], legend=False)
    plt.title(f"RMSE Comparison - {dataset_name}", fontsize=18, fontweight="bold")
    plt.ylabel("RMSE", fontsize=14)
    for i, v in enumerate(rmses):
        plt.text(i, v + 0.02 * max(rmses), f"{v:.4f}", ha='center', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig3_rmse_comparison.png"), dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # Fig 4: Explainability (SHAP + Causal DAG)
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    if shap_features and len(shap_features) > 0:
        features = [f[0] for f in shap_features]
        importances = [f[1] for f in shap_features]
        sns.barplot(x=importances, y=features, hue=features, palette="viridis", legend=False, ax=ax1)
    ax1.set_title("SHAP Feature Importance", fontsize=16, fontweight="bold")
    ax1.set_xlabel("Mean |SHAP Value|", fontsize=12)
    
    G = nx.DiGraph()
    if causal_edges and len(causal_edges) > 0:
        G.add_edges_from(causal_edges)
        pos = nx.spring_layout(G, seed=42)
        nx.draw(
            G, pos, ax=ax2, with_labels=True, node_color="#f1c40f", 
            node_size=2500, font_size=10, font_weight="bold", 
            edge_color="#7f8c8d", arrows=True, arrowsize=20
        )
    ax2.set_title("Causal DAG", fontsize=16, fontweight="bold")
    
    plt.suptitle(f"Layer 3: Causal XAI - {dataset_name}", fontsize=18, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig4_explainability.png"), dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # Fig 5: Combined All (2x2 Grid)
    # ---------------------------------------------------------
    import matplotlib.image as mpimg
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(f"Hệ thống AI Tài chính - {dataset_name}", fontsize=24, fontweight='bold', y=0.98)

    img1 = mpimg.imread(os.path.join(save_dir, "fig1_training_convergence.png"))
    img2 = mpimg.imread(os.path.join(save_dir, "fig2_risk_aware_prediction.png"))
    img3 = mpimg.imread(os.path.join(save_dir, "fig3_rmse_comparison.png"))
    img4 = mpimg.imread(os.path.join(save_dir, "fig4_explainability.png"))

    axes[0, 0].imshow(img1)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img2)
    axes[0, 1].axis('off')

    axes[1, 0].imshow(img3)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(img4)
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(os.path.join(save_dir, "fig5_combined_all.png"), dpi=300, bbox_inches='tight')
    plt.close()

    return True
