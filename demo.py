import torch

from module.utils.model_configuration import QuantMoEConfig
from module.quant_moe_model import QuantMoEModel

# ==========================================
# 6. 演示与测试 (Demo)
# ==========================================
if __name__ == "__main__":
    # 1. 初始化配置
    config = QuantMoEConfig(
        num_alphas=50,
        context_len=30,
        d_model=128,
        n_layers=2,
        use_feature_selection=True,
        selection_reg_lambda=0.05,
        router_noise=0.1
    )

    model = QuantMoEModel(config)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")

    # 2. 构造模拟数据
    bsz = 16
    x = torch.randn(bsz, 30, 50)  # [Batch, T, N]

    # 模拟前 10 个因子是无用的噪声，看看 Selector 能不能学着关掉它们
    # 这里只是随机输入，所以梯度只会随机更新，无法真正收敛，仅测试通路
    x[:, :, :10] = torch.randn(bsz, 30, 10) * 5.0

    factor_ids = torch.arange(50)
    # date_ids = torch.randint(0, 100, (bsz,))
    macro_features = torch.randn(bsz,50)
    labels = torch.randn(bsz, 50)  # 真实收益率

    # 3. 前向传播
    model.train()  # 确保 Gumbel Noise 和 Dropout 开启
    outputs = model(x, factor_ids, date_ids=None, labels=labels,macro_features=macro_features)

    # 4. 打印结果分析
    print("\n--- Training Metrics ---")
    for k, v in outputs.metrics.items():
        print(f"{k:<15}: {v:.6f}")

    print("\n--- Diagnostics ---")
    print(f"Gate Entropy    : {outputs.avg_gate_entropy:.4f} (Target: ~0.4-0.6)")
    print(f"Time Expert %   : {outputs.avg_time_ratio:.4f} (Target: Balanced or Regime-dependent)")

    print("\n--- Feature Selection (First 20 Factors) ---")
    # 初始状态下应该是 ~0.5 (Sigmoid(0))
    print(outputs.selected_mask[:20].tolist())

    # 5. 反向传播测试
    outputs.loss.backward()
    print("\n--- Gradient Check ---")
    # 检查特征选择器的梯度是否存在
    if model.feature_selector.mu.grad is not None:
        grad_norm = model.feature_selector.mu.grad.norm().item()
        print(f"Feature Selector Grad Norm: {grad_norm:.6f} (Should be > 0)")
    else:
        print("Error: No gradient flow to Feature Selector!")