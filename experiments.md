# Recommended Configs (t+1 vs t+5)

下面给出两套可落地配置（模型 + 预计算 macro state）：
- **t+1**：短期，内部 regime 为主（可选轻量 macro），router 较平滑。
- **t+5**：中期，强烈建议使用预计算 `macro_features`，router 更锋利，macro state 增强稳定性。

> 训练/eval 口径：Train 用 `DK_L` + `CSRankNorm`（rank-label），Valid/Test 用 `DK_I` raw label。早期 NaN 请确保预计算覆盖足够 warmup，或设 `market_state_strict=False`。

---

## t+1 配置（短期，内部 regime）

### Model (`model_conf["kwargs"]["model_config"]`)
- `d_model`: 64~128（依资源）
- `n_layers`: 2~4
- `n_heads`: 4
- `use_feature_selection`: False（建议先做 FiLM-only baseline；再做 STG ablation）
- `selection_reg_lambda`: 1e-5~1e-4（若开 STG；`reg_loss=z.sum(dim=-1).mean()` 值域 `[0,N]`）
- `selection_temperature`: 0.1（若开 STG）
- `selection_noise_std`: 0.5（若开 STG）
- `use_regime_time_embedding`: True  （regime-adaptive time embedding）
- `time_tau_init`: 5.0
- `use_regime_factor_gate`: True  （regime-adaptive factor gate）
- `factor_gate_scale`: 0.5
- `use_alibi`: False（推荐先关；短 T + regime time embedding 下通常足够，且可避免 MHA 的 `attn_mask` 展开开销；需要时再做 ablation 开启）
- `use_external_macro`: False  （t+1 可不依赖外部 macro）
- `regime_internal_mode`: "short"
- `regime_internal_lag`: 1
- `regime_internal_use_batch_stats`: True  （日度截面 batch）
- `regime_internal_tail_threshold`: 2.0
- `router_use_layer_summary`: True  （轻量层内摘要，提高 gate 自适应）
- `router_noise`: 0.05  （小噪声探索）
- `router_temperature`: 1.0
- `router_z_loss_coef`: 0.01  （防止 collapse，与代码默认值一致）
- `pooling_alpha`: 0.7
- `main_loss`: "mse"  （默认主 loss，可切换为 "ic" 或 "listmle"）
- `listmle_tau`: 1.0
- `rank_topk`: 5
- `huber_delta`: 1.0
- `loss_weights`: 默认

### Trainer (`trainer_config`)
- `lr`: 5e-4
- `batch_size`: 64~256（取决于截面规模）
- `n_epochs`: 20~40
- `early_stop`: 5
- `seed`: 42
- `use_warmup`: True, `warmup_ratio`: 0.05
- `num_workers`: 按机器设置
- `market_state_path`: 留空（内部 regime）
- `market_state_strict`: True（无 macro）

### Macro State 预计算
- 可选：不需要；若要尝试，简化版即可：`--pca_dim 8 --zscore_windows 20`

---

## t+5 配置（中期，外部 macro 驱动）

### Model (`model_conf["kwargs"]["model_config"]`)
- `d_model`: 128
- `n_layers`: 3~4
- `n_heads`: 4
- `use_feature_selection`: False（建议先做 FiLM-only baseline；再做 STG ablation）
- `selection_reg_lambda`: 1e-5~1e-4（若开 STG；`reg_loss=z.sum(dim=-1).mean()` 值域 `[0,N]`）
- `selection_temperature`: 0.1（若开 STG）
- `selection_noise_std`: 0.5（若开 STG）
- `use_regime_time_embedding`: True
- `time_tau_init`: 5.0
- `use_regime_factor_gate`: True
- `factor_gate_scale`: 0.5
- `use_alibi`: False（同上，建议先关；需要时再做 ablation）
- `use_external_macro`: True  （由 adapter 自动设置）
- `regime_internal_mode`: "long"  （仅作 fallback）
- `regime_internal_lag`: 5
- `regime_internal_use_batch_stats`: True
- `regime_internal_tail_threshold`: 2.0
- `router_use_layer_summary`: True
- `router_noise`: 0.1  （更大探索）
- `router_temperature`: 0.7~1.0  （更锋利 gate）
- `router_z_loss_coef`: 0.01  （防止 collapse）
- `pooling_alpha`: 0.7
- `main_loss`: "mse"  （默认主 loss，可切换为 "ic" 或 "listmle"）
- `listmle_tau`: 0.8~1.0  （可稍小以增强排序尖锐度）
- `rank_topk`: 5
- `huber_delta`: 1.0

### Trainer (`trainer_config`)
- `lr`: 5e-4
- `batch_size`: 64~128（截面更大时取小一些）
- `n_epochs`: 30~50
- `early_stop`: 5
- `seed`: 42
- `use_warmup`: True, `warmup_ratio`: 0.05
- `num_workers`: 按机器设置
- `market_state_path`: `market_state_csi300.pkl` 或 `market_state_csi800.pkl`
- `market_state_shift`: 0（t+1/t+5，T+k 预测；仅同日预测 T→T 才用 1）
- `market_state_strict`: True（推荐论文用），若早期 NaN 太多，可临时 False

### Macro State 预计算命令（示例：CSI300）
```bash
python scripts/precompute_market_state.py \
  --out market_state_csi300.pkl \
  --instruments csi300 \
  --pca_dim 16 \
  --state_delta_lags 1,5,10 \
  --add_market_ts \
  --market_ts_windows 5,20,60 \
  --zscore_windows 20,60,120 \
  --roll_mean 20 \
  --weight_field '$amount' \
  --filter_robust_z 6 \
  --filter_max_bad_frac 0.05 \
  --warmup_trading_days -1 \
  --no_norm 
```

### Macro State 预计算命令（示例：CSI800）
```bash
python scripts/precompute_market_state.py \
  --out market_state_csi800.pkl \
  --instruments csi800 \
  --pca_dim 16 \
  --state_delta_lags 1,5,10 \
  --add_market_ts \
  --market_ts_windows 5,20,60 \
  --zscore_windows 20,60,120 \
  --roll_mean 20 \
  --weight_field '$amount' \
  --trade_field '$amount' \
  --min_trade 1 \
  --filter_robust_z 6 \
  --filter_max_bad_frac 0.05 \
  --warmup_trading_days -1 \
  --no_norm 

```

> Note: `--warmup_trading_days -1` automatically extends the precompute start backward, ensuring rolling/zscore/Δstate/TS features are defined at training start. For t+1/t+5 (T+k) prediction, **do not** add `--market_ts_past_only` and keep `market_state_shift=0`; only same-day prediction needs the shift. If early data fields are missing, use an explicit number or set `market_state_strict=False`.

---

# Hypothesis Cards (quant-hypothesis)

## h-20260610-001: DropExtremeLabel（截面双侧 2.5%，仅 train/DK_L）提升 valid RankIC

- **Mechanism**: A 股涨跌停/公告跳空产生的极端 label 对 ranking 目标是纯噪声（多为不可交易的 limit 移动）；截掉截面 top/bottom 各 2.5% 后，CSZScoreNorm 的 z 值不再被厚尾拉伸，MSE 主 loss 的梯度从拟合尾部转向排序主体（quant-firm 视角：label de-noising 是监督侧最高 ROI 的修缮；MASTER 同款做法）。
- **History note**: `d6e82a9` 曾以 `qcut 0.05/0.95` 启用过本 processor，`8afb0d0` 整体移除但注释保留"2.5% 对齐 MASTER"意图。已验证（2026-06-10）安装版 qlib 的 `qlib.data.dataset.processor` **不存在** `DropExtremeLabel` 类 → 移除原因极可能是 crash 而非实验否决。ledger 60 天无重复（hash `2be4aa0cd0e0`）。
- **Falsifier**: paired n=3（seeds 42/43/44, 25e, vs full_135 同窗）`Δdaily_rank_ic < +0.003` → reject；或 `valid_score_std_last` 较 control 降幅 >30%（分布截断引发 score 塌缩）。
- **Config delta**: `learn_processors = [DropnaLabel, DropExtremeLabel(module_path=module.utils.processors, qcut 0.025/0.975), CSZScoreNorm(robust)]`；需新增 `module/utils/processors.py`（~30 行，按日截面分位截断，`is_for_infer()=False` 确保不进 DK_I）。
- **Minimal test (proxy)**: csi300, 25 epochs, 3 seeds paired。
- **Expected delta**: rank_ic 0.0786 → 0.080~0.082。
- **Budget**: proxy ≈2 GPU-hr；n=6 confirm 再 +2 GPU-hr。
- **Promotion criteria**: Δrank_ic ≥ +0.003 且 ≥2/3 seeds 为正 → n=6 四元组（RankIC / IR_cost / MaxDD_cost / post_peak_decay）无一降级。
- **Risk / failure mode**: train 截尾 vs valid 全量评估的分布偏移 → 盯 `worst5_day_rank_ic_mean`；leakage 无新类（learn-only processor），`/quant-leakage-audit` 须确认 DK_I 管道不受影响。

## h-20260610-002: 股票轴第三 expert（日截面 stock-stock attention）提升 RankIC

- **Mechanism**: 现有两 expert 只在单股内部的 T 轴 / N 轴做 attention，个股间关系仅剩 `layer_summary` 的 mean/std 标量广播 —— 模型对"同业联动 / 相对定价"结构性失明。第三 expert 在日截面 batch 上做 cross-stock attention，注入**新信息通路**，与已证伪的重加权类 lever（pool/readout）不同类（那些是对既有表示再加权）。MASTER 的主增益来源同型（LeCun 视角：关系归纳偏置；quant 视角：截面相对价值是确证存在的 alpha 结构）。
- **Falsifier**: paired n=3 `Δdaily_rank_ic < +0.003` → 直接停（pool 教训：n=3 即杀）；或三路 router 中 stock-expert gate 份额 < 0.05（学了没用上）；或 stock-attention 熵出现 six-nines 型退化。
- **Config delta**: `model_config {"use_stock_expert": true, "stock_expert_gate_init": -2.0}`（flag-guarded、默认 off，house 模式同 `temporal_readout`）。
- **Implementation sketch** (~120 行, `moe_block.py`): `s = mean_N(h[:,-1,:,:]) → [B,D]`；MHA over B（1-2 heads）；identity-start gated 残差广播回 `h`；router 2→3 logits。train/eval 均已按日成批（FixedDaily / DailyChunk），数据层零改动；薄日上采样的重复股票在 attention 中无害（自注意），diag 确认。
- **Minimal test (proxy)**: csi300, 25e, 3 seeds paired vs full_135。
- **Expected delta**: 0.0786 → 0.080~0.084（取决于同业结构 alpha 含量）。
- **Budget**: 实现 ~1 天 + proxy 2 GPU-hr + n=6 confirm 2 GPU-hr。
- **Promotion criteria**: house 规则 + stock-expert gate 份额 ∈ [0.1, 0.6]（确被路由使用）。
- **Risk / failure mode**: gen-gap 扩大（新容量用于记忆）→ 盯 `rank_ic_gap_last`；同日截面信息合法无 leakage，但 audit 须确认 valid loader 严格按日分组不跨日 shuffle。

## h-20260610-003 [DEFERRED]: 多 horizon 辅助头（t+1 / t+10）正则化共享表示，提升 t+5 RankIC

- **Mechanism**: 每样本监督位数 ×3，共享 backbone 被迫编码 return 期限结构而非单 horizon 的噪声实现 —— 监督饥饿（gen-gap=0.12 的症状）的处方之一；quant-firm 惯例 alpha 模型多 horizon 联训。
- **Falsifier**: t+5 主头 paired n=3 `Δdaily_rank_ic < +0.003` → reject。
- **Config delta**: handler `label = [t+5, t+1, t+10]`（**t+5 保持第 0 列**，eval 管道不动）+ `model {"aux_horizons": [1, 10], "aux_horizon_loss_weight": 0.25}`。
- **Implementation**: handler 多 label 列 + adapter 放开单 label 断言（现 raise on label_dim≠1）+ 两个辅助 head —— 改动面中等，故排在 001/002 之后；若 SSL/JEPA 立项，可并入其 Stage-2 微调臂。
- **Minimal test / Budget**: 实现 ~1 天 + proxy 2 GPU-hr。
- **Expected delta**: 0.0786 → 0.080±。
- **Promotion criteria**: house 规则（t+5 主头四元组）。
- **Risk / failure mode**: SignalRecord/SigAna 默认取 label 第 0 列 → 列序契约写进代码注释；3 列联合 DropnaLabel 使 train 末端 ~10 天样本被丢（t+10 NaN），可接受；`/quant-leakage-audit` 必查 t+10 与 `fit_end_time` 边界。

## h-20260627-001: Portfolio-IR 辅助损失（IR/return 半，去捆绑 turnover；readout OFF，保留 MSE 主 loss）抬升 with-cost IR

- **Mechanism**: ledger 84-85 已证模型在原始 Alpha158 上**打赢了经典学习器的 IC 天花板**（leakage-clean ridge/LGBM 在同样的 8 步窗口上 max 0.0592 < model 0.0786）→ IC 头部已耗尽，继续用架构追 RankIC 是低 EV。且纯 MSE 的 Hessian 对角 ⇒ 对 breadth `BR_eff` / turnover / TC **零梯度**（Grinold `IR=IC·√BR_eff·TC`），部署指标活在 MSE 的零空间。本辅助项是历史上**第一次**给目标注入 portfolio/IR 方向的正梯度：`total_loss += λ_ir·ramp·l_ir`，`l_ir = −mean_t( r_t / sqrt(detach(EMA_var)+eps) )`，`r_t` = 当日**去重**截面上 L1-归一化 dollar-neutral 多空组合的实现收益（由 per-day 预测分 p 与 realized label 构成）。**诚实定性（必须如实标注）**：方差分母 detached ⇒ 这是 soft long-short-return / IC-family aux，**非真 Sharpe**（对方差无梯度）。设计透镜：MASTER / diff-batch-Sharpe（Zhang-Zohren-Roberts 2020, arXiv:2005.13665）；Jeff-Dean 视角=把部署目标直接写进可微目标。
- **De-bundle note**: 这是 CMF+IR-cost 计划（ledger 82-83, `analysis/stock_expert/PORTFOLIO_ALIGNED_PLAN.md`）中 **C3 的去捆绑 + 去堆叠**版——剥离已被 6-regime KILL 的 CrossStockBlock readout（C1/C2）与 R1-R3 算子级，**readout OFF** 单测 IR-half，隔离"光是 portfolio 梯度能否抬 IR_wc"。明确区别于已 KILLED 的 h-20260610-002（那是架构 stock-expert；本卡是 objective-space 辅助损失，配置 delta 完全不同）。
- **Falsifier**: paired n=6 fresh seeds（42-47, from-scratch 25ep, vs g012 anchor 同窗）—— KILL 若任一：IC 与 rank_ic **均** flat 而仅 turnover/分布变动（mechanism≠benefit，本项目 τ/pool/readout/expert 反复出现的失败模）；或 IR_wc lift paired-t **p≥0.1**；或 RankIC < anchor−0.002；或 MaxDD_wc 比 anchor 深 >10%；或 IC < 0.0669。
- **Config delta**: `model_config {"time_tau_mlp_out_scale":0.5, "temporal_readout":null, "use_readout_stock_attn":false, "use_stock_expert":false, "main_loss":"mse", "loss_weights":{"mse":1.0}, "ir_aux_lambda":0.08, "ir_aux_ramp_frac":0.3, "ir_aux_var_eps":1e-6}`（`ir_aux_*` 为**新增字段**，需代码；`temporal_readout:null` = 裸 backbone）。
- **Required code (NOT env-only)**: 新增 stateful 损失项（per-day **去重** 多空组合收益 + detached EMA 方差 buffer + λ ramp）+ **segment-reset guard**：EMA buffer 在 valid/test forward **禁止更新**、且**禁止跨 train/valid 边界**（核心 leakage 面）。在 `module/quant_moe_model.py` loss 聚合处（~`:688-695`）加 `l_ir` 旁路（仿 `l_aux`/`l_reg`），字段走 `QuantMoEConfig`（`module/utils/model_configuration.py`）。
- **Minimal test (proxy)**: csi300, 25ep, seed 42 single（`/quant-minimal-repro`）—— 确认 `l_ir` 单调下降、IC 不崩（≥0.0669）、且 EMA buffer 在 valid 不更新（断言）→ 通过再 n=6。
- **Expected delta**: RankIC 持平（guardrail ≥anchor−0.002，**不指望涨**）；**IR_wc：anchor 1.75 → 目标正向提升且 paired-t p<0.1**；MaxDD_wc 不更深。
- **Budget**: 实现 ~0.5 天 + proxy ≈1.5 GPU-hr + n=6 ≈ 1 GPU-day。
- **Promotion criteria**: 向量门（非单指标）—— IR_wc lift paired-t p<0.1 AND RankIC≥anchor−0.002 AND MaxDD_wc 不深 >10% AND IC≥0.0669；survivor → `/quant-walk-forward` + `/quant-stress`（成本网格）方可标 **candidate**（绝不直接 promoted；valid==test ⇒ 单 split IR_wc 乐观，τ0.5 的 n=3→n=6 反转是前车之鉴）。
- **Risk / failure mode**: (1) mechanism≠benefit（激活机制但 IR 不动）→ LOO 须测 ΔIR_wc/ΔIC_mean（KILL line 已含）；(2) λ 过大压制 MSE → IC 崩，盯 IC≥0.0669 + score_std；(3) **leakage**：EMA buffer 跨 train/valid 边界或 valid 更新 → `/quant-leakage-audit` 必查 segment-reset guard；(4) 薄日 upsample 重复名扭曲组合收益 → unique-name mask；(5) 单 split + valid==test ⇒ IR_wc 乐观 → 必走 walk-forward。turnover/TC 半为**独立后续卡**（需连续两日 in-graph，当前 day-shuffle 使"昨日"=随机日）。

## h-20260628-001: 跨天梯度累积（grad_accum_steps K=2，step-matched 抬 n_epochs→40）能否抬升 valid RankIC

- **Mechanism**: 现 batch = 单日截面（`FixedDailyBatchSampler`，≤300 名/日），一次 fwd/bwd = 一天；K=1 时每天一次 `optimizer.step()`。单日截面高度同向（一个市场 move）⇒ K=1 的梯度在**时间/regime 方向方差极大**，而本模型已证 time-dominated。跨天累积（`grad_accum_steps`=K，`loss/K` 取**均值梯度** model_adapter.py:1738，scheduler 按 optimizer-step 计数 :2074）把每步梯度在 K 个不同交易日上平均，step 方差≈1/K：**若**当前 K=1 训练在 regime 方向噪声过大拖累收敛，small-K 平均可稳定优化轨迹。设计透镜：large-batch / gradient-noise（Keskar 2017，**反面见 Risk**）+ quant 视角（time-dominated 模型上的 regime-direction 梯度方差）。**关键：K>1 在全实验清单（experiment_inventory_unified + diagnostic_runs_inventory）中恒=1，从未跑过 ⇒ 全新 lever，非重复（hash `c269b85c56d8`，ledger 60 天无碰撞）。**
- **Falsifier**: single-seed proxy（s42, K=2, 40ep, vs g012 anchor 同窗）valid_rank_ic 较 anchor **跌 >0.005** → 立杀（large-batch 反噬成立）；或 **flat（|Δ|<0.003）** → 确认与 EMA/sampler 同属 below-detection-floor 的训练动力学空 lever（τ05 诊断 MDE +0.004–0.006）。n=6 阶段：`Δrank_ic < +0.004` 或 paired-t p≥0.1 → reject。
- **Config delta**（**trainer-only，零代码**）: 在 baseline_g012_scale05 anchor 之上 `QIB_TRAINER_OVERRIDES_JSON += {"grad_accum_steps": 2, "n_epochs": 40, "min_epochs": 40}`；model overrides = anchor 不变（τ scale=0.5 + d1pma g0.12）。
- **Step-budget 算账（必读，防混淆）**: 总 optimizer steps = epochs × days / K。Anchor = 25·D（K=1）。**K=2 + 40ep = 20·D ≈ 0.8× anchor**（近似 step-matched，干净隔离"跨天平均"本身）。⚠️ K=5 + 40ep = 8·D ≈ 0.32×，与 same-LR(5e-5) 叠加将**欠拟合**，把"步数变少/欠拟合"与"梯度方差降低"混为一谈 ⇒ PRIMARY 锁 **K=2**；更大 K 须同步 LR∝K 或 epoch∝K（昂贵），不在本卡。
- **Minimal test (proxy)**: csi300, K=2, 40ep, seed 42 single（`/quant-minimal-repro`）—— 看 valid_rank_ic 是否守住 anchor、**train/valid gap 是否扩大**（large-batch 反噬的直接读数）、train loss 是否到达 anchor 同水平（欠拟合排查）、router 不 collapse。
- **Expected delta vs current best**: RankIC 0.0786 → **持平±**（诚实先验：null-to-slightly-negative；**不指望涨，可能掉**）。
- **Budget**: proxy ≈1.5–3 GPU-hr（40ep 比 25ep proxy 略长）；n=6 **仅当 proxy 非负**才 +≈1 GPU-day。
- **Promotion criteria**: 向量门 —— n=6 `Δrank_ic ≥ +0.004`（过 MDE）AND paired-t p<0.1 AND ≥4/6 seeds 为正 AND regime-bucket 不 collapse。**先验弱**：更可能的产出是"确认空 lever"，本身即有价值（关掉一条训练动力学猜想，止血 GPU 流向 L-4/L-6）。
- **Risk / failure mode**: (1) **large-batch 泛化反噬（核心风险）**——降低 SGD 梯度噪声=去掉隐式正则，而本项目的墙正是 gen-gap（"finetune hurts valid"）→ 盯 `rank_ic_gap_last` / train-valid gap，扩大即坐实反噬；(2) **欠拟合混淆**——K=2/40ep 已 step-matched 规避，但仍盯 train loss 是否达 anchor 水平；(3) **below-detection-floor**——n=6 MDE +0.004–0.006，效应若 <0.004 不可辨；single-seed proxy 只能排除**大幅 negative**，不能确认 small positive（故 proxy 是 kill-gate 非 promote-gate）；(4) **无 leakage 新面**（trainer-only，数据/eval 管道零改；eval loader 已强制 accum_steps=1 model_adapter.py:1688）→ `/quant-leakage-audit` 走 fast-path，仅确认 grad accum 不触碰 valid forward。
- **RESULT 2026-06-29 (single-seed paired, step-matched, KILL)**: 见 ledger（proxy/reject）。control K=1/25ep vs treat K=2/50ep（seed42，同代码，各 25·D opt steps，model cfg 相同）：treat **训练拟合更深**（loss_mse 1.3294<1.3405，train rank_ic 0.186>0.169）却 **valid/test rank_ic 更低 0.0766 vs 0.0803（Δ−0.0038）**、IR_wc 1.62<1.81、AR_wc −0.99pp ⇒ **train-valid gap 扩大 = large-batch 泛化反噬**（卡里预登记的核心风险，干净复现）。非欠拟合（step-matched，且降得更深）。Δ−0.0038 单看 ~1σ seed 噪声，但"训练更好/验证更差"是**连贯机制**（非噪声）+ 先验弱负 ⇒ n=6 为 −EV，**不升 n=6，杀**。fresh control 0.0803 ≈ 归档 anchor 0.08163 ⇒ 工作树 +1548 行 drift 无害（~−0.001）。grad_accum K>1 = **closed lever**。**反转启示**：降梯度方差有害 ⇒ 未试的便宜方向是**增**随机性（daily batch<300 / 噪声注入 / dropout），非减。logs `logs/gradaccum_k2_s42/`。

## h-20260629-001: normalized-MSE（per-day CS z-score 双侧 ⇒ 纯 Pearson-IC loss）在 g012_d1pma 与 tau_scale_05 上能否抬 RankIC

- **Mechanism**: `cs_mse_loss(normalize=True)`（`losses.py:76-81`）把 pred 和 target **都**在 batch（=单日截面，FixedDailyBatchSampler）内 z-score 再 MSE。代数上 `L = 2(1 − per-day Pearson_IC)` ⇒ **主 loss 变成纯每日 IC 最大化（相关性 loss）**，pred 的 scale 与 common-mode 变自由。对比现行 plain MSE（`mse_normalize=False`）：row-83 已证 plain MSE 本就最大化 IC（`L*=1−IC²`）**但附带两个正则副约束**——dispersion-anchoring（`std(p)*≈IC·std(y)≈0.23`，即 confidence-proportional 收缩）+ common-mode 惩罚（`mean(p)→0`）；normalized-MSE 把这两个都**去掉**。设计透镜：metric-alignment（训你所测的相关性）vs 第 11 条 finding 的"MSE 副约束=正则器"。**`mse_normalize=True` 全清单从未跑过（恒 OFF）⇒ 新 lever，非重复（hash `aa44b8d43ced`）。**
- **Falsifier**: single-seed（s42, 25ep）vs 同代码同 seed 既有 control —— KILL 若 rank_ic 在**两个 setting 上都**较 control 跌 >0.005；或 mean IC 持平但 **ICIR/IR_wc 退化**（预登记失败模：去掉 confidence 收缩 ⇒ 低信号日过度 commit dispersion ⇒ IC_std↑，同 warm-q finding #9 的伤）。非负（rank_ic ≥ control 于 ≥1 setting 且 ICIR 不降）→ 才考虑 n=6。
- **Config delta**（**loss-only，零代码**，`mse_normalize` 为既有 flag）: 各 setting 在其 control 配置上仅加 `{"mse_normalize": true}`。
  - **Arm A** = anchor（g012_d1pma：τ0.5+d1pma+gate−2）+ `mse_normalize:true`，25ep，frictions-OFF。**Control = `gradaccum_ctrl_k1_25ep_s42`（rank_ic 0.0803, IR 1.81）**。
  - **Arm B** = readout-OFF τ0.5（`temporal_readout:""`, `ir_aux_lambda:0`）+ `mse_normalize:true`，25ep，COST-ON。**Control = `l6_ir000_s42`（rank_ic 0.0796, IC 0.0662, IR_wc 2.864, MaxDD_wc −0.0696）**。
- **Minimal test (proxy)**: 仅 **2 个新 treatment run**（A/B），seed 42，复用上面两个既有 same-code control（不重跑 control）。
- **Expected delta vs current best**: RankIC 持平±（先验 null-to-slightly-negative；主看是否出现 IC 平 / ICIR↓ 的去校准失败）。
- **Budget**: 2 × ~5 GPU-hr（单 seed 25ep，sequential）。
- **Promotion criteria**: 向量门 —— rank_ic ≥ control 于 ≥1 setting AND ICIR/IR_wc 不退 → n=6 paired 复核（先验弱，更可能确认"去正则伤 OOS"=第 11 条 pattern 第 6 例）。
- **Risk / failure mode**: (1) **去正则反噬**（核心，pattern 已 5×）——盯 train-valid gap + ICIR；(2) **dispersion 去校准**——盯 IC_mean vs IC_std 分解（IR_wc 的真凶，cf #9）；(3) 数值：z-score 分母 `std+eps`，薄日 upsample 重复名轻微偏置 batch 均值/方差（与 plain MSE 同病，非新 leakage）；(4) **leakage 无新面**——per-day 截面内归一、train-only loss、不跨日/跨 split（同 CSZScoreNorm 类）→ `/quant-leakage-audit` fast-path。
