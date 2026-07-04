# h-20260610-002 设计 — 第三方向 expert(股票轴),严格对称三-expert MoE

**设计修订(owner 2026-06-12,binding):** 不采用 identity-start gate / 零贡献起步。新 expert 是现有两个
expert 的**平等同伴**,必须与它们**机制与初始化完全一致**——readout 的 identity-start 逻辑在此**不适用**:
那里需要"嵌套已知最优 control"做归因;这里三个 expert 都从随机初始化出发,由 learned router 公平分配。
反面教训直接来自本项目:(a) g0.12 实验证实标量门控会**冻结在初始值**——零起步的 expert 在 router 的
rich-get-richer 动力学下永远起不来,得到的 null 是初始化伪影;(b) 原有两个 expert 在 from-scratch 时
谁也没有特权。**对称才是科学的 A/B:2-expert anchor vs 3-expert variant,差异只有"多一个方向"。**

## 1. 架构 — 严格轴对称(每个 expert 沿一条轴做注意力,其余轴作 batch)

现状 `moe_block.py`(每层一个 block,n_layers=2),`x:[B,T,N,D]`,B=当日截面股票数(~300),T=8,N=158,D=64:

| expert | rearrange | 注意力沿 | 序列长 | 组数 | 信息通路 |
|---|---|---|---|---|---|
| time_expert(现有) | `b t n d -> (b n) t d` | T | 8 | B·N | 自身历史 |
| factor_expert(现有) | `b t n d -> (b t) n d` | N | 158 | B·T | 自身因子间 |
| **stock_expert(新)** | `b t n d -> (t n) b d` | **B** | ~300 | T·N | **同日同行(关系/相对定价)— 当前架构完全缺失** |

```
x:[B,T,N,D] → norm1 → (FiLM/mask 同现状,先于全部 expert,天然作用于三者)
  out_time   = time_expert  ((b n) t d)   ← ParallelAttention(config), bias_time(ALiBi 可选)
  out_factor = factor_expert((b t) n d)   ← ParallelAttention(config), bias_factor
  out_stock  = stock_expert ((t n) b d)   ← ParallelAttention(config), bias=None(股票间无序,无位置偏置)
  gate = softmax(router(regime ⊕ layer_summary) / temp)        [B,3]   ← Linear(d/2, 2→3)
  fused = w_t·out_time + w_f·out_factor + w_s·out_stock
  x = residual + fused → FFN(同现状)
```

## 2. 初始化 — 与现有 expert 逐点相同,零特殊化

- `self.stock_expert = ParallelAttention(config)` — 与 time/factor **同一个类、同一行构造**;`post_init`
  的 `_init_weights` 自动给 MHA in_proj `normal(0.02)`(`quant_moe_model.py` nn.MultiheadAttention 分支),
  与另两个 expert 完全一致。**不加任何 gate、缩放、零置。**
- Router 末层 `nn.Linear(d_model//2, 3)`(原 2)——同样吃 `post_init` normal(0.02)+zero-bias ⇒ 初始
  logits≈0 ⇒ **初始 gate ≈ (1/3,1/3,1/3)**,精确对应原 2-expert 初始 ≈(1/2,1/2) 的对称状态。
- 唯一的 flag 是 `use_stock_expert: bool = False`(config 加参 + allow-处理,默认关 ⇒ baseline 逐字节
  不变)。flag 是 A/B 机制,不是科学偏置。

## 3. 代码改动清单(全部条件于 `use_stock_expert`,默认路径零改动)

1. **`module/utils/model_configuration.py`**:加 `use_stock_expert: bool = False`。
2. **`module/architecture/moe_block.py`**:
   - `__init__`:`n_experts = 3 if use_stock_expert else 2`;router 末层 `Linear(d//2, n_experts)`;
     条件构造 `self.stock_expert`。
   - `forward`:`fixed_05` 分支 → `full(1/n_experts)`;`router_override` 加 `"stock"` 与 **`"no_stock"`**
     (leave-one-out:gate 在 time+factor 上重归一,stock 置零——§6 监控层级的核心探针),`"uniform"`
     → `1/n_experts`;stock 分支 `rearrange(x,"b t n d -> (t n) b d")` → expert → 还原;fused 加第三项。
   - **路由机制本身不变(binding):** 仍是 soft/dense——softmax 加权、全 expert 求和,无 top-k 稀疏分发。
     `logit_margin` **仅作为监控指标**从 `|l0−l1|` 推广为 sorted **top1−top2**(2-expert 时两者数值恒等,
     向后兼容;3-expert 时 `|l0−l1|` 会无视 stock logit,丢失"router 决断度"语义)。不引入 Switch/GShard
     式 top-k 稀疏路由:那是 E≫10 的省算力手段,E=3 全算很便宜,引入会改变机制+需要 load-balancing
     loss,污染 A/B。
   - diag 新增见 §6(监控套件,分三层)。
3. **`module/quant_moe_model.py` 路由诊断(L438-453)**:`factor_ratio = 1−time_ratio` 仅在 2-expert 成立
   → 改为直接读 `gw[:,1]`,新增 `router_layer_X_stock_ratio = gw[:,2]`;`entropy_norm` 除数
   `math.log(2)` → `math.log(n_experts)`;`collapse_ratio` 改为基于 `gw.max(dim=1)`(>0.9)定义。
4. **`module/model_adapter.py:2478`** `gw[:,:,0]` 取 time_ratio 对 [L,B,3] 天然兼容,无须改;export 的
   attention-maps 对 stock expert 的 `return_attn` 需**子采样**(见 §5 成本)或首版直接跳过 stock 图。
5. **不改** sampler/work_flow/损失——信息有效性依赖"batch=单日截面"这一既有不变量(见 §4)。

## 4. 有效性不变量(必须写进代码注释/断言)

stock 轴注意力**只在 batch=同一天的截面时有意义**。现状已保证:train `sampled_daily`/FixedDailyBatchSampler
按日组 batch;valid/test/predict 路径逐日推理(`model_adapter` 按 day 迭代,`cnt_by_day`)。在 stock 分支
加一行防御性注释 + (可选) `assert` batch 内日期唯一(若 batch 携带日期信息则断言,否则文档化)。
**推理时一只股票的分数依赖同日 batch 组成** —— 与 MASTER 式 cross-stock attention 同性质,且训练/推理
的 batch 构造一致(整日截面),无 train-serve skew。脱离日截面 batch 的任何调用(如随机 batch 的
finetune 脚本)对 stock expert 无效——文档化为使用约束。

## 5. 成本与显存

- 参数:+1 个 ParallelAttention/层 ≈ 4·D²=16k ×2 层 ≈ **+33k**(与现有单 expert 同量级;总参数 ~+8%)。
- 计算:注意力 FLOPs ∝ 组数×L²×D — stock = 1264×300²×64 ≈ 7.3e9/层,约为 factor expert(2400×158²)
  的 ~2×,整体训练时间预估 **~1.3-1.6×**(5h → 7-8h/run)。
- 显存:`need_weights=False` 走 PyTorch SDPA 快径,不物化 [B·H,L,L] 概率矩阵——可行;`return_attn=True`
  的诊断路径会物化 1264×H×300² ≈ 大,**stock 注意力图导出必须子采样 (t,n) 组**(如每日固定 4 组)或
  首版跳过。1-epoch smoke 是显存的硬验证门。

## 6. 实验协议(对照 = 新锚点 τ0.5+d1pma g0.12)

- **Phase 0**:CPU 构造断言(3-expert 初始 gate≈1/3±噪声;`use_stock_expert=False` 与 anchor 逐字节同构
  ——state_dict 对比)+ 1-epoch GPU smoke(显存/管线/诊断键)。
- **Phase 1(n=1 screen)**:anchor+`use_stock_expert=true`,seed42,25ep,scale=0.5;对比 anchor seed42。
- **Phase 2(n=3 paired)**:seeds 42/43/44 vs anchor 三件套;四元组 {RankIC, IR_with_cost, MaxDD, ppd}。
- **Kill 线(预注册,沿用卡片)**:paired n=3 ΔRankIC < +0.003;或 stock-gate share < 0.05(**现在这是
  诚实信号**:router 从 1/3 起步、主动学会抛弃它 = 学习结论,非初始化伪影);或 stock 注意力退化
  (§6 L3 的 uniform-塌缩/self-塌缩,任一)。
- **解读预注册**:share→0 = "同日关系信息无增量"(诚实 kill);share 稳定>0.1 且 Δ≥+0.003 → n≥6 +
  HC kill-check(家训);share 高但 Δ≈0 → mechanism≠benefit,kill;**L2 的 no_stock 探针 Δ≈0 而
  share 高 = "router 在用但没带来收益"(τ-collapse 模式),kill。**

## 6b. 监控套件(三层裁决力——pool-readout 教训:权重统计量对"有用"是盲的,liveness 必须靠 probe)

**L1 收益(唯一裁决层)** — 四元组 paired ΔRankIC / ΔIR_with_cost / ΔMaxDD / Δppd vs anchor。kill 线只挂这里。

**L2 机制-liveness(归因层,可裁"机制死活",不可裁收益)**
| 指标 | 定义 | 读法 |
|---|---|---|
| `router_stock_advantage` | forced `router_override="stock"` 的 solo rank_ic − default(复用现有 forced-expert 诊断机制,自动入 diagnostic_matrix) | stock 单独值多少(类比 router_time_advantage=−0.015) |
| **`router_no_stock_delta`** | **leave-one-out**:`override="no_stock"`(time+factor 重归一)的 rank_ic − default | **stock 的边际贡献,免重训的最干净机制读数**——pool 战役里 PROBE-CEILING 的同款逻辑 |
| `router_layer_X_stock_ratio` | gate 第 3 列均值(逐 epoch 轨迹,从 0.333 出发) | 往哪走 = router 的投票;<0.05 触 kill |
| `expert_layer_X_stock_expert_norm` / `stock_contrib_norm` | 裸输出范数 / gate 加权贡献范数(镜像现有 time/factor 两列) | 贡献是否实质非零(对照 finding #5 的 contrib_norm_ratio 口径) |
| `expert_layer_X_cosine_ts` / `cosine_fs` | stock-vs-time、stock-vs-factor 输出余弦(现有 `expert_cosine` 只测 time-vs-factor,补全三角) | **新方向是否正交**——decorrelation 是 expert 存在理由(finding #5);cos≈1 = 冗余,即使 share 高也无增量 |
| `stock_winner_ratio` | contrib 范数最大者为 stock 的样本占比(镜像 `time_winner_ratio`) | 主导权分布 |

**L3 结构描述(仅解释,永不裁决——entropy 类指标的教训)**
| 指标 | 定义 | 病理信号 |
|---|---|---|
| `stock_attn_entropy_norm` | 注意力熵 / ln(B)(B 逐日变,按 batch 归一) | →1 = uniform-塌缩(无选择,six-nines 同款);→0 见下 |
| `stock_attn_self_frac` | 对角(自注意)质量占比 | →1 = self-塌缩,退化为逐股恒等,根本没用同行信息 |
| `stock_attn_hub_topk_share` | 入度最高 5% 股票吸收的注意力质量占比 | 高 = "全场都看龙头/指数权重股"(可解释的市场结构,非病理);配合 regime 分桶看是否随相关性 regime 移动 |
| 日级 `stock_ratio` 序列 → regime 分桶 | 接入现有 `st_disentangle_*` 导出 + regime-bucket 条件分析(`market_state_corr_*` 已有) | 机制合理性:高相关 regime 下 router 应更用 stock 通路 |

**工程约束:** L3 需要注意力权重,`return_attn` 物化 1264×H×300² 不可行——**子采样**:仅 t=末步 ×
固定 8 个 factor 槽位,且仅在 eval/诊断 pass 计算(不进训练步)。`model_adapter:2478` 的日级导出
`gw[:,:,0]` 旁加 `gw[:,:,2]`(stock 日级序列)。L2 全部是范数/gate 统计,零额外注意力物化,逐 epoch 可出。

## 7. 风险

- **R1 维度耦合**:B 轴长度逐日变化(~270-300)——MHA 对变长 L 无问题;但任何 per-L 参数化(如学习式
  位置偏置)被禁止,股票无序 ⇒ bias=None(置换等变,天然正确)。
- **R2 router 三路竞争初期不稳**:沿用现有 `router_noise`/`router_temperature`/z-loss 机制,不新增超参。
- **R3 诊断兼容**:§3.3 的三处 2-expert 硬编码必须同改,否则 3-expert 运行时 factor_ratio/entropy_norm
  静默错误(provenance 教训:先改读数,再开跑)。
- **R4 渗漏审计**:stock expert 只看**同日同 batch** 的其他股票的**当日及更早**表征(x 本身无未来),
  无新泄漏面;但按惯例先过 `/quant-leakage-audit` 再开跑。
