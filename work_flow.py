import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord

# ==========================================
# 1. Initialize (跟官方 yaml qlib_init 对齐)
# ==========================================
provider_uri = "~/.qlib/qlib_data/cn_data"
qlib.init(provider_uri=provider_uri, region=REG_CN)

# ==========================================
# 2. Data Config (跟官方 task.dataset 对齐)
# ==========================================
# 差异点：我们将 DatasetH 升级为 TSDatasetH 以支持时序模型
# 差异点：移除了 learn_processors 中的 DropnaLabel 以保护时间连续性
data_conf = {
    "class": "TSDatasetH",
    "module_path": "qlib.data.dataset",
    "kwargs": {
        "step_len": 32,  # 时序窗口，对应模型 context_len
        "handler": {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "start_time": "2008-01-01",
                "end_time": "2020-08-01",
                "fit_start_time": "2008-01-01",
                "fit_end_time": "2014-12-31",
                "instruments": "csi300",
                # [官方对齐] 推理预处理：去极值 + 填充
                "infer_processors": [
                    {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                    {"class": "Fillna", "kwargs": {"fields_group": "feature"}},                ],
                # [深度学习特化] 必须为 None，防止由 Dropna 导致的时间断裂
                "learn_processors": [],
                # [官方对齐] Label 定义
                "label": ["Ref($close, -1) / $close - 1"]
            }
        },
        "segments": {
            "train": ("2008-01-01", "2014-12-31"),
            "valid": ("2015-01-01", "2016-12-31"),
            "test": ("2017-01-01", "2020-08-01")
        }
    }
}

# ==========================================
# 3. Model Config (你的 RST-MoE)
# ==========================================
model_conf = {
    "class": "QlibQuantMoE",
    "module_path": "module.model_adapter",
    "kwargs": {
        "model_config": {
            "d_model": 32,
            "n_layers": 2,
            "use_feature_selection": True,
            # context_len 和 num_alphas 会自动探测
        },
        "trainer_config": {
            "lr": 5e-4,
            "n_epochs": 20,
            "batch_size": 256,  # 配合 FixedDailyBatchSampler
            "early_stop": 5,
            "num_workers": 0 # for debug only
        }
    }
}

# ==========================================
# 4. Strategy & Backtest Config (完全复制官方 port_analysis_config)
# ==========================================
port_conf = {
    "strategy": {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy.signal_strategy",
        "kwargs": {
            "signal": "<PRED>",  # 占位符，实际运行时会自动替换
            "topk": 50,
            "n_drop": 5
        }
    },
    "backtest": {
        "start_time": "2017-01-01",
        "end_time": "2020-08-01",
        "account": 100000000,
        "benchmark": "SH000300",
        "exchange_kwargs": {
            "freq": "day",
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5
        }
    }
}

# ==========================================
# 5. Execution Pipeline
# ==========================================
if __name__ == "__main__":
    # 实例化数据和模型
    dataset = init_instance_by_config(data_conf)
    model = init_instance_by_config(model_conf)

    # 启动实验
    with R.start(experiment_name="Official_Alignment_RST_MoE"):
        # 1. 记录超参
        R.log_params(**flatten_dict(model_conf))

        # 2. 训练
        print(">>> [Phase 1] Training Model...")
        model.fit(dataset)
        R.save_objects(model=model)

        # 3. 信号生成与分析 (IC/RankIC)
        # 注意：这里会自动在 Test 集 (2017-2020) 上跑
        print(">>> [Phase 2] Signal Analysis...")
        rec = R.get_recorder()
        SignalRecord(model, dataset, rec).generate()
        SigAnaRecord(rec).generate()

        # 4. 组合回测 (Backtest)
        print(">>> [Phase 3] Backtesting...")
        PortAnaRecord(rec, port_conf, "day").generate()

        # ==========================================
        # 6. Generate SOTA Report Table
        # ==========================================
        print("\n" + "=" * 80)
        print(f" EXPERIMENT REPORT: {rec.info['id']}")
        print("=" * 80)

        try:
            # Load Metrics
            sar = SigAnaRecord(rec)
            ic = sar.load("ic.pkl").mean()
            ric = sar.load("ric.pkl").mean()

            # Load Backtest Metrics (Needs to read from indicator file)
            # PortAnaRecord saves 'indicator_analysis_1day.pkl'
            par_path = rec.get_local_dir() / "indicator_analysis_1day.pkl"

            if par_path.exists():
                perf = pd.read_pickle(par_path)
                ann_ret = perf["annualized_return"].iloc[0]
                info_ratio = perf["information_ratio"].iloc[0]
                max_dd = perf["max_drawdown"].iloc[0]
            else:
                # Fallback if file not ready (rare)
                metrics = rec.list_metrics()
                ann_ret = metrics.get("excess_return_with_cost.annualized_return", 0)
                info_ratio = metrics.get("excess_return_with_cost.information_ratio", 0)
                max_dd = metrics.get("excess_return_with_cost.max_drawdown", 0)

            # Create DataFrame
            df_res = pd.DataFrame([{
                "Model": "RST-MoE",
                "Dataset": "Alpha158 (Official Split)",
                "Rank IC": f"{ric:.4f}",
                "ICIR": f"{ric / sar.load('ric.pkl').std():.2f}",
                "Ann. Ret": f"{ann_ret:.2%}",
                "Info Ratio": f"{info_ratio:.2f}",
                "Max DD": f"{max_dd:.2%}"
            }])

            print(df_res.to_markdown(index=False))
            print("-" * 80)
            print("Benchmarks (Reference for 2017-2020):")
            print("LightGBM  | Rank IC: ~0.080 | Ann. Ret: ~20% | Max DD: ~-10%")
            print("Linear    | Rank IC: ~0.050 | Ann. Ret: ~8%  | Max DD: ~-15%")

        except Exception as e:
            print(f"Report generation failed: {e}")
            print(f"Check raw results in: {rec.get_local_dir()}")