# -*- coding: utf-8 -*-
"""
RST-MoE Eval-Only Workflow (No Training)

功能：
1. 加载已训练好的模型（从 MLFlow recorder 中读取）
2. 运行 Signal 分析 + 组合回测（不需要重新训练）
3. 生成论文级实验报告

用法：
python work_flow_eval.py --recorder_id <your_recorder_id>
python work_flow_eval.py --recorder_id <your_recorder_id> --experiment_name <exp_name>
python work_flow_eval.py --recorder_id <your_recorder_id> --skip_visuals
python work_flow_eval.py --recorder_id <ANY_ID> --config_path dummy_config.py
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Optional, Dict
import importlib.util
import sys

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord

# Import shared utilities from work_flow.py
from work_flow import (
    data_conf as default_data_conf,
    port_conf as default_port_conf,
    generate_paper_report,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run RST-MoE evaluation and backtesting without training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python work_flow_eval.py --recorder_id e111a8e9223e4bddb8b80a04dd975c40
  python work_flow_eval.py --recorder_id e111a8e9223e4bddb8b80a04dd975c40 --skip_visuals
  python work_flow_eval.py --recorder_id e111a8e9223e4bddb8b80a04dd975c40 --segment valid
        """,
    )
    parser.add_argument(
        "--recorder_id",
        type=str,
        required=True,
        help="The recorder ID containing the trained model",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="Official_Alignment_RST_MoE",
        help="MLFlow experiment name (default: Official_Alignment_RST_MoE)",
    )
    parser.add_argument(
        "--segment",
        type=str,
        default="test",
        help="Evaluation segment: train, valid, or test (default: test)",
    )
    parser.add_argument(
        "--skip_visuals",
        action="store_true",
        help="Skip exporting visual diagnostics",
    )
    parser.add_argument(
        "--skip_backtest",
        action="store_true",
        help="Skip portfolio backtesting",
    )
    parser.add_argument(
        "--skip_report",
        action="store_true",
        help="Skip generating paper report",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="RST-MoE",
        help="Model name for the report (default: RST-MoE)",
    )
    parser.add_argument(
        "--use_saved_conf",
        action="store_true",
        help="Force using saved run_conf from the recorder when available",
    )
    parser.add_argument(
        "--no_saved_conf",
        action="store_true",
        help="Disable auto-using saved run_conf (always use default configs)",
    )
    parser.add_argument(
        "--provider_uri",
        type=str,
        default="~/.qlib/qlib_data/cn_data",
        help="Qlib data provider URI",
    )
    parser.add_argument(
        "--config_path",
        "-c",
        type=str,
        default=None,
        help="Path to a custom python config file (defining data_conf, port_conf)",
    )
    return parser.parse_args()


def load_run_conf_from_recorder(rec) -> dict | None:
    """Try to load run_conf from recorder, return None if not found."""
    try:
        conf = rec.load_object("run_conf_resolved")
        if isinstance(conf, dict) and conf:
            return conf
    except Exception:
        pass
    try:
        conf = rec.load_object("run_conf")
        if isinstance(conf, dict) and conf:
            return conf
    except Exception:
        pass
    return None


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Load a python config file as a dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    spec = importlib.util.spec_from_file_location("custom_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config from {config_path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules["custom_config"] = module
    spec.loader.exec_module(module)
    
    return {k: v for k, v in module.__dict__.items() if not k.startswith("__")}


def main():
    args = parse_args()

    # =============================================================================
    # 1. Initialize Qlib
    # =============================================================================
    print(f">>> [Phase 0] Initializing Qlib with provider_uri={args.provider_uri}")
    qlib.init(provider_uri=args.provider_uri, region=REG_CN)

    # =============================================================================
    # 2. Enter existing experiment/recorder to load model
    # =============================================================================
    print(f">>> [Phase 0] Loading model from experiment={args.experiment_name}, recorder_id={args.recorder_id}")

    # Use R.start to enter the existing recorder context
    with R.start(experiment_name=args.experiment_name, recorder_id=args.recorder_id, resume=True):
        rec = R.get_recorder()

        # Load trained model
        print(">>> [Phase 0] Loading trained model...")
        model = rec.load_object("model")
        print(f">>> [Phase 0] Model loaded: {type(model).__name__}")

        # Optionally load saved run_conf
        run_conf = None
        if args.no_saved_conf:
            run_conf = None
            print(">>> [Phase 0] Saved run_conf disabled (--no_saved_conf)")
        else:
            run_conf = load_run_conf_from_recorder(rec)
            if run_conf:
                if args.use_saved_conf:
                    print(">>> [Phase 0] Using saved run_conf from recorder (--use_saved_conf)")
                else:
                    print(">>> [Phase 0] Using saved run_conf from recorder (auto)")
            elif args.use_saved_conf:
                print(">>> [Phase 0] No saved run_conf found, using default configs")

        # Determine data and port configs
        if run_conf and "data_conf" in run_conf:
            data_conf = copy.deepcopy(run_conf["data_conf"])
        else:
            data_conf = copy.deepcopy(default_data_conf)

        if run_conf and "port_conf" in run_conf:
            port_conf = copy.deepcopy(run_conf["port_conf"])
        else:
            port_conf = copy.deepcopy(default_port_conf)

        # Override with custom config if provided
        if args.config_path:
            print(f">>> [Phase 0] Loading custom config from {args.config_path}")
            custom_cfg = load_config_from_file(args.config_path)
            if "data_conf" in custom_cfg:
                print(">>> [Phase 0] Overriding data_conf from custom config")
                data_conf = custom_cfg["data_conf"]
            if "port_conf" in custom_cfg:
                print(">>> [Phase 0] Overriding port_conf from custom config")
                port_conf = custom_cfg["port_conf"]

        # =============================================================================
        # 3. Create dataset
        # =============================================================================
        print(">>> [Phase 0] Creating dataset...")
        dataset = init_instance_by_config(data_conf)

        # =============================================================================
        # 4. Export Spatio-Temporal Visuals (optional)
        # =============================================================================
        if not args.skip_visuals:
            print(f">>> [Phase 1.1] Export Spatio-Temporal Visuals (segment={args.segment})...")
            try:
                model.export_visuals(
                    dataset,
                    segment=args.segment,
                    max_attn_days=4,
                    attn_layer=-1,
                    attn_layers="all",
                    target_dates=None,
                    prefix="st_disentangle",
                )
            except Exception as e:
                print(f">>> [Phase 1.1] Warning: export_visuals failed: {e}")
        else:
            print(">>> [Phase 1.1] Skipping visual export (--skip_visuals)")

        # =============================================================================
        # 5. Signal Generation & Analysis (IC / RankIC)
        # =============================================================================
        print(f">>> [Phase 2] Signal Analysis (segment={args.segment})...")
        SignalRecord(model, dataset, rec).generate()
        SigAnaRecord(rec).generate()

        # =============================================================================
        # 6. Portfolio Backtesting (optional)
        # =============================================================================
        if not args.skip_backtest:
            print(">>> [Phase 3] Backtesting...")
            PortAnaRecord(rec, port_conf, "day").generate()
        else:
            print(">>> [Phase 3] Skipping backtesting (--skip_backtest)")

        # =============================================================================
        # 7. Generate Paper Report (optional)
        # =============================================================================
        if not args.skip_report:
            print(">>> [Phase 4] Generate Paper-level Report...")
            generate_paper_report(
                rec,
                model_name=args.model_name,
                dataset=dataset,
                segment=args.segment,
            )
        else:
            print(">>> [Phase 4] Skipping report generation (--skip_report)")

        # Print summary
        print("\n" + "=" * 80)
        print("EVAL-ONLY WORKFLOW COMPLETED")
        print("=" * 80)
        print(f"Experiment: {args.experiment_name}")
        print(f"Recorder ID: {args.recorder_id}")
        print(f"Segment: {args.segment}")
        try:
            local_dir = Path(rec.get_local_dir())
            print(f"Results saved to: {local_dir}")
            report_path = local_dir / "kdd_report.md"
            if report_path.exists():
                print(f"Report: {report_path}")
        except Exception:
            pass
        print("=" * 80)


if __name__ == "__main__":
    main()
