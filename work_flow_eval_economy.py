# -*- coding: utf-8 -*-
"""
RST-MoE Economy Eval Workflow

Goal:
- Load a trained model from an existing recorder
- Run signal generation + IC/RankIC analysis
- (Optional) backtest

This is a minimal, fast evaluation pipeline without visuals or paper reports.

Usage:
  python work_flow_eval_economy.py --recorder_id <id>
  python work_flow_eval_economy.py --recorder_id <id> --run_backtest
  python work_flow_eval_economy.py --recorder_id <id> --use_saved_conf
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord, PortAnaRecord

from work_flow import data_conf as default_data_conf, port_conf as default_port_conf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal evaluation for RST-MoE (economy mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        "--use_saved_conf",
        action="store_true",
        help="Use saved run_conf from recorder if available",
    )
    parser.add_argument(
        "--run_backtest",
        action="store_true",
        help="Run portfolio backtest (optional)",
    )
    parser.add_argument(
        "--provider_uri",
        type=str,
        default="~/.qlib/qlib_data/cn_data",
        help="Qlib data provider URI",
    )
    return parser.parse_args()


def load_run_conf_from_recorder(rec) -> dict | None:
    try:
        conf = rec.load_object("run_conf")
        if isinstance(conf, dict) and conf:
            return conf
    except Exception:
        pass
    return None


def main() -> None:
    args = parse_args()

    print(f">>> [Phase 0] Init Qlib (provider_uri={args.provider_uri})")
    qlib.init(provider_uri=args.provider_uri, region=REG_CN)

    print(
        f">>> [Phase 0] Loading model from experiment={args.experiment_name}, recorder_id={args.recorder_id}"
    )

    with R.start(experiment_name=args.experiment_name, recorder_id=args.recorder_id, resume=True):
        rec = R.get_recorder()
        model = rec.load_object("model")
        print(f">>> [Phase 0] Model loaded: {type(model).__name__}")

        run_conf = None
        if args.use_saved_conf:
            run_conf = load_run_conf_from_recorder(rec)
            if run_conf:
                print(">>> [Phase 0] Using saved run_conf from recorder")
            else:
                print(">>> [Phase 0] No saved run_conf found, using default configs")

        if run_conf and "data_conf" in run_conf:
            data_conf = copy.deepcopy(run_conf["data_conf"])
        else:
            data_conf = copy.deepcopy(default_data_conf)

        if run_conf and "port_conf" in run_conf:
            port_conf = copy.deepcopy(run_conf["port_conf"])
        else:
            port_conf = copy.deepcopy(default_port_conf)

        print(">>> [Phase 1] Creating dataset...")
        dataset = init_instance_by_config(data_conf)

        print(">>> [Phase 2] Signal analysis...")
        SignalRecord(model, dataset, rec).generate()
        SigAnaRecord(rec).generate()

        if args.run_backtest:
            print(">>> [Phase 3] Backtesting...")
            PortAnaRecord(rec, port_conf, "day").generate()
        else:
            print(">>> [Phase 3] Backtesting skipped (--run_backtest not set)")

        print("\n" + "=" * 80)
        print("ECONOMY EVAL WORKFLOW COMPLETED")
        print("=" * 80)
        print(f"Experiment: {args.experiment_name}")
        print(f"Recorder ID: {args.recorder_id}")
        try:
            local_dir = Path(rec.get_local_dir())
            print(f"Results saved to: {local_dir}")
        except Exception:
            pass
        print("=" * 80)


if __name__ == "__main__":
    main()
