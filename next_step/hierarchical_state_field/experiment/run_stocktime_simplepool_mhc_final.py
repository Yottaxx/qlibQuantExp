from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from next_step.hierarchical_state_field.experiment import run_stocktime_simplepool_mhc as runner


FINAL_STEP = {
    "name": "stocktime_simplepool_mhc_v7_final",
    "label": "csi300_tri_scope_legacy_noinner_simplepool_mhc_v7",
    "workflow_module": (
        "next_step.hierarchical_state_field.experiment.configs."
        "ref_all_hsf_csi300_tri_scope_stocktime_simplepool_mhc_final"
    ),
    "intent": (
        "v7 ablation: legacy regime branch, no inner cross-stock attention, "
        "simple-pooling mHC-lite, higher dropout, lower learning rate."
    ),
}

runner.STEP = FINAL_STEP


def _final_design() -> dict[str, Any]:
    from next_step.hierarchical_state_field.experiment.configs import (
        ref_all_hsf_csi300_tri_scope_stocktime_simplepool_mhc_final as cfg,
    )

    model_cfg = cfg.model_conf["kwargs"]["model_config"]
    trainer_cfg = cfg.model_conf["kwargs"]["trainer_config"]
    model_keys = (
        "feature_tokenizer_add_factor_id",
        "router_aux_loss_type",
        "router_z_loss_coef",
        "use_hierarchical_state_field",
        "factor_gate_scale",
        "factor_gate_shift_scale",
        "router_summary_source",
        "router_summary_fusion_mode",
        "use_inner_cross_stock_attention",
        "inner_cross_stock_mode",
        "use_cross_stock_attention",
        "pooling_mode",
        "pooling_alpha",
        "pooling_summary_source",
        "temporal_pooling_mode",
        "temporal_mhc_mix_init",
        "temporal_mhc_mix_max",
        "dropout",
        "regime_macro_dropout",
    )
    trainer_keys = (
        "lr",
        "optimizer",
        "weight_decay",
        "adam_betas",
        "adam_eps",
        "adam_amsgrad",
        "adam_foreach",
        "adam_fused",
        "adamw_decay_matrix_only",
        "n_epochs",
        "early_stop",
        "train_stop_threshold",
        "seed",
    )
    design = {key: model_cfg.get(key) for key in model_keys}
    design.update({key: trainer_cfg.get(key) for key in trainer_keys})
    return design


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CSI300 legacy-regime no-inner simple-pooling mHC-lite v7 ablation.")
    parser.add_argument("--python", dest="python_exe", default=runner._default_python())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--market-state-path", default=runner.MARKET_STATE_PATH)
    parser.add_argument("--market-day-summary-path", default=runner.MARKET_DAY_SUMMARY_PATH)
    parser.add_argument(
        "--run-tag",
        default=dt.datetime.now().strftime("hsf_legacy_noinner_simplepool_mhc_v7_%Y%m%d_%H%M%S"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--with-visuals", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner.MARKET_STATE_PATH = str(args.market_state_path)
    runner.MARKET_DAY_SUMMARY_PATH = str(args.market_day_summary_path)

    run_dir = runner.RUNS_DIR / args.run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = runner._build_command(
        python_exe=args.python_exe,
        run_tag=args.run_tag,
        seed=args.seed,
        skip_visuals=not bool(args.with_visuals),
    )
    runner._write_text(run_dir / "commands.txt", f"[{FINAL_STEP['name']}]\n{runner._command_to_text(cmd)}\n")

    status: dict[str, Any] = {
        "run_tag": args.run_tag,
        "created_at": runner._now(),
        "project_root": str(runner.PROJECT_ROOT),
        "variant_script": str(runner.VARIANT_SCRIPT),
        "mlflow_tracking_uri": f"file:{runner.MLRUNS_DIR.resolve()}",
        "python_exe": args.python_exe,
        "seed": args.seed,
        "dry_run": bool(args.dry_run),
        "skip_visuals": not bool(args.with_visuals),
        "policy": "legacy_noinner_simplepool_mhc_v7",
        "baseline": "ref_all_hsf_csi300_tri_scope_stocktime_simplepool_mhc",
        "market_state_path": runner.MARKET_STATE_PATH,
        "market_day_summary_path": runner.MARKET_DAY_SUMMARY_PATH,
        "acceptance": {
            "best_rankic_min": 0.0615,
            "portfolio_ir_min": 0.71,
            "max_drawdown_floor": -0.135,
            "rankic_ir_min": 0.36,
        },
        "design": _final_design(),
        "steps": {
            FINAL_STEP["name"]: {
                "status": "planned",
                "workflow_module": FINAL_STEP["workflow_module"],
                "label": FINAL_STEP["label"],
                "intent": FINAL_STEP["intent"],
                "command": runner._command_to_text(cmd),
            }
        },
    }
    runner._write_json(run_dir / "status.json", status)
    runner._write_text(
        run_dir / "README.md",
        "# Legacy-Regime No-Inner Simple Pooling mHC-lite v7\n\n"
        "- Baseline: `ref_all_hsf_csi300_tri_scope_stocktime_simplepool_mhc`.\n"
        "- Deltas: HSF disabled, inner cross-stock disabled, factor ID added to tokenizer, router z-loss, router summary=batch, dropout 0.3, regime macro dropout 0.2, lr 1e-5, AdamW wd 0.01 matrix-only, betas (0.9, 0.998), n_epochs 40, early stop disabled.\n"
        "- Scope: CSI300 only, seed 42, tri-scope macro assets, simple static factor pooling.\n",
    )

    print(f"[RunDir] {run_dir}", flush=True)
    print(f"[Commands] {run_dir / 'commands.txt'}", flush=True)
    if args.dry_run:
        status["status"] = "dry_run_completed"
        status["finished_at"] = runner._now()
        runner._write_json(run_dir / "status.json", status)
        return 0

    status["status"] = "running"
    status["started_at"] = runner._now()
    runner._write_json(run_dir / "status.json", status)

    print(f"[Start] {FINAL_STEP['name']}", flush=True)
    returncode = runner._run_step(cmd=cmd, run_dir=run_dir, status=status)
    print(f"[Finish] {FINAL_STEP['name']} returncode={returncode}", flush=True)

    status["status"] = "completed" if returncode == 0 else "failed"
    status["finished_at"] = runner._now()
    status["exit_code"] = returncode
    runner._write_json(run_dir / "status.json", status)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
