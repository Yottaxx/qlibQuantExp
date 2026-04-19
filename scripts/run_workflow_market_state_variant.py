from __future__ import annotations

import argparse
import copy
import importlib
import json
from pathlib import Path
import re
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qlib.utils import flatten_dict, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord, SignalRecord

import work_flow as workflow_helpers
from module.utils.qlib_official_graphs import ensure_qlib_official_graphs


def _log(message: str) -> None:
    print(message, flush=True)


def _int_flag(value: str) -> int:
    ivalue = int(value)
    if ivalue not in (0, 1):
        raise argparse.ArgumentTypeError("Expected 0 or 1.")
    return ivalue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a workflow config module with market-state/day-summary overrides for controlled ablation runs.",
    )
    parser.add_argument(
        "--workflow_module",
        type=str,
        default="work_flow",
        help="Config module exposing data_conf/model_conf/port_conf. Defaults to work_flow.",
    )
    parser.add_argument("--market_state_path", type=str, required=True, help="Override trainer_config.market_state_path.")
    parser.add_argument(
        "--market_day_summary_path",
        type=str,
        default=None,
        help="Optional override for trainer_config.market_day_summary_path.",
    )
    parser.add_argument(
        "--router_summary_source",
        type=str,
        default=None,
        choices=["none", "batch", "day_asset"],
        help="Override model_config.router_summary_source.",
    )
    parser.add_argument(
        "--pooling_summary_source",
        type=str,
        default=None,
        choices=["none", "batch", "day_asset"],
        help="Override model_config.pooling_summary_source.",
    )
    parser.add_argument("--label", type=str, required=True, help="Short variant label saved into the recorder.")
    parser.add_argument(
        "--experiment_suffix",
        type=str,
        default="",
        help="Optional suffix appended to the experiment name for isolation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed override. If omitted, keep the config default.",
    )
    parser.add_argument("--use_hierarchical_state_field", type=_int_flag, default=None, help="Override model_config.use_hierarchical_state_field with 0/1.")
    parser.add_argument("--d_global_state", type=int, default=None, help="Override model_config.d_global_state.")
    parser.add_argument("--d_local_state", type=int, default=None, help="Override model_config.d_local_state.")
    parser.add_argument("--global_state_use_macro", type=_int_flag, default=None, help="Override model_config.global_state_use_macro with 0/1.")
    parser.add_argument("--global_state_use_day_summary", type=_int_flag, default=None, help="Override model_config.global_state_use_day_summary with 0/1.")
    parser.add_argument("--local_state_input_mode", type=str, default=None, help="Override model_config.local_state_input_mode.")
    parser.add_argument(
        "--state_fusion_mode",
        type=str,
        default=None,
        choices=["sum_norm", "branch_mlp_v1", "bounded_sum_v1"],
        help="Override model_config.state_fusion_mode.",
    )
    parser.add_argument("--local_router_scale_init", type=float, default=None)
    parser.add_argument("--local_pooling_scale_init", type=float, default=None)
    parser.add_argument("--local_pooling_alpha_scale_init", type=float, default=None)
    parser.add_argument("--local_film_scale_init", type=float, default=None)
    parser.add_argument("--local_scale_learnable", type=_int_flag, default=None)
    parser.add_argument("--router_use_global_state", type=_int_flag, default=None, help="Override model_config.router_use_global_state with 0/1.")
    parser.add_argument("--router_use_local_state", type=_int_flag, default=None, help="Override model_config.router_use_local_state with 0/1.")
    parser.add_argument("--film_use_global_state", type=_int_flag, default=None, help="Override model_config.film_use_global_state with 0/1.")
    parser.add_argument("--film_use_local_state", type=_int_flag, default=None, help="Override model_config.film_use_local_state with 0/1.")
    parser.add_argument("--pooling_use_global_state", type=_int_flag, default=None, help="Override model_config.pooling_use_global_state with 0/1.")
    parser.add_argument("--pooling_use_local_state", type=_int_flag, default=None, help="Override model_config.pooling_use_local_state with 0/1.")
    parser.add_argument(
        "--enable_local_counterfactual_diag",
        type=_int_flag,
        default=None,
        help="Override trainer_config.enable_local_counterfactual_diag with 0/1.",
    )
    parser.add_argument(
        "--enable_expert_advantage_diag",
        type=_int_flag,
        default=None,
        help="Override trainer_config.enable_expert_advantage_diag with 0/1.",
    )
    parser.add_argument(
        "--skip_visuals",
        action="store_true",
        help="Skip export_visuals to reduce runtime.",
    )
    return parser.parse_args()


def _load_workflow_module(module_name: str) -> Any:
    module_name = str(module_name or "work_flow").strip()
    return importlib.import_module(module_name or "work_flow")


def _clone_conf_with_overrides(
    args: argparse.Namespace,
    workflow_module: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    missing = [name for name in ("data_conf", "model_conf", "port_conf") if not hasattr(workflow_module, name)]
    if missing:
        raise AttributeError(
            f"workflow_module={args.workflow_module!r} is missing required config object(s): {missing}"
        )
    data_conf = copy.deepcopy(getattr(workflow_module, "data_conf"))
    model_conf = copy.deepcopy(getattr(workflow_module, "model_conf"))
    port_conf = copy.deepcopy(getattr(workflow_module, "port_conf"))

    model_k = ((model_conf.get("kwargs") or {}).get("model_config") or {})
    trainer_k = ((model_conf.get("kwargs") or {}).get("trainer_config") or {})
    trainer_k["market_state_path"] = str(args.market_state_path)
    if args.market_day_summary_path is not None:
        trainer_k["market_day_summary_path"] = str(args.market_day_summary_path)
    if args.router_summary_source is not None:
        model_k["router_summary_source"] = str(args.router_summary_source)
    if args.pooling_summary_source is not None:
        model_k["pooling_summary_source"] = str(args.pooling_summary_source)
    if args.seed is not None:
        trainer_k["seed"] = int(args.seed)
    if args.use_hierarchical_state_field is not None:
        model_k["use_hierarchical_state_field"] = bool(args.use_hierarchical_state_field)
    if args.d_global_state is not None:
        model_k["d_global_state"] = int(args.d_global_state)
    if args.d_local_state is not None:
        model_k["d_local_state"] = int(args.d_local_state)
    if args.global_state_use_macro is not None:
        model_k["global_state_use_macro"] = bool(args.global_state_use_macro)
    if args.global_state_use_day_summary is not None:
        model_k["global_state_use_day_summary"] = bool(args.global_state_use_day_summary)
    if args.local_state_input_mode is not None:
        model_k["local_state_input_mode"] = str(args.local_state_input_mode)
    if args.state_fusion_mode is not None:
        model_k["state_fusion_mode"] = str(args.state_fusion_mode)
    for key in (
        "local_router_scale_init",
        "local_pooling_scale_init",
        "local_pooling_alpha_scale_init",
        "local_film_scale_init",
    ):
        value = getattr(args, key, None)
        if value is not None:
            model_k[key] = float(value)
    if args.local_scale_learnable is not None:
        model_k["local_scale_learnable"] = bool(args.local_scale_learnable)
    if args.router_use_global_state is not None:
        model_k["router_use_global_state"] = bool(args.router_use_global_state)
    if args.router_use_local_state is not None:
        model_k["router_use_local_state"] = bool(args.router_use_local_state)
    if args.film_use_global_state is not None:
        model_k["film_use_global_state"] = bool(args.film_use_global_state)
    if args.film_use_local_state is not None:
        model_k["film_use_local_state"] = bool(args.film_use_local_state)
    if args.pooling_use_global_state is not None:
        model_k["pooling_use_global_state"] = bool(args.pooling_use_global_state)
    if args.pooling_use_local_state is not None:
        model_k["pooling_use_local_state"] = bool(args.pooling_use_local_state)
    if args.enable_local_counterfactual_diag is not None:
        trainer_k["enable_local_counterfactual_diag"] = bool(args.enable_local_counterfactual_diag)
    if args.enable_expert_advantage_diag is not None:
        trainer_k["enable_expert_advantage_diag"] = bool(args.enable_expert_advantage_diag)
    return data_conf, model_conf, port_conf


def _slug(value: Any, *, max_len: int = 120) -> str:
    text = str(value if value is not None else "na").strip()
    text = re.sub(r"[^A-Za-z0-9._=+-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return (text or "na")[:max_len]


def _local_build_experiment_name(model_k: dict[str, Any], trainer_k: dict[str, Any]) -> str:
    state_path = trainer_k.get("market_state_path", None)
    state_stem = Path(str(state_path)).stem if state_path else "none"
    use_hsf = bool(model_k.get("use_hierarchical_state_field", False))
    parts = [
        "Official_Alignment_RST_MoE",
        f"loss-{model_k.get('main_loss', 'na')}",
        f"d{model_k.get('d_model', 'na')}",
        f"l{model_k.get('n_layers', 'na')}",
        f"ctx{model_k.get('context_len', 'na')}",
        f"emb{model_k.get('value_embedding_type', 'na')}",
        f"time{int(bool(model_k.get('use_regime_time_embedding', True)))}",
        f"film{int(bool(model_k.get('use_regime_factor_gate', True)))}",
        f"rSrc{model_k.get('router_summary_source', 'none')}",
        f"pSrc{model_k.get('pooling_summary_source', 'none')}",
        f"macro{int(bool(model_k.get('use_external_macro', True)))}",
        f"hsf{int(use_hsf)}",
        f"mDrop{model_k.get('regime_macro_dropout', 'na')}",
        f"pool{model_k.get('pooling_alpha', 'na')}",
        f"ms{state_stem}",
    ]
    if use_hsf:
        parts.extend(
            [
                f"g{model_k.get('d_global_state', 'na')}",
                f"lS{model_k.get('d_local_state', 'na')}",
                f"gMacro{int(bool(model_k.get('global_state_use_macro', True)))}",
                f"gDay{int(bool(model_k.get('global_state_use_day_summary', True)))}",
                f"lMode{model_k.get('local_state_input_mode', 'na')}",
                f"fusion{model_k.get('state_fusion_mode', 'sum_norm')}",
                "rGL"
                f"{int(bool(model_k.get('router_use_global_state', True)))}"
                f"{int(bool(model_k.get('router_use_local_state', True)))}",
                "fGL"
                f"{int(bool(model_k.get('film_use_global_state', True)))}"
                f"{int(bool(model_k.get('film_use_local_state', True)))}",
                "pGL"
                f"{int(bool(model_k.get('pooling_use_global_state', True)))}"
                f"{int(bool(model_k.get('pooling_use_local_state', True)))}",
            ]
        )
    return "_".join(_slug(part) for part in parts)


def _build_experiment_name(model_conf: dict[str, Any], args: argparse.Namespace) -> str:
    model_k = ((model_conf.get("kwargs") or {}).get("model_config") or {})
    trainer_k = ((model_conf.get("kwargs") or {}).get("trainer_config") or {})
    helper = getattr(workflow_helpers, "_build_experiment_name", None)
    if callable(helper):
        exp_name = helper(model_k, trainer_k)
    else:
        exp_name = _local_build_experiment_name(model_k, trainer_k)
    suffix = str(args.experiment_suffix or "").strip()
    if suffix:
        exp_name = f"{exp_name}_{_slug(suffix)}"
    return exp_name


def _resolve_model_config(model: Any, fallback: dict[str, Any]) -> dict[str, Any]:
    helper = getattr(workflow_helpers, "_resolve_model_config", None)
    if callable(helper):
        return helper(model, fallback)
    resolved = copy.deepcopy(fallback)
    live = getattr(model, "model_config", None)
    if isinstance(live, dict):
        resolved.update(copy.deepcopy(live))
    net = getattr(model, "net", None) or getattr(model, "model", None)
    cfg = getattr(net, "config", None)
    if cfg is not None:
        for key in list(resolved.keys()):
            if hasattr(cfg, key):
                resolved[key] = getattr(cfg, key)
    return resolved


def _resolve_trainer_config(model: Any, fallback: dict[str, Any]) -> dict[str, Any]:
    helper = getattr(workflow_helpers, "_resolve_trainer_config", None)
    if callable(helper):
        return helper(model, fallback)
    resolved = copy.deepcopy(fallback)
    live = getattr(model, "trainer_config", None)
    if isinstance(live, dict):
        resolved.update(copy.deepcopy(live))
    for key in (
        "lr",
        "n_epochs",
        "batch_size",
        "eval_batch_size",
        "precision",
        "grad_accum_steps",
        "seed",
        "early_stop",
        "train_stop_key",
        "train_stop_threshold",
        "min_epochs",
        "consecutive_k",
        "num_workers",
        "market_state_path",
        "market_state_shift",
        "market_state_strict",
        "market_state_missing_policy",
        "market_day_summary_path",
        "market_day_summary_shift",
        "market_day_summary_strict",
        "market_day_summary_missing_policy",
        "use_warmup",
        "warmup_ratio",
        "warmup_steps",
        "debug_sanity_check",
        "strict_valid_data_key",
        "fill_nonfinite_feature",
        "enable_local_counterfactual_diag",
        "enable_expert_advantage_diag",
    ):
        if hasattr(model, key):
            resolved[key] = getattr(model, key)
    return resolved


def main() -> None:
    args = parse_args()
    workflow_module = _load_workflow_module(args.workflow_module)
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    _log(">>> [Phase 0] Cloning configs and applying market_state override...")
    data_conf, model_conf, port_conf = _clone_conf_with_overrides(args, workflow_module)

    _log(">>> [Phase 0] Creating dataset...")
    dataset = init_instance_by_config(data_conf)
    _log(">>> [Phase 0] Creating model...")
    model = init_instance_by_config(model_conf)
    port_conf_run = copy.deepcopy(port_conf)

    seg_rng = workflow_helpers._get_segment_range(data_conf, "test")
    if seg_rng is not None:
        seg_start, seg_end = seg_rng
        port_conf_run = workflow_helpers._clip_backtest_window(
            port_conf_run,
            start=seg_start,
            end=seg_end,
            reason="dataset.test segment",
        )

    exp_name = _build_experiment_name(model_conf, args)
    variant_meta = {
        "label": str(args.label),
        "workflow_module": str(args.workflow_module),
        "market_state_path": str(args.market_state_path),
        "market_day_summary_path": ((model_conf.get("kwargs") or {}).get("trainer_config") or {}).get("market_day_summary_path"),
        "router_summary_source": ((model_conf.get("kwargs") or {}).get("model_config") or {}).get("router_summary_source"),
        "pooling_summary_source": ((model_conf.get("kwargs") or {}).get("model_config") or {}).get("pooling_summary_source"),
        "use_hierarchical_state_field": bool(((model_conf.get("kwargs") or {}).get("model_config") or {}).get("use_hierarchical_state_field", False)),
        "d_global_state": ((model_conf.get("kwargs") or {}).get("model_config") or {}).get("d_global_state"),
        "d_local_state": ((model_conf.get("kwargs") or {}).get("model_config") or {}).get("d_local_state"),
        "global_state_use_macro": bool(((model_conf.get("kwargs") or {}).get("model_config") or {}).get("global_state_use_macro", True)),
        "global_state_use_day_summary": bool(((model_conf.get("kwargs") or {}).get("model_config") or {}).get("global_state_use_day_summary", True)),
        "local_state_input_mode": ((model_conf.get("kwargs") or {}).get("model_config") or {}).get("local_state_input_mode"),
        "state_fusion_mode": ((model_conf.get("kwargs") or {}).get("model_config") or {}).get("state_fusion_mode", "sum_norm"),
        "local_router_scale_init": ((model_conf.get("kwargs") or {}).get("model_config") or {}).get("local_router_scale_init"),
        "local_pooling_scale_init": ((model_conf.get("kwargs") or {}).get("model_config") or {}).get("local_pooling_scale_init"),
        "local_pooling_alpha_scale_init": ((model_conf.get("kwargs") or {}).get("model_config") or {}).get("local_pooling_alpha_scale_init"),
        "local_film_scale_init": ((model_conf.get("kwargs") or {}).get("model_config") or {}).get("local_film_scale_init"),
        "router_use_global_state": bool(((model_conf.get("kwargs") or {}).get("model_config") or {}).get("router_use_global_state", True)),
        "router_use_local_state": bool(((model_conf.get("kwargs") or {}).get("model_config") or {}).get("router_use_local_state", True)),
        "film_use_global_state": bool(((model_conf.get("kwargs") or {}).get("model_config") or {}).get("film_use_global_state", True)),
        "film_use_local_state": bool(((model_conf.get("kwargs") or {}).get("model_config") or {}).get("film_use_local_state", True)),
        "pooling_use_global_state": bool(((model_conf.get("kwargs") or {}).get("model_config") or {}).get("pooling_use_global_state", True)),
        "pooling_use_local_state": bool(((model_conf.get("kwargs") or {}).get("model_config") or {}).get("pooling_use_local_state", True)),
        "enable_local_counterfactual_diag": bool(((model_conf.get("kwargs") or {}).get("trainer_config") or {}).get("enable_local_counterfactual_diag", False)),
        "enable_expert_advantage_diag": bool(((model_conf.get("kwargs") or {}).get("trainer_config") or {}).get("enable_expert_advantage_diag", False)),
        "seed": ((model_conf.get("kwargs") or {}).get("trainer_config") or {}).get("seed"),
        "experiment_name": exp_name,
    }

    resolved_model_conf = None
    with R.start(experiment_name=exp_name):
        rec = R.get_recorder()
        _log(f">>> [Variant] label={variant_meta['label']}")
        _log(f">>> [Variant] experiment_name={exp_name}")
        _log(f">>> [Variant] recorder_id={rec.id}")
        _log(f">>> [Variant] market_state_path={variant_meta['market_state_path']}")
        _log(f">>> [Variant] market_day_summary_path={variant_meta['market_day_summary_path']}")
        _log(f">>> [Variant] router_summary_source={variant_meta['router_summary_source']}")
        _log(f">>> [Variant] pooling_summary_source={variant_meta['pooling_summary_source']}")
        _log(f">>> [Variant] use_hierarchical_state_field={variant_meta['use_hierarchical_state_field']}")
        _log(f">>> [Variant] global_state_use_macro={variant_meta['global_state_use_macro']}")
        _log(f">>> [Variant] global_state_use_day_summary={variant_meta['global_state_use_day_summary']}")
        _log(f">>> [Variant] local_state_input_mode={variant_meta['local_state_input_mode']}")
        _log(f">>> [Variant] state_fusion_mode={variant_meta['state_fusion_mode']}")

        R.log_params(**flatten_dict(model_conf))
        R.set_tags(
            variant_label=str(args.label),
            market_state_variant=str(Path(args.market_state_path).name),
            router_summary_source=str(variant_meta["router_summary_source"]),
            pooling_summary_source=str(variant_meta["pooling_summary_source"]),
            use_hierarchical_state_field=str(int(bool(variant_meta["use_hierarchical_state_field"]))),
            state_fusion_mode=str(variant_meta["state_fusion_mode"]),
        )
        R.save_objects(
            variant_meta=copy.deepcopy(variant_meta),
            run_conf={
                "data_conf": copy.deepcopy(data_conf),
                "model_conf": copy.deepcopy(model_conf),
                "port_conf": copy.deepcopy(port_conf_run),
            },
        )
        model.log_config_summary(stage="planned")

        _log(">>> [Phase 1] Training Model...")
        model.fit(dataset)

        resolved_model_k = _resolve_model_config(
            model, (model_conf.get("kwargs") or {}).get("model_config", {})
        )
        resolved_trainer_k = _resolve_trainer_config(
            model, (model_conf.get("kwargs") or {}).get("trainer_config", {})
        )
        resolved_model_conf = copy.deepcopy(model_conf)
        resolved_model_conf["kwargs"]["model_config"] = copy.deepcopy(resolved_model_k)
        resolved_model_conf["kwargs"]["trainer_config"] = copy.deepcopy(resolved_trainer_k)
        variant_meta["market_day_summary_path"] = resolved_trainer_k.get("market_day_summary_path")
        variant_meta["router_summary_source"] = resolved_model_k.get("router_summary_source")
        variant_meta["pooling_summary_source"] = resolved_model_k.get("pooling_summary_source")
        variant_meta["use_hierarchical_state_field"] = bool(resolved_model_k.get("use_hierarchical_state_field", False))
        variant_meta["d_global_state"] = resolved_model_k.get("d_global_state")
        variant_meta["d_local_state"] = resolved_model_k.get("d_local_state")
        variant_meta["global_state_use_macro"] = bool(resolved_model_k.get("global_state_use_macro", True))
        variant_meta["global_state_use_day_summary"] = bool(resolved_model_k.get("global_state_use_day_summary", True))
        variant_meta["local_state_input_mode"] = resolved_model_k.get("local_state_input_mode")
        variant_meta["state_fusion_mode"] = resolved_model_k.get("state_fusion_mode", "sum_norm")
        variant_meta["local_router_scale_init"] = resolved_model_k.get("local_router_scale_init")
        variant_meta["local_pooling_scale_init"] = resolved_model_k.get("local_pooling_scale_init")
        variant_meta["local_pooling_alpha_scale_init"] = resolved_model_k.get("local_pooling_alpha_scale_init")
        variant_meta["local_film_scale_init"] = resolved_model_k.get("local_film_scale_init")
        variant_meta["router_use_global_state"] = bool(resolved_model_k.get("router_use_global_state", True))
        variant_meta["router_use_local_state"] = bool(resolved_model_k.get("router_use_local_state", True))
        variant_meta["film_use_global_state"] = bool(resolved_model_k.get("film_use_global_state", True))
        variant_meta["film_use_local_state"] = bool(resolved_model_k.get("film_use_local_state", True))
        variant_meta["pooling_use_global_state"] = bool(resolved_model_k.get("pooling_use_global_state", True))
        variant_meta["pooling_use_local_state"] = bool(resolved_model_k.get("pooling_use_local_state", True))
        variant_meta["enable_local_counterfactual_diag"] = bool(
            resolved_trainer_k.get("enable_local_counterfactual_diag", False)
        )
        variant_meta["enable_expert_advantage_diag"] = bool(
            resolved_trainer_k.get("enable_expert_advantage_diag", False)
        )

        try:
            full_desc = (
                "[Model Full]\n" + workflow_helpers._pformat(resolved_model_k) + "\n\n"
                "[Trainer Full]\n" + workflow_helpers._pformat(resolved_trainer_k)
            )
            R.set_tags(**{"mlflow.note.content": full_desc})
        except Exception:
            pass

        R.save_objects(
            run_conf_resolved={
                "data_conf": copy.deepcopy(data_conf),
                "model_conf": copy.deepcopy(resolved_model_conf),
                "port_conf": copy.deepcopy(port_conf_run),
            }
        )
        R.save_objects(model=model)

        if not args.skip_visuals:
            _log(">>> [Phase 1.1] Export Spatio-Temporal Visuals...")
            model.export_visuals(
                dataset,
                segment="test",
                max_attn_days=4,
                attn_layer=-1,
                target_dates=None,
                prefix="st_disentangle",
                factor_use_last_time=True,
            )
        else:
            _log(">>> [Phase 1.1] Skipping visual export.")

        _log(">>> [Phase 2] Signal Analysis...")
        SignalRecord(model, dataset, rec).generate()
        SigAnaRecord(rec).generate()

        try:
            pred_df = rec.load_object("pred.pkl")
            pred_rng = workflow_helpers._infer_pred_date_range(pred_df)
            if pred_rng is not None:
                p_start, p_end = pred_rng
                port_conf_run = workflow_helpers._clip_backtest_window(
                    port_conf_run,
                    start=p_start,
                    end=p_end,
                    reason="pred.pkl datetime range",
                )
                conf_for_save = resolved_model_conf or model_conf
                R.save_objects(
                    run_conf_resolved={
                        "data_conf": copy.deepcopy(data_conf),
                        "model_conf": copy.deepcopy(conf_for_save),
                        "port_conf": copy.deepcopy(port_conf_run),
                    }
                )
        except Exception as e:
            _log(f">>> [WARN] Failed to align backtest window to pred.pkl: {e}")

        _log(">>> [Phase 3] Backtesting...")
        PortAnaRecord(rec, port_conf_run, "day").generate()

        _log(">>> [Phase 3.1] Export Qlib Official Graphs...")
        ensure_qlib_official_graphs(rec, dataset=dataset, segment="test", prefix="qlib", strict=False)

        workflow_helpers.print_metrics_summary(rec)

        _log(">>> [Phase 4] Generate Paper-level Report...")
        workflow_helpers.generate_paper_report(rec, model_name="RST-MoE", dataset=dataset, segment="test")

        final_meta = dict(variant_meta)
        final_meta["recorder_id"] = rec.id
        final_meta["artifact_uri"] = rec.get_local_dir()
        R.save_objects(variant_meta=final_meta)
        _log(">>> [Variant] final_meta")
        _log(json.dumps(final_meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
