"""
Shared run-ledger + project-state helpers used by every /quant-* skill.

Two stores:

1. ``experiments_ledger.jsonl`` (repo root) — append-only canonical log,
   one JSON object per row. Every skill that completes a meaningful step
   appends one row via :func:`append_ledger`.

2. ``memory/project_state.md`` (Claude session memory dir) — short,
   human-readable snapshot of the current best config and active failure
   modes, updated only on promotion events via :func:`update_project_state`.

Stdlib only, no torch / pandas import — safe to import from anywhere.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "experiments_ledger.jsonl"

# Claude project memory dir is OS-specific; resolve via env first, then fall back.
_DEFAULT_MEMORY = Path(
    os.environ.get(
        "QIB_MEMORY_DIR",
        Path.home()
        / ".claude"
        / "projects"
        / "C--Users-60585-PycharmProjects-qibMacV2"
        / "memory",
    )
)
PROJECT_STATE_PATH = _DEFAULT_MEMORY / "project_state.md"

VALID_STAGES = {
    "hypothesis",
    "leakage",
    "proxy",
    "ablation",
    "regime",
    "walkfwd",
    "cost",
    "risk",
    "paper",
    "recent",
}
VALID_VERDICTS = {"promote", "reject", "inconclusive", "pass", "fail", "info"}


def _now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def config_hash(config: Any) -> str:
    """SHA-1 of a JSON-canonicalized config; used to dedupe near-duplicate hypotheses."""
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def new_card_id(prefix: str = "h") -> str:
    """Deterministic-ish card id: ``h-YYYYMMDD-NNN`` where NNN is the day's seq."""
    today = _dt.date.today().strftime("%Y%m%d")
    seq = 1
    if LEDGER_PATH.exists():
        for row in iter_ledger():
            cid = row.get("card_id", "")
            if cid.startswith(f"{prefix}-{today}-"):
                try:
                    seq = max(seq, int(cid.rsplit("-", 1)[-1]) + 1)
                except ValueError:
                    pass
    return f"{prefix}-{today}-{seq:03d}"


def append_ledger(
    *,
    skill: str,
    stage: str,
    card_id: str,
    hypothesis: str = "",
    metrics: dict | None = None,
    verdict: str = "info",
    config: Any = None,
    wall_clock_s: float | None = None,
    mlflow_run: str | None = None,
    notes: str = "",
    extra: dict | None = None,
) -> dict:
    """Append a single row to the ledger and return it."""
    if stage not in VALID_STAGES:
        raise ValueError(f"stage must be one of {sorted(VALID_STAGES)}; got {stage!r}")
    if verdict not in VALID_VERDICTS:
        raise ValueError(f"verdict must be one of {sorted(VALID_VERDICTS)}; got {verdict!r}")

    row = {
        "ts": _now_iso(),
        "skill": skill,
        "stage": stage,
        "card_id": card_id,
        "hypothesis": hypothesis,
        "config_hash": config_hash(config) if config is not None else None,
        "git_sha": _git_sha(),
        "metrics": metrics or {},
        "verdict": verdict,
        "wall_clock_s": wall_clock_s,
        "mlflow_run": mlflow_run,
        "notes": notes,
    }
    if extra:
        row["extra"] = extra

    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def iter_ledger(path: Path | None = None) -> Iterable[dict]:
    """Yield ledger rows oldest-first; missing file → empty iterator."""
    p = path or LEDGER_PATH
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def find_duplicate_card(*, config: Any, since_days: int = 60) -> dict | None:
    """Return the most recent rejected/promoted row with the same config_hash, if any.

    Used by /quant-hypothesis to refuse near-duplicates of recently-decided cards.
    """
    target = config_hash(config)
    cutoff = _dt.datetime.now().astimezone() - _dt.timedelta(days=since_days)
    latest = None
    for row in iter_ledger():
        if row.get("config_hash") != target:
            continue
        if row.get("verdict") not in {"promote", "reject"}:
            continue
        try:
            ts = _dt.datetime.fromisoformat(row["ts"])
        except (KeyError, ValueError):
            continue
        if ts < cutoff:
            continue
        if latest is None or ts > _dt.datetime.fromisoformat(latest["ts"]):
            latest = row
    return latest


def update_project_state(
    *,
    best_config: dict,
    best_metrics: dict,
    last_promotions: list[dict] | None = None,
    failure_modes: list[str] | None = None,
) -> Path:
    """Overwrite ``memory/project_state.md`` with a compact snapshot.

    Called only on promotion events. Memory is for orientation, not archive —
    keep this file small (< 2 KB).
    """
    PROJECT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# RST-MoE Project State",
        f"_Updated {_now_iso()} • git {_git_sha()[:8]}_",
        "",
        "## Current best",
        "```json",
        json.dumps(best_metrics, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Best config (deltas from defaults only)",
        "```json",
        json.dumps(best_config, indent=2, ensure_ascii=False),
        "```",
        "",
    ]

    if last_promotions:
        lines.append("## Recent promotions / rejections")
        for p in last_promotions[-5:]:
            lines.append(
                f"- `{p.get('card_id', '?')}` {p.get('verdict', '?')} — "
                f"{p.get('hypothesis', '')[:80]}"
            )
        lines.append("")

    if failure_modes:
        lines.append("## Active failure modes")
        for fm in failure_modes:
            lines.append(f"- {fm}")
        lines.append("")

    PROJECT_STATE_PATH.write_text("\n".join(lines), encoding="utf-8")
    return PROJECT_STATE_PATH


def load_project_state() -> str:
    """Return the raw text of project_state.md, or '' if missing."""
    if not PROJECT_STATE_PATH.exists():
        return ""
    return PROJECT_STATE_PATH.read_text(encoding="utf-8")
