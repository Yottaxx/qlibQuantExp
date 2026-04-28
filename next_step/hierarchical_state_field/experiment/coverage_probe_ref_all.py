from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Qlib ref-all active membership raw-row coverage.")
    parser.add_argument("--date", action="append", dest="dates", default=[])
    parser.add_argument("--block-days", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--field-mode", choices=["raw3", "alpha5", "alpha_all"], default="raw3")
    parser.add_argument("--use-precompute-mask", action="store_true")
    parser.add_argument("--log", type=str, default="")
    return parser.parse_args()


def main() -> int:
    import qlib
    from qlib.data import D
    from qlib.contrib.data.handler import Alpha158

    import work_flow
    from scripts.precompute_market_state import _active_membership_mask_and_age

    args = parse_args()
    dates = args.dates or ["2008-01-02", "2008-12-31", "2011-04-25", "2015-07-08", "2020-03-31"]
    log_fh = None
    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = log_path.open("w", encoding="utf-8", buffering=1)

    def emit(*parts: object) -> None:
        print(*parts, flush=True)
        if log_fh is not None:
            print(*parts, file=log_fh, flush=True)

    try:
        qlib.init(provider_uri=work_flow.provider_uri, region=work_flow.REG_CN)
        spans = D.list_instruments(
            D.instruments("all"),
            start_time="2008-01-01",
            end_time="2022-12-31",
            freq="day",
            as_list=False,
        )
        cal = pd.DatetimeIndex(
            pd.to_datetime(D.calendar(start_time="2008-01-01", end_time="2022-12-31", freq="day"))
        ).normalize()
        calendar_pos = {pd.Timestamp(ts).normalize(): i for i, ts in enumerate(cal)}
        emit("n_span_instruments", len(spans))

        def active_for(day: str) -> list[str]:
            dt = pd.Timestamp(day).normalize()
            out: list[str] = []
            for inst, inst_spans in spans.items():
                for span in inst_spans or []:
                    if pd.Timestamp(span[0]).normalize() <= dt <= pd.Timestamp(span[1]).normalize():
                        out.append(str(inst))
                        break
            return out

        alpha_fields, _ = Alpha158.get_feature_config(None)
        alpha_fields = [str(x) for x in alpha_fields]
        if args.field_mode == "raw3":
            fields = ["$close", "$volume", "$amount"]
        elif args.field_mode == "alpha5":
            fields = alpha_fields[:5]
        else:
            fields = alpha_fields
        emit("field_mode", args.field_mode, "n_fields", len(fields), "field_sample", fields[:5])

        for day in dates:
            if int(args.block_days or 1) > 1:
                block_days = [
                    pd.Timestamp(dt).normalize()
                    for dt in D.calendar(start_time=day, end_time="2022-12-31", freq="day")[: int(args.block_days)]
                ]
                if not block_days:
                    continue
                active_by_day = {dtn: active_for(str(dtn.date())) for dtn in block_days}
                block_members = sorted({inst for members in active_by_day.values() for inst in members})
                expected_rows = sum(len(members) for members in active_by_day.values())
                got_pairs: set[tuple[pd.Timestamp, str]] = set()
                mask_pairs = 0
                rows = 0
                finite_rows = 0
                t0 = time.time()
                chunk = max(1, int(args.chunk_size or 800))
                for i in range(0, len(block_members), chunk):
                    sub = block_members[i : i + chunk]
                    df = D.features(
                        sub,
                        fields,
                        start_time=str(block_days[0].date()),
                        end_time=str(block_days[-1].date()),
                        freq="day",
                    )
                    if df is None or df.empty:
                        continue
                    rows += int(len(df))
                    finite_rows += int(df.notna().all(axis=1).sum())
                    names = list(df.index.names or [])
                    dt_level = names.index("datetime") if "datetime" in names else 1
                    inst_level = names.index("instrument") if "instrument" in names else 0
                    dts = pd.to_datetime(df.index.get_level_values(dt_level)).normalize()
                    insts = df.index.get_level_values(inst_level).astype(str)
                    for dtn, inst in zip(dts, insts):
                        if inst in active_by_day.get(pd.Timestamp(dtn).normalize(), []):
                            got_pairs.add((pd.Timestamp(dtn).normalize(), str(inst)))
                    if args.use_precompute_mask:
                        normalized_index = pd.MultiIndex.from_arrays(
                            [dts, insts],
                            names=["datetime", "instrument"],
                        )
                        dummy_env = type(
                            "DummyEnv",
                            (),
                            {
                                "reference_membership_spans": spans,
                                "calendar_pos": calendar_pos,
                            },
                        )()
                        mask, _ = _active_membership_mask_and_age(dummy_env, normalized_index)
                        mask_pairs += int(mask.sum())
                coverage = float(len(got_pairs) / expected_rows) if expected_rows else float("nan")
                emit(
                    "BLOCK_RESULT",
                    day,
                    "block_start",
                    str(block_days[0].date()),
                    "block_end",
                    str(block_days[-1].date()),
                    "block_days",
                    len(block_days),
                    "block_members",
                    len(block_members),
                    "expected_active_rows",
                    expected_rows,
                    "raw_rows",
                    rows,
                    "finite_rows",
                    finite_rows,
                    "active_pairs",
                    len(got_pairs),
                    "precompute_mask_pairs",
                    mask_pairs if args.use_precompute_mask else "na",
                    "coverage",
                    f"{coverage:.4f}",
                    "seconds",
                    f"{time.time() - t0:.2f}",
                )
                continue

            active = active_for(day)
            emit("DATE", day, "active", len(active))
            got: set[str] = set()
            rows = 0
            finite_rows = 0
            t0 = time.time()
            chunk = max(1, int(args.chunk_size or 800))
            for i in range(0, len(active), chunk):
                sub = active[i : i + chunk]
                df = D.features(sub, fields, start_time=day, end_time=day, freq="day")
                if df is None or df.empty:
                    continue
                rows += int(len(df))
                finite_rows += int(df.notna().all(axis=1).sum())
                names = list(df.index.names or [])
                inst_level = names.index("instrument") if "instrument" in names else 0
                got.update(str(x) for x in df.index.get_level_values(inst_level))
            coverage = float(len(got) / len(active)) if active else float("nan")
            missing = [inst for inst in active if inst not in got][:20]
            emit(
                "RESULT",
                day,
                "active",
                len(active),
                "raw_rows",
                rows,
                "finite_raw3_rows",
                finite_rows,
                "got_insts",
                len(got),
                "coverage",
                f"{coverage:.4f}",
                "seconds",
                f"{time.time() - t0:.2f}",
                "missing_sample",
                missing,
            )
    finally:
        if log_fh is not None:
            log_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
