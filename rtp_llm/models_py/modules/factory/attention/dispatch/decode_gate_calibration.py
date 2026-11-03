"""Offline calibration report generator for the signal-conditioned decode gate.

This command never updates runtime thresholds. A reviewer applies the selected
profile through normal code review after validating the generated report.

The records file is a list, or ``{"records": [...]}``, where every record has:
candidate/reference ``[batch, heads*dim]``, contribution ``[batch, heads]``,
fp8, kind and cohort. Run with ``python -m
rtp_llm.models_py.modules.factory.attention.dispatch.decode_gate_calibration``.
"""

import argparse
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import torch

from rtp_llm.models_py.modules.factory.attention.dispatch.precision_metrics import (
    DecodeGateConfig,
    evaluate_precision,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_records(path: Path):
    value = torch.load(path, map_location="cpu", weights_only=True)
    records = value.get("records") if isinstance(value, dict) else value
    if not isinstance(records, list) or not records:
        raise ValueError("calibration input must contain a non-empty records list")
    required = {"candidate", "reference", "contribution", "fp8", "kind", "cohort"}
    for index, record in enumerate(records):
        missing = required - set(record)
        if missing:
            raise ValueError(f"record {index} missing fields: {sorted(missing)}")
    return records


def _grid(value):
    keys = (
        "signal_split",
        "budget_coefficient",
        "gain_limit",
        "cosine_limit",
        "nrmse_limit",
        "ulp_limit",
    )
    missing = set(keys) - set(value)
    if missing:
        raise ValueError(f"calibration grid missing keys: {sorted(missing)}")
    values = [value[key] for key in keys]
    for combination in itertools.product(*values):
        yield DecodeGateConfig(
            roundoff_coefficient=float(value.get("roundoff_coefficient", 2**-9)),
            **dict(zip(keys, combination)),
        )


def _score(records, config, fp8):
    cohorts = defaultdict(list)
    kinds = {}
    for record in records:
        if bool(record["fp8"]) != fp8:
            continue
        verdict = evaluate_precision(
            record["candidate"],
            record["reference"],
            record["contribution"],
            fp8,
            config,
        )
        cohort = str(record["cohort"])
        cohorts[cohort].append(bool(verdict["overall_pass"]))
        kinds[cohort] = str(record["kind"])
    normal = [all(v) for c, v in cohorts.items() if kinds[c] == "normal"]
    faults = [not all(v) for c, v in cohorts.items() if kinds[c] != "normal"]
    normal_rate = sum(normal) / max(len(normal), 1)
    by_kind = defaultdict(list)
    for cohort, passed in cohorts.items():
        if kinds[cohort] != "normal":
            by_kind[kinds[cohort]].append(not all(passed))
    detection = {
        kind: sum(values) / len(values) for kind, values in sorted(by_kind.items())
    }
    minimum_detection = min(detection.values(), default=0.0)
    return {
        "normal_pass_rate": normal_rate,
        "fault_detection_rate": sum(faults) / max(len(faults), 1),
        "minimum_fault_detection_rate": minimum_detection,
        "fault_detection_by_kind": detection,
        "normal_cohorts": len(normal),
        "fault_cohorts": len(faults),
    }


def calibrate(records, grid, min_normal_pass, min_fault_detection):
    selected = {}
    all_results = {"bf16": [], "fp8": []}
    for fp8, key in ((False, "bf16"), (True, "fp8")):
        candidates = []
        for config in grid:
            score = _score(records, config, fp8)
            valid = (
                score["normal_pass_rate"] >= min_normal_pass
                and score["minimum_fault_detection_rate"] >= min_fault_detection
            )
            candidates.append(
                {
                    "profile": key,
                    "valid": valid,
                    "config": config.to_dict(),
                    **score,
                }
            )
        candidates.sort(
            key=lambda row: (
                not row["valid"],
                -row["minimum_fault_detection_rate"],
                -row["fault_detection_rate"],
                -row["normal_pass_rate"],
                row["config"]["budget_coefficient"],
            )
        )
        all_results[key] = candidates
        if not candidates or not candidates[0]["valid"]:
            raise RuntimeError(
                f"no calibration candidate satisfies constraints for {key}"
            )
        selected[key] = candidates[0]
    return selected, all_results


def _report(selected, input_path, grid_path):
    lines = [
        "# Decode Gate Calibration Report",
        "",
        f"- input: `{input_path}` sha256=`{_digest(input_path)}`",
        f"- grid: `{grid_path}` sha256=`{_digest(grid_path)}`",
        f"- script sha256: `{_digest(Path(__file__))}`",
        f"- torch: `{torch.__version__}`",
        "",
        "| profile | normal pass | minimum fault detection | config |",
        "| --- | ---: | ---: | --- |",
    ]
    for profile, result in selected.items():
        lines.append(
            f"| {profile} | {result['normal_pass_rate']:.4f} | "
            f"{result['minimum_fault_detection_rate']:.4f} | "
            f"`{json.dumps(result['config'], sort_keys=True)}` |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--grid", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-normal-pass", type=float, default=1.0)
    parser.add_argument("--min-fault-detection", type=float, default=0.95)
    args = parser.parse_args()

    records = _load_records(args.records)
    constraints = json.loads(args.grid.read_text())
    selected, all_results = calibrate(
        records,
        list(_grid(constraints)),
        args.min_normal_pass,
        args.min_fault_detection,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "decode_gate_profiles.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selected": selected,
                "source": {
                    "records_sha256": _digest(args.records),
                    "grid_sha256": _digest(args.grid),
                    "script_sha256": _digest(Path(__file__)),
                },
            },
            indent=2,
        )
        + "\n"
    )
    (args.output_dir / "calibration-results.json").write_text(
        json.dumps(all_results, indent=2) + "\n"
    )
    (args.output_dir / "calibration-report.md").write_text(
        _report(selected, args.records, args.grid)
    )


if __name__ == "__main__":
    main()
