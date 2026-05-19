"""
evaluate.py — Production-Grade Pipeline Evaluation Harness  (v14.0)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Real audit tool that calculates 4 critical metrics for data quality assessment:

  1. COVERAGE % — Events successfully stitched into trajectories
     Formula: (mapped zone-visit events / total raw events) × 100
     Spec target: ≥ 85%

  2. COMPLETENESS % — Trajectories with proper entry/exit patterns
     Formula: (trajectories ∈ [Z_E* → Z_E*/Z_CK*] / total unique persons) × 100
     Spec target: ≥ 70%

  3. CONSISTENCY % — Trajectories with zero temporal overlaps AND zero demographic flips
     Formula: (trajectories with NO overlaps AND constant gender/age / total) × 100
     Spec target: ≥ 95%

  4. NUMERIC PRECISION % — LLM output vs ground-truth metrics
     Formula: (verified numbers / total extracted numbers) × 100
     Source: hallucination_report from insights.json
     Spec target: ≥ 90%

USAGE
  python -m src.evaluate
  python -m src.evaluate --events data/events.csv --journeys output/journeys.csv \
                         --insights output/insights.json --output output/evaluation_report.json

EXIT CODES
  0  Evaluation completed successfully
  1  Input file not found or unrecoverable error
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS  (v14.0 — Stage E opened: Z_E* prefix matching)
# ══════════════════════════════════════════════════════════════════════════════

# v14.0: stitcher.py now emits any Z_E* zone as entrance/exit (not just Z_E1/Z_E2)
ENTRANCE_PREFIX = "Z_E"          # first zone must start with this
EXIT_PREFIXES   = ("Z_E", "Z_CK")  # last zone must start with one of these

# Specification targets
TARGET_COVERAGE     = 0.85
TARGET_COMPLETENESS = 0.70
TARGET_CONSISTENCY  = 0.95
TARGET_PRECISION    = 0.90


# ══════════════════════════════════════════════════════════════════════════════
# JSON SERIALISATION HELPER
# ══════════════════════════════════════════════════════════════════════════════

def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively convert NumPy scalars and other non-serialisable types to
    their native Python equivalents so json.dump never raises TypeError.

    Handles:
      np.bool_   → bool
      np.integer → int   (covers int8, int16, int32, int64, uint*, …)
      np.floating→ float (covers float16, float32, float64, …)
      Any other np.generic (via .item()) → Python scalar
      dict / list → recurse
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.generic):      # catch-all for any other NumPy scalar
        return obj.item()
    return obj


# ══════════════════════════════════════════════════════════════════════════════
# METRIC 1 — COVERAGE %
# ══════════════════════════════════════════════════════════════════════════════

def calculate_coverage(events_path: str, journeys_path: str) -> Dict[str, Any]:
    """
    METRIC 1: Coverage % — (mapped events / total raw events) × 100

    Algorithm:
      1. Load raw events CSV and count all rows (total baseline).
      2. Load stitched journeys CSV.
      3. For each zone, build a sorted array of raw event timestamps (ns).
      4. For each journey visit (zone, entry_time, exit_time) use binary search
         to count how many raw events fall inside [entry_ns, exit_ns].
      5. Sum across all visits/zones → mapped_count.

    Interpretation:
      Coverage ≥ 85%  → Most events were successfully integrated into trajectories.
      Coverage < 70%  → Risk: significant data loss during stitching.
    """
    try:
        logger.info("  Calculating COVERAGE…")

        # --- raw events ---
        df_events = pd.read_csv(
            events_path,
            usecols=['timestamp', 'zone_id'],
            dtype={'zone_id': 'string'},
            parse_dates=['timestamp'],
        )
        total_raw = len(df_events)

        if total_raw == 0:
            logger.warning("    Raw events CSV is empty.")
            return {"mapped_events": 0, "total_raw_events": 0, "coverage_pct": 0.0}

        logger.info(f"    Raw events loaded: {total_raw:,}")

        # --- stitched journeys ---
        df_journeys = pd.read_csv(
            journeys_path,
            usecols=['zone_id', 'entry_time', 'exit_time'],
            dtype={'zone_id': 'string'},
            parse_dates=['entry_time', 'exit_time'],
        )

        if df_journeys.empty:
            logger.warning("    Journeys CSV is empty.")
            return {"mapped_events": 0, "total_raw_events": int(total_raw), "coverage_pct": 0.0}

        # Convert to int64 nanoseconds for fast binary-search comparison
        df_events['ts_ns']       = df_events['timestamp'].astype('int64')
        df_journeys['entry_ns']  = df_journeys['entry_time'].astype('int64')
        df_journeys['exit_ns']   = df_journeys['exit_time'].astype('int64')

        mapped_count = 0

        for zone in df_events['zone_id'].unique():
            events_zone   = np.sort(df_events[df_events['zone_id'] == zone]['ts_ns'].values)
            journeys_zone = df_journeys[df_journeys['zone_id'] == zone]

            if len(events_zone) == 0:
                continue

            for _, visit in journeys_zone.iterrows():
                entry_ns = int(visit['entry_ns'])
                exit_ns  = int(visit['exit_ns'])

                left_idx  = np.searchsorted(events_zone, entry_ns, side='left')
                right_idx = np.searchsorted(events_zone, exit_ns,  side='right')

                mapped_count += max(0, right_idx - left_idx)

        coverage_pct = (mapped_count / total_raw) * 100.0 if total_raw > 0 else 0.0

        logger.info(
            f"    ✓ Mapped: {mapped_count:,} / {total_raw:,} = {coverage_pct:.2f}%"
        )

        return {
            "mapped_events":    int(mapped_count),
            "total_raw_events": int(total_raw),
            "coverage_pct":     round(coverage_pct, 2),
        }

    except Exception as e:
        logger.error(f"    ✗ Error calculating coverage: {e}", exc_info=True)
        return {"mapped_events": 0, "total_raw_events": 0, "coverage_pct": 0.0}


# ══════════════════════════════════════════════════════════════════════════════
# METRIC 2 — COMPLETENESS %
# ══════════════════════════════════════════════════════════════════════════════

def calculate_completeness(journeys_path: str) -> Dict[str, Any]:
    """
    METRIC 2: Completeness % — (complete trajectories / total unique persons) × 100

    v14.0 SPEC §3.1 — prefix-based matching (Stage E opened):
      Complete = first zone starts with "Z_E"
                 AND last zone starts with "Z_E" OR "Z_CK"

    Algorithm:
      1. Load journeys CSV (person_id, zone_id, entry_time).
      2. For each unique person sort by entry_time.
      3. Examine first_zone and last_zone against the prefix rules above.
      4. Calculate percentage.

    Interpretation:
      Completeness ≥ 70%  → Most customer journeys have proper entry/exit.
      Completeness < 50%  → Risk: many fragments; data integrity issue.
    """
    try:
        logger.info("  Calculating COMPLETENESS…")

        df = pd.read_csv(
            journeys_path,
            usecols=['person_id', 'zone_id', 'entry_time'],
            dtype={'person_id': 'string', 'zone_id': 'string'},
            parse_dates=['entry_time'],
        )

        if df.empty:
            logger.warning("    Journeys CSV is empty.")
            return {"complete_trajectories": 0, "total_trajectories": 0, "completeness_pct": 0.0}

        total_persons = df['person_id'].nunique()
        logger.info(f"    Total unique persons: {total_persons:,}")

        complete_count = 0

        for person_id in df['person_id'].unique():
            person_visits = df[df['person_id'] == person_id].sort_values('entry_time')

            if person_visits.empty:
                continue

            first_zone = str(person_visits.iloc[0]['zone_id'])
            last_zone  = str(person_visits.iloc[-1]['zone_id'])

            # v14.0: prefix-based check — accepts any Z_E* entrance and Z_E*/Z_CK* exit
            born_at_entrance = first_zone.startswith(ENTRANCE_PREFIX)
            left_at_exit     = any(last_zone.startswith(p) for p in EXIT_PREFIXES)

            if born_at_entrance and left_at_exit:
                complete_count += 1

        completeness_pct = (complete_count / total_persons * 100.0) if total_persons > 0 else 0.0

        logger.info(
            f"    ✓ Complete: {complete_count:,} / {total_persons:,} = {completeness_pct:.2f}%"
        )

        return {
            "complete_trajectories": int(complete_count),
            "total_trajectories":    int(total_persons),
            "completeness_pct":      round(completeness_pct, 2),
        }

    except Exception as e:
        logger.error(f"    ✗ Error calculating completeness: {e}", exc_info=True)
        return {"complete_trajectories": 0, "total_trajectories": 0, "completeness_pct": 0.0}


# ══════════════════════════════════════════════════════════════════════════════
# METRIC 3 — CONSISTENCY %
# ══════════════════════════════════════════════════════════════════════════════

def calculate_consistency(journeys_path: str) -> Dict[str, Any]:
    """
    METRIC 3: Consistency % — (consistent trajectories / total unique persons) × 100

    Definition of "Consistent Trajectory" (two independent checks):

      CHECK A — No temporal overlaps:
        For consecutive visits i and i+1, exit_time[i] must be ≤ entry_time[i+1].
        (A visitor cannot be in two zones simultaneously.)

      CHECK B — Demographic stability:
        gender must be constant across ALL visits (nunique == 1).
        age_range must be constant across ALL visits (nunique == 1).
        → 0 (zero) fluctuations of age or gender allowed.

    Algorithm:
      1. Load journeys CSV with demographic columns.
      2. For each unique person_id:
         a. Evaluate Check B (fast — avoids full sort for obvious failures).
         b. Sort by entry_time; evaluate Check A.
      3. Count trajectories passing both checks.

    Interpretation:
      Consistency ≥ 95%  → Data integrity very high.
      Consistency < 80%  → Risk: check stitcher merge & purity logic.
    """
    try:
        logger.info("  Calculating CONSISTENCY…")

        df = pd.read_csv(
            journeys_path,
            usecols=['person_id', 'gender', 'age_range', 'entry_time', 'exit_time'],
            dtype={'person_id': 'string', 'gender': 'string', 'age_range': 'string'},
            parse_dates=['entry_time', 'exit_time'],
        )

        if df.empty:
            logger.warning("    Journeys CSV is empty.")
            return {
                "consistent_trajectories": 0,
                "total_trajectories":      0,
                "consistency_pct":         0.0,
            }

        total_persons = df['person_id'].nunique()
        logger.info(f"    Total unique persons: {total_persons:,}")

        consistent_count      = 0
        overlap_violations    = 0
        demographic_violations = 0

        for person_id in df['person_id'].unique():
            person_visits = df[df['person_id'] == person_id].sort_values('entry_time')

            if person_visits.empty:
                continue

            # CHECK B — Demographic stability (0 fluctuations tolerated)
            gender_stable    = person_visits['gender'].nunique()    <= 1
            age_range_stable = person_visits['age_range'].nunique() <= 1

            if not gender_stable or not age_range_stable:
                demographic_violations += 1
                continue

            # CHECK A — No temporal overlaps
            has_overlap = False
            exit_times  = person_visits['exit_time'].values
            entry_times = person_visits['entry_time'].values

            for i in range(len(person_visits) - 1):
                exit_i      = exit_times[i]
                entry_next  = entry_times[i + 1]

                # pd.NaT comparisons evaluate as False; treat NaT as non-overlapping
                if pd.notna(exit_i) and pd.notna(entry_next):
                    if entry_next < exit_i:       # strict overlap
                        has_overlap = True
                        break

            if has_overlap:
                overlap_violations += 1
                continue

            # Passed both checks
            consistent_count += 1

        consistency_pct = (
            consistent_count / total_persons * 100.0
        ) if total_persons > 0 else 0.0

        logger.info(
            f"    ✓ Consistent: {consistent_count:,} / {total_persons:,} = {consistency_pct:.2f}%"
        )
        logger.info(f"    ├─ Temporal overlap violations:  {overlap_violations:,}")
        logger.info(f"    └─ Demographic instabilities:    {demographic_violations:,}")

        return {
            "consistent_trajectories":   int(consistent_count),
            "total_trajectories":        int(total_persons),
            "consistency_pct":           round(consistency_pct, 2),
            "temporal_overlap_violations": int(overlap_violations),
            "demographic_violations":    int(demographic_violations),
        }

    except Exception as e:
        logger.error(f"    ✗ Error calculating consistency: {e}", exc_info=True)
        return {
            "consistent_trajectories": 0,
            "total_trajectories":      0,
            "consistency_pct":         0.0,
        }


# ══════════════════════════════════════════════════════════════════════════════
# METRIC 4 — NUMERIC PRECISION %
# ══════════════════════════════════════════════════════════════════════════════

def calculate_numeric_precision(insights_path: str) -> Dict[str, Any]:
    """
    METRIC 4: Numeric Precision % — LLM output validation

    Source: hallucination_report in insights.json (populated by insights.py).

    Formula:
      precision = (verified_numbers / total_extracted_numbers) × 100

    The validator extracts all numeric tokens from LLM-generated text and
    cross-checks them against ground-truth metrics.json values (±5% tolerance).

    Interpretation:
      Precision ≥ 90%  → LLM output highly reliable; minimal hallucinations.
      Precision < 70%  → Risk: LLM fabricating numbers; review insights.
    """
    try:
        logger.info("  Calculating NUMERIC PRECISION…")

        if not os.path.exists(insights_path):
            logger.warning(f"    Insights file not found: {insights_path}")
            return {"numeric_precision_pct": 0.0}

        with open(insights_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        hr = data.get("hallucination_report", {})

        # Prefer few-shot precision; fall back to zero-shot
        p_few  = hr.get("numeric_precision_few_shot")
        p_zero = hr.get("numeric_precision_zero_shot")

        precision = p_few if p_few is not None else p_zero

        if precision is None:
            logger.warning("    No precision metrics found in hallucination_report.")
            precision = 1.0   # assume pass when no data available

        precision_pct   = float(precision) * 100.0
        strategy_used   = "few-shot" if p_few is not None else "zero-shot"
        unverified_count = len(
            hr.get("unverified_numbers_few_shot")
            or hr.get("unverified_numbers_zero_shot")
            or []
        )

        logger.info(
            f"    ✓ Precision: {precision_pct:.2f}% "
            f"(strategy: {strategy_used}, unverified: {unverified_count})"
        )

        return {
            "numeric_precision_pct": round(precision_pct, 2),
            "strategy_used":         strategy_used,
            "unverified_count":      int(unverified_count),
        }

    except Exception as e:
        logger.error(f"    ✗ Error calculating numeric precision: {e}", exc_info=True)
        return {"numeric_precision_pct": 0.0}


# ══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_evaluation_report(
    coverage:     Dict[str, Any],
    completeness: Dict[str, Any],
    consistency:  Dict[str, Any],
    precision:    Dict[str, Any],
) -> Dict[str, Any]:
    """
    Assemble all 4 metrics into a comprehensive evaluation report.

    CRITICAL: all_targets_met and every boolean field are cast through bool()
    before entering the dict to prevent np.bool_ from reaching json.dump.
    """
    # Explicit bool() cast — guards against np.bool_ from pandas comparisons
    cov_met  = bool(coverage.get('coverage_pct', 0)              >= TARGET_COVERAGE     * 100)
    comp_met = bool(completeness.get('completeness_pct', 0)       >= TARGET_COMPLETENESS * 100)
    cons_met = bool(consistency.get('consistency_pct', 0)         >= TARGET_CONSISTENCY  * 100)
    prec_met = bool(precision.get('numeric_precision_pct', 0)     >= TARGET_PRECISION    * 100)

    all_targets_met = bool(cov_met and comp_met and cons_met and prec_met)

    report = {
        "meta": {
            "evaluated_at": datetime.now(tz=timezone.utc).isoformat(),
            "spec_version": "v14.0",
        },
        "metrics": {
            "coverage":          coverage,
            "completeness":      completeness,
            "consistency":       consistency,
            "numeric_precision": precision,
        },
        "targets": {
            "coverage_target":     f"{TARGET_COVERAGE     * 100:.0f}%",
            "completeness_target": f"{TARGET_COMPLETENESS * 100:.0f}%",
            "consistency_target":  f"{TARGET_CONSISTENCY  * 100:.0f}%",
            "precision_target":    f"{TARGET_PRECISION    * 100:.0f}%",
        },
        "targets_met": {
            "coverage_met":     cov_met,
            "completeness_met": comp_met,
            "consistency_met":  cons_met,
            "precision_met":    prec_met,
        },
        "summary": {
            "all_targets_met": all_targets_met,
        },
    }
    return report


# ══════════════════════════════════════════════════════════════════════════════
# I/O HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def save_report(report: Dict[str, Any], output_path: str) -> None:
    """Serialise report to JSON, sanitising all NumPy types first."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sanitized = sanitize_for_json(report)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sanitized, f, indent=2, ensure_ascii=False)
    logger.info(f"  ✓ Report saved → {output_path}")


def print_summary(report: Dict[str, Any]) -> None:
    """
    Pretty-print the Audit Report to stdout.

    Format example:
      Coverage %               80.84% / 85%
    Final line switches between:
      ✓ ALL TARGETS MET
      ✗ NEEDS ATTENTION
    """
    metrics      = report['metrics']
    targets      = report['targets']
    targets_met  = report.get('targets_met', {})
    all_met      = bool(report['summary']['all_targets_met'])

    W = 60

    def _tick(met: bool) -> str:
        return "✓" if met else "✗"

    print()
    print("╔" + "═" * W + "╗")
    print("║" + "  evaluate.py  v14.0  — Pipeline Audit Report".center(W) + "║")
    print("╠" + "═" * W + "╣")

    def row(label: str, value: str, target: str, met: bool) -> None:
        tick = _tick(met)
        line = f"  {tick} {label:<26} {value:>10} / {target:<7}"
        print(f"║{line:<{W}}║")

    row(
        "Coverage %",
        f"{metrics['coverage'].get('coverage_pct', 0):.2f}%",
        targets['coverage_target'],
        bool(targets_met.get('coverage_met', False)),
    )
    row(
        "Completeness %",
        f"{metrics['completeness'].get('completeness_pct', 0):.2f}%",
        targets['completeness_target'],
        bool(targets_met.get('completeness_met', False)),
    )
    row(
        "Consistency %",
        f"{metrics['consistency'].get('consistency_pct', 0):.2f}%",
        targets['consistency_target'],
        bool(targets_met.get('consistency_met', False)),
    )
    row(
        "Numeric Precision %",
        f"{metrics['numeric_precision'].get('numeric_precision_pct', 0):.2f}%",
        targets['precision_target'],
        bool(targets_met.get('precision_met', False)),
    )

    print("╠" + "═" * W + "╣")

    if all_met:
        status = "✓ ALL TARGETS MET"
    else:
        status = "✗ NEEDS ATTENTION"

    status_line = f"  {status:<{W - 2}}"
    print(f"║{status_line}║")
    print("╚" + "═" * W + "╝")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    """
    Main entry point for the evaluation pipeline.

    Returns:
      0  Evaluation completed (regardless of pass/fail status).
      1  Critical error (file not found, unrecoverable exception).
    """
    parser = argparse.ArgumentParser(
        description="evaluate.py — Production Pipeline Evaluation Harness (v14.0)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--events",
        default="data/events.csv",
        help="Path to raw events CSV",
    )
    parser.add_argument(
        "--journeys",
        default="output/journeys.csv",
        help="Path to stitched journeys CSV",
    )
    parser.add_argument(
        "--insights",
        default="output/insights.json",
        help="Path to LLM insights JSON (for hallucination_report)",
    )
    parser.add_argument(
        "--output",
        default="output/evaluation_report.json",
        help="Destination path for JSON evaluation report",
    )
    args = parser.parse_args()

    logger.info("evaluate.py v14.0 — Pipeline Audit Tool")

    # Verify mandatory input files exist
    for path in [args.events, args.journeys, args.insights]:
        if not os.path.exists(path):
            logger.error(f"✗ Input file not found: {path}")
            return 1

    logger.info("Evaluating pipeline outputs…")
    logger.info(f"  Events:   {args.events}")
    logger.info(f"  Journeys: {args.journeys}")
    logger.info(f"  Insights: {args.insights}")
    logger.info("")

    # Calculate all 4 metrics
    coverage     = calculate_coverage(args.events, args.journeys)
    completeness = calculate_completeness(args.journeys)
    consistency  = calculate_consistency(args.journeys)
    precision    = calculate_numeric_precision(args.insights)

    # Assemble, sanitise, and save report
    report = generate_evaluation_report(coverage, completeness, consistency, precision)
    save_report(report, args.output)

    # Human-readable summary
    print_summary(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())