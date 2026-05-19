"""
analytics.py — Phase 2 Analytics Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Consumes journeys.csv produced by stitcher.py and emits a compact metrics.json
safe for LLM context windows (≤ 8 k tokens).  Purely deterministic Pandas math.
No LLM, no randomness, no external I/O beyond the two CSV/JSON files.

PIPELINE
  1. Traffic / Hour     — unique visitors + total visits, by hour-of-day
  2. Funnel / Conversion — % of unique visitors who reach Z_CK
  3. Dwell Time         — mean + P25/P50/P90 per zone; top non-entrance zone
  4. Anomaly Detection  — IQR + 3-sigma on hourly traffic & per-zone dwell

USAGE
  python -m src.analytics
  python -m src.analytics --input output/journeys.csv --output output/metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

CHECKOUT_ZONE   = "Z_CK"
ENTRANCE_PREFIX = "Z_E"

# Anomaly detection — SPEC §4.2: 6-day baseline → flag Day 7 deviations > 2σ
ANOMALY_BASELINE_DAYS = 6       # number of days used to build the baseline
ANOMALY_SIGMA         = 2.0     # deviation threshold (spec-mandated)
MIN_ZONE_OBS          = 10      # minimum zone observations for dwell anomalies


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _native(obj: Any) -> Any:
    """Recursively convert numpy / pandas scalars to native Python types."""
    if isinstance(obj, dict):
        return {k: _native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def _pct(numerator: int, denominator: int, decimals: int = 2) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator * 100, decimals)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD & VALIDATE
# ══════════════════════════════════════════════════════════════════════════════

def load_journeys(path: str) -> pd.DataFrame:
    required = {
        "person_id", "zone_id", "entry_time", "exit_time",
        "dwell_s", "gender", "hour_of_day",
    }
    df = pd.read_csv(path, parse_dates=["entry_time", "exit_time"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"journeys.csv is missing columns: {missing}")

    df["dwell_s"] = pd.to_numeric(df["dwell_s"], errors="coerce")
    df["hour_of_day"] = pd.to_numeric(df["hour_of_day"], errors="coerce")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# METRIC 1 — TRAFFIC / HOUR
# ══════════════════════════════════════════════════════════════════════════════

def compute_traffic(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Global footfall split into two explicitly documented metrics:

    • unique_visitors (per hour): person_id.nunique() — each person counted
      once per hour regardless of how many zone visits they made in that hour.
      Denominator = distinct person_ids in the hour.

    • total_visits (per hour): row count — every zone-visit row for the hour.
      A single person visiting 3 zones contributes 3 to total_visits.
      Denominator = zone-visit events (rows) in the hour.

    These two metrics are intentionally different and labelled accordingly in
    the JSON output.  top_zone_visits (in the dwell section) uses the same
    row-count denominator as total_visits.
    """
    hourly = (
        df.groupby("hour_of_day")
        .agg(
            unique_visitors=("person_id", "nunique"),
            total_visits=("person_id", "count"),
        )
        .reset_index()
        .sort_values("hour_of_day")
    )

    peak_row = hourly.loc[hourly["unique_visitors"].idxmax()]
    quiet_row = hourly.loc[hourly["unique_visitors"].idxmin()]

    top10 = (
        hourly.sort_values("unique_visitors", ascending=False)
        .head(10)
        .to_dict(orient="records")
    )

    return {
        "total_unique_visitors": int(df["person_id"].nunique()),
        "total_visits":          int(len(df)),
        # DENOMINATOR NOTE: total_visits counts every zone-visit row (a person
        # in 3 zones = 3 visits).  unique_visitors counts distinct person_ids.
        # top_zone_visits in the dwell section uses the same row-count basis.
        "denominator_note":      "total_visits = zone-visit rows; unique_visitors = distinct person_ids",
        "peak_hour":             int(peak_row["hour_of_day"]),
        "peak_unique_visitors":  int(peak_row["unique_visitors"]),
        "quiet_hour":            int(quiet_row["hour_of_day"]),
        "quiet_unique_visitors": int(quiet_row["unique_visitors"]),
        "by_hour_top10":         _native(top10),
    }


# ══════════════════════════════════════════════════════════════════════════════
# METRIC 2 — FUNNEL & CONVERSION
# ══════════════════════════════════════════════════════════════════════════════

def compute_funnel(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Conversion = % of unique person_ids who appear in at least one Z_CK row.
    Also surfaces gender-split conversion for colour.
    """
    all_visitors     = set(df["person_id"].unique())
    checkout_visitors = set(df.loc[df["zone_id"] == CHECKOUT_ZONE, "person_id"].unique())
    n_total          = len(all_visitors)
    n_converted      = len(checkout_visitors)

    # Gender-split conversion
    gender_splits: Dict[str, Any] = {}
    for g, gdf in df.groupby("gender"):
        g_all  = set(gdf["person_id"].unique())
        g_ck   = set(
            gdf.loc[gdf["zone_id"] == CHECKOUT_ZONE, "person_id"].unique()
        )
        gender_splits[str(g)] = {
            "visitors":        len(g_all),
            "converted":       len(g_ck),
            "conversion_rate": _pct(len(g_ck), len(g_all)),
        }

    return {
        "unique_visitors":    n_total,
        "reached_checkout":   n_converted,
        "conversion_rate":    _pct(n_converted, n_total),
        "gender_breakdown":   gender_splits,
    }


# ══════════════════════════════════════════════════════════════════════════════
# METRIC 3 — DWELL TIME
# ══════════════════════════════════════════════════════════════════════════════

def compute_dwell(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Per-zone dwell statistics (mean, P25, P50, P90) and overall summary.
    Top Zone = highest footfall zone excluding entrances (Z_E*) and checkout (Z_CK).
    """
    # Drop rows with invalid dwell or zero dwell (zero-dwell rows inflate/deflate stats)
    valid = df.dropna(subset=["dwell_s"])
    valid = valid[valid["dwell_s"] > 0]

    # Per-zone stats
    zone_stats = (
        valid.groupby("zone_id")["dwell_s"]
        .agg(
            count="count",
            mean="mean",
            p25=lambda x: x.quantile(0.25),
            p50=lambda x: x.quantile(0.50),
            p90=lambda x: x.quantile(0.90),
        )
        .round(2)
        .reset_index()
        .sort_values("count", ascending=False)
    )

    # Top zone: exclude entrances and checkout, rank by footfall (visit count)
    interior = zone_stats[
        ~zone_stats["zone_id"].str.startswith(ENTRANCE_PREFIX)
        & (zone_stats["zone_id"] != CHECKOUT_ZONE)
    ]
    top_zone_row = interior.iloc[0] if len(interior) > 0 else None
    top_zone = str(top_zone_row["zone_id"]) if top_zone_row is not None else None

    # Overall dwell (all zones)
    overall = valid["dwell_s"]

    # Top-10 zones by footfall for JSON (compact)
    top10_zones = _native(zone_stats.head(10).to_dict(orient="records"))

    return {
        "overall": {
            "mean_s": round(float(overall.mean()), 2),
            "p25_s":  round(float(overall.quantile(0.25)), 2),
            "p50_s":  round(float(overall.quantile(0.50)), 2),
            "p90_s":  round(float(overall.quantile(0.90)), 2),
        },
        "top_zone":       top_zone,
        # DENOMINATOR NOTE: top_zone_visits = zone-visit row count (same basis
        # as traffic.total_visits).  Not unique visitors.
        "top_zone_visits": int(top_zone_row["count"]) if top_zone_row is not None else 0,
        "top_zone_denominator": "zone-visit rows (same as traffic.total_visits)",
        "by_zone_top10":  top10_zones,
    }


# ══════════════════════════════════════════════════════════════════════════════
# METRIC 4 — STATISTICAL ANOMALIES
# ══════════════════════════════════════════════════════════════════════════════

def compute_anomalies(df: pd.DataFrame, traffic: Dict, dwell: Dict) -> Dict[str, Any]:
    """
    SPEC §4.2 — Anomaly Detection (6-day baseline vs Day 7).

    Algorithm:
      1. Sort all data chronologically and identify the 7 calendar days present.
         Days 1–6 are the baseline period; Day 7 is the evaluation period.
         If fewer than 7 days of data exist, return an explanatory result.

      2. HOURLY TRAFFIC ANOMALIES (spec-mandated method):
         • For each hour-of-day (0–23), compute mean and std of unique-visitor
           counts across baseline days 1–6 (up to 6 data points per hour).
         • Flag any hour in Day 7 whose unique-visitor count deviates more than
           ANOMALY_SIGMA (2σ) from the baseline mean for that hour.
         • Hours with no baseline observations are skipped (cannot establish a
           baseline for them).

      3. PER-ZONE DWELL ANOMALIES (unchanged method — spec does not override):
         • Compare each zone's mean dwell_s against the store-wide mean±σ.
         • Zones with < MIN_ZONE_OBS observations are skipped.

    Denominators:
      • Unique visitors (person_id.nunique) per hour per day for traffic.
      • dwell_s rows per zone for dwell (same basis as dwell.top_zone_visits).
    """
    anomalies: list = []

    # ── Identify calendar days ─────────────────────────────────────────────────
    if "visit_date" not in df.columns:
        # Derive from entry_time if visit_date column absent
        df = df.copy()
        df["visit_date"] = pd.to_datetime(df["entry_time"]).dt.date

    sorted_days = sorted(df["visit_date"].unique())
    n_days = len(sorted_days)

    baseline_meta: Dict[str, Any] = {
        "method":           "6-day baseline vs Day 7 (SPEC §4.2)",
        "sigma_threshold":  ANOMALY_SIGMA,
        "n_days_available": n_days,
    }

    if n_days < 2:
        baseline_meta["warning"] = (
            f"Only {n_days} day(s) of data available; "
            "need ≥ 2 to establish any baseline. No anomalies flagged."
        )
        return {
            "n_anomalies":    0,
            "baseline_meta":  baseline_meta,
            "anomaly_list":   [],
        }

    # Use the last day as Day 7 (evaluation) and all prior days as baseline.
    day7      = sorted_days[-1]
    baseline_days = sorted_days[:-1]          # days 1…(n-1)

    if len(baseline_days) < ANOMALY_BASELINE_DAYS:
        baseline_meta["warning"] = (
            f"Only {len(baseline_days)} baseline day(s) available "
            f"(spec expects {ANOMALY_BASELINE_DAYS}). "
            "Proceeding with available data; results may be less reliable."
        )

    baseline_meta["baseline_days"] = [str(d) for d in baseline_days]
    baseline_meta["evaluation_day"] = str(day7)

    # ── A: Hourly traffic anomalies (6-day baseline vs Day 7) ─────────────────
    df_base = df[df["visit_date"].isin(baseline_days)]
    df_day7 = df[df["visit_date"] == day7]

    # Baseline: unique visitors per (day, hour)
    base_hourly = (
        df_base.groupby(["visit_date", "hour_of_day"])["person_id"]
        .nunique()
        .reset_index()
        .rename(columns={"person_id": "uv"})
    )

    # Baseline stats per hour (mean and std across days)
    base_stats = (
        base_hourly.groupby("hour_of_day")["uv"]
        .agg(baseline_mean="mean", baseline_std="std")
        .reset_index()
    )
    # std is NaN when only 1 observation; fill with 0 (no spread → any deviation flags)
    base_stats["baseline_std"] = base_stats["baseline_std"].fillna(0.0)

    # Day 7: unique visitors per hour
    day7_hourly = (
        df_day7.groupby("hour_of_day")["person_id"]
        .nunique()
        .reset_index()
        .rename(columns={"person_id": "uv_day7"})
    )

    merged = day7_hourly.merge(base_stats, on="hour_of_day", how="left")

    for _, row in merged.iterrows():
        hour        = int(row["hour_of_day"])
        uv_day7     = int(row["uv_day7"])
        b_mean      = row["baseline_mean"]
        b_std       = row["baseline_std"]

        if pd.isna(b_mean):
            # No baseline observations for this hour — cannot flag.
            continue

        threshold_hi = b_mean + ANOMALY_SIGMA * b_std
        threshold_lo = b_mean - ANOMALY_SIGMA * b_std

        if uv_day7 > threshold_hi:
            anomalies.append({
                "type":           "traffic",
                "subtype":        "high",
                "hour":           hour,
                "day":            str(day7),
                "value":          uv_day7,
                "baseline_mean":  round(float(b_mean), 2),
                "baseline_std":   round(float(b_std), 2),
                "threshold_hi":   round(float(threshold_hi), 2),
                "sigma":          ANOMALY_SIGMA,
                "detail": (
                    f"Day 7 hour {hour:02d}:00 — unusually busy "
                    f"({uv_day7} UV > baseline {b_mean:.1f} + {ANOMALY_SIGMA}σ={threshold_hi:.1f})"
                ),
            })
        elif uv_day7 < threshold_lo:
            anomalies.append({
                "type":           "traffic",
                "subtype":        "low",
                "hour":           hour,
                "day":            str(day7),
                "value":          uv_day7,
                "baseline_mean":  round(float(b_mean), 2),
                "baseline_std":   round(float(b_std), 2),
                "threshold_lo":   round(float(threshold_lo), 2),
                "sigma":          ANOMALY_SIGMA,
                "detail": (
                    f"Day 7 hour {hour:02d}:00 — unusually quiet "
                    f"({uv_day7} UV < baseline {b_mean:.1f} − {ANOMALY_SIGMA}σ={threshold_lo:.1f})"
                ),
            })

    # ── B: Per-zone dwell anomalies (store-wide 3σ, spec does not override) ───
    valid = df.dropna(subset=["dwell_s"])
    valid = valid[valid["dwell_s"] > 0]

    store_mean = valid["dwell_s"].mean()
    store_std  = valid["dwell_s"].std()

    if store_std > 0:
        zone_means = (
            valid.groupby("zone_id")["dwell_s"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "zone_mean", "count": "n"})
        )
        for zone, row in zone_means.iterrows():
            if row["n"] < MIN_ZONE_OBS:
                continue
            z_score = abs(row["zone_mean"] - store_mean) / store_std
            if z_score > 3.0:
                direction = "high" if row["zone_mean"] > store_mean else "low"
                anomalies.append({
                    "type":      "dwell",
                    "subtype":   direction,
                    "zone":      str(zone),
                    "value":     round(float(row["zone_mean"]), 2),
                    "z_score":   round(float(z_score), 2),
                    "n":         int(row["n"]),
                    "detail": (
                        f"Zone {zone} — dwell {direction} outlier "
                        f"(mean={row['zone_mean']:.1f}s, z={z_score:.2f})"
                    ),
                })
    else:
        store_mean = float("nan")
        store_std  = float("nan")

    return {
        "n_anomalies":         len(anomalies),
        "baseline_meta":       baseline_meta,
        "dwell_store_mean_s":  round(float(store_mean), 2) if not pd.isna(store_mean) else None,
        "dwell_store_std_s":   round(float(store_std),  2) if not pd.isna(store_std)  else None,
        "anomaly_list":        anomalies,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(metrics: Dict[str, Any]) -> None:
    W = 60
    funnel    = metrics["funnel"]
    dwell     = metrics["dwell"]
    anomalies = metrics["anomalies"]
    traffic   = metrics["traffic"]

    print()
    print("╔" + "═" * W + "╗")
    print("║" + "  analytics.py  — Phase 2 Summary".center(W) + "║")
    print("╠" + "═" * W + "╣")

    def row(label: str, value: str) -> None:
        inner = f"  {label:<32}{value:>24}  "
        print(f"║{inner}║")

    row("Total Unique Visitors",   f"{traffic['total_unique_visitors']:,}")
    row("Total Visits",            f"{traffic['total_visits']:,}")
    row("Peak Hour",               f"{traffic['peak_hour']:02d}:00  ({traffic['peak_unique_visitors']:,} UV)")
    print("╠" + "═" * W + "╣")
    row("conversion_rate",         f"{funnel['conversion_rate']:.2f}%")
    row("Reached Checkout",        f"{funnel['reached_checkout']:,} / {funnel['unique_visitors']:,}")
    print("╠" + "═" * W + "╣")
    row("top_zone",                str(dwell["top_zone"]))
    row("Top Zone Visits",         f"{dwell['top_zone_visits']:,}")
    row("Overall Median Dwell",    f"{dwell['overall']['p50_s']:.1f} s")
    row("Overall P90 Dwell",       f"{dwell['overall']['p90_s']:.1f} s")
    print("╠" + "═" * W + "╣")
    row("n_anomalias detectadas",  str(anomalies["n_anomalies"]))

    if anomalies["anomaly_list"]:
        print("║" + "  Top anomalies:".ljust(W) + "║")
        for a in anomalies["anomaly_list"][:5]:
            detail = a["detail"][:W - 4]
            print(f"║  {'·'} {detail:<{W-4}}║")

    print("╚" + "═" * W + "╝")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run(input_path: str, output_path: str) -> Dict[str, Any]:
    print(f"  [analytics] Loading {input_path}…")
    df = load_journeys(input_path)
    print(f"  [analytics] {len(df):,} rows, {df['person_id'].nunique():,} unique visitors.")

    traffic   = compute_traffic(df)
    funnel    = compute_funnel(df)
    dwell     = compute_dwell(df)
    anomalies = compute_anomalies(df, traffic, dwell)

    metrics: Dict[str, Any] = {
        "meta": {
            "source_file":  os.path.basename(input_path),
            "n_rows":       int(len(df)),
            "generated_by": "analytics.py",
        },
        "traffic":   traffic,
        "funnel":    funnel,
        "dwell":     dwell,
        "anomalies": anomalies,
    }

    # Sanitise all numpy types before serialisation
    metrics = _native(metrics)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)
    print(f"  [analytics] → {output_path}")

    print_summary(metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="analytics.py — Phase 2 Deterministic Analytics Engine"
    )
    parser.add_argument(
        "--input",  default="output/journeys.csv",
        help="Path to journeys.csv (default: output/journeys.csv)",
    )
    parser.add_argument(
        "--output", default="output/metrics.json",
        help="Output path for metrics.json (default: output/metrics.json)",
    )
    args = parser.parse_args()

    try:
        run(args.input, args.output)
    except FileNotFoundError as exc:
        print(f"[ERROR] Input file not found: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()