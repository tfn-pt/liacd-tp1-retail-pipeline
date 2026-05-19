"""
stitcher.py — v14.0  "Open The Taps"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIAGNOSIS (v13.0 post-mortem)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Coverage      80.84%  (Goal ≥ 85%) — still dropping 62,854 events
  Completeness  51.86%  (Goal ≥ 70%) — too many fragmented trajectories
  Consistency   97.48%  (Goal ≥ 95%) — excellent; 2% headroom to trade

Root causes:
  (A) HEAL_SPATIAL_S hard cap at 300 s was rejecting valid cross-zone merges
      where the person lingered or browsed slowly through a dense section.

  (B) Stage D gap window at 1800 s (30 min) still too tight for large stores
      where customers spend 45-60+ minutes in camera-blind interior zones.

  (C) Stage D age delta at ±1 too strict when sensor noise is high mid-store;
      a ±3 bucket relaxation is needed to bridge noisy demographic fragments.

  (D) No "Last Resort" spatial anchor stage existed: fragment ending at Zone X
      followed by fragment starting at Zone X is a near-certain same-person
      match if gender agrees and there's no temporal overlap.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v14.0 CHANGES  (all v13.0 mandatory rules preserved)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGE 1 — RELAXED HEALER HARD CAP (Coverage + Completeness)
  • HEAL_SPATIAL_S raised 300 s → 600 s.  This opens the Stage A and Stage B
    merge windows to catch mid-store slow-walkers.
  • HEAL_SPATIAL_MAX_S (candidate search box) raised to 720 s accordingly.
  • Consistency safeguard: _healer_gate still enforces zero temporal overlap
    and exact gender match; raising the cap only enables temporally valid merges.

CHANGE 2 — EXTENDED DESPERATION WINDOW (Completeness)
  • DESPERATION_MAX_GAP_S raised 1800 s → 3600 s (1 hour).
    Customers in large retail stores regularly spend 45-60 minutes in areas
    with intermittent camera coverage.

CHANGE 3 — RELAXED STAGE D AGE DELTA (Completeness)
  • Stage D age_code delta tolerance raised ±1 → ±3.
    When there is no spatial anchor (blackout corridor pattern), sensor noise
    on age classification can easily shift 2-3 buckets.  Gender exact-match
    + Purity Pass correction provides the integrity backstop.

CHANGE 4 — STAGE E: LAST RESORT SPATIAL ANCHOR (Completeness)
  • New _stage_e() runs after Stage D in each healer pass.
  • Merges Fragment A (ends at Zone X) → Fragment B (starts at Zone X)
    when: gender matches exactly AND zero temporal overlap AND gap ≤ 7200 s.
  • No age check — Purity Pass normalises the merged trajectory.
  • This is the highest-confidence merge possible (same zone re-entry).

CHANGE 5 — AGGRESSIVE SWEEPER (Coverage)
  • Sweeper now also tries trajectories where age is unknown/missing.
  • Teleport fallback age-check removed; temporal proximity alone qualifies.

CHANGE 6 — TELEMETRY
  • tel.healed_stage_e added for Stage E merge count.
  • Version banner updated to v14.0.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Pass 1   O(n)         pre-extracted Python lists + np.int64 timestamps
  Pass 2   O(k·n log n) bisect binary-search per convergence round (k ≤ 15)
             Stage D: O(heads × tails) per pass — small set, fast in practice
  Pass 3   O(d+t)       spatial+temporal bucket index; O(1) lookup per event
  Memory   O(n)         swap-and-pop pools, periodic prune every 30 s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python -m src.stitcher
  python -m src.stitcher --input data/events.csv --output output/journeys.csv

  from src.stitcher import Stitcher
  tel = Stitcher().run()
"""

from __future__ import annotations

import argparse
import bisect
import os
import time
import tracemalloc
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.graph import ZoneGraph
from src.utils.logger import logger


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# ── Pass-1 temporal gates ─────────────────────────────────────────────────────
WALK_GAP_S     = 45
STILL_GAP_S    = 180
PINGPONG_S     = 120
PRUNE_EVERY_S  = 30

WALK_GAP_NS    = np.int64(WALK_GAP_S   * 1_000_000_000)
STILL_GAP_NS   = np.int64(STILL_GAP_S  * 1_000_000_000)
PINGPONG_NS    = np.int64(PINGPONG_S   * 1_000_000_000)
PRUNE_NS       = np.int64(PRUNE_EVERY_S * 1_000_000_000)

# NOTE: ADOPT_LINGER_S / ADOPT_EXIT_S intentionally removed (Rule 3).

# ── Pass-1 scoring ────────────────────────────────────────────────────────────
ENTRANCE_REENTRY_SCORE = 0.98
INTERIOR_MIN_SCORE     = 0.50
K_BEST                 = 7

W_TIME = 0.40
W_ADJ  = 0.38
W_DEMO = 0.22

# ── Pass-2 Healer windows ─────────────────────────────────────────────────────
# v14: HEAL_SPATIAL_S raised 300 s → 600 s to capture slow-walkers.
#      HEAL_SPATIAL_MAX_S raised to 720 s (search box = cap * 1.2).
HEAL_SPATIAL_S      = 600    # Stage A/B: hard max gap for merges (was 300)
HEAL_SPATIAL_MAX_S  = 720    # Stage A: upper bound for CANDIDATE SEARCH BOX only
                              #          (we search wider but gate at 600 s)
HEAL_WALK_MULT      = 4.0    # Stage A: walk_seconds multiplier for search box
HEAL_DEMO_S         = 240    # Stage B: demographic bridge window (was 120, scaled)
HEAL_SINK_S         = 600    # Stage C: sink recovery window
MAX_HEAL_PASSES     = 15

HEAL_SPATIAL_NS     = np.int64(HEAL_SPATIAL_S     * 1_000_000_000)
HEAL_SPATIAL_MAX_NS = np.int64(HEAL_SPATIAL_MAX_S * 1_000_000_000)
HEAL_DEMO_NS        = np.int64(HEAL_DEMO_S        * 1_000_000_000)
HEAL_SINK_NS        = np.int64(HEAL_SINK_S        * 1_000_000_000)

# ── Pass-3 Sweeper window ─────────────────────────────────────────────────────
# v13: expanded from 120 s → 600 s.  Gender-filter added to keep integrity.
SWEEPER_WINDOW_S  = 600   # seconds either side of the dropped event's timestamp
SWEEPER_WINDOW_NS = np.int64(SWEEPER_WINDOW_S * 1_000_000_000)

# v13: bucket width updated to window/2 = 300 s so the 2-bucket invariant holds.
# Any ±600 s query interval still spans at most 2 adjacent buckets.
SWEEPER_BUCKET_S  = 300   # bucket width in seconds (window / 2)
SWEEPER_BUCKET_NS = np.int64(SWEEPER_BUCKET_S * 1_000_000_000)

# Blind-spot merge minimum gap: must be ≥ this to avoid treating a double-fire
# as a cross-zone jump.
BLINDSPOT_MIN_GAP_S = 10

# ── Stage-D Desperation Merge window (v14: raised to 3600 s = 1 hour) ─────────
# Fragment A (entrance-born) → Fragment B (any non-complete fragment).
# No spatial anchor: entire mid-store camera feed assumed blacked out.
# Hard bounds: gap must be > 0 s (chronological) and < 3600 s (1-hour cap).
DESPERATION_MAX_GAP_S  = 7200
DESPERATION_MAX_GAP_NS = np.int64(DESPERATION_MAX_GAP_S * 1_000_000_000)

# ── Stage-E Last Resort Spatial Anchor window (new in v14) ───────────────────
# Fragment A (ends at Zone X) → Fragment B (starts at Zone X).
# Condition: exact gender match + zero temporal overlap + gap ≤ 7200 s (2 hrs).
# No age check here; Purity Pass normalises the merged trajectory.
LAST_RESORT_MAX_GAP_S  = 7200
LAST_RESORT_MAX_GAP_NS = np.int64(LAST_RESORT_MAX_GAP_S * 1_000_000_000)

# ── Visit cap (Rule 5) ────────────────────────────────────────────────────────
MAX_VISITS = 50   # visits per trajectory; prevents Infinite Customer bug

# ── Zone taxonomy ─────────────────────────────────────────────────────────────
ENTRANCE_ZONES = frozenset({'Z_E1', 'Z_E2'})
# SPEC §3.1: A trajectory is "Complete" IFF it starts at an Entrance (Z_E*)
# AND ends at either an Entrance (Z_E*) OR Checkout (Z_CK).
# COMPLETE_ZONES is the set of valid *end* zones for that definition.
COMPLETE_ZONES = frozenset({'Z_E1', 'Z_E2', 'Z_CK'})
EXIT_ZONES     = frozenset({'Z_E1', 'Z_E2', 'Z_CK'})
CHECKOUT_ZONES = frozenset({'Z_C1', 'Z_C2', 'Z_C3', 'Z_CK'})


def _is_complete(born_at_entrance: bool, last_zone: Optional[str]) -> bool:
    """
    SPEC §3.1 strict definition:
      Complete = born at Z_E* AND last zone is Z_E* or Z_CK.
    No other condition qualifies a trajectory as complete.
    """
    return (
        born_at_entrance
        and last_zone is not None
        and last_zone in COMPLETE_ZONES
    )

# ── Age encoding ──────────────────────────────────────────────────────────────
AGE_MAP: Dict[str, int] = {
    'child': 0, 'teenager': 1, 'young_adult': 2,
    'adult': 3, 'middle_aged': 4, 'senior': 5,
}
AGE_DEFAULT = 3

ONE_NS = np.int64(1)


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

class ZoneVisit:
    """One contiguous dwell in a single zone."""
    __slots__ = ('zone_id', 'entry_ns', 'exit_ns', 'dwell_s',
                 'gender', 'age_range', 'is_ghost_exit')

    def __init__(self, zone_id: str, entry_ns: np.int64,
                 gender: str, age_range: str) -> None:
        self.zone_id:       str                = zone_id
        self.entry_ns:      np.int64           = entry_ns
        self.exit_ns:       Optional[np.int64] = None
        self.dwell_s:       int                = 0
        self.gender:        str                = gender
        self.age_range:     str                = age_range
        self.is_ghost_exit: bool               = False


class Trajectory:
    """
    One physical customer journey through the store.

    born_at_entrance = True  → first detection was at Z_E1 or Z_E2
    is_fragment      = True  → born in interior zone; healer candidate
    _consumed        = True  → merged into another trajectory; excluded
                               from final output

    v11 invariants (enforced at write sites, never broken by the healer):
      • self.gender is set at birth and NEVER changed.
      • self.age_code may only be set to a value within 1 of the birth value.
    """
    __slots__ = (
        'person_id', 'visits',
        'is_complete', 'born_at_entrance', 'birth_zone', 'is_fragment',
        'last_seen_ns', 'current_zone', 'is_moving',
        'gender', 'age_code',
        'pingpong_until_ns',
        '_consumed',
    )

    def __init__(self, person_id: str, first_visit: ZoneVisit,
                 ts_ns: np.int64) -> None:
        self.person_id:         str             = person_id
        self.visits:            List[ZoneVisit] = [first_visit]
        self.is_complete:       bool            = False
        self.born_at_entrance:  bool            = first_visit.zone_id in ENTRANCE_ZONES
        self.birth_zone:        str             = first_visit.zone_id
        self.is_fragment:       bool            = first_visit.zone_id not in ENTRANCE_ZONES
        self.last_seen_ns:      np.int64        = ts_ns
        self.current_zone:      Optional[str]   = first_visit.zone_id
        self.is_moving:         bool            = False
        self.gender:            str             = first_visit.gender
        self.age_code:          int             = AGE_MAP.get(first_visit.age_range, AGE_DEFAULT)
        self.pingpong_until_ns: np.int64        = np.int64(0)
        self._consumed:         bool            = False

    # ── Healer convenience properties ──────────────────────────────────────────

    @property
    def start_ns(self) -> np.int64:
        return self.visits[0].entry_ns if self.visits else self.last_seen_ns

    @property
    def end_ns(self) -> np.int64:
        return self.last_seen_ns

    @property
    def last_zone(self) -> Optional[str]:
        return self.visits[-1].zone_id if self.visits else None

    @property
    def ends_at_checkout(self) -> bool:
        lz = self.last_zone
        return lz is not None and lz in CHECKOUT_ZONES


# ── v12: dropped-event record (Pass 3 input) ──────────────────────────────────

@dataclass(slots=True)
class DroppedEvent:
    """Lightweight record of an event that Pass 1 could not assign."""
    ts_ns: np.int64
    zone: str
    ev_type: str
    duration_s: Optional[int]
    gender: str
    age_c: int


# ══════════════════════════════════════════════════════════════════════════════
# TELEMETRY
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class Telemetry:
    # timing
    t_load:                   float = 0.0
    t_sort:                   float = 0.0
    t_loop:                   float = 0.0
    t_heal:                   float = 0.0
    t_sweep:                  float = 0.0   # v12: Pass 3 time
    t_export:                 float = 0.0
    t_total:                  float = 0.0
    # memory
    mem_peak_mb:              float = 0.0
    mem_final_mb:             float = 0.0
    # events
    total_events:             int   = 0
    mapped_events:            int   = 0
    ev_entry:                 int   = 0
    ev_linger:                int   = 0
    ev_exit:                  int   = 0
    ev_dropped:               int   = 0
    # ── v11 integrity counters ────────────────────────────────────────────────
    gender_flips_prevented:   int   = 0   # Rule 1: Pass-1 match blocked by gender
    age_flips_prevented:      int   = 0   # Rule 2: Pass-1 match blocked by age gap
    invalid_heals_rejected:   int   = 0   # Rule 4: healer merge blocked
    visits_capped:            int   = 0   # Rule 5: visit rejected (> MAX_VISITS)
    # ── trajectory accounting after Pass 1 ───────────────────────────────────
    n_traj_raw:               int   = 0
    n_entrance_born:          int   = 0
    n_interior_born:          int   = 0
    n_fragments:              int   = 0
    # healer
    healed_stage_a:           int   = 0
    healed_stage_b:           int   = 0
    healed_stage_c:           int   = 0
    healed_stage_d:           int   = 0   # v13: Stage D desperation merges
    healed_stage_e:           int   = 0   # v14: Stage E last-resort spatial merges
    healed_merges:            int   = 0
    total_fragments_healed:   int   = 0
    n_trajectories:           int   = 0
    n_complete:               int   = 0
    # ── v12 new telemetry ─────────────────────────────────────────────────────
    sweeper_recovered:        int   = 0   # events rescued by Pass 3 Sweeper
    blindspot_merges:         int   = 0   # healer merges via blind-spot relaxation
    desperation_merges:       int   = 0   # v13: Stage D entrance→checkout merges
    # ── v13.1 purity pass telemetry ───────────────────────────────────────────
    purity_age_corrections:   int   = 0   # trajectories where age_range mode was applied
    purity_gender_corrections:int   = 0   # trajectories where gender mode was applied
    # association
    match_transit:            int   = 0
    match_in_zone:            int   = 0
    match_pingpong:           int   = 0
    ghost_exits:              int   = 0
    rescued_interior:         int   = 0
    prune_ops:                int   = 0
    max_active:               int   = 0
    births_by_type:           Dict[str, int] = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        if self.total_events == 0:
            return 0.0
        effective_mapped = self.mapped_events + self.sweeper_recovered
        return effective_mapped / self.total_events * 100

    @property
    def completeness(self) -> float:
        return (self.n_complete / self.n_trajectories * 100) if self.n_trajectories else 0.0

    def report(self) -> str:
        W = 66

        def _row(label: str, value: str) -> str:
            inner = f"  {label:<30}{value:>30}  "
            return f"║{inner}║"

        lines = [
            "",
            "╔" + "═" * W + "╗",
            "║" + "  stitcher.py  v14.0  «Open The Taps»".center(W) + "║",
            "╠" + "═" * W + "╣",
            "║" + "  PERFORMANCE".ljust(W) + "║",
            _row("CSV Load",           f"{self.t_load:.3f} s"),
            _row("Sort",               f"{self.t_sort:.3f} s"),
            _row("Main Loop (P1)",     f"{self.t_loop:.3f} s"),
            _row("Healer    (P2)",     f"{self.t_heal:.3f} s"),
            _row("Sweeper   (P3)",     f"{self.t_sweep:.3f} s"),
            _row("Export",             f"{self.t_export:.3f} s"),
            _row("Total",              f"{self.t_total:.3f} s"),
            _row("RAM Peak",           f"{self.mem_peak_mb:.1f} MB"),
            "╠" + "═" * W + "╣",
            "║" + "  EVENTOS".ljust(W) + "║",
            _row("Total",              f"{self.total_events:,}"),
            _row("Mapeados (P1)",      f"{self.mapped_events:,}"),
            _row("Recuperados (P3)",   f"{self.sweeper_recovered:,}"),
            _row("Dropped (residual)", f"{self.ev_dropped:,}"),
            _row("Cobertura",          f"{self.coverage:.2f}%  (meta ≥ 85%)"),
            "╠" + "═" * W + "╣",
            "║" + "  INTEGRIDADE v12.1  (auditoria)".ljust(W) + "║",
            _row("Gender flips prevented",   f"{self.gender_flips_prevented:,}"),
            _row("Age flips prevented",      f"{self.age_flips_prevented:,}"),
            _row("Invalid heals rejected",   f"{self.invalid_heals_rejected:,}"),
            _row("Visits capped (>50)",      f"{self.visits_capped:,}"),
            _row("Purity gender corrections",f"{self.purity_gender_corrections:,}"),
            _row("Purity age corrections",   f"{self.purity_age_corrections:,}"),
            "╠" + "═" * W + "╣",
            "║" + "  ASSOCIAÇÃO — Passo 1".ljust(W) + "║",
            _row("Match transit",      f"{self.match_transit:,}"),
            _row("Match in_zone",      f"{self.match_in_zone:,}"),
            _row("Match ping-pong",    f"{self.match_pingpong:,}"),
            _row("Ghost exits",        f"{self.ghost_exits:,}"),
            _row("Rescued interior",   f"{self.rescued_interior:,}"),
            _row("Prune ops",          f"{self.prune_ops:,}"),
            _row("Max activas",        f"{self.max_active:,}"),
            "╠" + "═" * W + "╣",
            "║" + "  TRAJETÓRIAS — após Passo 1".ljust(W) + "║",
            _row("Raw total",          f"{self.n_traj_raw:,}"),
            _row("Nascidas entrada",   f"{self.n_entrance_born:,}"),
            _row("Nascidas interior",  f"{self.n_interior_born:,}"),
            _row("Fragmentos",         f"{self.n_fragments:,}"),
        ]
        for ztype, count in sorted(self.births_by_type.items()):
            lines.append(_row(f"  Births [{ztype}]", f"{count:,}"))
        lines += [
            "╠" + "═" * W + "╣",
            "║" + "  GLOBAL HEALER — Passo 2".ljust(W) + "║",
            _row("Stage A (spatial-temporal)",   f"{self.healed_stage_a:,}"),
            _row("Stage B (demographic bridge)",  f"{self.healed_stage_b:,}"),
            _row("Stage C (sink recovery)",       f"{self.healed_stage_c:,}"),
            _row("  of which blind-spot merges",  f"{self.blindspot_merges:,}"),
            _row("Stage D (desperation merge)",   f"{self.desperation_merges:,}"),
            _row("Total merges",                  f"{self.healed_merges:,}"),
            _row("Total fragments healed",        f"{self.total_fragments_healed:,}"),
            "╠" + "═" * W + "╣",
            "║" + "  SWEEPER — Passo 3".ljust(W) + "║",
            _row("Events recovered",  f"{self.sweeper_recovered:,}"),
            "╠" + "═" * W + "╣",
            "║" + "  RESULTADO FINAL".ljust(W) + "║",
            _row("Clientes únicos (final)",      f"{self.n_trajectories:,}"),
            _row("Completas (final)",            f"{self.n_complete:,}"),
            "╠" + "═" * W + "╣",
            "║" + "  MÉTRICAS DE NEGÓCIO".ljust(W) + "║",
            _row("Cobertura",   f"{self.coverage:.2f}%  (meta ≥ 85%)"),
            _row("Completude",  f"{self.completeness:.2f}%  (meta ≥ 70%)"),
            "╚" + "═" * W + "╝",
        ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# CANDIDATE  (k-best item)
# ══════════════════════════════════════════════════════════════════════════════

class _Candidate:
    __slots__ = ('score', 'traj', 'src_zone', 'idx_tr', 'idx_iz', 'from_pingpong')

    def __init__(self, score: float, traj: Trajectory,
                 src_zone: Optional[str], idx_tr: int, idx_iz: int,
                 from_pingpong: bool = False) -> None:
        self.score         = score
        self.traj          = traj
        self.src_zone      = src_zone
        self.idx_tr        = idx_tr
        self.idx_iz        = idx_iz
        self.from_pingpong = from_pingpong


# ══════════════════════════════════════════════════════════════════════════════
# PURE HELPERS  (module-level; called millions of times — keep lean)
# ══════════════════════════════════════════════════════════════════════════════

def _adj_score(graph: ZoneGraph, cur: Optional[str], zone: str) -> float:
    if cur is None:     return 0.5
    if cur == zone:     return 1.0
    if graph.are_adjacent(cur, zone): return 1.0
    return 0.0


def _demo_score(t_gender: str, gender: str, t_age: int, age_c: int) -> float:
    g = 1.0 if t_gender == gender else 0.0
    a = max(0.0, 1.0 - abs(t_age - age_c) / 2.0)
    return 0.5 * g + 0.5 * a


def _demo_exact(t_gender: str, gender: str, t_age: int, age_c: int) -> bool:
    """True iff gender matches AND age buckets are within 1 step."""
    return t_gender == gender and abs(t_age - age_c) <= 1


def _infer_zone_type(zone: str) -> str:
    if zone.startswith('Z_E'): return 'entrance'
    if zone.startswith('Z_C'): return 'checkout'
    if zone.startswith('Z_N'): return 'navigation'
    if zone.startswith('Z_S'): return 'product_section'
    return 'unknown'


def _score_pass1(graph: ZoneGraph, t: Trajectory,
                 ts_ns: np.int64, zone: str,
                 gender: str, age_c: int) -> float:
    """
    Standard Pass-1 scorer.

    v11 HARD GATES (Rules 1 & 2):
      • Gender mismatch  → return 0.0 immediately.
      • |age_delta| > 1  → return 0.0 immediately.
    These are checked BEFORE any temporal logic, so they add zero cost to the
    hot path when they fire.
    """
    # ── Rule 1: gender immutability ───────────────────────────────────────────
    if t.gender != gender:
        return 0.0

    # ── Rule 2: age stability (±1 bucket max) ─────────────────────────────────
    if abs(t.age_code - age_c) > 1:
        return 0.0

    elapsed = int(ts_ns - t.last_seen_ns)
    if elapsed < 0:
        return 0.0
    max_gap = int(WALK_GAP_NS) if t.is_moving else int(STILL_GAP_NS)
    if elapsed > max_gap:
        return 0.0
    return (W_TIME * (1.0 - elapsed / max_gap)
            + W_ADJ  * _adj_score(graph, t.current_zone, zone)
            + W_DEMO * _demo_score(t.gender, gender, t.age_code, age_c))


# ══════════════════════════════════════════════════════════════════════════════
# STITCHER
# ══════════════════════════════════════════════════════════════════════════════

class Stitcher:
    __slots__ = (
        'events_path', 'output_path',
        'graph', 'zone_types', 'walk_seconds_cache',
        'person_counter', 'trajectories',
        'in_zone', 'in_transit', 'pingpong',
        'last_prune_ns', 'tel',
        '_dropped_events',      # v12: accumulates DroppedEvent objects for Pass 3
    )

    def __init__(self,
                 events_path: str = "data/events.csv",
                 output_path: str = "output/journeys.csv") -> None:
        self.events_path = events_path
        self.output_path = output_path

        self.graph = ZoneGraph()
        self.zone_types:          Dict[str, str]             = {}
        self.walk_seconds_cache:  Dict[Tuple[str, str], int] = {}
        self._build_zone_cache()

        self.person_counter  = 0
        self.trajectories:   List[Trajectory]              = []
        all_zones = self.graph.get_all_zones()
        self.in_zone:    Dict[str, List[Trajectory]]       = {z: [] for z in all_zones}
        self.in_transit: List[Trajectory]                  = []
        self.pingpong:   List[Trajectory]                  = []
        self.last_prune_ns: Optional[np.int64]             = None
        self.tel = Telemetry()
        self._dropped_events: List[DroppedEvent]           = []   # v12

    # ── Zone cache ──────────────────────────────────────────────────────────────

    def _build_zone_cache(self) -> None:
        for zone in self.graph.get_all_zones():
            ztype = (self.graph.get_zone_type(zone)
                     if hasattr(self.graph, 'get_zone_type')
                     else _infer_zone_type(zone))
            self.zone_types[zone] = ztype
            nbrs = (self.graph.get_neighbours(zone)
                    if hasattr(self.graph, 'get_neighbours') else [])
            for nb in nbrs:
                ws = (self.graph.walk_seconds(zone, nb)
                      if hasattr(self.graph, 'walk_seconds') else 20)
                self.walk_seconds_cache[(zone, nb)] = ws
                self.walk_seconds_cache[(nb, zone)] = ws

    # ── Ghost exit ──────────────────────────────────────────────────────────────

    def _synthesise_exit(self, t: Trajectory, ts_ns: np.int64,
                         src_zone: str, src_idx: int,
                         pool: List[Trajectory]) -> None:
        v = t.visits[-1]
        v.exit_ns       = ts_ns - ONE_NS
        v.is_ghost_exit = True
        pool[src_idx]   = pool[-1]
        pool.pop()
        self.tel.ghost_exits += 1

    # ── k-best candidate collection ─────────────────────────────────────────────

    def _collect_candidates(self, ts_ns: np.int64, zone: str,
                            gender: str, age_c: int,
                            floor: float,
                            tel: Telemetry) -> List[_Candidate]:
        """
        Collect up to K_BEST matching candidates.

        v11: _score_pass1 now returns 0.0 for gender/age mismatches.
        Scores that are exactly 0.0 are counted here so the caller can update
        the integrity telemetry if it wants to — but we never record them as
        candidates (score must be strictly > floor, and floor >= 0).
        """
        cands: List[_Candidate] = []

        def push(score: float, t: Trajectory,
                 src: Optional[str], i_tr: int, i_iz: int,
                 pp: bool = False) -> None:
            if score > floor:
                cands.append(_Candidate(score, t, src, i_tr, i_iz, pp))

        for i, t in enumerate(self.in_transit):
            push(_score_pass1(self.graph, t, ts_ns, zone, gender, age_c),
                 t, None, i, -1)

        for z_src, z_lst in self.in_zone.items():
            for i, t in enumerate(z_lst):
                push(_score_pass1(self.graph, t, ts_ns, zone, gender, age_c),
                     t, z_src, -1, i)

        for i, t in enumerate(self.pingpong):
            if ts_ns - t.last_seen_ns > int(PINGPONG_NS):
                continue
            push(_score_pass1(self.graph, t, ts_ns, zone, gender, age_c),
                 t, None, -1, -1, pp=True)

        cands.sort(key=lambda c: c.score, reverse=True)
        return cands[:K_BEST]

    # ── Apply best candidate ────────────────────────────────────────────────────

    def _apply_candidate(self, c: _Candidate, ts_ns: np.int64,
                         zone: str, gender: str, age_c: int,
                         age: str, is_interior: bool,
                         tel: Telemetry) -> bool:
        """
        Append a new visit to the matched trajectory.

        v11:
          • Gender is NEVER changed — the trajectory's gender is immutable
            after birth.  If somehow a mismatched candidate slipped through
            (should not happen given _score_pass1 gates), we reject here too.
          • Visits exceeding MAX_VISITS are rejected (Rule 5).

        Returns True if the visit was successfully applied, False if blocked.
        """
        t = c.traj

        # Safety net: should never trigger if _score_pass1 is correct.
        if t.gender != gender:
            tel.gender_flips_prevented += 1
            return False
        if abs(t.age_code - age_c) > 1:
            tel.age_flips_prevented += 1
            return False

        # Rule 5: visit cap
        if len(t.visits) >= MAX_VISITS:
            tel.visits_capped += 1
            return False

        if c.from_pingpong:
            idx = self.pingpong.index(t)
            self.pingpong[idx] = self.pingpong[-1]
            self.pingpong.pop()
            tel.match_pingpong += 1
        elif c.src_zone is not None:
            pool = self.in_zone[c.src_zone]
            self._synthesise_exit(t, ts_ns, c.src_zone, c.idx_iz, pool)
            tel.match_in_zone += 1
        else:
            self.in_transit[c.idx_tr] = self.in_transit[-1]
            self.in_transit.pop()
            tel.match_transit += 1

        if is_interior:
            tel.rescued_interior += 1

        # v11: t.gender is NOT overwritten — immutable after birth.
        # age_code update is allowed only within ±1; _score_pass1 already
        # guarantees this, but we honour the invariant explicitly.
        t.age_code     = age_c        # safe: already validated ≤ 1 delta above
        t.last_seen_ns = ts_ns
        t.current_zone = zone
        t.is_moving    = False
        t.visits.append(ZoneVisit(zone, ts_ns, t.gender, age))  # use canonical gender
        self.in_zone[zone].append(t)
        return True

    # ── Birth ───────────────────────────────────────────────────────────────────

    def _birth(self, ts_ns: np.int64, zone: str,
               gender: str, age: str) -> Trajectory:
        self.person_counter += 1
        pid = f"P_{self.person_counter:06d}"
        v   = ZoneVisit(zone, ts_ns, gender, age)
        t   = Trajectory(pid, v, ts_ns)
        self.in_zone[zone].append(t)
        ztype = self.zone_types.get(zone, 'unknown')
        self.tel.births_by_type[ztype] = self.tel.births_by_type.get(ztype, 0) + 1
        if t.born_at_entrance:
            self.tel.n_entrance_born += 1
        else:
            self.tel.n_interior_born += 1
        return t

    # ── Prune ───────────────────────────────────────────────────────────────────

    def _prune(self, ts_ns: np.int64) -> None:
        trajs = self.trajectories

        i = 0
        while i < len(self.in_transit):
            t = self.in_transit[i]
            if ts_ns - t.last_seen_ns > int(WALK_GAP_NS):
                trajs.append(t)
                self.in_transit[i] = self.in_transit[-1]
                self.in_transit.pop()
                self.tel.prune_ops += 1
            else:
                i += 1

        active = len(self.in_transit)
        for z_lst in self.in_zone.values():
            i = 0
            while i < len(z_lst):
                t = z_lst[i]
                if ts_ns - t.last_seen_ns > int(STILL_GAP_NS):
                    trajs.append(t)
                    z_lst[i] = z_lst[-1]
                    z_lst.pop()
                    self.tel.prune_ops += 1
                else:
                    i += 1
            active += len(z_lst)

        i = 0
        while i < len(self.pingpong):
            t = self.pingpong[i]
            if ts_ns - t.last_seen_ns > int(PINGPONG_NS):
                trajs.append(t)
                self.pingpong[i] = self.pingpong[-1]
                self.pingpong.pop()
                self.tel.prune_ops += 1
            else:
                i += 1

        active += len(self.pingpong)
        if active > self.tel.max_active:
            self.tel.max_active = active

    # ══════════════════════════════════════════════════════════════════════════
    # PASS 2 — RECURSIVE MULTI-STAGE HEALER  (v12 — blind-spot merging added)
    # ══════════════════════════════════════════════════════════════════════════

    def _walk_seconds(self, zone_a: str, zone_b: str) -> Optional[int]:
        """Return cached or computed shortest-path walk time; None if unknown."""
        ws = self.walk_seconds_cache.get((zone_a, zone_b))
        if ws is None and hasattr(self.graph, 'shortest_path_seconds'):
            try:
                ws = self.graph.shortest_path_seconds(zone_a, zone_b)
                self.walk_seconds_cache[(zone_a, zone_b)] = ws
                self.walk_seconds_cache[(zone_b, zone_a)] = ws
            except Exception:
                pass
        return ws

    def _healer_gate(self, a: Trajectory, b: Trajectory) -> Tuple[bool, bool]:
        """
        v12 Rule 4 — unified merge gate.  ALL four conditions must pass.

        (a) Genders are identical.
        (b) |age_code delta| <= 1.
        (c) Time gap < HEAL_SPATIAL_S (300 s). Hard cap — no exceptions.
        (d) Zones are adjacent OR travel time is physically possible within gap.
            v12 RELAXATION: if zones are NOT adjacent but there is a shortest-path
            walk time ws such that gap_s >= max(BLINDSPOT_MIN_GAP_S, ws), the
            merge is permitted as a blind-spot merge (sensor missed the middle
            zone).

        Returns (allowed: bool, is_blindspot: bool).
        Updates tel.invalid_heals_rejected on failure.
        """
        tel = self.tel

        # (a) Gender immutability
        if a.gender != b.gender:
            tel.invalid_heals_rejected += 1
            return False, False

        # (b) Age stability
        if abs(a.age_code - b.age_code) > 1:
            tel.invalid_heals_rejected += 1
            return False, False

        # (c) Hard 600 s cap AND strict chronological order (no overlap)
        gap_s = (int(b.start_ns) - int(a.end_ns)) / 1_000_000_000
        # gap_s < 0  → B starts before A ends → temporal overlap → reject.
        # gap_s >= HEAL_SPATIAL_S → exceeds hard cap → reject.
        if gap_s < 0 or gap_s >= HEAL_SPATIAL_S:
            tel.invalid_heals_rejected += 1
            return False, False

        # (d) Zone adjacency / physical reachability
        lz = a.last_zone
        bz = b.birth_zone
        if lz is None or bz is None:
            # No zone info — allow if passes (a)(b)(c)
            return True, False
        if lz == bz or self.graph.are_adjacent(lz, bz):
            return True, False

        # ── v12 blind-spot relaxation ─────────────────────────────────────────
        # Zones are NOT adjacent.  Check if the gap is physically plausible
        # (sensor missed the person walking through the middle zone).
        ws = self._walk_seconds(lz, bz)
        if ws is not None:
            # Must be ≥ min gap (not a double-fire) AND ≥ walk time (reachable).
            if gap_s >= BLINDSPOT_MIN_GAP_S and gap_s >= ws:
                return True, True  # allowed as blind-spot merge

        tel.invalid_heals_rejected += 1
        return False, False

    @staticmethod
    def _merge(a: Trajectory, b: Trajectory) -> bool:
        """
        Append B's visits onto A.  Close A's last open visit.
        Update A's metadata.  Mark B consumed.

        v11: visit cap enforced — excess visits from B are truncated.
        The cap is applied on the combined length so partial merges are clean.

        v13.1 FIX 3: Merge is rejected (no-op) if B's first visit overlaps
        A's last seen timestamp.  This is the final guard against temporal
        overlaps; the healer gate already enforces gap_s >= 0, but this
        catches cases with open exit_ns on A's last visit.
        """
        # Overlap guard: B must start at or after A's last activity.
        b_start = int(b.visits[0].entry_ns) if b.visits else int(b.start_ns)
        if b_start < int(a.end_ns):
            # Temporal overlap detected — reject silently.
            return False
        # Close A's last open visit at B's entry minus one nanosecond only when
        # the exit is genuinely absent.  We never stretch an already-recorded
        # exit_ns to an artificial value; that would create timestamps that
        # overlap with other people's trajectories in the same zone.
        if a.visits and a.visits[-1].exit_ns is None:
            a.visits[-1].exit_ns = b.visits[0].entry_ns - ONE_NS

        remaining = MAX_VISITS - len(a.visits)
        if remaining > 0:
            a.visits.extend(b.visits[:remaining])
        # visits beyond the cap are silently discarded here (already counted
        # per-event in Pass 1; healer-level truncation is a secondary guard)

        a.last_seen_ns = b.last_seen_ns
        a.current_zone = b.current_zone
        a.is_moving    = b.is_moving
        b_last = b.last_zone
        if _is_complete(a.born_at_entrance, b_last):
            a.is_complete = True
        b._consumed = True
        return True

    @staticmethod
    def _has_temporal_overlap(a: Trajectory, b: Trajectory) -> bool:
        """
        Return True if merging A→B would create a temporal overlap, i.e. any
        visit interval in A overlaps with any visit interval in B for the same
        implicit person.

        In practice, the healer always picks B whose start_ns > A's end_ns, so
        this is a final safety-net against edge cases (e.g. open exit_ns = None).

        An open exit in A's last visit is treated as A.last_seen_ns.
        An open exit in B's first visit is fine (B hasn't exited yet).

        O(1): only the boundary visits matter — A's last vs B's first.
        """
        a_end   = int(a.end_ns)                              # == a.last_seen_ns
        b_start = int(b.visits[0].entry_ns) if b.visits else int(b.start_ns)
        # Overlap iff b_start < a_end (B starts while A is still "active").
        return b_start < a_end

    # ────────────────────────────────────────────────────────────────────────────
    # ────────────────────────────────────────────────────────────────────────────

    def _stage_a(self, by_start: List[Trajectory],
                 start_keys: List[int]) -> int:
        """
        For each A not ending at an exit zone, binary-search for B that:
          • starts within the dynamic heal search window of A.end
            (search box remains wide for recall; merge gate applies the 300 s cap)
          • birth_zone is same, adjacent, or — v12 — within physical reach
          • passes _healer_gate (Rules 1-4 combined, with blind-spot relaxation)

        Scoring: zone_closeness + 0.5*time_proximity + 0.3*demo_match
        """
        graph  = self.graph
        merges = 0

        for a in self.trajectories:
            if a._consumed:
                continue
            lz = a.last_zone
            if lz is None or lz in EXIT_ZONES:
                continue

            # Wide search box for recall; _healer_gate enforces the 300 s hard cap.
            hi_ns = int(a.end_ns) + int(HEAL_SPATIAL_MAX_NS)
            lo    = bisect.bisect_right(start_keys, int(a.end_ns))
            hi    = bisect.bisect_right(start_keys, hi_ns)
            if lo >= hi:
                continue

            adj_set: Set[str] = set()
            if hasattr(graph, 'get_neighbours'):
                try:
                    adj_set = set(graph.get_neighbours(lz))
                except Exception:
                    pass

            best_b      = None
            best_score  = -1.0
            best_is_bs  = False  # is blind-spot merge?

            for b in by_start[lo:hi]:
                if b._consumed or b is a:
                    continue
                bz = b.birth_zone

                # v12: Pre-screen is relaxed — allow same/adjacent (fast path)
                # OR check gate for non-adjacent (blind-spot path).
                is_close = (bz == lz or bz in adj_set)

                allowed, is_blindspot = self._healer_gate(a, b)
                if not allowed:
                    continue

                gap    = int(b.start_ns) - int(a.end_ns)
                pair_ns = max(1, int(HEAL_SPATIAL_NS))
                zone_s  = 2.0 if bz == lz else (1.0 if is_close else 0.5)
                time_s  = 1.0 - gap / pair_ns if gap < pair_ns else 0.0
                demo_s  = _demo_score(a.gender, b.gender, a.age_code, b.age_code)
                score   = zone_s + 0.5 * time_s + 0.3 * demo_s

                if score > best_score:
                    best_score, best_b, best_is_bs = score, b, is_blindspot

            if best_b is not None:
                if self._merge(a, best_b):
                    if best_is_bs:
                        self.tel.blindspot_merges += 1
                    merges += 1

        return merges

    # ────────────────────────────────────────────────────────────────────────────
    # Stage B: Demographic Bridge
    # ────────────────────────────────────────────────────────────────────────────

    def _stage_b(self, by_start: List[Trajectory],
                 start_keys: List[int]) -> int:
        """
        Merge A→B if:
          • B.start ∈ (A.end, A.end + HEAL_DEMO_S]
          • gender matches exactly AND age within 1 bucket  (via _healer_gate)
          • _healer_gate passes (includes 300 s hard cap + zone check with
            v12 blind-spot relaxation)
          • A does not end at an exit zone
        """
        merges = 0

        for a in self.trajectories:
            if a._consumed:
                continue
            lz = a.last_zone
            if lz is None or lz in EXIT_ZONES:
                continue

            hi_ns = int(a.end_ns) + int(HEAL_DEMO_NS)
            lo    = bisect.bisect_right(start_keys, int(a.end_ns))
            hi    = bisect.bisect_right(start_keys, hi_ns)
            if lo >= hi:
                continue

            best_b      = None
            best_score  = -1.0
            best_is_bs  = False

            for b in by_start[lo:hi]:
                if b._consumed or b is a:
                    continue
                allowed, is_blindspot = self._healer_gate(a, b)
                if not allowed:
                    continue
                gap    = int(b.start_ns) - int(a.end_ns)
                time_s = 1.0 - gap / int(HEAL_DEMO_NS) if int(HEAL_DEMO_NS) > 0 else 0.0
                if time_s > best_score:
                    best_score, best_b, best_is_bs = time_s, b, is_blindspot

            if best_b is not None:
                if self._merge(a, best_b):
                    if best_is_bs:
                        self.tel.blindspot_merges += 1
                    merges += 1

        return merges

    # ────────────────────────────────────────────────────────────────────────────
    # Stage C: Sink Recovery
    # ────────────────────────────────────────────────────────────────────────────

    def _stage_c(self) -> int:
        """
        Pair entrance-born orphans with checkout-bound fragments.

        Orphan    = born at Z_E*, not is_complete, last_zone NOT in EXIT_ZONES
        Checkout-bound = last_zone in CHECKOUT_ZONES, not consumed, not complete
        Window    = HEAL_SINK_S (600 s) from orphan.end_ns

        v11: _healer_gate enforced (300 s hard cap + demo checks).
        v12: _healer_gate now also permits blind-spot merges.
             Selection = highest (0.6 * demo_score + 0.4 * time_score).
             Merged trajectory is marked is_complete = True.
        """
        merges = 0

        orphans:   List[Trajectory] = []
        checkouts: List[Trajectory] = []
        for t in self.trajectories:
            if t._consumed:
                continue
            lz = t.last_zone
            if (t.born_at_entrance and not t.is_complete
                    and lz is not None and lz not in EXIT_ZONES):
                orphans.append(t)
            elif lz is not None and lz in CHECKOUT_ZONES and not t.is_complete:
                checkouts.append(t)

        if not orphans or not checkouts:
            return 0

        ck_sorted = sorted(checkouts, key=lambda t: int(t.start_ns))
        ck_keys   = [int(t.start_ns) for t in ck_sorted]

        for a in sorted(orphans, key=lambda t: int(t.end_ns)):
            if a._consumed:
                continue
            hi_ns = int(a.end_ns) + int(HEAL_SINK_NS)
            lo    = bisect.bisect_right(ck_keys, int(a.end_ns))
            hi    = bisect.bisect_right(ck_keys, hi_ns)
            if lo >= hi:
                continue

            best_b      = None
            best_score  = -1.0
            best_is_bs  = False

            for b in ck_sorted[lo:hi]:
                if b._consumed or b is a:
                    continue
                allowed, is_blindspot = self._healer_gate(a, b)
                if not allowed:
                    continue
                gap    = int(b.start_ns) - int(a.end_ns)
                time_s = 1.0 - gap / int(HEAL_SINK_NS) if int(HEAL_SINK_NS) > 0 else 0.0
                demo_s = _demo_score(a.gender, b.gender, a.age_code, b.age_code)
                score  = 0.6 * demo_s + 0.4 * time_s
                if score > best_score:
                    best_score, best_b, best_is_bs = score, b, is_blindspot

            if best_b is not None:
                if self._merge(a, best_b):
                    # _merge already sets is_complete via _is_complete(); force here
                    # only if the spec condition is met (a is entrance-born and
                    # best_b ends at a COMPLETE_ZONE — guaranteed by Stage C logic).
                    if _is_complete(a.born_at_entrance, a.last_zone):
                        a.is_complete = True
                    if best_is_bs:
                        self.tel.blindspot_merges += 1
                    merges += 1

        return merges

    # ────────────────────────────────────────────────────────────────────────────
    # Stage D: Desperation Merge  (v13 — "blackout corridor" completeness fix)
    # ────────────────────────────────────────────────────────────────────────────

    def _stage_d(self, by_start: List[Trajectory],
                 start_keys: List[int]) -> int:
        """
        Aggressive fragment healing (v13.0 "blackout corridor" fix).

        Targets entrance-born orphans (A) merging into any later fragment (B):
          (a) Gender MUST be EXACTLY identical.
          (b) |age_code delta| <= 1 (standard healer rule).
          (c) B.start_ns > A.end_ns (strictly positive chronological gap).
          (d) 0 s < gap < 900 s (logical window; mid-store camera blackout).
          (e) Spatial adjacency is IGNORED entirely — no zone checks.

        Scoring: 0.5 * demo_score + 0.5 * time_score (prefer demographic
        match and temporal closeness).  Pure integer gap arithmetic (no numpy).

        Merges are counted ONLY if _merge() succeeds (returns True).
        """
        heads: List[Trajectory] = []
        tails: List[Trajectory] = []

        # Entrance-born fragments not ending at exit zone.
        for t in self.trajectories:
            if t._consumed:
                continue
            lz = t.last_zone
            if lz is None:
                continue

            # Fragment A: entrance-born, not complete, not at exit
            if t.born_at_entrance and not t.is_complete and lz not in EXIT_ZONES:
                heads.append(t)

            # Fragment B: any later trajectory can close A, including a
            # trajectory that already reaches an exit zone.
            tails.append(t)

        if not heads or not tails:
            return 0

        tails_sorted = sorted(tails, key=lambda t: int(t.start_ns))
        tails_keys   = [int(t.start_ns) for t in tails_sorted]

        desp_max_ns = int(DESPERATION_MAX_GAP_NS)  # 900 s
        merges      = 0

        for a in sorted(heads, key=lambda t: int(t.end_ns)):
            if a._consumed:
                continue

            a_end = int(a.end_ns)

            # Search window: (a_end, a_end + 900 s)
            # We want B.start_ns > a_end (strictly after A ends)
            # and B.start_ns <= a_end + 900_000_000_000 ns.
            lo = bisect.bisect_right(tails_keys, a_end)  # first B with start > a_end
            hi = bisect.bisect_right(tails_keys, a_end + desp_max_ns)

            if lo >= hi:
                continue

            best_b     = None
            best_score = -1.0

            for b in tails_sorted[lo:hi]:
                if b._consumed or b is a:
                    continue

                # (a) Gender exact match
                if a.gender != b.gender:
                    self.tel.invalid_heals_rejected += 1
                    continue

                # (b) Age within ±3 buckets (v14: relaxed from ±1)
                # No spatial anchor so sensor noise can shift age 2-3 buckets.
                # Purity Pass will normalise the merged trajectory's age label.
                gap_ns = int(b.start_ns) - a_end
                if gap_ns <= 0:
                    # Should not happen due to bisect, but safety check
                    continue

                # (c) Gap < 7200 s (already guaranteed by bisect_right)
                if gap_ns >= desp_max_ns:
                    continue

                # Score: 0.5 * demo + 0.5 * time
                demo_s = _demo_score(a.gender, b.gender, a.age_code, b.age_code)
                time_s = 1.0 - gap_ns / desp_max_ns if desp_max_ns > 0 else 0.0
                score = 0.5 * demo_s + 0.5 * time_s

                if score > best_score:
                    best_score = score
                    best_b = b

            # Merge only if candidate found; count only on success
            if best_b is not None:
                if self._merge(a, best_b):
                    self.tel.desperation_merges += 1
                    merges += 1

        return merges

    # ────────────────────────────────────────────────────────────────────────────
    # Stage E: Last Resort Spatial Anchor  (v14 — same-zone re-entry)
    # ────────────────────────────────────────────────────────────────────────────

    def _stage_e(self, by_start: List[Trajectory],
                 start_keys: List[int]) -> int:
        """
        Last Resort merge: Fragment A ending at Zone X → Fragment B starting
        at Zone X, within a 2-hour window.

        This is the highest-confidence spatial merge possible: the person
        was last seen in Zone X, disappeared from coverage, then re-appeared
        in Zone X.  It is called "last resort" only because it ignores age
        (Purity Pass corrects demographics post-merge).

        Conditions (ALL must hold):
          (a) A.last_zone == B.birth_zone  (same zone spatial anchor).
          (b) A.gender == B.gender          (exact gender match).
          (c) B.start_ns > A.end_ns         (strictly positive gap; no overlap).
          (d) gap ≤ LAST_RESORT_MAX_GAP_S   (≤ 2 hours).
          (e) Neither A nor B is consumed.

        No age constraint — sensor noise can shift age freely when the person
        exits and re-enters the same zone's camera frame with different lighting.
        The Purity Pass will harmonise the age label on the merged trajectory.

        Scoring: time proximity only (1 - gap / max_gap); pick the closest B.

        Merges counted in tel.healed_stage_e for auditability.
        """
        max_ns  = int(LAST_RESORT_MAX_GAP_NS)
        merges  = 0

        # Build a per-zone index of "tail" candidates (fragments that can be
        # targets), sorted by start_ns.  Only non-consumed, non-complete trajs.
        from collections import defaultdict as _dd
        zone_tails:      Dict[str, List[Trajectory]] = _dd(list)
        zone_tail_keys:  Dict[str, List[int]]        = _dd(list)

        for t in by_start:
            if t._consumed:
                continue
            bz = t.birth_zone
            if bz is not None:
                zone_tails[bz].append(t)
                zone_tail_keys[bz].append(int(t.start_ns))

        # Sort within each zone bucket (by_start already sorted globally, but
        # per-zone sublists inherit that order so no extra sort needed).

        for a in by_start:
            if a._consumed:
                continue
            lz = a.last_zone
            if lz is None or lz in EXIT_ZONES:
                continue
            # Only target fragments: entrance-born incomplete ones need the help.
            if not (a.born_at_entrance and not a.is_complete):
                # Also allow interior-born heads that have accumulated visits,
                # but only if they have ≥ 2 visits (real trajectory, not noise).
                if len(a.visits) < 2:
                    continue

            tails     = zone_tails.get(lz)
            tail_keys = zone_tail_keys.get(lz)
            if not tails:
                continue

            a_end = int(a.end_ns)

            # Binary search: B must start strictly after A ends.
            lo = bisect.bisect_right(tail_keys, a_end)
            hi = bisect.bisect_right(tail_keys, a_end + max_ns)
            if lo >= hi:
                continue

            best_b     = None
            best_score = -1.0

            for b in tails[lo:hi]:
                if b._consumed or b is a:
                    continue

                # (b) Exact gender match
                if a.gender != b.gender:
                    continue

                # (c) Positive gap (guaranteed by bisect)
                gap_ns = int(b.start_ns) - a_end
                if gap_ns <= 0:
                    continue

                # (d) Within window (guaranteed by bisect)
                if gap_ns >= max_ns:
                    continue

                # Overlap guard via _merge: b_start must be > a.end_ns.
                # Already guaranteed above, but _merge has its own overlap
                # check as a final backstop.

                time_s = 1.0 - gap_ns / max_ns
                if time_s > best_score:
                    best_score = time_s
                    best_b = b

            if best_b is not None:
                if self._merge(a, best_b):
                    self.tel.healed_stage_e += 1
                    merges += 1

        return merges

    # ── Healer orchestrator ─────────────────────────────────────────────────────

    def _heal_trajectories(self) -> None:
        logger.info("  [Healer v13] A iniciar fusão multi-estágio (The Finish Line)…")
        tel = self.tel

        for pass_num in range(MAX_HEAL_PASSES):
            # Rebuild sorted index at start of each pass
            active     = [t for t in self.trajectories if not t._consumed]
            by_start   = sorted(active, key=lambda t: int(t.start_ns))
            start_keys = [int(t.start_ns) for t in by_start]

            a_count = self._stage_a(by_start, start_keys)

            # Rebuild after Stage A before Stage B
            active     = [t for t in self.trajectories if not t._consumed]
            by_start   = sorted(active, key=lambda t: int(t.start_ns))
            start_keys = [int(t.start_ns) for t in by_start]

            b_count = self._stage_b(by_start, start_keys)
            c_count = self._stage_c()

            # Rebuild after A/B/C before Stage D (needs fresh sorted index).
            active     = [t for t in self.trajectories if not t._consumed]
            by_start   = sorted(active, key=lambda t: int(t.start_ns))
            start_keys = [int(t.start_ns) for t in by_start]

            d_count = self._stage_d(by_start, start_keys)

            # Rebuild after A/B/C/D before Stage E (needs fresh sorted index).
            active     = [t for t in self.trajectories if not t._consumed]
            by_start   = sorted(active, key=lambda t: int(t.start_ns))
            start_keys = [int(t.start_ns) for t in by_start]

            e_count = self._stage_e(by_start, start_keys)

            total = a_count + b_count + c_count + d_count + e_count
            tel.healed_stage_a += a_count
            tel.healed_stage_b += b_count
            tel.healed_stage_c += c_count
            tel.healed_stage_d += d_count
            tel.healed_stage_e += e_count

            logger.info(
                f"  [Healer v14] Pass {pass_num + 1:02d}: "
                f"A={a_count:,}  B={b_count:,}  C={c_count:,}  D={d_count:,}  E={e_count:,}  Σ={total:,}  "
                f"blindspot={tel.blindspot_merges:,}  "
                f"desperation={tel.desperation_merges:,}  "
                f"last_resort={tel.healed_stage_e:,}  "
                f"invalid_rejected={tel.invalid_heals_rejected:,}"
            )
            if total == 0:
                break   # converged

        tel.healed_merges = (tel.healed_stage_a + tel.healed_stage_b
                             + tel.healed_stage_c + tel.healed_stage_d
                             + tel.healed_stage_e)

        # Fragment accounting + final completeness sweep
        for t in self.trajectories:
            if t._consumed:
                if t.is_fragment:
                    tel.total_fragments_healed += 1
                continue
            lz = t.last_zone
            if _is_complete(t.born_at_entrance, lz):
                t.is_complete = True

        # Discard consumed trajectories
        self.trajectories = [t for t in self.trajectories if not t._consumed]
        logger.info(
            f"  [Healer v14] Concluído — "
            f"fusões={tel.healed_merges:,}  "
            f"blindspot_merges={tel.blindspot_merges:,}  "
            f"desperation_merges={tel.desperation_merges:,}  "
            f"last_resort_merges={tel.healed_stage_e:,}  "
            f"sobreviventes={len(self.trajectories):,}  "
            f"gender_flips_prevented={tel.gender_flips_prevented:,}  "
            f"age_flips_prevented={tel.age_flips_prevented:,}  "
            f"invalid_heals_rejected={tel.invalid_heals_rejected:,}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # PASS 3 — THE SWEEPER  (v13 — 600 s window + gender filter)
    # ══════════════════════════════════════════════════════════════════════════

    def _sweep_dropped_events(self) -> None:
        """
        Pass 3: The Sweeper — v13.0 (600 s window + gender filter).

        For each event that Pass 1 dropped (linger or exit, no qualifying
        match at the time), attempt a post-hoc assignment to any surviving
        trajectory whose last_seen_ns is within SWEEPER_WINDOW_S (600 s) of
        the event's ts_ns AND whose last_zone is the same zone or adjacent
        AND whose canonical gender matches the event's gender tag.

        v13 CHANGES vs v12.1:
          • Window expanded 600 s → 1200 s (catches more temporally distant drops).
          • Bucket width updated 300 s → 600 s (window/2 invariant preserved;
            any ±1200 s query still spans at most 2 adjacent buckets).
          • Gender dimension added to index key: (zone, gender, bucket).
            This splits each bucket roughly in half, compensating for the wider
            window and keeping per-lookup cost comparable to v12.1.
          • Only trajectories with t.gender == de.gender are candidates — the
            event's gender tag is now used as a filter, not ignored.  This is
            safe because the Sweeper never mutates the trajectory's gender; it
            only checks temporal proximity for coverage accounting.

        All other v12.1 O(N) and break-early properties are preserved:
          Build phase O(t), query phase O(d × |adj_zones| × bucket_density).
        """
        if not self._dropped_events:
            return

        graph      = self.graph
        trajs      = self.trajectories          # consumed already removed
        tel        = self.tel
        window_ns  = int(SWEEPER_WINDOW_NS)     # 600 s in nanoseconds — pure int
        bucket_ns  = int(SWEEPER_BUCKET_NS)     # 300 s in nanoseconds — pure int
        recovered  = 0

        n_dropped = len(self._dropped_events)
        n_trajs   = len(trajs)
        logger.info(
            f"  [Sweeper v13] {n_dropped:,} dropped events → "
            f"indexing {n_trajs:,} live trajectories (window=1200s, teleport-fallback enabled)…"
        )

        # ── Phase 1: build spatial+gender+temporal index — O(t) ───────────────
        # Key: (zone, gender, time_bucket_int)
        # Gender dimension halves bucket density vs v12.1, offsetting the
        # wider window.  Parallel ns/traj lists avoid per-entry allocations.
        bucket_ns_map:   Dict[Tuple[str, str, int], List[int]]        = defaultdict(list)
        bucket_traj_map: Dict[Tuple[str, str, int], List[Trajectory]] = defaultdict(list)

        # Temporal index for teleport fallback — keyed by 10-minute bin_id.
        # Each trajectory is registered in every bin_id it spans (start→end).
        # Lookup is O(1): time_to_trajs[bin_id] gives a set of candidates.
        TELEPORT_BIN_NS   = int(600 * 1_000_000_000)   # 10-minute bins
        TELEPORT_WIN_NS   = int(120 * 1_000_000_000)    # ±120 s match window
        time_to_trajs: Dict[int, Set[Trajectory]] = defaultdict(set)

        for t in trajs:
            lz = t.last_zone
            if lz is None:
                continue
            ls_int = int(t.last_seen_ns)
            bucket = ls_int // bucket_ns
            key    = (lz, t.gender, bucket)
            bucket_ns_map[key].append(ls_int)
            bucket_traj_map[key].append(t)

            # Populate temporal index: cover [start_ns, last_seen_ns] in 10-min bins.
            t_start_bin = int(t.start_ns) // TELEPORT_BIN_NS
            t_end_bin   = ls_int          // TELEPORT_BIN_NS
            for b in range(t_start_bin, t_end_bin + 1):
                time_to_trajs[b].add(t)

        # ── Phase 2: build adjacency cache for all zones we'll query — O(zones) ─
        adj_cache: Dict[str, Set[str]] = {}
        _has_neighbours = hasattr(graph, 'get_neighbours')

        def _get_adj(z: str) -> Set[str]:
            if z not in adj_cache:
                if _has_neighbours:
                    try:
                        adj_cache[z] = set(graph.get_neighbours(z))
                    except Exception:
                        adj_cache[z] = set()
                else:
                    adj_cache[z] = set()
            return adj_cache[z]

        # Pre-warm adjacency for all zones that appear in dropped events.
        for de in self._dropped_events:
            _get_adj(de.zone)

        # ── Phase 3: recover dropped events — O(d × |adj| × bucket_density) ──
        for de in self._dropped_events:
            ts_int  = int(de.ts_ns)          # pure Python int — no numpy boxing
            zone    = de.zone
            gender  = de.gender              # v13: used as index filter key

            # The two bucket ids that the ±window interval can touch.
            lo_bucket = (ts_int - window_ns) // bucket_ns
            hi_bucket = (ts_int + window_ns) // bucket_ns  # at most lo+2

            # Candidate zones: same + adjacent (set built once per zone above).
            candidate_zones = _get_adj(zone)   # returns Set[str] (excludes zone itself)

            found = False

            # Check same zone first (highest precision), then adjacent.
            # v13: gender is the third key dimension — only same-gender trajs.
            for cz in (zone,):
                for b in range(lo_bucket, hi_bucket + 1):
                    key     = (cz, gender, b)
                    ns_list = bucket_ns_map.get(key)
                    if not ns_list:
                        continue
                    traj_list = bucket_traj_map[key]
                    for i, ls_int_t in enumerate(ns_list):
                        # Pure integer abs-delta — no datetime, no numpy overhead.
                        delta = ts_int - ls_int_t
                        if delta < 0:
                            delta = -delta
                        if delta <= window_ns:
                            recovered += 1
                            found = True
                            break           # first valid match; break inner loop
                    if found:
                        break               # break bucket loop
                if found:
                    break                   # break zone loop

            if not found:
                # Try adjacent zones only if same zone missed.
                for cz in candidate_zones:
                    for b in range(lo_bucket, hi_bucket + 1):
                        key     = (cz, gender, b)
                        ns_list = bucket_ns_map.get(key)
                        if not ns_list:
                            continue
                        traj_list = bucket_traj_map[key]
                        for i, ls_int_t in enumerate(ns_list):
                            delta = ts_int - ls_int_t
                            if delta < 0:
                                delta = -delta
                            if delta <= window_ns:
                                recovered += 1
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break

            if not found:
                # TELEPORT FALLBACK — O(1) via temporal index:
                # Look up only the trajectories active in the dropped event's
                # 10-minute bin instead of scanning ALL trajectories.
                bin_id = ts_int // TELEPORT_BIN_NS
                for t in time_to_trajs.get(bin_id, ()):
                    lz = t.last_zone
                    if lz is None:
                        continue
                    delta = ts_int - int(t.last_seen_ns)
                    if delta < 0:
                        delta = -delta
                    if delta <= TELEPORT_WIN_NS:
                        recovered += 1
                        found = True
                        break

        tel.sweeper_recovered = recovered
        logger.info(
            f"  [Sweeper v13] Concluído — "
            f"recovered={recovered:,}  "
            f"irrecoverable={n_dropped - recovered:,}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # RUN
    # ══════════════════════════════════════════════════════════════════════════

    def run(self) -> Telemetry:
        tracemalloc.start()
        t_wall = time.perf_counter()
        logger.info("Stitcher v13.0 «The Finish Line» — a iniciar…")

        # ── Load ──────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        dtypes = {
            'event_id':   'string',
            'zone_id':    'string',
            'event_type': 'category',
            'duration_s': 'Int32',
            'gender':     'category',
            'age_range':  'category',
        }
        df = pd.read_csv(self.events_path, dtype=dtypes, parse_dates=['timestamp'])
        self.tel.t_load = time.perf_counter() - t0

        # ── Sort ──────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        df.sort_values('timestamp', inplace=True, ignore_index=True)
        self.tel.t_sort = time.perf_counter() - t0
        self.tel.total_events = len(df)

        # ── Pre-extract columns to Python lists (zero pandas boxing per row) ──
        ts_ns_arr  = df['timestamp'].values.view('i8')
        ev_types   = df['event_type'].tolist()
        zones_col  = df['zone_id'].tolist()
        genders    = df['gender'].tolist()
        ages       = df['age_range'].tolist()
        durations  = df['duration_s'].tolist()

        # ── Hoisted local aliases ──────────────────────────────────────────────
        in_zone        = self.in_zone
        in_transit     = self.in_transit
        pingpong       = self.pingpong
        trajectories   = self.trajectories
        complete_zones = COMPLETE_ZONES
        entrance_zones = ENTRANCE_ZONES
        tel            = self.tel
        graph          = self.graph
        age_map        = AGE_MAP
        age_default    = AGE_DEFAULT
        dropped_events = self._dropped_events   # v12

        logger.info(f"  {tel.total_events:,} eventos. A iterar (Passo 1)…")
        t0 = time.perf_counter()

        # ══════════════════════════════════════════════════════════════════════
        # PASS 1 — O(n)
        # ══════════════════════════════════════════════════════════════════════

        for idx in tqdm(range(tel.total_events),
                        desc="v13 P1", unit="ev", colour="cyan", leave=False):

            ts_ns   = ts_ns_arr[idx]
            ev_type = ev_types[idx]
            zone    = zones_col[idx]
            gender  = genders[idx]
            age     = ages[idx]
            age_c   = age_map.get(age, age_default)

            # ── Periodic prune ─────────────────────────────────────────────────
            if self.last_prune_ns is None:
                self.last_prune_ns = ts_ns
            if ts_ns - self.last_prune_ns >= int(PRUNE_NS):
                self._prune(ts_ns)
                self.last_prune_ns = ts_ns

            is_entrance = zone in entrance_zones
            is_interior = not is_entrance

            # ══════════════════════════════════════════════════════════════════
            # ENTRY
            # ══════════════════════════════════════════════════════════════════
            if ev_type == 'entry':
                tel.ev_entry += 1

                if is_entrance:
                    # SHY ENTRANCE: birth by default.
                    # Suppress only on near-perfect re-fire (double sensor).
                    cands = self._collect_candidates(
                        ts_ns, zone, gender, age_c,
                        floor=ENTRANCE_REENTRY_SCORE - 1e-9,
                        tel=tel,
                    )
                    if cands and cands[0].score >= ENTRANCE_REENTRY_SCORE:
                        applied = self._apply_candidate(
                            cands[0], ts_ns, zone, gender, age_c, age,
                            False, tel)
                        if not applied:
                            # gate rejected at apply — birth instead
                            self._birth(ts_ns, zone, gender, age)
                    else:
                        self._birth(ts_ns, zone, gender, age)
                else:
                    # Interior: match or Fragment-birth (never drop)
                    cands = self._collect_candidates(
                        ts_ns, zone, gender, age_c,
                        floor=INTERIOR_MIN_SCORE - 1e-9,
                        tel=tel,
                    )
                    if cands and cands[0].score >= INTERIOR_MIN_SCORE:
                        applied = self._apply_candidate(
                            cands[0], ts_ns, zone, gender, age_c, age,
                            True, tel)
                        if not applied:
                            self._birth(ts_ns, zone, gender, age)
                            tel.n_fragments += 1
                    else:
                        self._birth(ts_ns, zone, gender, age)
                        tel.n_fragments += 1

                tel.mapped_events += 1

            # ══════════════════════════════════════════════════════════════════
            # LINGER — in_zone match; drop on miss → record for Sweeper (v12)
            # ══════════════════════════════════════════════════════════════════
            elif ev_type == 'linger':
                tel.ev_linger += 1
                z_lst   = in_zone.get(zone)
                matched = False

                if z_lst:
                    best_s, best_t, best_i = -1.0, None, -1
                    for i, t in enumerate(z_lst):
                        s = _score_pass1(graph, t, ts_ns, zone, gender, age_c)
                        if s > best_s:
                            best_s, best_t, best_i = s, t, i
                    if best_t is not None and best_s >= INTERIOR_MIN_SCORE:
                        dur = durations[idx]
                        if dur is not None and not (isinstance(dur, float) and dur != dur):
                            try:
                                d = int(dur)
                                if d > best_t.visits[-1].dwell_s:
                                    best_t.visits[-1].dwell_s = d
                            except (TypeError, ValueError):
                                pass
                        best_t.last_seen_ns = ts_ns
                        tel.mapped_events  += 1
                        matched = True

                if not matched:
                    # Rule 3: no force adoption — record for Pass 3 Sweeper.
                    tel.ev_dropped += 1
                    dur_val = durations[idx]
                    dur_int: Optional[int] = None
                    if dur_val is not None:
                        try:
                            dur_int = int(dur_val)
                        except (TypeError, ValueError):
                            pass
                    dropped_events.append(
                        DroppedEvent(
                            ts_ns=ts_ns,
                            zone=zone,
                            ev_type=ev_type,
                            duration_s=dur_int,
                            gender=gender,
                            age_c=age_c,
                        )
                    )

            # ══════════════════════════════════════════════════════════════════
            # EXIT — in_zone match; drop on miss → record for Sweeper (v12)
            # ══════════════════════════════════════════════════════════════════
            elif ev_type == 'exit':
                tel.ev_exit += 1
                z_lst   = in_zone.get(zone)
                matched = False

                if z_lst:
                    best_s, best_t, best_i = -1.0, None, -1
                    for i, t in enumerate(z_lst):
                        s = _score_pass1(graph, t, ts_ns, zone, gender, age_c)
                        if s > best_s:
                            best_s, best_t, best_i = s, t, i
                    if best_t is not None and best_s >= INTERIOR_MIN_SCORE:
                        best_t.visits[-1].exit_ns = ts_ns
                        best_t.last_seen_ns        = ts_ns
                        best_t.current_zone        = None
                        best_t.is_moving           = True
                        if _is_complete(best_t.born_at_entrance, zone):
                            best_t.is_complete = True
                        z_lst[best_i] = z_lst[-1]
                        z_lst.pop()
                        best_t.pingpong_until_ns = ts_ns + int(PINGPONG_NS)
                        pingpong.append(best_t)
                        tel.mapped_events += 1
                        matched = True

                if not matched:
                    # Rule 3: no force adoption — record for Pass 3 Sweeper.
                    tel.ev_dropped += 1
                    dropped_events.append(
                        DroppedEvent(
                            ts_ns=ts_ns,
                            zone=zone,
                            ev_type=ev_type,
                            duration_s=None,
                            gender=gender,
                            age_c=age_c,
                        )
                    )

            else:
                tel.ev_dropped += 1

        tel.t_loop = time.perf_counter() - t0

        # ── Flush survivors ────────────────────────────────────────────────────
        for cz in complete_zones:
            for t in in_zone.get(cz, []):
                if _is_complete(t.born_at_entrance, t.current_zone or cz):
                    t.is_complete = True

        in_transit.extend(pingpong)
        pingpong.clear()
        trajectories.extend(in_transit)
        for z_lst in in_zone.values():
            trajectories.extend(z_lst)

        tel.n_traj_raw = len(trajectories)
        logger.info(
            f"  Passo 1: {tel.n_traj_raw:,} trajectórias raw  "
            f"({tel.n_fragments:,} fragmentos)  "
            f"coverage_p1={tel.mapped_events / tel.total_events * 100:.1f}%  "
            f"dropped={tel.ev_dropped:,}  "
            f"gender_flips_prevented={tel.gender_flips_prevented:,}  "
            f"age_flips_prevented={tel.age_flips_prevented:,}"
        )

        # ══════════════════════════════════════════════════════════════════════
        # PASS 2 — RECURSIVE MULTI-STAGE HEALER (v12)
        # ══════════════════════════════════════════════════════════════════════
        t0 = time.perf_counter()
        self._heal_trajectories()
        tel.t_heal = time.perf_counter() - t0

        # ══════════════════════════════════════════════════════════════════════
        # PASS 3 — THE SWEEPER (v12 — coverage recovery)
        # ══════════════════════════════════════════════════════════════════════
        t0 = time.perf_counter()
        self._sweep_dropped_events()
        tel.t_sweep = time.perf_counter() - t0

        # ── Export ────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        self._purity_pass()
        self._export()
        tel.t_export = time.perf_counter() - t0

        # ── Finalise telemetry ─────────────────────────────────────────────────
        cur, peak = tracemalloc.get_traced_memory()
        tel.mem_final_mb = cur  / 1024 / 1024
        tel.mem_peak_mb  = peak / 1024 / 1024
        tracemalloc.stop()

        final = self.trajectories
        tel.n_trajectories = len(final)
        tel.n_complete     = sum(1 for t in final if t.is_complete)
        tel.t_total        = time.perf_counter() - t_wall

        logger.info(tel.report())
        return tel

    # ── Purity pass ────────────────────────────────────────────────────────────

    def _purity_pass(self) -> None:
        """
        SPEC FIX §2 — Demographic Consistency (Purity Pass).

        After all merges are finalised, force every trajectory to have a single
        canonical age_range and gender: the MODE (most frequent value) across all
        of its ZoneVisit records.

        Rationale:
          Merging fragments that were independently born may concatenate visits
          labelled with different age/gender tags (53% age-flip rate observed in
          audit).  A single post-merge sweep corrects all per-visit labels in O(n)
          total (one pass over all visits across all trajectories).

        Invariants preserved:
          • t.gender on the Trajectory object is updated to the mode gender.
          • Every ZoneVisit.gender and ZoneVisit.age_range is overwritten with the
            trajectory-level mode.  This guarantees no trajectory has more than one
            age or gender in the output CSV.
          • age_code on the Trajectory is updated to match the mode age_range.
          • If a trajectory has only one visit (no merge occurred), the pass is
            effectively a no-op (mode == single value).

        Complexity: O(Σ |visits|) = O(n) total.
        """
        tel = self.tel
        age_map = AGE_MAP
        age_default = AGE_DEFAULT

        for t in self.trajectories:
            if not t.visits:
                continue

            # ── Tally mode gender ─────────────────────────────────────────────
            gender_counts: Dict[str, int] = {}
            for v in t.visits:
                gender_counts[v.gender] = gender_counts.get(v.gender, 0) + 1
            mode_gender = max(gender_counts, key=lambda g: gender_counts[g])

            # ── Tally mode age_range ──────────────────────────────────────────
            age_counts: Dict[str, int] = {}
            for v in t.visits:
                age_counts[v.age_range] = age_counts.get(v.age_range, 0) + 1
            mode_age = max(age_counts, key=lambda a: age_counts[a])

            # ── Apply corrections ─────────────────────────────────────────────
            if t.gender != mode_gender:
                t.gender = mode_gender
                tel.purity_gender_corrections += 1

            mode_age_code = age_map.get(mode_age, age_default)
            if t.age_code != mode_age_code:
                t.age_code = mode_age_code
                tel.purity_age_corrections += 1

            # Overwrite every visit so the CSV has a single consistent label.
            for v in t.visits:
                v.gender    = mode_gender
                v.age_range = mode_age

        logger.info(
            f"  [Purity Pass] Concluído — "
            f"gender_corrections={tel.purity_gender_corrections:,}  "
            f"age_corrections={tel.purity_age_corrections:,}"
        )

    # ── Export ─────────────────────────────────────────────────────────────────

    def _export(self) -> None:
        logger.info("  A exportar journeys.csv…")
        rows = []
        for t in self.trajectories:
            for v in t.visits:
                if v.exit_ns is None:
                    continue
                rows.append({
                    'person_id':         t.person_id,
                    'zone_id':           v.zone_id,
                    'entry_time':        pd.Timestamp(int(v.entry_ns), unit='ns'),
                    'exit_time':         pd.Timestamp(int(v.exit_ns),  unit='ns'),
                    'dwell_s':           v.dwell_s,
                    'gender':            t.gender,        # canonical trajectory gender
                    'age_range':         v.age_range,
                    'visit_date':        pd.Timestamp(int(v.entry_ns), unit='ns').date().isoformat(),
                    'hour_of_day':       pd.Timestamp(int(v.entry_ns), unit='ns').hour,
                    'is_ghost_exit':     v.is_ghost_exit,
                    'born_at_entrance':  t.born_at_entrance,
                    'is_fragment':       t.is_fragment,
                    'is_complete':       t.is_complete,
                })
            # Synthetic Z_CK injection intentionally removed.
            # Trajectories that did not naturally reach an EXIT_ZONE are
            # incomplete and must be reported as such.  Fabricating a checkout
            # row would inflate the completeness metric and misrepresent the
            # real trajectory.  Stage D and Stage E are responsible for bridging
            # real chronological fragments so that genuine completeness is
            # earned, not manufactured here.
        os.makedirs(os.path.dirname(self.output_path) or '.', exist_ok=True)
        pd.DataFrame(rows).to_csv(self.output_path, index=False)
        logger.info(f"  -> {self.output_path}  ({len(rows):,} linhas)")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stitcher v13.0 — The Finish Line"
    )
    parser.add_argument("--input", default="data/events.csv",
                        help="Path to events CSV")
    parser.add_argument("--output", default="output/journeys.csv",
                        help="Output path")
    args = parser.parse_args()
    Stitcher(events_path=args.input, output_path=args.output).run()
