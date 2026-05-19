# TP1: From Raw Detections to Real Intelligence

**LIACD (Laboratórios de Introdução à Análise e Ciência de Dados) — 2025/2026**

---

## 📋 Overview

This project implements a production-grade pipeline for converting raw optical detection events from a retail store into meaningful customer journey trajectories. The system employs a **strict separation architecture** where deterministic Python modules handle mathematical processing while a local LLM (via Ollama) generates interpretative insights.

### Key Metrics
- **250,015 raw events** processed across 7 days and 23 store zones
- **3,960 unique visitors** reconstructed via trajectory stitching
- **100% numeric precision** in LLM-generated insights (anti-hallucination scrubber)
- **96.46% consistency** in trajectory temporal and demographic stability

---

## 📁 Project Structure

```
tp1/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── evaluate.py                        # Pipeline audit harness (v14.0)
│
├── data/
│   └── events.csv                     # Raw detection events (entry/linger/exit)
│
├── src/
│   ├── stitcher.py                    # Multi-stage trajectory assembly (Pass 1, Healer, Sweeper)
│   ├── analytics.py                   # Statistical aggregation & anomaly detection
│   ├── insights.py                    # LLM interface & prompt engineering
│   ├── report.py                      # Markdown report formatting
│   └── utils/
│       ├── graph.py                   # Zone topology & adjacency graphs
│       └── logger.py                  # Unified logging utilities
│
├── prompts/
│   ├── prompt_estrategia_A_zero_shot.txt      # Direct JSON-to-insights
│   └── prompt_estrategia_B_few_shot.txt       # Context-anchored examples
│
└── output/
    ├── journeys.csv                   # Stitched customer trajectories
    ├── metrics.json                   # Aggregated operational metrics
    ├── insights.json                  # LLM insights with hallucination_report
    ├── RELATORIO_TECNICO.md           # Full 8-section technical report (Portuguese)
    └── evaluation_report.json          # Audit metrics (Coverage/Completeness/Consistency/Precision)
```

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.11+**
- **Ollama** running locally with `llama3.1:8b` model loaded
- Dependencies listed in `requirements.txt`

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

#### 1. Execute Full Pipeline (Recommended)
```bash
# Runs all stages: stitching → analytics → insights generation → evaluation
python src/stitcher.py
python src/analytics.py
python src/insights.py
python evaluate.py
```

#### 2. Individual Module Execution

**Stitching (Pass 1 + Healer + Sweeper + Purity):**
```bash
python src/stitcher.py --events data/events.csv --output output/journeys.csv
```

**Analytics (Aggregation + Anomaly Detection):**
```bash
python src/analytics.py --journeys output/journeys.csv --output output/metrics.json
```

**Insights (LLM-based synthesis):**
```bash
python src/insights.py --metrics output/metrics.json --output output/insights.json
```

**Evaluation (Audit Report):**
```bash
python evaluate.py \
  --events data/events.csv \
  --journeys output/journeys.csv \
  --insights output/insights.json \
  --output output/evaluation_report.json
```

---

## 🔧 Core Modules

### `src/stitcher.py`
**Purpose:** Reconstruct customer journeys from fragmented detection events.

**Algorithm Overview:**
1. **Pass 1 (O(n) Linear):** Initial trajectory assignment using K-best matching with weighted scoring
   - Time proximity (40%)
   - Zone adjacency (38%)
   - Demographics (22%)

2. **Healer (Iterative):** Multi-stage fusion of orphaned fragments
   - Stage A: Strict spatio-temporal merging (600s window)
   - Stage B: Demographic bridge (240s window)
   - Stage C: Sink recovery (checkout consolidation)
   - Stage D: Desperation merge (2-hour blackout tolerance)
   - Stage E: Spatial anchor reentry (same zone recovery)

3. **Sweeper:** Recover dropped events via 3D indexing (zone, gender, time_bucket)

4. **Purity Pass:** Stabilize demographics by mode selection

**Key Hyperparameters:**
- `INTERIOR_MIN_SCORE = 0.50` (acceptance threshold)
- `MAX_HEAL_PASSES = 15` (iteration limit)
- `K_BEST = 7` (candidate pool size)

### `src/analytics.py`
**Purpose:** Aggregate metrics and detect anomalies.

**Outputs:**
- Traffic volume by hour/zone
- Conversion funnel (entry → checkout)
- Dwell time (median + P90, outlier-robust)
- Operational anomalies (2-sigma detection)

**Anomaly Detection:**
Uses first 6 days as baseline, tests Day 7 against normal distribution:
$$\text{Anomalous} \iff |X_{\text{observed}} - \mu| > 2\sigma$$

### `src/insights.py`
**Purpose:** Generate executive insights via LLM with anti-hallucination protection.

**Two Prompting Strategies:**
1. **Zero-Shot (Strategy A):** Direct JSON → insights (66.7% precision baseline)
2. **Few-Shot (Strategy B):** Example-anchored generation (68.8% precision baseline)

**Anti-Hallucination Scrubber:**
- Extracts all numeric tokens from LLM output
- Cross-validates against `metrics.json` (±10% tolerance)
- Removes/regenerates unverified claims

**Final Precision:** 100.0% after scrubbing

### `evaluate.py`
**Purpose:** Audit pipeline against 4 specification metrics.

**Metrics:**
| Metric | Formula | Target | Actual |
|--------|---------|--------|--------|
| **Coverage** | (mapped events / total events) × 100 | ≥85% | 78.07% |
| **Completeness** | (proper entry/exit / unique persons) × 100 | ≥70% | 60.35% |
| **Consistency** | (no overlaps + stable demographics / unique persons) × 100 | ≥95% | 96.46% ✓ |
| **Numeric Precision** | (verified numbers / extracted numbers) × 100 | ≥90% | 100.00% ✓ |

**Root Causes of Shortfalls:**
- **Coverage (78.07%):** 4.2% orphaned events + 2.8% blackout periods
- **Completeness (60.35%):** 32% non-official entry points + 8% untracked exits

---

## 📊 Data Formats

### Input: `data/events.csv`
```csv
timestamp,zone_id,event_type,gender,age_range
2026-05-13 08:15:30,Z_E1,entry,male,adult
2026-05-13 08:15:45,Z_N1,linger,male,adult
2026-05-13 08:16:02,Z_N1,exit,male,adult
...
```

### Output: `output/journeys.csv`
```csv
person_id,zone_id,entry_time,exit_time,gender,age_range
P_001,Z_E1,2026-05-13 08:15:30,2026-05-13 08:15:42,male,adult
P_001,Z_N1,2026-05-13 08:15:45,2026-05-13 08:16:02,male,adult
...
```

### Metrics: `output/metrics.json`
```json
{
  "traffic": {
    "unique_visitors": 3960,
    "total_visits": 57886,
    "peak_hour": "12:00",
    "peak_users": 518
  },
  "conversion": {
    "checkout_reached": 1433,
    "conversion_rate": 36.19
  },
  "dwell_time": {
    "median_seconds": 32.0,
    "p90_seconds": 255.0
  },
  "anomalies": [...]
}
```

---

## 🛠️ Configuration

### Prompting Strategy Selection
**In `src/insights.py`:**
```python
STRATEGY = "few_shot"  # Options: "zero_shot", "few_shot"
```

### Ollama Connection
**Default:** `http://localhost:11434`  
**Override in `src/insights.py`:**
```python
client = ollama.Client(host="your_host:port")
```

### Zone Graph Topology
**In `src/utils/graph.py`:**
Update `ZoneGraph` adjacency matrix if store layout changes.

---

## 📈 Performance & Scalability

| Metric | Value |
|--------|-------|
| Total Processing Time | ~8 minutes |
| Memory Peak | ~850 MB (3.96K trajectories) |
| Events/Second Throughput | ~520 events/sec |
| Scalability | Linear (O(n) stitching) |

---

## ⚠️ Known Limitations

1. **Blackout Sensitivity:** Simultaneous camera failure across multiple zones destroys trajectory continuity
2. **Demographic Volatility:** 8% gender + 12% age classification error can cause false merges
3. **Missing Ground Truth:** No transaction-level validation (POS integration required)
4. **Zone Coverage:** Unmonitored exits (emergency doors, staff passages) reduce completeness to 60%

---

## 🔍 Troubleshooting

### Issue: Ollama Connection Refused
```bash
# Verify Ollama is running
ollama list

# Pull model if missing
ollama pull llama3.1:8b
```

### Issue: Low Coverage / Completeness
- Check zone graph topology in `utils/graph.py`
- Review blackout periods in logs
- Inspect fragment statistics in stitcher output

### Issue: High Hallucination Rate (< 90% precision)
- Switch to `few_shot` strategy
- Increase `HALLUCINATION_TOL` tolerance (current: ±10%)
- Review prompt templates in `prompts/`

---

## 📚 Technical Documentation

**Full Technical Report:** See [`output/RELATORIO_TECNICO.md`](output/RELATORIO_TECNICO.md)

### Key Sections:
- **§7:** Problem formulation & architecture decisions
- **§8:** Stitching algorithm (Pass 1, Healer stages, Sweeper)
- **§9:** Analytics pipeline & dwell time robustness
- **§10:** LLM prompting strategies & anti-hallucination
- **§11:** Evaluation results & honest assessment of limitations
- **§12:** Future research directions (HMM, POS integration, group detection)

---

## 📝 References

### Dependencies
- **pandas** 2.2.2: Tabular data processing
- **numpy** 1.26.4: Vectorized operations & binary search
- **networkx** 3.3: Zone topology graphs
- **pydantic** 2.7.1: Data validation
- **ollama** 0.2.1: LLM interface

### Key Publications Referenced
- Multi-Object Tracking (MOT) literature
- Statistical process control (SPC) for anomaly detection
- Prompt engineering best practices for local LLMs

---

## 📄 License & Attribution

**Academic Project:** LIACD 2025/2026  
**Author:** [Student Name]  
**Institution:** [University Name]

---

## 🤝 Support

For issues or improvements, refer to the diagnostic output in:
- `iteration_log.txt` (per-stage progress)
- `output/evaluation_report.json` (metric details)
- Console logs (grepped for ✓/✗ status indicators)

---

**Last Updated:** 19 May 2026  
**Pipeline Version:** 1.0 (Stable)
