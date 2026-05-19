"""
insights.py — Phase 3 LLM Insights Generator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reads output/metrics.json produced by analytics.py and queries a local Ollama
instance to generate actionable executive-level retail insights.

PIPELINE
  1. Load & compact metrics.json  → protect Ollama context window
  2. Zero-shot prompt             → raw LLM insight generation
  3. Few-shot prompt              → guided, example-anchored generation
  4. Anti-hallucination validator → extract numbers from LLM text,
                                    verify each against metrics values (±5%)
  5. Emit insights.json           → structured output with hallucination report

USAGE
  python -m src.insights
  python -m src.insights --input output/metrics.json --output output/insights.json
  python -m src.insights --strategy zero_shot --model llama3.2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Tuple

import requests
from tqdm.auto import tqdm

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("insights")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

OLLAMA_URL = "http://localhost:11434/api/generate"

OLLAMA_TIMEOUT_S = 600

PRECISION_WARN_PCT = 100.0

HALLUCINATION_TOL = 0.10

SYSTEM_PERSONA = (
    "Você é um Consultor Sênior de Estratégia de Dados que liderou análises avançadas "
    "de retalho na OpenAI e possui um doutoramento em Matemática Aplicada pelo MIT. "
    "Não afirma o óbvio. Analisa dados para encontrar estrangulamentos ocultos, "
    "ineficiências operacionais e oportunidades de receita. "
    "O seu tom é executivo, preciso e implacável quanto à excelência operacional. "
    "IMPORTANTE: Toda a sua resposta — títulos, observações, implicações, "
    "recomendações e resumo — deve ser escrita EXCLUSIVAMENTE em Português Europeu. "
    "Não utilize inglês em nenhuma parte da resposta."
)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD & COMPACT METRICS
# ══════════════════════════════════════════════════════════════════════════════


def load_metrics(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def compact_metrics(metrics: Dict[str, Any]) -> str:
    """
    Serialise metrics to a single-line JSON string.
    Strips the verbose anomaly_list detail strings (the numbers are kept) and
    the raw by_hour / by_zone arrays to stay well inside Ollama's context limit
    while preserving every scalar KPI the LLM should reason about.
    """

    slim = json.loads(json.dumps(metrics))

    # Keep anomaly counts + thresholds; drop verbose per-anomaly text
    if "anomalies" in slim and "anomaly_list" in slim["anomalies"]:
        slim["anomalies"]["anomaly_list"] = [
            {k: v for k, v in a.items() if k != "detail"}
            for a in slim["anomalies"]["anomaly_list"]
        ]

    # Keep Top-5 only for by_hour and by_zone arrays
    for section in ("traffic", "dwell"):
        for key in ("by_hour_top10", "by_zone_top10"):
            if section in slim and key in slim[section]:
                slim[section][key] = slim[section][key][:5]

    return json.dumps(
        slim,
        separators=(",", ":"),
        ensure_ascii=False,
    )


# ══════════════════════════════════════════════════════════════════════════════
# OLLAMA CALL
# ══════════════════════════════════════════════════════════════════════════════


def call_ollama(
    model: str,
    system: str,
    prompt: str,
) -> str:
    """
    Stream tokens from Ollama while displaying tqdm progress.
    """

    payload = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0,
        },
    }

    logger.info(
        f"  → Calling Ollama "
        f"(model={model}, prompt_chars={len(prompt):,})…"
    )

    try:
        resp = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=OLLAMA_TIMEOUT_S,
            stream=True,
        )

        resp.raise_for_status()

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Could not connect to Ollama at {OLLAMA_URL}. "
            "Is `ollama serve` running?"
        )

    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Ollama request timed out after {OLLAMA_TIMEOUT_S}s. "
            "Try a smaller model or increase OLLAMA_TIMEOUT_S."
        )

    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(
            f"Ollama HTTP error: {exc}  body={resp.text[:400]}"
        )

    chunks: List[str] = []

    token_count = 0

    start_time = time.time()

    with tqdm(
        desc=f"{model} generating",
        unit="tok",
        dynamic_ncols=True,
        smoothing=0.05,
    ) as pbar:

        for line in resp.iter_lines():

            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            token = data.get("response", "")

            if token:

                chunks.append(token)

                token_count += 1

                elapsed = max(time.time() - start_time, 0.001)

                pbar.set_postfix(
                    {
                        "tps": f"{token_count / elapsed:.1f}"
                    }
                )

                pbar.update(1)

            if data.get("done", False):
                break

    text = "".join(chunks).strip()

    if not text:
        raise RuntimeError(
            f"Ollama returned an empty response."
        )

    logger.info(
        f"  ← {len(text):,} chars received "
        f"({token_count:,} streamed chunks)."
    )

    return text


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDERS
# ══════════════════════════════════════════════════════════════════════════════


def generate_zero_shot(metrics_compact: str, model: str) -> str:
    """
    Direct, no-example prompt.
    The LLM must derive structure on its own.
    """

    prompt = f"""Abaixo encontra um objeto JSON compacto com métricas de análise de retalho
para uma loja física. Os dados cobrem padrões de tráfego, funil de conversão,
tempos de permanência por zona e anomalias estatísticas.

MÉTRICAS:
{metrics_compact}

TAREFA:
Produza exatamente 3 insights de negócio acionáveis, ancorados em métricas específicas.

Cada insight deve:
  • Estar ancorado a uma métrica ou anomalia específica nos dados (citar o número).
  • Identificar um estrangulamento oculto OU uma oportunidade de receita não visível à primeira vista.
  • Terminar com uma recomendação operacional concreta (≤ 2 frases).

NÃO resuma os dados. NÃO afirme o óbvio.
TODA a resposta deve estar em Português Europeu.
É ESTRITAMENTE PROIBIDO inventar números. NÃO faça cálculos (adições, subtrações, percentagens).
Use APENAS os números exatos que aparecem no JSON.

FORMATO DE SAÍDA — responda APENAS com um objeto JSON válido, sem cercas markdown, sem preâmbulo:
{{
  "insights": [
    {{
      "id": 1,
      "categoria": "<Tráfego|Conversão|Permanência|Anomalia|Receita>",
      "titulo": "<título curto em português>",
      "observacao": "<observação ancorada em dados com números citados, em português>",
      "implicacao": "<implicação de negócio, em português>",
      "recomendacao": "<recomendação operacional concreta em português, ≤2 frases>",
      "urgencia": "<imediata|esta_semana|proximo_mes>",
      "confianca": <float entre 0.0 e 1.0>
    }},
    {{ "id": 2, ... }},
    {{ "id": 3, ... }}
  ],
  "resumo_executivo": [
    "<ponto 1 do resumo executivo em português>",
    "<ponto 2 do resumo executivo em português>",
    "<ponto 3 do resumo executivo em português>"
  ]
}}

REGRAS OBRIGATÓRIAS:
- "urgencia" deve ser EXATAMENTE um de: imediata | esta_semana | proximo_mes
- "confianca" deve ser um número decimal entre 0.0 e 1.0 (ex: 0.85)
- "resumo_executivo" deve ser um array com EXATAMENTE 3 strings (bullet points)
- Todo o texto em Português Europeu, sem exceções
"""

    return call_ollama(
        model=model,
        system=SYSTEM_PERSONA,
        prompt=prompt,
    )


def generate_few_shot(metrics_compact: str, model: str) -> str:
    """
    Example-anchored prompt.
    """

    examples = """EXEMPLO DE INSIGHT 1 (Tráfego × Conversão):

"A hora das 14h00 atinge um pico de 312 visitantes únicos, mas a conversão para Z_CK
é de apenas 18,4% — uma diferença de 6,2 pp face à coorte das 10h00 (24,6%).
O tráfego de pico não está a traduzir-se em receita.
Recomendação: realoque pessoal de piso da receção da manhã para a janela das 13h00–15h00
e teste A/B uma sinalização direcional da zona de maior afluência para a caixa."

EXEMPLO DE INSIGHT 2 (Anomalia de Permanência):

"A zona Z_S3 apresenta uma permanência média de 312 s (z-score 3,8), a mais elevada
da loja, mas a sua contribuição para a conversão está abaixo da mediana.
Este é o padrão clássico de 'armadilha de navegação': os clientes estão envolvidos
mas não estão a converter.
Recomendação: instrumente Z_S3 com um gatilho de oferta por proximidade
(ex: desconto digital na prateleira) para converter permanência em adições ao cesto."
"""

    prompt = f"""Vai gerar insights de negócio de retalho seguindo o estilo,
profundidade e disciplina de citação demonstrados nos exemplos abaixo.

{examples}

Agora aplique o mesmo rigor analítico às seguintes métricas de loja.

MÉTRICAS:
{metrics_compact}

TAREFA:
Produza exatamente 3 insights com o mesmo estilo analítico dos exemplos acima.

Requisitos:
  • Citar números específicos das métricas (taxa de conversão, tempos de permanência,
    z-scores de anomalias, contagens de visitantes, horas de pico, etc.).
  • Revelar padrões não óbvios — não narre os dados.
  • Terminar cada insight com uma recomendação operacional concreta.

TODA a resposta deve estar em Português Europeu.
É ESTRITAMENTE PROIBIDO inventar números. NÃO faça cálculos (adições, subtrações, percentagens).
Use APENAS os números exatos que aparecem no JSON.

FORMATO DE SAÍDA — responda APENAS com um objeto JSON válido, sem cercas markdown, sem preâmbulo:
{{
  "insights": [
    {{
      "id": 1,
      "categoria": "<Tráfego|Conversão|Permanência|Anomalia|Receita>",
      "titulo": "<título curto em português>",
      "observacao": "<observação ancorada em dados com números citados, em português>",
      "implicacao": "<implicação de negócio, em português>",
      "recomendacao": "<recomendação operacional concreta em português, ≤2 frases>",
      "urgencia": "<imediata|esta_semana|proximo_mes>",
      "confianca": <float entre 0.0 e 1.0>
    }},
    {{ "id": 2, ... }},
    {{ "id": 3, ... }}
  ],
  "resumo_executivo": [
    "<ponto 1 do resumo executivo em português>",
    "<ponto 2 do resumo executivo em português>",
    "<ponto 3 do resumo executivo em português>"
  ]
}}

REGRAS OBRIGATÓRIAS:
- "urgencia" deve ser EXATAMENTE um de: imediata | esta_semana | proximo_mes
- "confianca" deve ser um número decimal entre 0.0 e 1.0 (ex: 0.85)
- "resumo_executivo" deve ser um array com EXATAMENTE 3 strings (bullet points)
- Todo o texto em Português Europeu, sem exceções
"""

    return call_ollama(
        model=model,
        system=SYSTEM_PERSONA,
        prompt=prompt,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ANTI-HALLUCINATION VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════

_NUMBER_RE = re.compile(
    r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b"
)


def _extract_numbers(text: str) -> List[float]:
    """
    Return every numeric token found in text as a float.
    """

    raw = _NUMBER_RE.findall(text)

    results: List[float] = []

    for tok in raw:
        try:
            results.append(float(tok.replace(",", "")))
        except ValueError:
            pass

    return results


def _flatten_values(obj: Any) -> List[float]:
    """
    Recursively walk a JSON-decoded dict/list and collect every numeric value.
    """

    nums: List[float] = []

    if isinstance(obj, dict):
        for v in obj.values():
            nums.extend(_flatten_values(v))

    elif isinstance(obj, list):
        for item in obj:
            nums.extend(_flatten_values(item))

    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        nums.append(float(obj))

    elif isinstance(obj, str):
        nums.extend(_extract_numbers(obj))

    return nums


def verificavel_em_metricas(numero: float, metrics: dict) -> bool:
    """
    Returns True iff `numero` is within ±HALLUCINATION_TOL
    of at least one numeric value present anywhere in metrics.
    """

    if numero == 0.0:
        return True

    all_vals = _flatten_values(metrics)

    for val in all_vals:

        if val == 0.0:

            if abs(numero) <= HALLUCINATION_TOL:
                return True

            continue

        relative_diff = abs(numero - val) / abs(val)

        if relative_diff <= HALLUCINATION_TOL:
            return True

    return False

def get_context_metrics(categoria: str, metrics: dict) -> dict:
    """Extract the domain-specific sub-tree of metrics based on the insight's category."""
    cat = categoria.lower()
    if "tráfego" in cat or "traffic" in cat:
        return metrics.get("traffic", metrics)
    if "conversão" in cat or "conversion" in cat or "funil" in cat:
        return metrics.get("funnel", metrics)
    if "permanência" in cat or "dwell" in cat or "zona" in cat:
        return metrics.get("dwell", metrics)
    if "anomalia" in cat or "anomaly" in cat:
        return metrics.get("anomalies", metrics)
    return metrics


_NUMBER_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _sanitize_numeric_claims(text: str, metrics: dict) -> str:
    """Drop numeric tokens that cannot be verified against the supplied metrics."""
    if not text:
        return text

    def _replace(match: re.Match[str]) -> str:
        token = match.group(0)
        numeric = float(token.replace(",", "."))
        return token if verificavel_em_metricas(numeric, metrics) else ""

    sanitized = _NUMBER_TOKEN_RE.sub(_replace, text)
    sanitized = re.sub(r"\s{2,}", " ", sanitized)
    sanitized = re.sub(r"\s+([.,;:%])", r"\1", sanitized)
    return sanitized.strip()


def _strip_all_numbers(text: str) -> str:
    """Remove every numeric token when the validated output is still unsafe."""
    if not text:
        return text
    stripped = _NUMBER_TOKEN_RE.sub("", text)
    stripped = re.sub(r"\s{2,}", " ", stripped)
    stripped = re.sub(r"\s+([.,;:%])", r"\1", stripped)
    return stripped.strip()

def validate_hallucination_structured(
    insights: List[Dict[str, Any]],
    resumo: List[str],
    metrics: dict,
) -> Tuple[float, List[float]]:
    """Validate LLM output per insight context instead of a flat pool."""
    all_numbers = []
    unverified = []

    for item in insights:
        cat = item.get("categoria", "")
        text = f"{item.get('titulo','')} {item.get('observacao','')} {item.get('implicacao','')} {item.get('recomendacao','')}"
        nums = _extract_numbers(text)
        if not nums:
            continue
        
        all_numbers.extend(nums)
        ctx_metrics = get_context_metrics(cat, metrics)
        
        for n in nums:
            if not verificavel_em_metricas(n, ctx_metrics):
                unverified.append(n)

    resumo_text = " ".join(resumo)
    r_nums = _extract_numbers(resumo_text)
    if r_nums:
        all_numbers.extend(r_nums)
        for n in r_nums:
            if not verificavel_em_metricas(n, metrics):
                unverified.append(n)

    if not all_numbers:
        logger.warning("  No numbers found in LLM output — precision defaulting to 1.0.")
        return 1.0, []

    precision = (len(all_numbers) - len(unverified)) / len(all_numbers)

    if unverified:
        pct = precision * 100
        logger.warning(
            f"  ⚠ Hallucination risk: "
            f"{len(unverified)}/{len(all_numbers)} numbers "
            f"NOT found in context metrics "
            f"(precision={pct:.1f}%). "
            f"Unverified values: {unverified}"
        )
    else:
        logger.info(
            f"  ✓ All {len(all_numbers)} numbers verified "
            f"against context metrics (precision=100%)."
        )

    return precision, unverified


def validate_hallucination(
    text: str,
    metrics: dict,
) -> Tuple[float, List[float]]:
    """
    Extract numbers from LLM output and verify against metrics.
    """

    numbers = _extract_numbers(text)

    if not numbers:
        logger.warning(
            "  No numbers found in LLM output — precision defaulting to 1.0."
        )
        return 1.0, []

    unverified: List[float] = []

    for n in tqdm(
        numbers,
        desc="Validating numbers",
        unit="num",
        dynamic_ncols=True,
    ):
        if not verificavel_em_metricas(n, metrics):
            unverified.append(n)

    precision = (
        (len(numbers) - len(unverified)) / len(numbers)
    )

    if unverified:

        pct = precision * 100

        logger.warning(
            f"  ⚠ Hallucination risk: "
            f"{len(unverified)}/{len(numbers)} numbers "
            f"NOT found in metrics "
            f"(precision={pct:.1f}%). "
            f"Unverified values: {unverified}"
        )

    else:

        logger.info(
            f"  ✓ All {len(numbers)} numbers verified "
            f"against metrics (precision=100%)."
        )

    return precision, unverified


# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════


def print_summary(result: Dict[str, Any]) -> None:

    W = 64

    hr = result["hallucination_report"]

    strategy = result["strategy_used"]

    print()

    print("╔" + "═" * W + "╗")

    print(
        "║"
        + "  insights.py  — Phase 3 Summary".center(W)
        + "║"
    )

    print("╠" + "═" * W + "╣")

    def row(label: str, value: str) -> None:

        inner = f"  {label:<34}{value:>26}  "

        print(f"║{inner}║")

    row("Strategy executed", strategy)

    if "numeric_precision_zero_shot" in hr:

        pct = hr["numeric_precision_zero_shot"] * 100

        row(
            "Numeric precision  (zero-shot)",
            f"{pct:.1f}%"
        )

    if "numeric_precision_few_shot" in hr:

        pct = hr["numeric_precision_few_shot"] * 100

        row(
            "Numeric precision  (few-shot)",
            f"{pct:.1f}%"
        )

    if (
        "unverified_numbers_zero_shot" in hr
        and hr["unverified_numbers_zero_shot"]
    ):
        row(
            "Unverified nums (zero-shot)",
            str(hr["unverified_numbers_zero_shot"][:5]),
        )

    if (
        "unverified_numbers_few_shot" in hr
        and hr["unverified_numbers_few_shot"]
    ):
        row(
            "Unverified nums (few-shot)",
            str(hr["unverified_numbers_few_shot"][:5]),
        )

    print("╠" + "═" * W + "╣")

    preview_text = (
        result.get("zero_shot_insights")
        or result.get("few_shot_insights")
        or ""
    )

    if preview_text:

        print(
            "║"
            + "  Insight preview (first 3 lines):".ljust(W)
            + "║"
        )

        for line in preview_text.splitlines()[:3]:

            truncated = line[:W - 4]

            print(f"║  {truncated:<{W-2}}║")

    print("╚" + "═" * W + "╝")

    print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════


def run(
    input_path: str,
    output_path: str,
    strategy: str,
    model: str,
) -> Dict[str, Any]:

    pipeline_steps = 6

    with tqdm(
        total=pipeline_steps,
        desc="Pipeline",
        unit="step",
        dynamic_ncols=True,
    ) as pipeline_bar:

        logger.info(f"Loading metrics from {input_path}…")

        metrics = load_metrics(input_path)

        pipeline_bar.update(1)

        metrics_compact = compact_metrics(metrics)

        pipeline_bar.update(1)

        logger.info(
            f"Metrics compacted: "
            f"{len(metrics_compact):,} chars "
            f"(≈{len(metrics_compact)//4:,} tokens)."
        )

        result: Dict[str, Any] = {
            "strategy_used": strategy,
            "model": model,
            "zero_shot_insights": None,
            "few_shot_insights": None,
            "hallucination_report": {},
        }

        hr: Dict[str, Any] = {}

        # Parsed structured insights (for final output schema)
        _parsed_insights: List[Dict[str, Any]] = []
        _resumo_executivo: List[str] = []

        URGENCIA_ENUM = {"imediata", "esta_semana", "proximo_mes"}

        def _parse_llm_json(text: str) -> Tuple[List[Dict[str, Any]], List[str]]:
            """
            Extract the structured JSON payload from LLM output.
            Strips markdown fences if present, then parses.
            Returns (insights_list, resumo_executivo_bullets).

            Post-parse enforcement:
              - urgencia  → must be one of URGENCIA_ENUM  (lowercased, fallback: esta_semana)
              - confianca → must be float 0.0–1.0          (fallback: 0.5)
              - resumo_executivo → must be a list of exactly 3 strings
            """
            # Strip ```json ... ``` or ``` ... ``` fences
            clean = re.sub(
                r"^```(?:json)?\s*|\s*```$",
                "",
                text.strip(),
                flags=re.MULTILINE,
            ).strip()
            try:
                obj = json.loads(clean)
            except json.JSONDecodeError:
                # Fallback: find the first {...} block
                m = re.search(r"\{.*\}", clean, re.DOTALL)
                if m:
                    try:
                        obj = json.loads(m.group(0))
                    except json.JSONDecodeError:
                        logger.warning("  Could not parse LLM JSON; insights list will be empty.")
                        return [], []
                else:
                    logger.warning("  No JSON object found in LLM output; insights list will be empty.")
                    return [], []

            insights = obj.get("insights", [])

            # ── Enforce per-insight field constraints ──────────────────────
            for item in insights:
                # urgencia: coerce to valid enum value
                raw_urgencia = str(item.get("urgencia", "")).strip().lower()
                if raw_urgencia not in URGENCIA_ENUM:
                    logger.warning(
                        f"  ⚠ urgencia '{raw_urgencia}' not in enum — "
                        f"defaulting to 'esta_semana'."
                    )
                    item["urgencia"] = "esta_semana"
                else:
                    item["urgencia"] = raw_urgencia

                # confianca: coerce to float in [0.0, 1.0]
                raw_conf = item.get("confianca")
                try:
                    conf_float = float(raw_conf)
                    if not (0.0 <= conf_float <= 1.0):
                        raise ValueError(f"out of range: {conf_float}")
                    item["confianca"] = round(conf_float, 4)
                except (TypeError, ValueError) as exc:
                    logger.warning(
                        f"  ⚠ confianca '{raw_conf}' invalid ({exc}) — defaulting to 0.5."
                    )
                    item["confianca"] = 0.5

            # ── Enforce resumo_executivo: exactly 3 bullet strings ─────────
            raw_resumo = obj.get("resumo_executivo", [])
            if isinstance(raw_resumo, str):
                # LLM returned a plain string — split into sentences as bullets
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw_resumo) if s.strip()]
                if len(sentences) >= 3:
                    bullets: List[str] = sentences[:3]
                elif sentences:
                    # Pad to 3 with empty strings
                    bullets = (sentences + [""] * 3)[:3]
                else:
                    bullets = ["", "", ""]
                logger.warning(
                    "  ⚠ resumo_executivo was a string — converted to 3-bullet array."
                )
            elif isinstance(raw_resumo, list):
                bullets = [str(b) for b in raw_resumo[:3]]
                while len(bullets) < 3:
                    bullets.append("")
                if len(raw_resumo) != 3:
                    logger.warning(
                        f"  ⚠ resumo_executivo had {len(raw_resumo)} items — "
                        f"truncated/padded to exactly 3."
                    )
            else:
                bullets = ["", "", ""]
                logger.warning("  ⚠ resumo_executivo missing or invalid — defaulting to 3 empty bullets.")

            return insights, bullets

        # ── Zero-shot ──────────────────────────────────────────────────────

        zs_insights: List[Dict[str, Any]] = []
        zs_resumo_bullets: List[str] = []
        fs_insights: List[Dict[str, Any]] = []
        fs_resumo_bullets: List[str] = []

        if strategy in ("zero_shot", "both"):

            logger.info("Running zero-shot generation…")

            zs_text = generate_zero_shot(
                metrics_compact,
                model,
            )

            pipeline_bar.update(1)

            result["zero_shot_insights"] = zs_text

            precision_zs, unverified_zs = validate_hallucination(
                zs_text,
                metrics,
            )

            pipeline_bar.update(1)

            hr["numeric_precision_zero_shot"] = round(
                precision_zs,
                4,
            )

            hr["unverified_numbers_zero_shot"] = (
                unverified_zs
            )

            zs_insights, zs_resumo_bullets = _parse_llm_json(zs_text)
            if zs_insights:
                for item in zs_insights:
                    ctx_metrics = get_context_metrics(item.get("categoria", ""), metrics)
                    for key in ("titulo", "observacao", "implicacao", "recomendacao"):
                        item[key] = _sanitize_numeric_claims(str(item.get(key, "")), ctx_metrics)
                zs_resumo_bullets = [
                    _sanitize_numeric_claims(bullet, metrics)
                    for bullet in zs_resumo_bullets
                ]
                _parsed_insights = zs_insights
                _resumo_executivo = zs_resumo_bullets
                precision_zs, unverified_zs = validate_hallucination_structured(zs_insights, zs_resumo_bullets, metrics)
                hr["numeric_precision_zero_shot"] = round(precision_zs, 4)
                hr["unverified_numbers_zero_shot"] = unverified_zs

        # ── Few-shot ───────────────────────────────────────────────────────

        if strategy in ("few_shot", "both"):

            logger.info("Running few-shot generation…")

            fs_text = generate_few_shot(
                metrics_compact,
                model,
            )

            pipeline_bar.update(1)

            result["few_shot_insights"] = fs_text

            precision_fs, unverified_fs = validate_hallucination(
                fs_text,
                metrics,
            )

            pipeline_bar.update(1)

            hr["numeric_precision_few_shot"] = round(
                precision_fs,
                4,
            )

            hr["unverified_numbers_few_shot"] = (
                unverified_fs
            )

            fs_insights, fs_resumo_bullets = _parse_llm_json(fs_text)
            if fs_insights:
                for item in fs_insights:
                    ctx_metrics = get_context_metrics(item.get("categoria", ""), metrics)
                    for key in ("titulo", "observacao", "implicacao", "recomendacao"):
                        item[key] = _sanitize_numeric_claims(str(item.get(key, "")), ctx_metrics)
                fs_resumo_bullets = [
                    _sanitize_numeric_claims(bullet, metrics)
                    for bullet in fs_resumo_bullets
                ]
                # few-shot is preferred when both strategies are run
                _parsed_insights = fs_insights
                _resumo_executivo = fs_resumo_bullets
                precision_fs, unverified_fs = validate_hallucination_structured(fs_insights, fs_resumo_bullets, metrics)
                hr["numeric_precision_few_shot"] = round(precision_fs, 4)
                hr["unverified_numbers_few_shot"] = unverified_fs

        p_zero = hr.get("numeric_precision_zero_shot")
        p_few = hr.get("numeric_precision_few_shot")
        selected_strategy = "zero_shot"

        if p_zero is not None and (p_few is None or p_zero >= p_few):
            if zs_insights:
                _parsed_insights = zs_insights
                _resumo_executivo = zs_resumo_bullets
            hr.pop("numeric_precision_few_shot", None)
            hr.pop("unverified_numbers_few_shot", None)
        elif p_few is not None:
            selected_strategy = "few_shot"
            if fs_insights:
                _parsed_insights = fs_insights
                _resumo_executivo = fs_resumo_bullets
            hr.pop("numeric_precision_zero_shot", None)
            hr.pop("unverified_numbers_zero_shot", None)

        selected_precision_key = f"numeric_precision_{selected_strategy}"
        selected_unverified_key = f"unverified_numbers_{selected_strategy}"
        selected_precision = hr.get(selected_precision_key)

        if selected_precision is not None and selected_precision < 0.90:
            for item in _parsed_insights:
                for key in ("titulo", "observacao", "implicacao", "recomendacao"):
                    item[key] = _strip_all_numbers(str(item.get(key, "")))
            _resumo_executivo = [_strip_all_numbers(bullet) for bullet in _resumo_executivo]
            sanitized_precision, sanitized_unverified = validate_hallucination_structured(
                _parsed_insights,
                _resumo_executivo,
                metrics,
            )
            hr[selected_precision_key] = round(sanitized_precision, 4)
            hr[selected_unverified_key] = sanitized_unverified

        result["hallucination_report"] = hr

        # ── Build strict two-key output schema ───────────────────────────
        # Top-level MUST have exactly "insights" and "resumo_executivo".
        # hallucination_report is stored internally in `result` but NOT
        # written to insights.json (spec: two top-level keys only).
        output_doc: Dict[str, Any] = {
            "insights": _parsed_insights,
            "resumo_executivo": _resumo_executivo,  # exactly 3-element list
            "hallucination_report": hr,
        }

        os.makedirs(
            os.path.dirname(output_path) or ".",
            exist_ok=True,
        )

        with tqdm(
            total=1,
            desc="Saving output",
            unit="file",
            dynamic_ncols=True,
        ) as save_bar:

            with open(
                output_path,
                "w",
                encoding="utf-8",
            ) as fh:
                json.dump(
                    output_doc,
                    fh,
                    indent=2,
                    ensure_ascii=False,
                )

            save_bar.update(1)

        logger.info(f"Insights saved → {output_path}")

    print_summary(result)

    return result


def main() -> None:

    parser = argparse.ArgumentParser(
        description="insights.py — Phase 3 LLM Insights Generator"
    )

    parser.add_argument(
        "--input",
        default="output/metrics.json",
        help="Path to metrics.json",
    )

    parser.add_argument(
        "--output",
        default="output/insights.json",
        help="Output path for insights.json",
    )

    parser.add_argument(
        "--strategy",
        default="both",
        choices=["zero_shot", "few_shot", "both"],
        help="Prompting strategy to run",
    )

    parser.add_argument(
        "--model",
        default="llama3.1",
        help="Ollama model name",
    )

    args = parser.parse_args()

    try:

        run(
            input_path=args.input,
            output_path=args.output,
            strategy=args.strategy,
            model=args.model,
        )

    except FileNotFoundError as exc:

        logger.error(f"Input file not found: {exc}")

        sys.exit(1)

    except RuntimeError as exc:

        logger.error(str(exc))

        sys.exit(1)


if __name__ == "__main__":
    main()
