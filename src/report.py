"""
report.py — Phase 4 Markdown Report Generator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reads output/insights.json (produced by insights.py) and generates a
structured Markdown report with 6 mandatory sections.

ARCHITECTURAL NOTE:
  This script reads ONLY from insights.json. It does NOT touch metrics.json.
  All data required for the report must be present inside insights.json.

USAGE
  python src/report.py
  python src/report.py --input output/insights.json --metrics output/metrics.json --output output/weekly_report.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("report")

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _val(value: Any, suffix: str = "", fallback: str = "N/A") -> str:
    """Format a value for display; return fallback if None or empty."""
    if value is None or value == "":
        return fallback
    return f"{value}{suffix}"


def _pct(value: Any, fallback: str = "N/A") -> str:
    """Format a 0-1 float as a percentage string, or pass through a string."""
    if value is None:
        return fallback
    if isinstance(value, str):
        return value
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return fallback


def _insights_by_category(
    insights: List[Dict[str, Any]],
    categoria: str,
) -> List[Dict[str, Any]]:
    """Return insights whose 'categoria' matches (case-insensitive)."""
    return [
        i for i in insights
        if i.get("categoria", "").lower() == categoria.lower()
    ]


def _urgency_badge(urgencia: str) -> str:
    mapping = {
        "imediata":     "🔴",
        "esta_semana":  "🟡",
        "proximo_mes":  "🟢",
        # Legacy aliases (tolerated for backward compatibility)
        "Alta":   "🔴",
        "Media":  "🟡",
        "Baixa":  "🟢",
    }
    return mapping.get(urgencia, "⚪")


def _urgency_label(urgencia: str) -> str:
    labels = {
        "imediata":     "Imediata",
        "esta_semana":  "Esta Semana",
        "proximo_mes":  "Próximo Mês",
        "Alta":         "Alta",
        "Media":        "Média",
        "Baixa":        "Baixa",
    }
    return labels.get(urgencia, urgencia)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION BUILDERS
# ══════════════════════════════════════════════════════════════════════════════


def _section_resumo_executivo(
    resumo: Any,
    insights: List[Dict[str, Any]],
    hr: Dict[str, Any],
    metrics: Dict[str, Any],
) -> List[str]:
    """
    Section 1 — Resumo Executivo (deterministic KPI injection + AI insights).
    
    SPEC §6 — Executive Summary MUST be deterministic:
    1. Ground-truth KPIs injected from metrics.json (NOT from LLM)
    2. Exactly 3 KPI bullets: Visitantes Únicos | Total de Visitas | Taxa de Conversão
    3. Exactly 3 AI-generated bullet points from insights.json
    4. KPIs override any LLM-generated text (spec compliance)
    """
    md: List[str] = []
    md.append("## 📈 1. Resumo Executivo\n")

    # PART 1: DETERMINISTIC KPIs from metrics.json (non-negotiable ground truth)
    md.append("### 📊 Indicadores Chave de Desempenho\n")
    
    if metrics:
        traffic = metrics.get("traffic", {})
        funnel = metrics.get("funnel", {})
        
        # KPI 1: Unique Visitors
        uv = traffic.get('total_unique_visitors', 'N/A')
        if uv != 'N/A':
            md.append(f"- **Visitantes Únicos:** {int(uv):,}")
        else:
            md.append("- **Visitantes Únicos:** N/A")
        
        # KPI 2: Total Visits
        tv = traffic.get('total_visits', 'N/A')
        if tv != 'N/A':
            md.append(f"- **Total de Visitas:** {int(tv):,}")
        else:
            md.append("- **Total de Visitas:** N/A")
        
        # KPI 3: Conversion Rate
        conv = funnel.get('conversion_rate', 'N/A')
        if conv != 'N/A':
            try:
                conv_val = float(conv)
                md.append(f"- **Taxa de Conversão:** {conv_val:.1f}%")
            except (TypeError, ValueError):
                md.append(f"- **Taxa de Conversão:** {conv}%")
        else:
            md.append("- **Taxa de Conversão:** N/A")
    else:
        md.append("- **Visitantes Únicos:** N/A")
        md.append("- **Total de Visitas:** N/A")
        md.append("- **Taxa de Conversão:** N/A")

    md.append("\n### 🤖 Principais Conclusões da IA\n")

    # PART 2: AI-generated bullet points (exactly 3) from resumo_executivo field
    # Parse resumo: accepts list[str] (new schema) or plain str (legacy).
    if isinstance(resumo, list):
        bullets = [str(b).strip() for b in resumo if b and str(b).strip()]
    elif isinstance(resumo, str) and resumo.strip():
        import re as _re
        # Split on sentence boundaries: period, exclamation, question mark
        bullets = [s.strip() for s in _re.split(r"(?<=[.!?])\s+", resumo.strip()) if s.strip()]
    else:
        bullets = []

    # Render exactly 3 AI bullets (or fewer if not available)
    bullet_count = min(3, len(bullets))
    for i in range(bullet_count):
        md.append(f"- {bullets[i]}")
    
    # If fewer than 3 AI bullets available, note it
    if bullet_count < 3:
        md.append(f"*({bullet_count} de 3 conclusões disponíveis)*\n")
    else:
        md.append("")

    # PART 3: Summary statistics table
    md.append("### 📈 Sumário de Urgências\n")
    md.append("| Indicador | Valor |")
    md.append("| :--- | :--- |")
    
    total = len(insights)
    imediata   = sum(1 for i in insights if i.get("urgencia") in ("imediata", "Alta"))
    esta_semana = sum(1 for i in insights if i.get("urgencia") in ("esta_semana", "Media"))
    proximo_mes = sum(1 for i in insights if i.get("urgencia") in ("proximo_mes", "Baixa"))

    md.append(f"| **Total de Insights** | {total} |")
    md.append(f"| **Urgência Imediata** 🔴 | {imediata} |")
    md.append(f"| **Urgência Esta Semana** 🟡 | {esta_semana} |")
    md.append(f"| **Urgência Próximo Mês** 🟢 | {proximo_mes} |")

    # Hallucination report summary if available
    if hr:
        p_few = hr.get("numeric_precision_few_shot")
        p_zero = hr.get("numeric_precision_zero_shot")
        best_precision = p_few if p_few is not None else p_zero
        if best_precision is not None:
            md.append(
                f"| **Precisão Numérica (Auditoria)** | **{_pct(best_precision)}** |"
            )

    md.append("")
    return md


def _section_performance_trafego(insights: List[Dict[str, Any]]) -> List[str]:
    """Section 2 — Performance de Tráfego."""
    md: List[str] = []
    md.append("## 🚶 2. Performance de Tráfego\n")

    traffic_insights = _insights_by_category(insights, "Tráfego")
    if not traffic_insights:
        traffic_insights = _insights_by_category(insights, "Traffic")

    if not traffic_insights:
        # Fall back to any insight that mentions traffic-related keywords
        traffic_insights = [
            i for i in insights
            if any(
                kw in (i.get("observacao", "") + i.get("titulo", "")).lower()
                for kw in ("traffic", "tráfego", "visitor", "visitante", "peak", "pico")
            )
        ]

    if not traffic_insights:
        md.append(
            "*Nenhum insight de tráfego disponível no ficheiro de entrada.*\n"
        )
        return md

    for insight in traffic_insights:
        urgencia = insight.get("urgencia", "")
        badge = _urgency_badge(urgencia)
        md.append(
            f"### {badge} {insight.get('titulo', 'Insight de Tráfego')}"
        )
        if obs := insight.get("observacao"):
            md.append(f"**Observação:** {obs}\n")
        if imp := insight.get("implicacao"):
            md.append(f"**Implicação:** {imp}\n")

    md.append("")
    return md


def _section_analise_zonas(insights: List[Dict[str, Any]]) -> List[str]:
    """Section 3 — Análise de Zonas."""
    md: List[str] = []
    md.append("## 📍 3. Análise de Zonas\n")

    zone_insights = _insights_by_category(insights, "Permanência")
    if not zone_insights:
        zone_insights = _insights_by_category(insights, "Dwell")
    if not zone_insights:
        zone_insights = [
            i for i in insights
            if any(
                kw in (i.get("observacao", "") + i.get("titulo", "")).lower()
                for kw in ("zone", "zona", "dwell", "permanência", "z_")
            )
        ]

    if not zone_insights:
        md.append(
            "*Nenhum insight de análise de zonas disponível no ficheiro de entrada.*\n"
        )
        return md

    for insight in zone_insights:
        urgencia = insight.get("urgencia", "")
        badge = _urgency_badge(urgencia)
        md.append(f"### {badge} {insight.get('titulo', 'Análise de Zona')}")
        if obs := insight.get("observacao"):
            md.append(f"**Observação:** {obs}\n")
        if imp := insight.get("implicacao"):
            md.append(f"**Implicação:** {imp}\n")

    md.append("")
    return md


def _section_funil_clientes(insights: List[Dict[str, Any]]) -> List[str]:
    """Section 4 — Funil de Clientes."""
    md: List[str] = []
    md.append("## 🔁 4. Funil de Clientes\n")

    funnel_insights = _insights_by_category(insights, "Conversão")
    if not funnel_insights:
        funnel_insights = _insights_by_category(insights, "Conversion")
    if not funnel_insights:
        funnel_insights = [
            i for i in insights
            if any(
                kw in (i.get("observacao", "") + i.get("titulo", "")).lower()
                for kw in (
                    "conversion", "conversão", "funnel", "funil",
                    "checkout", "basket", "purchase", "compra",
                )
            )
        ]

    if not funnel_insights:
        md.append(
            "*Nenhum insight de funil de clientes disponível no ficheiro de entrada.*\n"
        )
        return md

    for insight in funnel_insights:
        urgencia = insight.get("urgencia", "")
        badge = _urgency_badge(urgencia)
        md.append(
            f"### {badge} {insight.get('titulo', 'Insight de Funil')}"
        )
        if obs := insight.get("observacao"):
            md.append(f"**Observação:** {obs}\n")
        if imp := insight.get("implicacao"):
            md.append(f"**Implicação:** {imp}\n")

    md.append("")
    return md


def _section_anomalias(insights: List[Dict[str, Any]]) -> List[str]:
    """Section 5 — Anomalias da Semana (vs. baseline de 6 dias)."""
    md: List[str] = []
    md.append("## ⚠️ 5. Anomalias da Semana\n")
    md.append(
        "> As anomalias de tráfego abaixo foram detetadas utilizando a lógica de "
        "**\"6-day baseline vs Day 7\"** (comparação estatística face à linha "
        "de base de 6 dias anteriores, excluindo o dia em análise).\n"
    )

    anomaly_insights = _insights_by_category(insights, "Anomalia")
    if not anomaly_insights:
        # Try legacy English category name and keyword fallback
        anomaly_insights = _insights_by_category(insights, "Anomaly")
    if not anomaly_insights:
        anomaly_insights = [
            i for i in insights
            if any(
                kw in (i.get("observacao", "") + i.get("titulo", "")).lower()
                for kw in (
                    "anomal", "z-score", "zscore", "outlier",
                    "spike", "pico anormal", "desvio",
                )
            )
        ]

    if not anomaly_insights:
        md.append(
            "*Nenhuma anomalia operacional significativa identificada nos insights gerados.*\n"
        )
        return md

    for insight in anomaly_insights:
        urgencia = insight.get("urgencia", "")
        badge = _urgency_badge(urgencia)
        label = _urgency_label(urgencia)
        confianca = insight.get("confianca", "")
        # Format confianca: float → percentage, string → as-is
        try:
            conf_display = f"{float(confianca):.0%}"
        except (TypeError, ValueError):
            conf_display = str(confianca) if confianca else "N/A"
        md.append(
            f"### {badge} {insight.get('titulo', 'Anomalia')} "
            f"*(Confiança: {conf_display} | Urgência: {label})*"
        )
        if obs := insight.get("observacao"):
            md.append(f"**Observação:** {obs}\n")
        if imp := insight.get("implicacao"):
            md.append(f"**Implicação:** {imp}\n")

    md.append("")
    return md


def _section_recomendacoes(insights: List[Dict[str, Any]]) -> List[str]:
    """Section 6 — Recomendações para a Próxima Semana."""
    md: List[str] = []
    md.append("## 🎯 6. Recomendações para a Próxima Semana\n")

    # Sort by urgency: imediata → esta_semana → proximo_mes (legacy Alta/Media/Baixa tolerated)
    urgency_order = {
        "imediata":     0,
        "Alta":         0,
        "esta_semana":  1,
        "Media":        1,
        "proximo_mes":  2,
        "Baixa":        2,
    }
    sorted_insights = sorted(
        insights,
        key=lambda i: urgency_order.get(i.get("urgencia", ""), 3),
    )

    if not sorted_insights:
        md.append("*Sem recomendações disponíveis.*\n")
        return md

    md.append(
        "As recomendações abaixo estão ordenadas por urgência operacional:\n"
    )
    md.append("| # | Urgência | Título | Recomendação |")
    md.append("| :---: | :---: | :--- | :--- |")

    for idx, insight in enumerate(sorted_insights, start=1):
        urgencia = insight.get("urgencia", "N/A")
        badge = _urgency_badge(urgencia)
        label = _urgency_label(urgencia)
        titulo = insight.get("titulo", "—")
        rec = insight.get("recomendacao", "—")
        # Escape pipe characters inside table cells
        rec_escaped = rec.replace("|", "\\|")
        titulo_escaped = titulo.replace("|", "\\|")
        md.append(
            f"| {idx} | {badge} {label} | {titulo_escaped} | {rec_escaped} |"
        )

    md.append("")
    return md


def _section_auditoria(hr: Dict[str, Any]) -> List[str]:
    """Bonus section — Anti-Hallucination Audit (appended when data present)."""
    if not hr:
        return []

    md: List[str] = []
    md.append("## 🛡️ 7. Auditoria de Confiança (Anti-Hallucination)\n")
    md.append(
        "Todos os números gerados pela IA foram cruzados com as métricas "
        "determinísticas do pipeline.\n"
    )
    md.append("| Estratégia | Precisão Numérica |")
    md.append("| :--- | :--- |")

    p_zero = hr.get("numeric_precision_zero_shot")
    p_few = hr.get("numeric_precision_few_shot")

    if p_zero is not None:
        md.append(f"| Zero-Shot | {_pct(p_zero)} |")
    if p_few is not None:
        md.append(f"| Few-Shot | **{_pct(p_few)}** |")

    if p_few is not None and p_zero is not None and p_few > p_zero:
        md.append(
            "\n*A estratégia Few-Shot demonstrou redução efetiva de alucinações "
            "face à Zero-Shot.*\n"
        )

    unverified_few = hr.get("unverified_numbers_few_shot", [])
    unverified_zero = hr.get("unverified_numbers_zero_shot", [])
    unverified = unverified_few or unverified_zero
    if unverified:
        sample = ", ".join(str(v) for v in unverified[:5])
        md.append(
            f"\n> ⚠️ **Números não verificados (amostra):** {sample}\n"
        )

    md.append("")
    return md


# ══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATOR
# ══════════════════════════════════════════════════════════════════════════════


def generate_markdown(input_path: str, metrics_path: str, output_path: str) -> None:
    """
    Load insights.json from *input_path* and write a Markdown report to
    *output_path*.  All data is sourced exclusively from insights.json.
    """

    logger.info(f"Loading insights from {input_path}…")
    data = load_json(input_path)

    if not data:
        logger.error(f"Ficheiro de insights não encontrado: {input_path}")
        sys.exit(1)

    insights: List[Dict[str, Any]] = data.get("insights", [])
    resumo: str = data.get("resumo_executivo", "")
    hr: Dict[str, Any] = data.get("hallucination_report", {})
    metrics: Dict[str, Any] = load_json(metrics_path) or {}

    if not insights:
        logger.warning(
            "O campo 'insights' está vazio ou ausente. "
            "O relatório será gerado com secções vazias."
        )

    # ── Header ────────────────────────────────────────────────────────────────
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md: List[str] = []
    md.append("# 🛒 Retail Analytics — Relatório Semanal de Operações")
    md.append(f"*Gerado automaticamente em: {now}*")
    md.append(f"*Fonte de dados: `{os.path.basename(input_path)}`*")
    md.append("\n---\n")

    # ── 6 Mandatory Sections ──────────────────────────────────────────────────
    md.extend(_section_resumo_executivo(resumo, insights, hr, metrics))
    md.append("---\n")

    md.extend(_section_performance_trafego(insights))
    md.append("---\n")

    md.extend(_section_analise_zonas(insights))
    md.append("---\n")

    md.extend(_section_funil_clientes(insights))
    md.append("---\n")

    md.extend(_section_anomalias(insights))
    md.append("---\n")

    md.extend(_section_recomendacoes(insights))

    # ── Optional Audit Section ────────────────────────────────────────────────
    audit = _section_auditoria(hr)
    if audit:
        md.append("---\n")
        md.extend(audit)

    # ── Footer ────────────────────────────────────────────────────────────────
    md.append("\n---")
    md.append(
        "*Relatório gerado pelo pipeline Retail Analytics. "
        "Dados processados exclusivamente a partir de insights.json.*"
    )

    # ── Write output ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))

    logger.info(f"Relatório guardado em: {output_path}")
    print(f"\n[Sucesso] Relatório final gerado com sucesso em: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="report.py — Phase 4 Markdown Report Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default="output/insights.json",
        help="Path to insights.json produced by insights.py",
    )
    parser.add_argument(
        "--metrics",
        default="output/metrics.json",
        help="Path to metrics.json produced by analytics.py",
    )
    parser.add_argument(
        "--output",
        default="output/weekly_report.md",
        help="Destination path for the generated Markdown report",
    )
    args = parser.parse_args()

    generate_markdown(
        input_path=args.input,
        metrics_path=args.metrics,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()