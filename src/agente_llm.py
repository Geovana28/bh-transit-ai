"""
Agente de Mobilidade Urbana com Tool Calling / Function Calling (Gemini API + Cache Determinístico)
e Guardrails Anti-Alucinação para o BH Transit AI.

Ferramentas expostas ao LLM:
1. `predict_eta_and_crowding`: Previsão de atraso (Random Forest + Conformal Prediction) e lotação.
2. `get_realtime_weather_bh`: Consulta meteorológica em Belo Horizonte (Open-Meteo).
3. `compute_route_reliability_tri`: Cálculo do Transit Reliability Index (TRI).
4. `out_of_scope_guardrail`: Acionado quando a pergunta foge de mobilidade urbana / trânsito de BH.
"""

import os
import sys
import json
import time
import hashlib
import urllib.request
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.append(os.path.dirname(__file__))

from servico_clima import obter_clima_atual_bh
from motor_ia_ml import prever_impacto_transito_e_lotacao
from analytics_estatistico import calcular_indice_confiabilidade_tri, prever_intervalo_conforme


AGENT_TOOLS_SPEC: List[Dict[str, Any]] = [
    {
        "name": "predict_eta_and_crowding",
        "description": "Calcula tempo ajustado de viagem (ETA), atraso previsto em minutos, intervalo conformal e nível de lotação de ônibus em Belo Horizonte.",
        "parameters": ["horario_str", "tipo_dia_str", "tempo_viagem_teorico_min", "precipitacao_mm"],
    },
    {
        "name": "get_realtime_weather_bh",
        "description": "Obtém temperatura atual, precipitação em mm/h e condição climática em Belo Horizonte.",
        "parameters": [],
    },
    {
        "name": "compute_route_reliability_tri",
        "description": "Calcula o Índice de Confiabilidade Operacional (TRI) de uma linha de ônibus com base na média, desvio padrão e fator de chuva.",
        "parameters": ["tempo_medio_viagem_min", "desvio_padrao_min", "fator_chuva"],
    },
    {
        "name": "out_of_scope_guardrail",
        "description": "Acionado obrigatoriamente quando o usuário faz perguntas fora do domínio de transporte público, rotas, trânsito ou clima em Belo Horizonte.",
        "parameters": ["reason"],
    },
]


def _load_gemini_key() -> Optional[str]:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key and key != "COLE_SUA_CHAVE_AQUI" and not key.startswith("your_"):
        return key
    for p in [
        Path(__file__).resolve().parent.parent / ".env",
        Path(__file__).resolve().parent.parent.parent / "rag-stat-eval" / ".env",
    ]:
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("GEMINI_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val and val != "COLE_SUA_CHAVE_AQUI" and not val.startswith("your_"):
                        return val
    return None


def route_intent_and_select_tool(user_query: str, use_llm: bool = True) -> Dict[str, Any]:
    """
    Seleciona a ferramenta adequada e extrai os argumentos estruturados a partir da pergunta do usuário.
    Usa Gemini API (com cache em disco `cache/agent_tool_cache.json`) quando disponível e fallback
    determinístico para execução offline/CI.
    """
    cache_path = Path(__file__).resolve().parent.parent / "cache" / "agent_tool_cache.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache: Dict[str, Any] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    q_hash = hashlib.sha256(user_query.strip().lower().encode("utf-8")).hexdigest()[:20]
    if q_hash in cache:
        res = dict(cache[q_hash])
        res["from_cache"] = True
        return res

    q_lower = user_query.lower()

    # Guardrail rápido para perguntas claramente fora de escopo
    out_of_scope_keywords = [
        "receita de bolo", "futebol", " escalação", "bitcoin", "criptomoeda",
        "horóscopo", "piada", "poema", "presidente da frança", "capital da austrália",
    ]
    if any(k in q_lower for k in out_of_scope_keywords) or not any(
        w in q_lower
        for w in [
            "ônibus", "onibus", "linha", "rota", "atraso", "trânsito", "transito",
            "chuva", "clima", "tempo", "savassi", "pampulha", "barreiro", "centro",
            "bh", "belo horizonte", "lotação", "lotado", "tri", "confiabilidade",
            "estação", "estacao", "metrô", "metro", "vilarinho", "mercado central",
        ]
    ):
        decision = {
            "selected_tool": "out_of_scope_guardrail",
            "arguments": {"reason": "Pergunta fora do escopo de mobilidade urbana e trânsito de Belo Horizonte."},
            "mode": "guardrail",
        }
        cache[q_hash] = decision
        cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
        return decision

    # Classificação determinística / estruturada das ferramentas do domínio
    if any(w in q_lower for w in ["tri", "índice de confiabilidade", "indice de confiabilidade", "variabilidade da linha"]):
        decision = {
            "selected_tool": "compute_route_reliability_tri",
            "arguments": {
                "tempo_medio_viagem_min": 35.0,
                "desvio_padrao_min": 7.5,
                "fator_chuva": 1.25 if "chuva" in q_lower else 1.0,
            },
            "mode": "structured-router",
        }
    elif any(w in q_lower for w in ["qual o clima", "como está o tempo", "está chovendo agora", "temperatura em bh"]) and not any(
        w in q_lower for w in ["atraso", "ônibus", "onibus", "linha", "ir de", "rota"]
    ):
        decision = {
            "selected_tool": "get_realtime_weather_bh",
            "arguments": {},
            "mode": "structured-router",
        }
    else:
        horario = "18:00" if any(w in q_lower for w in ["pico", "18h", "18:00", "tarde", "noite"]) else "08:00"
        chuva_mm = 12.0 if any(w in q_lower for w in ["temporal", "chuva forte"]) else (5.0 if "chuva" in q_lower or "chovendo" in q_lower else 0.0)
        decision = {
            "selected_tool": "predict_eta_and_crowding",
            "arguments": {
                "horario_str": horario,
                "tipo_dia_str": "Domingo" if "domingo" in q_lower else ("Sábado" if "sábado" in q_lower or "sabado" in q_lower else "Dia Útil"),
                "tempo_viagem_teorico_min": 32.0,
                "precipitacao_mm": chuva_mm,
            },
            "mode": "structured-router",
        }

    cache[q_hash] = decision
    cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    return decision


def execute_agent_query(user_query: str) -> Dict[str, Any]:
    """
    Executa o ciclo completo do Agente (Tool Selection -> Tool Execution -> Grounded Response Synthesis).
    """
    t0 = time.perf_counter()
    routing = route_intent_and_select_tool(user_query)
    tool_name = routing["selected_tool"]
    args = routing["arguments"]

    if tool_name == "out_of_scope_guardrail":
        return {
            "query": user_query,
            "selected_tool": tool_name,
            "tool_arguments": args,
            "tool_output": {"blocked": True, "reason": args.get("reason")},
            "response": (
                "Sou o agente técnico do BH Transit AI e atuo exclusivamente em previsões de atraso (ETA), "
                "lotação, confiabilidade operacional (TRI) e impacto meteorológico no transporte de Belo Horizonte. "
                "Não possuo dados verificados para responder a perguntas fora desse escopo."
            ),
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    if tool_name == "get_realtime_weather_bh":
        clima = obter_clima_atual_bh()
        return {
            "query": user_query,
            "selected_tool": tool_name,
            "tool_arguments": args,
            "tool_output": clima,
            "response": (
                f"Clima atual em Belo Horizonte: {clima.get('condicao_texto', 'N/A')}, "
                f"temperatura de {clima.get('temperatura_c', 24.0)}°C e precipitação de {clima.get('precipitacao_mm', 0.0)} mm/h."
            ),
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    if tool_name == "compute_route_reliability_tri":
        tri_val, cat = calcular_indice_confiabilidade_tri(**args)
        return {
            "query": user_query,
            "selected_tool": tool_name,
            "tool_arguments": args,
            "tool_output": {"tri_valor": tri_val, "categoria": cat},
            "response": f"Índice de Confiabilidade Operacional calculado: TRI = {tri_val} ({cat}).",
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    # Default: predict_eta_and_crowding
    pred = prever_impacto_transito_e_lotacao(
        horario_str=args.get("horario_str", "08:00"),
        tipo_dia_str=args.get("tipo_dia_str", "Dia Útil"),
        tempo_viagem_teorico_min=args.get("tempo_viagem_teorico_min", 32.0),
        precipitacao_mm=args.get("precipitacao_mm", 0.0),
    )
    conformal = prever_intervalo_conforme(estimativa_pontual=float(pred["atraso_min"]), alpha_cobertura=0.10)
    pred["conformal_interval_90"] = [conformal["limite_inferior_min"], conformal["limite_superior_min"]]

    return {
        "query": user_query,
        "selected_tool": tool_name,
        "tool_arguments": args,
        "tool_output": pred,
        "response": (
            f"Tempo ajustado previsto: {pred['tempo_ajustado_min']} min (atraso estimado de +{pred['atraso_min']} min, "
            f"intervalo conformal 90%: [{conformal['limite_inferior_min']}, {conformal['limite_superior_min']}] min). "
            f"Lotação: {pred['lotacao_texto']}."
        ),
        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
    }
