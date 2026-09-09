"""
Serviço de Coleta e Análise de Clima em Tempo Real para Belo Horizonte (MG).

Integração com a API Pública do Open-Meteo para obter dados meteorológicos
instantâneos (temperatura, precipitação, umidade, vento e código WMO) e 
calcular o impacto no tráfego urbano.
"""

from typing import Dict, Any, Optional
import logging
import json

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    import urllib.request
    import urllib.parse
    import urllib.error

logger = logging.getLogger(__name__)

# Coordenadas Oficiais de Belo Horizonte (Praça Sete / Centro)
LAT_BH: float = -19.9167
LON_BH: float = -43.9345

# Mapeamento oficial dos códigos meteorológicos da WMO (World Meteorological Organization)
WMO_WEATHER_CODES: Dict[int, Dict[str, Any]] = {
    0: {"descricao": "Céu Limpo", "emoji": "☀️", "fator_chuva": 1.0, "alerta": "Tempo bom e seco."},
    1: {"descricao": "Predominantemente Ensolarado", "emoji": "🌤️", "fator_chuva": 1.0, "alerta": "Tempo estável."},
    2: {"descricao": "Parcialmente Nublado", "emoji": "⛅", "fator_chuva": 1.0, "alerta": "Condições normais."},
    3: {"descricao": "Encoberto", "emoji": "☁️", "fator_chuva": 1.02, "alerta": "Céu nublado, trânsito regular."},
    45: {"descricao": "Nevoeiro / Neblina", "emoji": "🌫️", "fator_chuva": 1.08, "alerta": "Visibilidade reduzida nos pontos altos da capital."},
    48: {"descricao": "Nevoeiro com Depósito", "emoji": "🌫️", "fator_chuva": 1.08, "alerta": "Visibilidade reduzida."},
    51: {"descricao": "Garoa Leve", "emoji": "🌦️", "fator_chuva": 1.10, "alerta": "Pista escorregadia."},
    53: {"descricao": "Garoa Moderada", "emoji": "🌧️", "fator_chuva": 1.15, "alerta": "Garoa contínua nos corredores urbanos."},
    55: {"descricao": "Garoa Densa", "emoji": "🌧️", "fator_chuva": 1.20, "alerta": "Atenção em descidas e vias expressas."},
    61: {"descricao": "Chuva Fraca", "emoji": "🌦️", "fator_chuva": 1.15, "alerta": "Chuva leve, velocidade reduzida."},
    63: {"descricao": "Chuva Moderada", "emoji": "🌧️", "fator_chuva": 1.25, "alerta": "Chuva moderada em BH. Lentidão em avenidas principais."},
    65: {"descricao": "Chuva Forte", "emoji": "🌧️🚨", "fator_chuva": 1.40, "alerta": "Alerta Defesa Civil: Risco de retenções nas vias Amazonas, Cristiano Machado e Vilarinho."},
    80: {"descricao": "Pancadas de Chuva Leves", "emoji": "🌦️", "fator_chuva": 1.15, "alerta": "Pancadas isoladas na Grande BH."},
    81: {"descricao": "Pancadas de Chuva Moderadas", "emoji": "🌧️", "fator_chuva": 1.25, "alerta": "Pancadas de chuva com lentidão localizada."},
    82: {"descricao": "Pancadas de Chuva Violentas / Temporal", "emoji": "⛈️🚨", "fator_chuva": 1.55, "alerta": "ALERTA MÁXIMO DEFESA CIVIL BH: Temporal severo! Alto risco de alagamentos e vias bloqueadas."},
    95: {"descricao": "Tempestade com Trovoadas", "emoji": "⛈️", "fator_chuva": 1.45, "alerta": "Tempestade elétrica e chuva forte. Evite trechos baixos."},
    96: {"descricao": "Tempestade com Granizo Leve", "emoji": "⛈️❄️", "fator_chuva": 1.50, "alerta": "Granizo pontual na Região Metropolitana."},
    99: {"descricao": "Tempestade Severa com Granizo", "emoji": "⛈️🚨", "fator_chuva": 1.60, "alerta": "Tempestade severa. Trânsito crítico em todas as regionais."}
}


def obter_clima_atual_bh(timeout_segundos: int = 4) -> Dict[str, Any]:
    """
    Consulta a API Open-Meteo para obter os dados meteorológicos atuais de Belo Horizonte.

    Args:
        timeout_segundos (int): Tempo limite de requisição HTTP (padrão: 4s).

    Returns:
        Dict[str, Any]: Dicionário estruturado com temperatura, precipitação, umidade,
            vento, código WMO, status da requisição e recomendações de tráfego.
    """
    base_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": str(LAT_BH),
        "longitude": str(LON_BH),
        "current": "temperature_2m,relative_humidity_2m,precipitation,rain,weather_code,wind_speed_10m",
        "timezone": "America/Sao_Paulo"
    }

    data: Dict[str, Any] = {}

    try:
        if HAS_REQUESTS:
            resp = requests.get(base_url, params=params, timeout=timeout_segundos)
            resp.raise_for_status()
            data = resp.json()
        else:
            query_str = urllib.parse.urlencode(params)
            full_url = f"{base_url}?{query_str}"
            req = urllib.request.Request(full_url, headers={"User-Agent": "BHTransitAI/2.0"})
            with urllib.request.urlopen(req, timeout=timeout_segundos) as response:
                content = response.read().decode('utf-8')
                data = json.loads(content)

        current = data.get("current", {})

        temp_c = float(current.get("temperature_2m", 24.0))
        umidade_pct = int(current.get("relative_humidity_2m", 65))
        precipitacao_mm = float(current.get("precipitation", 0.0))
        vento_kmh = float(current.get("wind_speed_10m", 10.0))
        code_wmo = int(current.get("weather_code", 0))

        info_wmo = WMO_WEATHER_CODES.get(code_wmo, {
            "descricao": "Tempo Estável",
            "emoji": "☀️",
            "fator_chuva": 1.0,
            "alerta": "Condições típicas de trânsito em BH."
        })

        if precipitacao_mm > 15.0:
            fator_calculado = 1.50
            alerta_dinamico = "⛈️🚨 Volume extremo de chuva (> 15mm/h). Alerta máximo de saturação viária."
        elif precipitacao_mm > 5.0:
            fator_calculado = 1.30
            alerta_dinamico = "🌧️ Chuva acumulada intensa. Redução de velocidade nos corredores expressos."
        elif precipitacao_mm > 0.5:
            fator_calculado = 1.15
            alerta_dinamico = "🌦️ Chuva leve/moderada em andamento."
        else:
            fator_calculado = info_wmo["fator_chuva"]
            alerta_dinamico = info_wmo["alerta"]

        return {
            "sucesso": True,
            "fonte": "Open-Meteo API (Tempo Real)",
            "cidade": "Belo Horizonte - MG",
            "temperatura_c": temp_c,
            "umidade_pct": umidade_pct,
            "precipitacao_mm": precipitacao_mm,
            "vento_kmh": vento_kmh,
            "codigo_wmo": code_wmo,
            "descricao": info_wmo["descricao"],
            "emoji": info_wmo["emoji"],
            "fator_chuva": fator_calculado,
            "alerta": alerta_dinamico,
            "horario_consulta": current.get("time", "")
        }

    except Exception as e:
        logger.warning(f"Não foi possível consultar API Open-Meteo ({e}). Utilizando modo determinístico de contingência.")
        return {
            "sucesso": False,
            "fonte": "Simulação / Fallback Offline",
            "cidade": "Belo Horizonte - MG",
            "temperatura_c": 23.5,
            "umidade_pct": 60,
            "precipitacao_mm": 0.0,
            "vento_kmh": 12.0,
            "codigo_wmo": 1,
            "descricao": "Ensolarado / Tempo Bom (Offline)",
            "emoji": "☀️",
            "fator_chuva": 1.0,
            "alerta": "Modo offline: Tráfego operando em condições meteorológicas normais.",
            "horario_consulta": "Offline"
        }
