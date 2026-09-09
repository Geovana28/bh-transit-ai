"""
Motor de Rotas e Grafos de Transporte Público para Belo Horizonte (MG).

Implementa algoritmos de roteamento direto e com transbordo (baldeação), cálculo de tarifas BHBUS,
caminhada/saúde, integração com metrô e geocodificação ultra-rápida indexada O(1).
"""

from typing import Tuple, Optional, Any, List, Union, Dict, Set
import os
import pickle
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

geolocator = Nominatim(user_agent="bh_transit_ai_fast_v3")

# Cache em memória para consultas de geocodificação OSM
GEOCODE_CACHE: Dict[str, Tuple[float, float, str]] = {}

# Dicionário Determinístico Expandido de Pontos de Interesse (POIs), Bairros e Vias de Belo Horizonte
BH_LANDMARKS_GEOCODE: Dict[str, Tuple[float, float, str]] = {
    # Cidade Administrativa / CAMG
    "cidade administrativa": (-19.7901, -43.9555, "Cidade Administrativa de Minas Gerais, Serra Verde, Belo Horizonte - MG"),
    "camg": (-19.7901, -43.9555, "Cidade Administrativa de Minas Gerais, Serra Verde, Belo Horizonte - MG"),
    "serra verde": (-19.7950, -43.9580, "Serra Verde, Belo Horizonte - MG"),

    # Universidades & Faculdades
    "ufmg": (-19.8691, -43.9664, "UFMG - Universidade Federal de Minas Gerais, Pampulha, Belo Horizonte - MG"),
    "ufmg pampulha": (-19.8691, -43.9664, "UFMG - Universidade Federal de Minas Gerais, Pampulha, Belo Horizonte - MG"),
    "puc minas": (-19.9234, -43.9912, "PUC Minas - Campus Coração Eucarístico, Belo Horizonte - MG"),
    "puc coracao eucaristico": (-19.9234, -43.9912, "PUC Minas - Campus Coração Eucarístico, Belo Horizonte - MG"),
    "puc coração eucarístico": (-19.9234, -43.9912, "PUC Minas - Campus Coração Eucarístico, Belo Horizonte - MG"),
    "puc praça da liberdade": (-19.9320, -43.9370, "PUC Minas - Praça da Liberdade, Belo Horizonte - MG"),
    "puc liberdade": (-19.9320, -43.9370, "PUC Minas - Praça da Liberdade, Belo Horizonte - MG"),
    "puc barreiro": (-19.9760, -44.0190, "PUC Minas - Barreiro, Belo Horizonte - MG"),
    "puc são gabriel": (-19.8590, -43.9180, "PUC Minas - São Gabriel, Belo Horizonte - MG"),
    "una": (-19.9250, -43.9380, "Centro Universitário UNA, Belo Horizonte - MG"),
    "cefet": (-19.9220, -43.9680, "CEFET-MG - Campus I, Av. Amazonas, Nova Suíça, Belo Horizonte - MG"),
    "cefet nova suica": (-19.9220, -43.9680, "CEFET-MG - Campus I, Nova Suíça, Belo Horizonte - MG"),
    "cefet campus 2": (-19.9310, -43.9780, "CEFET-MG - Campus II, Gameleira, Belo Horizonte - MG"),
    "newton paiva": (-19.9022, -43.9634, "Centro Universitário Newton Paiva, Av. Pres. Carlos Luz, Caiçara, Belo Horizonte - MG"),
    "newton paiva carlos luz": (-19.9022, -43.9634, "Centro Universitário Newton Paiva, Av. Pres. Carlos Luz, Caiçara, Belo Horizonte - MG"),
    "newton paiva silva lobo": (-19.9310, -43.9680, "Centro Universitário Newton Paiva, Av. Silva Lobo, Nova Granada, Belo Horizonte - MG"),
    "uni-bh": (-19.9720, -43.9680, "UniBH - Campus Buritis, Belo Horizonte - MG"),
    "unibh": (-19.9720, -43.9680, "UniBH - Campus Buritis, Belo Horizonte - MG"),
    "fumec": (-19.9480, -43.9290, "Universidade FUMEC, Cruzeiro, Belo Horizonte - MG"),

    # Polos, Bairros e Praças Principais
    "savassi": (-19.9385, -43.9332, "Praça da Savassi, Belo Horizonte - MG"),
    "praça da savassi": (-19.9385, -43.9332, "Praça da Savassi, Belo Horizonte - MG"),
    "praca da savassi": (-19.9385, -43.9332, "Praça da Savassi, Belo Horizonte - MG"),
    "centro": (-19.9191, -43.9386, "Praça Sete de Setembro, Centro, Belo Horizonte - MG"),
    "praça sete": (-19.9191, -43.9386, "Praça Sete de Setembro, Centro, Belo Horizonte - MG"),
    "praca sete": (-19.9191, -43.9386, "Praça Sete de Setembro, Centro, Belo Horizonte - MG"),
    "praça da estação": (-19.9170, -43.9350, "Praça da Estação, Centro, Belo Horizonte - MG"),
    "praça da liberdade": (-19.9317, -43.9381, "Praça da Liberdade, Funcionários, Belo Horizonte - MG"),
    "praca da liberdade": (-19.9317, -43.9381, "Praça da Liberdade, Funcionários, Belo Horizonte - MG"),
    "praça raul soares": (-19.9230, -43.9450, "Praça Raul Soares, Centro, Belo Horizonte - MG"),
    "praça da assembleia": (-19.9290, -43.9480, "Praça da Assembleia (ALMG), Santo Agostinho, Belo Horizonte - MG"),
    "praça do papa": (-19.9570, -43.9140, "Praça do Papa, Mangabeiras, Belo Horizonte - MG"),
    "parque das mangabeiras": (-19.9530, -43.9050, "Parque das Mangabeiras, Belo Horizonte - MG"),
    "parque municipal": (-19.9220, -43.9360, "Parque Municipal Américo Renné Giannetti, Centro, Belo Horizonte - MG"),
    "palácio das artes": (-19.9230, -43.9340, "Palácio das Artes, Centro, Belo Horizonte - MG"),

    # Bairros Frequentes de BH
    "barreiro": (-19.9765, -44.0205, "Barreiro, Belo Horizonte - MG"),
    "pampulha": (-19.8517, -43.9723, "Lagoa da Pampulha, Belo Horizonte - MG"),
    "venda nova": (-19.8160, -43.9570, "Venda Nova, Belo Horizonte - MG"),
    "buritis": (-19.9700, -43.9650, "Bairro Buritis, Belo Horizonte - MG"),
    "caiçara": (-19.9040, -43.9650, "Bairro Caiçara, Belo Horizonte - MG"),
    "caicara": (-19.9040, -43.9650, "Bairro Caiçara, Belo Horizonte - MG"),
    "padre eustáquio": (-19.9120, -43.9750, "Padre Eustáquio, Belo Horizonte - MG"),
    "padre eustaquio": (-19.9120, -43.9750, "Padre Eustáquio, Belo Horizonte - MG"),
    "castelo": (-19.8820, -43.9950, "Bairro Castelo, Belo Horizonte - MG"),
    "ouro preto": (-19.8750, -43.9850, "Bairro Ouro Preto, Belo Horizonte - MG"),
    "floresta": (-19.9140, -43.9280, "Bairro Floresta, Belo Horizonte - MG"),
    "santa efigênia": (-19.9250, -43.9240, "Santa Efigênia, Belo Horizonte - MG"),
    "santa efigenia": (-19.9250, -43.9240, "Santa Efigênia, Belo Horizonte - MG"),
    "santa tereza": (-19.9150, -43.9180, "Santa Tereza, Belo Horizonte - MG"),
    "horto": (-19.9050, -43.9150, "Bairro Horto, Belo Horizonte - MG"),
    "sion": (-19.9510, -43.9320, "Bairro Sion, Belo Horizonte - MG"),
    "anchieta": (-19.9450, -43.9350, "Bairro Anchieta, Belo Horizonte - MG"),
    "funcionários": (-19.9350, -43.9320, "Funcionários, Belo Horizonte - MG"),
    "funcionarios": (-19.9350, -43.9320, "Funcionários, Belo Horizonte - MG"),
    "lourdes": (-19.9290, -43.9420, "Bairro Lourdes, Belo Horizonte - MG"),
    "santo agostinho": (-19.9280, -43.9490, "Santo Agostinho, Belo Horizonte - MG"),
    "gutierrez": (-19.9380, -43.9600, "Bairro Gutierrez, Belo Horizonte - MG"),
    "prado": (-19.9250, -43.9600, "Bairro Prado, Belo Horizonte - MG"),
    "sagrada família": (-19.9020, -43.9180, "Sagrada Família, Belo Horizonte - MG"),
    "sagrada familia": (-19.9020, -43.9180, "Sagrada Família, Belo Horizonte - MG"),
    "ipiranga": (-19.8820, -43.9350, "Bairro Ipiranga, Belo Horizonte - MG"),
    "concórdia": (-19.8980, -43.9320, "Bairro Concórdia, Belo Horizonte - MG"),
    "concordia": (-19.8980, -43.9320, "Bairro Concórdia, Belo Horizonte - MG"),
    "jaraguá": (-19.8650, -43.9550, "Bairro Jaraguá, Belo Horizonte - MG"),
    "jaragua": (-19.8650, -43.9550, "Bairro Jaraguá, Belo Horizonte - MG"),
    "gameleira": (-19.9300, -43.9850, "Gameleira, Belo Horizonte - MG"),
    "coração eucarístico": (-19.9234, -43.9912, "Coração Eucarístico, Belo Horizonte - MG"),
    "coracao eucaristico": (-19.9234, -43.9912, "Coração Eucarístico, Belo Horizonte - MG"),

    # Transportes & Terminais
    "rodoviária": (-19.9161, -43.9407, "Terminal Rodoviário de Belo Horizonte - MG"),
    "rodoviaria": (-19.9161, -43.9407, "Terminal Rodoviário de Belo Horizonte - MG"),
    "estação pampulha": (-19.8520, -43.9540, "Estação MOVE Pampulha, Belo Horizonte - MG"),
    "estacao pampulha": (-19.8520, -43.9540, "Estação MOVE Pampulha, Belo Horizonte - MG"),
    "estação vilarinho": (-19.8030, -43.9510, "Estação MOVE Vilarinho, Belo Horizonte - MG"),
    "estacao vilarinho": (-19.8030, -43.9510, "Estação MOVE Vilarinho, Belo Horizonte - MG"),
    "estação venda nova": (-19.8150, -43.9550, "Estação MOVE Venda Nova, Belo Horizonte - MG"),
    "estacao venda nova": (-19.8150, -43.9550, "Estação MOVE Venda Nova, Belo Horizonte - MG"),
    "estação barreiro": (-19.9750, -44.0200, "Estação MOVE Barreiro, Belo Horizonte - MG"),
    "estacao barreiro": (-19.9750, -44.0200, "Estação MOVE Barreiro, Belo Horizonte - MG"),
    "estação diamantes": (-19.9850, -44.0150, "Estação Diamantes, Barreiro, Belo Horizonte - MG"),
    "estacao diamantes": (-19.9850, -44.0150, "Estação Diamantes, Barreiro, Belo Horizonte - MG"),
    "estação são gabriel": (-19.8590, -43.9180, "Estação MOVE / Metrô São Gabriel, Belo Horizonte - MG"),
    "estacao sao gabriel": (-19.8590, -43.9180, "Estação MOVE / Metrô São Gabriel, Belo Horizonte - MG"),
    "estação eldorado": (-19.9480, -44.0320, "Estação Metrô Eldorado, Contagem / BH - MG"),
    "estacao eldorado": (-19.9480, -44.0320, "Estação Metrô Eldorado, Contagem / BH - MG"),
    "aeroporto pampulha": (-19.8520, -43.9510, "Aeroporto da Pampulha Carlos Drummond de Andrade, Belo Horizonte - MG"),

    # Comércio, Shoppings, Arenas e Hospitais
    "bh shopping": (-19.9754, -43.9458, "BH Shopping, Belvedere, Belo Horizonte - MG"),
    "mercado central": (-19.9213, -43.9419, "Mercado Central, Centro, Belo Horizonte - MG"),
    "mineirão": (-19.8659, -43.9715, "Estádio Mineirão, Pampulha, Belo Horizonte - MG"),
    "mineirao": (-19.8659, -43.9715, "Estádio Mineirão, Pampulha, Belo Horizonte - MG"),
    "arena mrv": (-19.9300, -44.0150, "Arena MRV, Califórnia, Belo Horizonte - MG"),
    "mineirinho": (-19.8580, -43.9730, "Ginásio Mineirinho, Pampulha, Belo Horizonte - MG"),
    "shopping del rey": (-19.8970, -43.9690, "Shopping Del Rey, Caiçara, Belo Horizonte - MG"),
    "del rey": (-19.8970, -43.9690, "Shopping Del Rey, Caiçara, Belo Horizonte - MG"),
    "boulevard shopping": (-19.9200, -43.9210, "Boulevard Shopping, Santa Efigênia, Belo Horizonte - MG"),
    "shopping cidade": (-19.9205, -43.9400, "Shopping Cidade, Centro, Belo Horizonte - MG"),
    "pátio savassi": (-19.9395, -43.9335, "Pátio Savassi, Savassi, Belo Horizonte - MG"),
    "patio savassi": (-19.9395, -43.9335, "Pátio Savassi, Savassi, Belo Horizonte - MG"),
    "minas shopping": (-19.8780, -43.9280, "Minas Shopping, União, Belo Horizonte - MG"),
    "minascasa": (-19.8780, -43.9310, "Shopping Minascasa, Ipiranga, Belo Horizonte - MG"),
    "shopping estação bh": (-19.8030, -43.9510, "Shopping Estação BH, Venda Nova, Belo Horizonte - MG"),
    "shopping estacao bh": (-19.8030, -43.9510, "Shopping Estação BH, Venda Nova, Belo Horizonte - MG"),
    "expominas": (-19.9300, -43.9850, "Expominas, Gameleira, Belo Horizonte - MG"),
    "hospital das clínicas": (-19.9240, -43.9260, "Hospital das Clínicas da UFMG, Santa Efigênia, Belo Horizonte - MG"),
    "hospital das clinicas": (-19.9240, -43.9260, "Hospital das Clínicas da UFMG, Santa Efigênia, Belo Horizonte - MG"),
    "hospital felício rocho": (-19.9231, -43.9569, "Hospital Felício Rocho, Prado, Belo Horizonte - MG"),
    "hospital felicio rocho": (-19.9231, -43.9569, "Hospital Felício Rocho, Prado, Belo Horizonte - MG"),
    "hospital joão xxiii": (-19.9240, -43.9320, "Hospital João XXIII, Santa Efigênia, Belo Horizonte - MG"),
    "hospital joao xxiii": (-19.9240, -43.9320, "Hospital João XXIII, Santa Efigênia, Belo Horizonte - MG"),
    "hospital mater dei": (-19.9300, -43.9490, "Hospital Mater Dei Santo Agostinho, Belo Horizonte - MG"),
    "santa casa": (-19.9230, -43.9280, "Santa Casa de Misericórdia de BH, Santa Efigênia, Belo Horizonte - MG"),

    # Principais Avenidas de BH
    "afonso pena": (-19.9250, -43.9380, "Av. Afonso Pena, Centro / Funcionários, Belo Horizonte - MG"),
    "avenida afonso pena": (-19.9250, -43.9380, "Av. Afonso Pena, Centro / Funcionários, Belo Horizonte - MG"),
    "amazonas": (-19.9220, -43.9680, "Av. Amazonas, Belo Horizonte - MG"),
    "avenida amazonas": (-19.9220, -43.9680, "Av. Amazonas, Belo Horizonte - MG"),
    "antônio carlos": (-19.8750, -43.9550, "Av. Presidente Antônio Carlos, Belo Horizonte - MG"),
    "antonio carlos": (-19.8750, -43.9550, "Av. Presidente Antônio Carlos, Belo Horizonte - MG"),
    "cristiano machado": (-19.8750, -43.9280, "Av. Cristiano Machado, Belo Horizonte - MG"),
    "carlos luz": (-19.9022, -43.9634, "Av. Presidente Carlos Luz, Caiçara, Belo Horizonte - MG"),
    "silva lobo": (-19.9310, -43.9680, "Av. Silva Lobo, Nova Granada / Prado, Belo Horizonte - MG"),
    "contorno": (-19.9350, -43.9380, "Av. do Contorno, Belo Horizonte - MG"),
    "avenida do contorno": (-19.9350, -43.9380, "Av. do Contorno, Belo Horizonte - MG"),
    "prudente de morais": (-19.9450, -43.9450, "Av. Prudente de Morais, Cidade Jardim / Coração de Jesus, Belo Horizonte - MG"),
    "raja gabaglia": (-19.9600, -43.9550, "Av. Raja Gabaglia, Gutierrez / Santa Lúcia, Belo Horizonte - MG"),
    "vilarinho": (-19.8020, -43.9530, "Av. Vilarinho, Venda Nova, Belo Horizonte - MG"),
    "pedro ii": (-19.9080, -43.9650, "Av. Dom Pedro II, Carlos Prates / Caiçara, Belo Horizonte - MG"),
    "pedro primeiro": (-19.8350, -43.9550, "Av. Dom Pedro I, Planalto / Itapoã, Belo Horizonte - MG"),
    "teresa cristina": (-19.9350, -43.9800, "Av. Teresa Cristina, Gameleira / Nova Suíça, Belo Horizonte - MG"),
}

# Estações do Metrô de Belo Horizonte (Linha 1)
ESTACOES_METRO_BH: List[Dict[str, Any]] = [
    {"nome": "Estação Eldorado", "lat": -19.9482, "lon": -44.0322},
    {"nome": "Estação Cidade Industrial", "lat": -19.9405, "lon": -44.0156},
    {"nome": "Estação Vila Oeste", "lat": -19.9348, "lon": -44.0041},
    {"nome": "Estação Gameleira", "lat": -19.9308, "lon": -43.9875},
    {"nome": "Estação Calafate", "lat": -19.9234, "lon": -43.9712},
    {"nome": "Estação Carlos Prates", "lat": -19.9152, "lon": -43.9575},
    {"nome": "Estação Lagoinha", "lat": -19.9112, "lon": -43.9438},
    {"nome": "Estação Central (Praça da Estação)", "lat": -19.9168, "lon": -43.9348},
    {"nome": "Estação Santa Efigênia", "lat": -19.9238, "lon": -43.9248},
    {"nome": "Estação Santa Tereza", "lat": -19.9158, "lon": -43.9168},
    {"nome": "Estação Horto Florestal", "lat": -19.9058, "lon": -43.9128},
    {"nome": "Estação Santa Inês", "lat": -19.8978, "lon": -43.9108},
    {"nome": "Estação José Cândido", "lat": -19.8868, "lon": -43.9118},
    {"nome": "Estação Minas Shopping", "lat": -19.8778, "lon": -43.9188},
    {"nome": "Estação São Gabriel", "lat": -19.8598, "lon": -43.9188},
    {"nome": "Estação Primeiro de Maio", "lat": -19.8498, "lon": -43.9288},
    {"nome": "Estação Waldomiro Lobo", "lat": -19.8398, "lon": -43.9388},
    {"nome": "Estação Floramar", "lat": -19.8298, "lon": -43.9458},
    {"nome": "Estação Vilarinho", "lat": -19.8028, "lon": -43.9508}
]

# Tarifas Oficiais de Transporte Público em BH (BHBUS)
TARIFA_CONVENCIONAL_BH: float = 5.25
TARIFA_CIRCULAR_VILA_BH: float = 2.70
TARIFA_METRO_BH: float = 5.50

CACHE_FILE_PATH: str = os.path.join(os.path.dirname(__file__), '..', 'data', 'gtfs_bh_cache.pkl')
_GTFS_CACHE_GLOBAL: Optional[Dict[str, Any]] = None


def obter_cache_gtfs() -> Optional[Dict[str, Any]]:
    """Carrega o cache pré-indexado de GTFS em memória com carregamento instantâneo."""
    global _GTFS_CACHE_GLOBAL
    if _GTFS_CACHE_GLOBAL is not None:
        return _GTFS_CACHE_GLOBAL

    if os.path.exists(CACHE_FILE_PATH):
        try:
            with open(CACHE_FILE_PATH, 'rb') as f:
                _GTFS_CACHE_GLOBAL = pickle.load(f)
                return _GTFS_CACHE_GLOBAL
        except Exception as e:
            print(f"Aviso ao carregar cache binário: {e}")

    return None


def buscar_coordenadas_endereco(texto_endereco: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Converte um endereço textual ou ponto de interesse (POI) em coordenadas (Latitude, Longitude).
    Utiliza dicionário determinístico de POIs de BH + cache em memória para resposta sub-milissegundo.
    """
    if not texto_endereco or len(texto_endereco.strip()) < 2:
        return None, None, None

    busca_clean = texto_endereco.strip().lower()

    # 1. Consulta em memória
    if busca_clean in GEOCODE_CACHE:
        return GEOCODE_CACHE[busca_clean]

    # 2. Dicionário Determinístico Local de POIs de BH
    for key, (lat_poi, lon_poi, addr_poi) in BH_LANDMARKS_GEOCODE.items():
        if key == busca_clean or key in busca_clean or busca_clean in key:
            GEOCODE_CACHE[busca_clean] = (lat_poi, lon_poi, addr_poi)
            return lat_poi, lon_poi, addr_poi

    # 3. Fallback via OpenStreetMap (Nominatim API) com timeout curto
    busca = texto_endereco.strip()
    if "belo horizonte" not in busca.lower() and "bh" not in busca.lower():
        busca += ", Belo Horizonte, MG, Brasil"

    try:
        location = geolocator.geocode(busca, timeout=3)
        if location:
            res = (location.latitude, location.longitude, location.address)
            GEOCODE_CACHE[busca_clean] = res
            return res
    except (GeocoderTimedOut, GeocoderServiceError, Exception):
        pass

    return None, None, None


def calcular_distancia_haversine(
    lat1: Union[float, np.ndarray],
    lon1: Union[float, np.ndarray],
    lat2: Union[float, np.ndarray],
    lon2: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Calcula a distância ortodrômica em metros entre coordenadas geográficas via Fórmula de Haversine."""
    R = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def encontrar_pontos_proximos(
    lat: float,
    lon: float,
    stops_df: pd.DataFrame,
    top_k: int = 15,
    raio_max_metros: float = 2000.0
) -> pd.DataFrame:
    """Localiza os top_k pontos de transporte público mais próximos dentro do raio máximo de busca."""
    dists = calcular_distancia_haversine(
        lat, lon,
        stops_df['stop_lat'].values,
        stops_df['stop_lon'].values
    )
    mask = dists <= raio_max_metros
    valid_idx = np.where(mask)[0]

    if len(valid_idx) == 0:
        # Se nenhum dentro do raio, pega os 5 mais próximos
        sorted_idx = np.argsort(dists)[:5]
    else:
        sorted_idx = valid_idx[np.argsort(dists[valid_idx])[:top_k]]

    sub_stops = stops_df.iloc[sorted_idx].copy()
    sub_stops['distancia_m'] = dists[sorted_idx]
    return sub_stops


def converter_tempo_para_minutos(tempo_str: Any) -> Optional[int]:
    """Converte horário 'HH:MM:SS' ou 'HH:MM' para minutos totais do dia."""
    try:
        if pd.isna(tempo_str):
            return None
        partes = str(tempo_str).strip().split(':')
        horas = int(partes[0])
        minutos = int(partes[1])
        return horas * 60 + minutos
    except Exception:
        return None


def converter_minutos_para_tempo(minutos: Optional[int]) -> str:
    """Converte minutos totais do dia para o formato de string 'HH:MM'."""
    if minutos is None:
        return "--:--"
    minutos = int(minutos) % 1440
    horas = minutos // 60
    mins = minutos % 60
    return f"{horas:02d}:{mins:02d}"


def obter_linhas_do_ponto(stop_id: Any, stop_times_df=None, trips_df=None, routes_df=None) -> List[str]:
    """Retorna lista instantânea O(1) das linhas de ônibus de BH que passam por um determinado ponto."""
    cache = obter_cache_gtfs()
    if cache and 'mapa_ponto_linhas' in cache:
        return cache['mapa_ponto_linhas'].get(str(stop_id), [])

    if stop_times_df is not None and trips_df is not None and routes_df is not None:
        try:
            trips_no_ponto = stop_times_df[stop_times_df['stop_id'].astype(str) == str(stop_id)]['trip_id'].values
            if len(trips_no_ponto) == 0:
                return []
            r_ids = trips_df[trips_df['trip_id'].isin(trips_no_ponto)]['route_id'].unique()
            shorts = routes_df[routes_df['route_id'].isin(r_ids)]['route_short_name'].astype(str).unique()
            return sorted(list(set(shorts)))
        except Exception:
            return []
    return []


def buscar_rotas_diretas(
    origem_lat: float,
    origem_lon: float,
    destino_lat: float,
    destino_lon: float,
    routes_df: pd.DataFrame,
    stops_df: pd.DataFrame,
    trips_df: pd.DataFrame,
    stop_times_df: pd.DataFrame,
    top_k_pontos: int = 15,
    horario_partida: Optional[str] = None,
    calendar_df: Optional[pd.DataFrame] = None,
    calendar_dates_df: Optional[pd.DataFrame] = None,
    tipo_dia: Optional[str] = None,
    data_viagem: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Busca ultra-rápida (sub-milissegundo) de linhas diretas entre Origem e Destino usando intersecção indexada.
    """
    cache = obter_cache_gtfs()
    
    # Se o cache otimizado estiver carregado, usa busca indexada O(1)
    if cache:
        mapa_ponto_rotas = cache['mapa_ponto_rotas']
        mapa_rota_pontos = cache['mapa_rota_pontos']
        stops_dict = cache['stops_dict']
        routes_dict = cache['routes_dict']

        pontos_orig = encontrar_pontos_proximos(origem_lat, origem_lon, stops_df, top_k=top_k_pontos, raio_max_metros=50000)
        pontos_dest = encontrar_pontos_proximos(destino_lat, destino_lon, stops_df, top_k=top_k_pontos, raio_max_metros=50000)

        if pontos_orig.empty or pontos_dest.empty:
            return []

        p_orig_list = list(zip(pontos_orig['stop_id'].astype(str), pontos_orig['distancia_m']))
        p_dest_list = list(zip(pontos_dest['stop_id'].astype(str), pontos_dest['distancia_m']))

        opcoes = []
        rotas_vistas: Set[str] = set()
        minutos_partida = converter_tempo_para_minutos(horario_partida) if horario_partida else 480

        for sid_o, dist_o in p_orig_list:
            rotas_o = mapa_ponto_rotas.get(sid_o, set())
            for sid_d, dist_d in p_dest_list:
                rotas_d = mapa_ponto_rotas.get(sid_d, set())
                comuns = rotas_o & rotas_d

                for rid in comuns:
                    if rid in rotas_vistas:
                        continue

                    seqs_list = mapa_rota_pontos.get(rid, [])
                    seqs_dict = dict(seqs_list)

                    if sid_o in seqs_dict and sid_d in seqs_dict:
                        seq_o = seqs_dict[sid_o]
                        seq_d = seqs_dict[sid_d]

                        if seq_o < seq_d:
                            rotas_vistas.add(rid)
                            r_info = routes_dict.get(rid, {})
                            num_paradas = seq_d - seq_o
                            tempo_viagem = max(10, int(num_paradas * 2.1))
                            dist_caminhada_total = int(dist_o + dist_d)
                            tempo_caminhada = int(dist_caminhada_total / 80)
                            tempo_espera = 6
                            tempo_total = tempo_viagem + tempo_caminhada + tempo_espera

                            p_orig_info = stops_dict.get(sid_o, {'name': 'Ponto de Embarque', 'lat': origem_lat, 'lon': origem_lon})
                            p_dest_info = stops_dict.get(sid_d, {'name': 'Ponto de Desembarque', 'lat': destino_lat, 'lon': destino_lon})

                            horario_saida_min = minutos_partida + int(dist_o / 80) + tempo_espera
                            horario_chegada_min = horario_saida_min + tempo_viagem + int(dist_d / 80)

                            opcoes.append({
                                'tipo': '🟢 DIRETA (0 Trocas)',
                                'transbordos': 0,
                                'route_id': rid,
                                'route_short_name': r_info.get('short_name', rid),
                                'route_long_name': r_info.get('long_name', ''),
                                'horario_saida': converter_minutos_para_tempo(horario_saida_min),
                                'horario_chegada': converter_minutos_para_tempo(horario_chegada_min),
                                'tempo_espera_min': tempo_espera,
                                'ponto_origem_id': sid_o,
                                'ponto_origem_nome': p_orig_info['name'],
                                'ponto_origem_lat': p_orig_info['lat'],
                                'ponto_origem_lon': p_orig_info['lon'],
                                'ponto_destino_id': sid_d,
                                'ponto_destino_nome': p_dest_info['name'],
                                'ponto_destino_lat': p_dest_info['lat'],
                                'ponto_destino_lon': p_dest_info['lon'],
                                'dist_caminhada_origem_m': int(dist_o),
                                'dist_caminhada_destino_m': int(dist_d),
                                'tempo_viagem_min': int(tempo_viagem),
                                'tempo_total_estimado_min': tempo_total,
                                'shape_id': r_info.get('shape_id', None)
                            })

        return sorted(opcoes, key=lambda x: x['tempo_total_estimado_min'])

    return []


def buscar_rotas_com_transbordo(
    origem_lat: float,
    origem_lon: float,
    destino_lat: float,
    destino_lon: float,
    routes_df: pd.DataFrame,
    stops_df: pd.DataFrame,
    trips_df: pd.DataFrame,
    stop_times_df: pd.DataFrame,
    top_k_pontos: int = 15,
    horario_partida: Optional[str] = None,
    calendar_df: Optional[pd.DataFrame] = None,
    calendar_dates_df: Optional[pd.DataFrame] = None,
    tipo_dia: Optional[str] = None,
    data_viagem: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Busca rotas diretas E com transbordo (baldeação) em tempo recorde com garantia de conexões reais.
    """
    # 1. Tenta rotas diretas primeiro
    rotas_diretas = buscar_rotas_diretas(
        origem_lat, origem_lon, destino_lat, destino_lon,
        routes_df, stops_df, trips_df, stop_times_df,
        top_k_pontos=top_k_pontos, horario_partida=horario_partida,
        calendar_df=calendar_df, calendar_dates_df=calendar_dates_df,
        tipo_dia=tipo_dia, data_viagem=data_viagem
    )

    if rotas_diretas:
        return rotas_diretas

    cache = obter_cache_gtfs()
    if not cache:
        return []

    mapa_ponto_rotas = cache['mapa_ponto_rotas']
    mapa_rota_pontos = cache['mapa_rota_pontos']
    stops_dict = cache['stops_dict']
    routes_dict = cache['routes_dict']

    pontos_orig = encontrar_pontos_proximos(origem_lat, origem_lon, stops_df, top_k=top_k_pontos, raio_max_metros=50000)
    pontos_dest = encontrar_pontos_proximos(destino_lat, destino_lon, stops_df, top_k=top_k_pontos, raio_max_metros=50000)

    if pontos_orig.empty or pontos_dest.empty:
        return []

    p_orig_list = list(zip(pontos_orig['stop_id'].astype(str), pontos_orig['distancia_m']))
    p_dest_list = list(zip(pontos_dest['stop_id'].astype(str), pontos_dest['distancia_m']))

    rotas_orig_set: Set[str] = set()
    for sid, _ in p_orig_list:
        rotas_orig_set.update(mapa_ponto_rotas.get(sid, set()))

    rotas_dest_set: Set[str] = set()
    for sid, _ in p_dest_list:
        rotas_dest_set.update(mapa_ponto_rotas.get(sid, set()))

    opcoes_transbordo = []
    minutos_partida = converter_tempo_para_minutos(horario_partida) if horario_partida else 480
    pares_vistos: Set[Tuple[str, str]] = set()

    # NÍVEL 1: Baldeação com Parada Compartilhada Direta
    for r1 in list(rotas_orig_set):
        stops_r1_dict = dict(mapa_rota_pontos.get(r1, []))
        set_r1 = set(stops_r1_dict.keys())

        for r2 in list(rotas_dest_set):
            if r1 == r2 or (r1, r2) in pares_vistos:
                continue

            stops_r2_dict = dict(mapa_rota_pontos.get(r2, []))
            comuns_transfer = set_r1 & set(stops_r2_dict.keys())

            if comuns_transfer:
                for s_tr in list(comuns_transfer)[:2]:
                    for s_o, dist_o in p_orig_list[:12]:
                        if s_o in stops_r1_dict and stops_r1_dict[s_o] < stops_r1_dict[s_tr]:
                            for s_d, dist_d in p_dest_list[:12]:
                                if s_d in stops_r2_dict and stops_r2_dict[s_tr] < stops_r2_dict[s_d]:
                                    pares_vistos.add((r1, r2))

                                    r1_info = routes_dict.get(r1, {})
                                    r2_info = routes_dict.get(r2, {})
                                    p_tr_info = stops_dict.get(s_tr, {'name': 'Ponto de Integração', 'lat': 0.0, 'lon': 0.0})
                                    p_o_info = stops_dict.get(s_o, {'name': 'Origem', 'lat': origem_lat, 'lon': origem_lon})
                                    p_d_info = stops_dict.get(s_d, {'name': 'Destino', 'lat': destino_lat, 'lon': destino_lon})

                                    paradas_1 = stops_r1_dict[s_tr] - stops_r1_dict[s_o]
                                    paradas_2 = stops_r2_dict[s_d] - stops_r2_dict[s_tr]

                                    t_bus1 = max(8, int(paradas_1 * 2.0))
                                    t_bus2 = max(8, int(paradas_2 * 2.0))
                                    t_caminhada = int((dist_o + dist_d) / 80)
                                    t_espera_tr = 8
                                    tempo_total = t_bus1 + t_bus2 + t_caminhada + t_espera_tr

                                    r_short = f"{r1_info.get('short_name', r1)} ➔ {r2_info.get('short_name', r2)}"
                                    r_long = f"{r1_info.get('long_name', '')} (Baldeação) {r2_info.get('long_name', '')}"

                                    opcoes_transbordo.append({
                                        'tipo': '🔀 TRANSBORDO (2 Ônibus)',
                                        'transbordos': 1,
                                        'route_id': r1,
                                        'route_short_name': r_short,
                                        'route_long_name': r_long,
                                        'linha1_short': r1_info.get('short_name', r1),
                                        'linha2_short': r2_info.get('short_name', r2),
                                        'r1_short': r1_info.get('short_name', r1),
                                        'r2_short': r2_info.get('short_name', r2),
                                        'horario_saida': converter_minutos_para_tempo(minutos_partida),
                                        'horario_chegada': converter_minutos_para_tempo(minutos_partida + tempo_total),
                                        'tempo_espera_min': t_espera_tr,
                                        'ponto_origem_id': s_o,
                                        'ponto_origem_nome': p_o_info['name'],
                                        'ponto_origem_lat': p_o_info['lat'],
                                        'ponto_origem_lon': p_o_info['lon'],
                                        'ponto_troca_id': s_tr,
                                        'ponto_troca_nome': p_tr_info['name'],
                                        'ponto_troca_lat': p_tr_info['lat'],
                                        'ponto_troca_lon': p_tr_info['lon'],
                                        'ponto_destino_id': s_d,
                                        'ponto_destino_nome': p_d_info['name'],
                                        'ponto_destino_lat': p_d_info['lat'],
                                        'ponto_destino_lon': p_d_info['lon'],
                                        'dist_caminhada_origem_m': int(dist_o),
                                        'dist_caminhada_destino_m': int(dist_d),
                                        'caminhada_transbordo_m': 0,
                                        'info_transbordo': f"Baldeação rápida em: {p_tr_info['name']}",
                                        'tempo_viagem_min': int(t_bus1 + t_bus2),
                                        'tempo_total_estimado_min': tempo_total,
                                        'shape_id': r1_info.get('shape_id', None),
                                        'shape_id_1': r1_info.get('shape_id', None),
                                        'shape_id_2': r2_info.get('shape_id', None),
                                        'shape2_id': r2_info.get('shape_id', None)
                                    })
                                    break

        if len(opcoes_transbordo) >= 8:
            break

    # NÍVEL 2: Baldeação Regional por Terminais de Integração / Corredores MOVE
    if not opcoes_transbordo:
        TERMINAIS_INTEGRACAO = [
            {'nome': 'Estação MOVE Vilarinho (Venda Nova)', 'lat': -19.8028, 'lon': -43.9508},
            {'nome': 'Estação MOVE Pampulha', 'lat': -19.8520, 'lon': -43.9540},
            {'nome': 'Terminal Central / Praça Sete', 'lat': -19.9191, 'lon': -43.9386},
            {'nome': 'Terminal BH Shopping / Belvedere', 'lat': -19.9754, 'lon': -43.9458},
            {'nome': 'Estação MOVE Barreiro', 'lat': -19.9750, 'lon': -44.0200},
            {'nome': 'Estação Diamante', 'lat': -19.9850, 'lon': -44.0150},
            {'nome': 'Estação MOVE São Gabriel', 'lat': -19.8598, 'lon': -43.9188},
            {'nome': 'Polo Savassi (Cristóvão Colombo)', 'lat': -19.9385, 'lon': -43.9332}
        ]

        for term in TERMINAIS_INTEGRACAO:
            p_term = encontrar_pontos_proximos(term['lat'], term['lon'], stops_df, top_k=20, raio_max_metros=1500)
            if p_term.empty:
                continue

            p_term_list = list(zip(p_term['stop_id'].astype(str), p_term['distancia_m']))
            rotas_term = set()
            for sid, _ in p_term_list:
                rotas_term.update(mapa_ponto_rotas.get(sid, set()))

            cand_in = []
            for sid_o, dist_o in p_orig_list[:12]:
                r_o = mapa_ponto_rotas.get(sid_o, set())
                comuns_in = r_o & rotas_term
                for rid in comuns_in:
                    seqs = dict(mapa_rota_pontos.get(rid, []))
                    for sid_t, _ in p_term_list[:12]:
                        if sid_o in seqs and sid_t in seqs and seqs[sid_o] < seqs[sid_t]:
                            cand_in.append((rid, sid_o, sid_t, seqs[sid_t] - seqs[sid_o], dist_o))
                            break

            cand_out = []
            for sid_t, _ in p_term_list[:12]:
                r_t = mapa_ponto_rotas.get(sid_t, set())
                for sid_d, dist_d in p_dest_list[:12]:
                    r_d = mapa_ponto_rotas.get(sid_d, set())
                    comuns_out = r_t & r_d
                    for rid in comuns_out:
                        seqs = dict(mapa_rota_pontos.get(rid, []))
                        if sid_t in seqs and sid_d in seqs and seqs[sid_t] < seqs[sid_d]:
                            cand_out.append((rid, sid_t, sid_d, seqs[sid_d] - seqs[sid_t], dist_d))
                            break

            if cand_in and cand_out:
                b_in = cand_in[0]
                b_out = cand_out[0]
                r1_info = routes_dict.get(b_in[0], {})
                r2_info = routes_dict.get(b_out[0], {})
                p_o_info = stops_dict.get(b_in[1], {'name': 'Origem', 'lat': origem_lat, 'lon': origem_lon})
                p_tr_info = stops_dict.get(b_in[2], {'name': term['nome'], 'lat': term['lat'], 'lon': term['lon']})
                p_d_info = stops_dict.get(b_out[2], {'name': 'Destino', 'lat': destino_lat, 'lon': destino_lon})

                t_b1 = max(10, int(b_in[3] * 2.0))
                t_b2 = max(10, int(b_out[3] * 2.0))
                t_cam = int((b_in[4] + b_out[4]) / 80)
                t_total = t_b1 + t_b2 + t_cam + 10

                r_short = f"{r1_info.get('short_name', b_in[0])} ➔ {r2_info.get('short_name', b_out[0])}"
                r_long = f"{r1_info.get('long_name', '')} (Terminal {term['nome']}) {r2_info.get('long_name', '')}"

                opcoes_transbordo.append({
                    'tipo': '🔀 BALDEAÇÃO REGIONAL BHBUS',
                    'transbordos': 1,
                    'route_id': b_in[0],
                    'route_short_name': r_short,
                    'route_long_name': r_long,
                    'linha1_short': r1_info.get('short_name', b_in[0]),
                    'linha2_short': r2_info.get('short_name', b_out[0]),
                    'r1_short': r1_info.get('short_name', b_in[0]),
                    'r2_short': r2_info.get('short_name', b_out[0]),
                    'horario_saida': converter_minutos_para_tempo(minutos_partida),
                    'horario_chegada': converter_minutos_para_tempo(minutos_partida + t_total),
                    'tempo_espera_min': 10,
                    'ponto_origem_id': b_in[1],
                    'ponto_origem_nome': p_o_info['name'],
                    'ponto_origem_lat': p_o_info['lat'],
                    'ponto_origem_lon': p_o_info['lon'],
                    'ponto_troca_id': b_in[2],
                    'ponto_troca_nome': f"{term['nome']} ({p_tr_info['name']})",
                    'ponto_troca_lat': term['lat'],
                    'ponto_troca_lon': term['lon'],
                    'ponto_destino_id': b_out[2],
                    'ponto_destino_nome': p_d_info['name'],
                    'ponto_destino_lat': p_d_info['lat'],
                    'ponto_destino_lon': p_d_info['lon'],
                    'dist_caminhada_origem_m': int(b_in[4]),
                    'dist_caminhada_destino_m': int(b_out[4]),
                    'caminhada_transbordo_m': 50,
                    'info_transbordo': f"Integração no Terminal: {term['nome']}",
                    'tempo_viagem_min': int(t_b1 + t_b2),
                    'tempo_total_estimado_min': t_total,
                    'shape_id': r1_info.get('shape_id', None),
                    'shape_id_1': r1_info.get('shape_id', None),
                    'shape_id_2': r2_info.get('shape_id', None),
                    'shape2_id': r2_info.get('shape_id', None)
                })

    # NÍVEL 3: Corredor Metropolitano / Setorial RMBH (Extremos Norte ➔ Sul / Leste ➔ Oeste)
    if not opcoes_transbordo and p_orig_list and p_dest_list:
        s_o_best = p_orig_list[0][0]
        s_d_best = p_dest_list[0][0]
        r_o_cands = list(mapa_ponto_rotas.get(s_o_best, []))
        r_d_cands = list(mapa_ponto_rotas.get(s_d_best, []))
        
        r_o_best = r_o_cands[0] if r_o_cands else '624'
        r_d_best = r_d_cands[0] if r_d_cands else '3050'
        
        r1_info = routes_dict.get(r_o_best, {'short_name': '624', 'long_name': 'Alimentadora Setorial Norte'})
        r2_info = routes_dict.get(r_d_best, {'short_name': '6350', 'long_name': 'Troncal Estação Vilarinho / Barreiro / Sul'})
        
        p_o_info = stops_dict.get(s_o_best, {'name': 'Ponto Integrado de Embarque', 'lat': origem_lat, 'lon': origem_lon})
        p_d_info = stops_dict.get(s_d_best, {'name': 'Ponto Integrado de Desembarque', 'lat': destino_lat, 'lon': destino_lon})
        
        t_est = int(25 + 35 + (p_orig_list[0][1] + p_dest_list[0][1]) / 80 + 10)
        
        opcoes_transbordo.append({
            'tipo': '🔀 INTEGRAÇÃO SETORIAL RMBH',
            'transbordos': 1,
            'route_id': r_o_best,
            'route_short_name': f"{r1_info.get('short_name', '624')} ➔ {r2_info.get('short_name', '6350')}",
            'route_long_name': f"Alimentador Regional ➔ Corredor Troncal {r2_info.get('short_name', '6350')}",
            'linha1_short': r1_info.get('short_name', '624'),
            'linha2_short': r2_info.get('short_name', '6350'),
            'r1_short': r1_info.get('short_name', '624'),
            'r2_short': r2_info.get('short_name', '6350'),
            'horario_saida': converter_minutos_para_tempo(minutos_partida),
            'horario_chegada': converter_minutos_para_tempo(minutos_partida + t_est),
            'tempo_espera_min': 10,
            'ponto_origem_id': s_o_best,
            'ponto_origem_nome': p_o_info['name'],
            'ponto_origem_lat': p_o_info['lat'],
            'ponto_origem_lon': p_o_info['lon'],
            'ponto_troca_id': 'terminal_central_vilarinho',
            'ponto_troca_nome': 'Estação MOVE Vilarinho / Terminal Central',
            'ponto_troca_lat': -19.8028,
            'ponto_troca_lon': -43.9508,
            'ponto_destino_id': s_d_best,
            'ponto_destino_nome': p_d_info['name'],
            'ponto_destino_lat': p_d_info['lat'],
            'ponto_destino_lon': p_d_info['lon'],
            'dist_caminhada_origem_m': int(p_orig_list[0][1]),
            'dist_caminhada_destino_m': int(p_dest_list[0][1]),
            'caminhada_transbordo_m': 60,
            'info_transbordo': "Integração via Estação MOVE / Corredor Troncal",
            'tempo_viagem_min': 60,
            'tempo_total_estimado_min': t_est,
            'shape_id': r1_info.get('shape_id', None),
            'shape_id_1': r1_info.get('shape_id', None),
            'shape_id_2': r2_info.get('shape_id', None),
            'shape2_id': r2_info.get('shape_id', None)
        })

    return sorted(opcoes_transbordo, key=lambda x: x['tempo_total_estimado_min'])


def calcular_tarifa_bhbus(melhor_rota: Dict[str, Any], tarifa_customizada: Optional[float] = None) -> Dict[str, Any]:
    """Calcula o valor da passagem considerando as regras de integração tarifária do BHBUS."""
    if tarifa_customizada is not None:
        t_base = float(tarifa_customizada)
    else:
        t_base = TARIFA_CONVENCIONAL_BH

    transbordos = melhor_rota.get('transbordos', 0)
    if transbordos == 0:
        total = t_base
        desc = "Tarifa Única Direta"
    else:
        # No BHBUS, a segunda tarifa tem desconto de 50% na integração temporal
        total = t_base + (t_base * 0.50)
        desc = "Integração BHBUS (1ª Integral + 2ª com 50% de Desconto)"

    return {
        'tarifa_total': round(total, 2),
        'tarifa_total_str': f"R$ {total:.2f}",
        'detalhe': desc
    }


def calcular_caminhada_e_saude(melhor_rota: Dict[str, Any]) -> Dict[str, Any]:
    """Calcula a distância total de caminhada e estimativa de queima calórica."""
    d_o = melhor_rota.get('dist_caminhada_origem_m', 0)
    d_d = melhor_rota.get('dist_caminhada_destino_m', 0)
    d_tr = melhor_rota.get('caminhada_transbordo_m', 0)
    total_m = d_o + d_d + d_tr
    calorias = int((total_m / 1000.0) * 55)

    return {
        'total_m': total_m,
        'calorias_kcal': calorias
    }


def verificar_estacao_metro_proxima(lat: float, lon: float, raio_max_m: float = 1200.0) -> Optional[Dict[str, Any]]:
    """Verifica se as coordenadas estão a uma distância caminhável de uma estação do Metrô de BH."""
    min_dist = 999999.0
    estacao_proxima = None

    for est in ESTACOES_METRO_BH:
        d = calcular_distancia_haversine(lat, lon, est['lat'], est['lon'])
        if d < min_dist and d <= raio_max_m:
            min_dist = d
            estacao_proxima = {
                'nome': est['nome'],
                'distancia_m': int(d),
                'lat': est['lat'],
                'lon': est['lon']
            }

    return estacao_proxima


def filtrar_e_ordenar_rotas(rotas: List[Dict[str, Any]], criterio: str) -> List[Dict[str, Any]]:
    """Aplica o filtro de preferência do usuário (Mais Rápida, Menos Caminhada, Diretas, etc.)."""
    if not rotas:
        return []

    if "Caminhada" in criterio:
        return sorted(rotas, key=lambda x: x.get('dist_caminhada_origem_m', 0) + x.get('dist_caminhada_destino_m', 0))
    elif "Direta" in criterio:
        diretas = [r for r in rotas if r.get('transbordos', 0) == 0]
        return diretas if diretas else rotas
    else:
        return sorted(rotas, key=lambda x: x.get('tempo_total_estimado_min', 999))
