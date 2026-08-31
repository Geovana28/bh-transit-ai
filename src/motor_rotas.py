from typing import Tuple, Optional, Any, List, Union
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

geolocator = Nominatim(user_agent="bh_transit_ai_app_v2")

# Dicionário Determinístico de Pontos de Interesse (POIs) e Vias Notáveis de Belo Horizonte
BH_LANDMARKS_GEOCODE: dict[str, Tuple[float, float, str]] = {
    # Cidade Administrativa / CAMG
    "cidade administrativa": (-19.7901, -43.9555, "Cidade Administrativa de Minas Gerais, Serra Verde, Belo Horizonte - MG"),
    "camg": (-19.7901, -43.9555, "Cidade Administrativa de Minas Gerais, Serra Verde, Belo Horizonte - MG"),
    "serra verde": (-19.7950, -43.9580, "Serra Verde, Belo Horizonte - MG"),

    # Newton Paiva / Carlos Luz / Caiçara
    "newton paiva": (-19.9022, -43.9634, "Centro Universitário Newton Paiva, Av. Pres. Carlos Luz, Caiçara, Belo Horizonte - MG"),
    "newton paiva carlos luz": (-19.9022, -43.9634, "Centro Universitário Newton Paiva, Av. Pres. Carlos Luz, Caiçara, Belo Horizonte - MG"),
    "carlos luz": (-19.9022, -43.9634, "Av. Presidente Carlos Luz, Caiçara, Belo Horizonte - MG"),
    "caiçara": (-19.9040, -43.9650, "Bairro Caiçara, Belo Horizonte - MG"),
    "caicara": (-19.9040, -43.9650, "Bairro Caiçara, Belo Horizonte - MG"),

    # Universidades
    "ufmg": (-19.8691, -43.9664, "UFMG - Universidade Federal de Minas Gerais, Pampulha, Belo Horizonte - MG"),
    "ufmg pampulha": (-19.8691, -43.9664, "UFMG - Universidade Federal de Minas Gerais, Pampulha, Belo Horizonte - MG"),
    "puc minas": (-19.9234, -43.9912, "PUC Minas - Campus Coração Eucarístico, Belo Horizonte - MG"),
    "puc coracao eucaristico": (-19.9234, -43.9912, "PUC Minas - Campus Coração Eucarístico, Belo Horizonte - MG"),
    "puc praça da liberdade": (-19.9320, -43.9370, "PUC Minas - Praça da Liberdade, Belo Horizonte - MG"),
    "una": (-19.9250, -43.9380, "Centro Universitário UNA, Belo Horizonte - MG"),
    "cefet": (-19.9220, -43.9680, "CEFET-MG - Campus I, Av. Amazonas, Nova Suíça, Belo Horizonte - MG"),

    # Polos, Bairros e Praças Principais
    "savassi": (-19.9385, -43.9332, "Praça da Savassi, Belo Horizonte - MG"),
    "praça da savassi": (-19.9385, -43.9332, "Praça da Savassi, Belo Horizonte - MG"),
    "centro": (-19.9191, -43.9386, "Praça Sete de Setembro, Centro, Belo Horizonte - MG"),
    "praça sete": (-19.9191, -43.9386, "Praça Sete de Setembro, Centro, Belo Horizonte - MG"),
    "praca sete": (-19.9191, -43.9386, "Praça Sete de Setembro, Centro, Belo Horizonte - MG"),
    "barreiro": (-19.9765, -44.0205, "Barreiro, Belo Horizonte - MG"),
    "pampulha": (-19.8517, -43.9723, "Lagoa da Pampulha, Belo Horizonte - MG"),
    "mineirão": (-19.8659, -43.9715, "Estádio Mineirão, Pampulha, Belo Horizonte - MG"),
    "mineirao": (-19.8659, -43.9715, "Estádio Mineirão, Pampulha, Belo Horizonte - MG"),
    "mineirinho": (-19.8580, -43.9730, "Ginásio Mineirinho, Pampulha, Belo Horizonte - MG"),
    "venda nova": (-19.8160, -43.9570, "Venda Nova, Belo Horizonte - MG"),
    "vilarinho": (-19.8020, -43.9530, "Av. Vilarinho, Venda Nova, Belo Horizonte - MG"),

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

    # Comércio, Shoppings e Cultura
    "bh shopping": (-19.9754, -43.9458, "BH Shopping, Belvedere, Belo Horizonte - MG"),
    "mercado central": (-19.9213, -43.9419, "Mercado Central, Centro, Belo Horizonte - MG"),
    "shopping del rey": (-19.8970, -43.9690, "Shopping Del Rey, Caiçara, Belo Horizonte - MG"),
    "del rey": (-19.8970, -43.9690, "Shopping Del Rey, Caiçara, Belo Horizonte - MG"),
    "boulevard shopping": (-19.9200, -43.9210, "Boulevard Shopping, Santa Efigênia, Belo Horizonte - MG"),
    "shopping cidade": (-19.9205, -43.9400, "Shopping Cidade, Centro, Belo Horizonte - MG"),
    "pátio savassi": (-19.9395, -43.9335, "Pátio Savassi, Belo Horizonte - MG"),
    "patio savassi": (-19.9395, -43.9335, "Pátio Savassi, Belo Horizonte - MG"),
    "minascasa": (-19.8780, -43.9310, "Shopping Minascasa, Ipiranga, Belo Horizonte - MG"),
    "expominas": (-19.9300, -43.9850, "Expominas, Gameleira, Belo Horizonte - MG"),
    "hospital das clínicas": (-19.9240, -43.9260, "Hospital das Clínicas da UFMG, Santa Efigênia, Belo Horizonte - MG"),
    "hospital das clinicas": (-19.9240, -43.9260, "Hospital das Clínicas da UFMG, Santa Efigênia, Belo Horizonte - MG"),
    "praça da liberdade": (-19.9317, -43.9381, "Praça da Liberdade, Funcionários, Belo Horizonte - MG"),
    "praca da liberdade": (-19.9317, -43.9381, "Praça da Liberdade, Funcionários, Belo Horizonte - MG"),
    "parque das mangabeiras": (-19.9530, -43.9050, "Parque das Mangabeiras, Belo Horizonte - MG"),
    "palácio das artes": (-19.9230, -43.9340, "Palácio das Artes, Parque Municipal, Belo Horizonte - MG"),
}

def buscar_coordenadas_endereco(texto_endereco: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Converte um endereço textual ou ponto de interesse (POI) em coordenadas (Latitude, Longitude).

    Utiliza primeiramente um dicionário determinístico de POIs de Belo Horizonte para 
    evitar erros de limite de requisições (HTTP 429) e garantir respostas instantâneas.

    Args:
        texto_endereco (str): Endereço ou nome do local a ser buscado.

    Returns:
        Tuple[Optional[float], Optional[float], Optional[str]]: Tupla (latitude, longitude, endereço_formatado).
    """
    if not texto_endereco or len(texto_endereco.strip()) < 3:
        return None, None, None

    busca_clean = texto_endereco.strip().lower()

    # 1. Consulta ao Dicionário Determinístico Local de POIs de BH
    for key, (lat_poi, lon_poi, addr_poi) in BH_LANDMARKS_GEOCODE.items():
        if key in busca_clean or busca_clean in key:
            return lat_poi, lon_poi, addr_poi

    # 2. Fallback via OpenStreetMap (Nominatim API)
    busca = texto_endereco.strip()
    if "belo horizonte" not in busca.lower() and "bh" not in busca.lower():
        busca += ", Belo Horizonte, MG, Brasil"

    try:
        location = geolocator.geocode(busca, timeout=10)
        if location:
            return location.latitude, location.longitude, location.address
        
        location_broad = geolocator.geocode(texto_endereco + ", Brasil", timeout=10)
        if location_broad:
            return location_broad.latitude, location_broad.longitude, location_broad.address
            
    except (GeocoderTimedOut, GeocoderServiceError, Exception) as e:
        print(f"Erro ao buscar endereço via OSM: {e}")
        
    return None, None, None

def calcular_distancia_haversine(
    lat1: Union[float, np.ndarray],
    lon1: Union[float, np.ndarray],
    lat2: Union[float, np.ndarray],
    lon2: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calcula a distância ortodrômica em metros entre coordenadas geográficas via Fórmula de Haversine.

    Notação Matemática:
        a = sin^2(dlat/2) + cos(lat1) * cos(lat2) * sin^2(dlon/2)
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        d = R * c

    Complexidade Assintótica:
        O(N) operações vetoriais para N pontos de entrada.

    Returns:
        Union[float, np.ndarray]: Distância em metros.
    """
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
    top_k: int = 20,
    raio_max_metros: float = 2500.0
) -> pd.DataFrame:
    """
    Localiza os top_k pontos de transporte público mais próximos dentro do raio máximo de busca.

    Complexidade Assintótica:
        O(N log K) onde N é o número total de paradas de ônibus e K é o limite de retorno.

    Args:
        lat (float): Latitude de origem/destino.
        lon (float): Longitude de origem/destino.
        stops_df (pd.DataFrame): DataFrame das paradas GTFS (stops.txt).
        top_k (int): Número de paradas a retornar.
        raio_max_metros (float): Raio máximo em metros.

    Returns:
        pd.DataFrame: DataFrame ordenado pela distância geográfica crescente.
    """
    stops = stops_df.copy()
    stops['distancia_m'] = calcular_distancia_haversine(
        lat, lon, 
        stops['stop_lat'].values, 
        stops['stop_lon'].values
    )
    return stops[stops['distancia_m'] <= raio_max_metros].sort_values('distancia_m').head(top_k)

def converter_tempo_para_minutos(tempo_str):
    """Converte horário 'HH:MM:SS' ou 'HH:MM' para minutos totais do dia"""
    try:
        if pd.isna(tempo_str):
            return None
        partes = str(tempo_str).strip().split(':')
        horas = int(partes[0])
        minutos = int(partes[1])
        return horas * 60 + minutos
    except Exception:
        return None

def converter_minutos_para_tempo(minutos):
    """Converte minutos totais do dia para o formato de string 'HH:MM'"""
    if minutos is None:
        return "--:--"
    minutos = int(minutos) % 1440 # Garante loop de 24h
    horas = minutos // 60
    mins = minutos % 60
    return f"{horas:02d}:{mins:02d}"

def obter_service_ids_por_tipo_dia(calendar_df, calendar_dates_df, tipo_dia, data_viagem=None):
    """
    Retorna a lista de service_ids válidos baseados no tipo de dia (Dia Útil, Sábado, Domingo/Feriado).
    Se data_viagem for fornecida (ex: '20240907'), verifica exceções em calendar_dates.txt.
    """
    if calendar_df is None:
        return None
        
    c_df = calendar_df.copy()
    
    # 1. Verifica se a data exata tem exceções especificadas em calendar_dates.txt (ex: Feriado)
    if calendar_dates_df is not None and data_viagem:
        data_str = str(data_viagem).replace("-", "")
        exc = calendar_dates_df[calendar_dates_df['date'].astype(str) == data_str]
        if not exc.empty:
            added_services = exc[exc['exception_type'] == 1]['service_id'].values
            if len(added_services) > 0:
                return list(added_services)

    # 2. Senão, filtra pelo tipo de dia escolhido (Dia Útil, Sábado, Domingo/Feriado)
    if "Sábado" in str(tipo_dia) or "Sabado" in str(tipo_dia):
        services = c_df[c_df['saturday'] == 1]['service_id'].values
    elif "Domingo" in str(tipo_dia) or "Feriado" in str(tipo_dia):
        services = c_df[c_df['sunday'] == 1]['service_id'].values
    else: # Dia útil por padrão
        services = c_df[(c_df['monday'] == 1) | (c_df['tuesday'] == 1) | (c_df['wednesday'] == 1) | (c_df['thursday'] == 1) | (c_df['friday'] == 1)]['service_id'].values
        
    return list(services)

def buscar_rotas_diretas(origem_lat, origem_lon, destino_lat, destino_lon, routes_df, stops_df, trips_df, stop_times_df, top_k_pontos=20, horario_partida=None, calendar_df=None, calendar_dates_df=None, tipo_dia=None, data_viagem=None):
    """Busca linhas de ônibus diretas que conectam a Origem até o Destino (sem transbordo), considerando horário de partida e calendário do tipo de dia."""
    
    # Filtragem por tipo de dia (Dia Útil, Sábado, Domingo/Feriado)
    if calendar_df is not None and tipo_dia is not None:
        service_ids_validos = obter_service_ids_por_tipo_dia(calendar_df, calendar_dates_df, tipo_dia, data_viagem)
        if service_ids_validos:
            trips_df = trips_df[trips_df['service_id'].isin(service_ids_validos)].copy()

    pontos_origem = encontrar_pontos_proximos(origem_lat, origem_lon, stops_df, top_k=top_k_pontos)
    pontos_destino = encontrar_pontos_proximos(destino_lat, destino_lon, stops_df, top_k=top_k_pontos)
    
    if pontos_origem.empty or pontos_destino.empty or trips_df.empty:
        return []

    ids_origem = set(pontos_origem['stop_id'].astype(str))
    ids_destino = set(pontos_destino['stop_id'].astype(str))

    st_origem = stop_times_df[stop_times_df['stop_id'].astype(str).isin(ids_origem)].copy()
    st_destino = stop_times_df[stop_times_df['stop_id'].astype(str).isin(ids_destino)].copy()

    cruzamento = pd.merge(
        st_origem, st_destino, 
        on='trip_id', 
        suffixes=('_origem', '_destino')
    )

    viagens_validas = cruzamento[cruzamento['stop_sequence_origem'] < cruzamento['stop_sequence_destino']].copy()

    if viagens_validas.empty:
        return []

    minutos_partida = converter_tempo_para_minutos(horario_partida) if horario_partida else None

    # Adiciona aux de minutos para ordenação por horário
    if 'departure_time_origem' in viagens_validas.columns:
        viagens_validas['minutos_origem'] = viagens_validas['departure_time_origem'].apply(converter_tempo_para_minutos)
        if minutos_partida is not None:
            # Filtra apenas viagens que saem a partir do horário de partida escolhido (ou próximo)
            viagens_futuras = viagens_validas[viagens_validas['minutos_origem'] >= minutos_partida]
            if not viagens_futuras.empty:
                viagens_validas = viagens_futuras

    viagens_com_trips = pd.merge(viagens_validas, trips_df[['trip_id', 'route_id', 'shape_id']], on='trip_id')
    viagens_com_rotas = pd.merge(viagens_com_trips, routes_df, on='route_id')

    opcoes_rotas = []
    
    for route_id, grupo in viagens_com_rotas.groupby('route_id'):
        # Ordena para pegar a próxima viagem disponível
        if 'minutos_origem' in grupo.columns and minutos_partida is not None:
            grupo = grupo.sort_values('minutos_origem')
            
        exemplo = grupo.iloc[0]
        
        ponto_orig = stops_df[stops_df['stop_id'].astype(str) == str(exemplo['stop_id_origem'])].iloc[0]
        ponto_dest = stops_df[stops_df['stop_id'].astype(str) == str(exemplo['stop_id_destino'])].iloc[0]
        
        dist_orig_m = calcular_distancia_haversine(origem_lat, origem_lon, ponto_orig['stop_lat'], ponto_orig['stop_lon'])
        dist_dest_m = calcular_distancia_haversine(destino_lat, destino_lon, ponto_dest['stop_lat'], ponto_dest['stop_lon'])
        
        min_orig = converter_tempo_para_minutos(exemplo.get('departure_time_origem', None))
        min_dest = converter_tempo_para_minutos(exemplo.get('arrival_time_destino', None))
        
        tempo_viagem_min = 25
        tempo_espera_min = 0

        if min_orig and min_dest and min_dest > min_orig:
            tempo_viagem_min = min_dest - min_orig
            
        if minutos_partida is not None and min_orig and min_orig >= minutos_partida:
            tempo_espera_min = min_orig - minutos_partida

        horario_saida_str = converter_minutos_para_tempo(min_orig) if min_orig else "--:--"
        horario_chegada_str = converter_minutos_para_tempo(min_dest) if min_dest else "--:--"
            
        tempo_total = int(tempo_espera_min + tempo_viagem_min + (dist_orig_m + dist_dest_m) / 80)

        opcoes_rotas.append({
            'tipo': '🟢 DIRETA (0 Trocas)',
            'transbordos': 0,
            'route_id': route_id,
            'route_short_name': str(exemplo['route_short_name']),
            'route_long_name': str(exemplo['route_long_name']),
            'horario_saida': horario_saida_str,
            'horario_chegada': horario_chegada_str,
            'tempo_espera_min': int(tempo_espera_min),
            'ponto_origem_id': exemplo['stop_id_origem'],
            'ponto_origem_nome': ponto_orig['stop_name'],
            'ponto_origem_lat': ponto_orig['stop_lat'],
            'ponto_origem_lon': ponto_orig['stop_lon'],
            'ponto_destino_id': exemplo['stop_id_destino'],
            'ponto_destino_nome': ponto_dest['stop_name'],
            'ponto_destino_lat': ponto_dest['stop_lat'],
            'ponto_destino_lon': ponto_dest['stop_lon'],
            'dist_caminhada_origem_m': int(dist_orig_m),
            'dist_caminhada_destino_m': int(dist_dest_m),
            'tempo_viagem_min': int(tempo_viagem_min),
            'tempo_total_estimado_min': tempo_total,
            'shape_id': exemplo.get('shape_id', None)
        })

    return sorted(opcoes_rotas, key=lambda x: x['tempo_total_estimado_min'])

def buscar_rotas_com_transbordo(origem_lat, origem_lon, destino_lat, destino_lon, routes_df, stops_df, trips_df, stop_times_df, top_k_pontos=20, horario_partida=None, calendar_df=None, calendar_dates_df=None, tipo_dia=None, data_viagem=None):
    """
    Busca rotas diretas E rotas com 1 baldeação/transbordo (2 ônibus).
    Garante que o ponto de baldeação seja REALMENTE compartilhado pelas duas linhas ou em quarteirões vizinhos!
    Considera o horário de partida e tipo de dia selecionados.
    """
    rotas_diretas = buscar_rotas_diretas(
        origem_lat, origem_lon, destino_lat, destino_lon, 
        routes_df, stops_df, trips_df, stop_times_df, 
        top_k_pontos=top_k_pontos, horario_partida=horario_partida,
        calendar_df=calendar_df, calendar_dates_df=calendar_dates_df,
        tipo_dia=tipo_dia, data_viagem=data_viagem
    )
    
    if rotas_diretas:
        return rotas_diretas

    # Filtragem por tipo de dia (Dia Útil, Sábado, Domingo/Feriado)
    if calendar_df is not None and tipo_dia is not None:
        service_ids_validos = obter_service_ids_por_tipo_dia(calendar_df, calendar_dates_df, tipo_dia, data_viagem)
        if service_ids_validos:
            trips_df = trips_df[trips_df['service_id'].isin(service_ids_validos)].copy()

    pontos_origem = encontrar_pontos_proximos(origem_lat, origem_lon, stops_df, top_k=top_k_pontos, raio_max_metros=2500)
    pontos_destino = encontrar_pontos_proximos(destino_lat, destino_lon, stops_df, top_k=top_k_pontos, raio_max_metros=2500)

    if pontos_origem.empty or pontos_destino.empty or trips_df.empty:
        return []

    ids_origem = set(pontos_origem['stop_id'].astype(str))
    ids_destino = set(pontos_destino['stop_id'].astype(str))

    st_origem = stop_times_df[stop_times_df['stop_id'].astype(str).isin(ids_origem)]
    trips_origem = trips_df[trips_df['trip_id'].isin(st_origem['trip_id'])]
    routes_origem_ids = list(set(trips_origem['route_id']))

    st_destino = stop_times_df[stop_times_df['stop_id'].astype(str).isin(ids_destino)]
    trips_destino = trips_df[trips_df['trip_id'].isin(st_destino['trip_id'])]
    routes_destino_ids = list(set(trips_destino['route_id']))

    opcoes_transbordo = []
    minutos_partida = converter_tempo_para_minutos(horario_partida) if horario_partida else None

    # Procura pares de linhas r1 (origem) e r2 (destino) que se conectam em pontos REAIS
    for r1_id in routes_origem_ids[:5]:
        t1_df = trips_df[trips_df['route_id'] == r1_id]
        if t1_df.empty:
            continue
        st1_stops = set(stop_times_df[stop_times_df['trip_id'].isin(t1_df['trip_id'])]['stop_id'].astype(str))

        for r2_id in routes_destino_ids[:5]:
            if r1_id == r2_id:
                continue
                
            t2_df = trips_df[trips_df['route_id'] == r2_id]
            if t2_df.empty:
                continue
            st2_stops = set(stop_times_df[stop_times_df['trip_id'].isin(t2_df['trip_id'])]['stop_id'].astype(str))

            # 1. TENTA INTERSECÇÃO DIRETA (PARADA IDÊNTICA COMPARTILHADA PELAS DUAS LINHAS)
            em_comum = list(st1_stops.intersection(st2_stops))
            
            p_troca = None
            caminhada_transbordo_m = 0
            
            if em_comum:
                # Pega o primeiro ponto de parada compartilhado no GTFS
                p_troca = stops_df[stops_df['stop_id'].astype(str) == em_comum[0]].iloc[0]
                p_troca_desembarque = p_troca
                p_troca_embarque = p_troca
            else:
                # 2. TENTA PONTOS VIZINHOS (Caminhada curta de até 300 metros entre paradas vizinhas)
                stops_r1 = stops_df[stops_df['stop_id'].astype(str).isin(st1_stops)]
                stops_r2 = stops_df[stops_df['stop_id'].astype(str).isin(st2_stops)]
                
                # Procura o par de pontos de r1 e r2 com menor distância física entre si
                min_d = 999999
                best_pair = None
                
                for _, s1 in stops_r1.head(15).iterrows():
                    distancias = calcular_distancia_haversine(s1['stop_lat'], s1['stop_lon'], stops_r2['stop_lat'].values, stops_r2['stop_lon'].values)
                    min_idx = np.argmin(distancias)
                    d = distancias[min_idx]
                    if d < min_d and d <= 300: # Max 300m de caminhada entre pontos na baldeação
                        min_d = d
                        best_pair = (s1, stops_r2.iloc[min_idx])
                
                if best_pair:
                    p_troca_desembarque, p_troca_embarque = best_pair
                    p_troca = p_troca_desembarque
                    caminhada_transbordo_m = int(min_d)

            if p_troca is not None:
                r1 = routes_df[routes_df['route_id'] == r1_id].iloc[0]
                r2 = routes_df[routes_df['route_id'] == r2_id].iloc[0]
                
                shape1 = t1_df.iloc[0].get('shape_id', None)
                shape2 = t2_df.iloc[0].get('shape_id', None)

                p_orig = pontos_origem.iloc[0]
                p_dest = pontos_destino.iloc[0]
                
                dist_o = p_orig['distancia_m']
                dist_d = p_dest['distancia_m']
                
                info_transbordo = f"Baldeação em: {p_troca_desembarque['stop_name']}"
                if caminhada_transbordo_m > 0:
                    info_transbordo += f" (Caminhada de {caminhada_transbordo_m}m até {p_troca_embarque['stop_name']})"
                
                horario_saida_str = converter_minutos_para_tempo(minutos_partida) if minutos_partida else "Em breve"
                horario_chegada_str = converter_minutos_para_tempo(minutos_partida + 48) if minutos_partida else "Em breve"

                opcoes_transbordo.append({
                    'tipo': '🔀 TRANSBORDO (2 Ônibus)',
                    'transbordos': 1,
                    'linha1_short': str(r1['route_short_name']),
                    'linha1_long': str(r1['route_long_name']),
                    'linha2_short': str(r2['route_short_name']),
                    'linha2_long': str(r2['route_long_name']),
                    'route_short_name': f"{r1['route_short_name']} ➔ (Troca) ➔ {r2['route_short_name']}",
                    'route_long_name': f"1º Ônibus: Linha {r1['route_short_name']} | {info_transbordo} | 2º Ônibus: Linha {r2['route_short_name']}",
                    'horario_saida': horario_saida_str,
                    'horario_chegada': horario_chegada_str,
                    'tempo_espera_min': 0,
                    'ponto_origem_id': p_orig['stop_id'],
                    'ponto_origem_nome': p_orig['stop_name'],
                    'ponto_origem_lat': p_orig['stop_lat'],
                    'ponto_origem_lon': p_orig['stop_lon'],
                    'ponto_troca_nome': p_troca_desembarque['stop_name'],
                    'ponto_troca_lat': p_troca_desembarque['stop_lat'],
                    'ponto_troca_lon': p_troca_desembarque['stop_lon'],
                    'ponto_destino_id': p_dest['stop_id'],
                    'ponto_destino_nome': p_dest['stop_name'],
                    'ponto_destino_lat': p_dest['stop_lat'],
                    'ponto_destino_lon': p_dest['stop_lon'],
                    'dist_caminhada_origem_m': int(dist_o),
                    'dist_caminhada_destino_m': int(dist_d),
                    'caminhada_transbordo_m': caminhada_transbordo_m,
                    'tempo_viagem_min': 40,
                    'tempo_total_estimado_min': int(40 + (dist_o + dist_d + caminhada_transbordo_m) / 80 + 8),
                    'shape_id_1': shape1,
                    'shape_id_2': shape2
                })
                
                if len(opcoes_transbordo) >= 5:
                    break

    return sorted(opcoes_transbordo, key=lambda x: (x['transbordos'], x['tempo_total_estimado_min']))


# ==============================================================================
# ESTAÇÕES DO METRÔ DE BELO HORIZONTE (LINHA 1) E INTEGRAÇÃO INTERMODAL
# ==============================================================================
ESTACOES_METRO_BH = [
    {"nome": "Estação Eldorado", "lat": -19.9515, "lon": -44.0157},
    {"nome": "Estação Cidade Industrial", "lat": -19.9443, "lon": -44.0042},
    {"nome": "Estação Vila Oeste", "lat": -19.9372, "lon": -43.9930},
    {"nome": "Estação Gameleira", "lat": -19.9298, "lon": -43.9775},
    {"nome": "Estação Calafate", "lat": -19.9215, "lon": -43.9658},
    {"nome": "Estação Carlos Prates", "lat": -19.9142, "lon": -43.9535},
    {"nome": "Estação Lagoinha", "lat": -19.9078, "lon": -43.9452},
    {"nome": "Estação Central", "lat": -19.9167, "lon": -43.9356},
    {"nome": "Estação Santa Efigênia", "lat": -19.9248, "lon": -43.9272},
    {"nome": "Estação Santa Teresa", "lat": -19.9205, "lon": -43.9189},
    {"nome": "Estação Horto Florestal", "lat": -19.9125, "lon": -43.9118},
    {"nome": "Estação Santa Inês", "lat": -19.9038, "lon": -43.9025},
    {"nome": "Estação José Cândido da Silveira", "lat": -19.8942, "lon": -43.9002},
    {"nome": "Estação Minas Shopping", "lat": -19.8825, "lon": -43.9128},
    {"nome": "Estação São Gabriel", "lat": -19.8655, "lon": -43.9247},
    {"nome": "Estação Primeiro de Maio", "lat": -19.8522, "lon": -43.9352},
    {"nome": "Estação Waldomiro Lobo", "lat": -19.8438, "lon": -43.9432},
    {"nome": "Estação Floramar", "lat": -19.8325, "lon": -43.9512},
    {"nome": "Estação Vilarinho", "lat": -19.8182, "lon": -43.9575}
]

def verificar_estacao_metro_proxima(lat, lon, raio_max_m=1200):
    """Verifica se há alguma estação do Metrô de BH próxima da coordenada dada"""
    estacao_mais_proxima = None
    menor_dist = float('inf')
    for est in ESTACOES_METRO_BH:
        dist = calcular_distancia_haversine(lat, lon, est['lat'], est['lon'])
        if dist <= raio_max_m and dist < menor_dist:
            menor_dist = dist
            estacao_mais_proxima = {
                'nome': est['nome'],
                'distancia_m': int(dist),
                'tempo_caminhada_min': int(dist / 75.0) # ~4.5 km/h
            }
    return estacao_mais_proxima


# ==============================================================================
# CALCULADORA DE TARIFA BHBUS E ECONOMIA FINANCEIRA
# ==============================================================================
TARIFA_CONVENCIONAL_BH = 6.25
TARIFA_CIRCULAR_VILA_BH = 6.00

def e_linha_circular_ou_suplementar(route_short_name):
    """Identifica se a linha é circular ou suplementar/vila (Tarifa R$ 6,00)"""
    sn = str(route_short_name).strip().upper()
    if sn.startswith('S') or sn.startswith('C') or sn in ['102', '103', '105', '201', '301', '503', '601', '901']:
        return True
    return False

def calcular_tarifa_bhbus(rota, tarifa_customizada=None):
    """
    Calcula o custo total da viagem com suporte a linhas municipais BH, linhas metropolitanas (R$ 8,45) ou tarifa customizada.
    - Tarifa convencional BH: R$ 6,25
    - Tarifa circular / vilas BH: R$ 6,00
    - Tarifa metropolitana / customizada: R$ 8,45 (ou valor fornecido pelo usuário)
    - Integração em Estações Fechadas / MOVE: 100% Gratuita (R$ 0,00 no 2º ônibus).
    - Integração BHBUS em Ruas (até 90 min): 50% de desconto no 2º ônibus.
    """
    is_transbordo = rota.get('transbordos', 0) > 0
    sn1 = rota.get('linha1_short', rota.get('route_short_name', ''))
    sn2 = rota.get('linha2_short', '')
    ponto_troca = str(rota.get('ponto_troca_nome', '')).lower()
    caminhada_troca_m = rota.get('caminhada_transbordo_m', 0)

    if tarifa_customizada and float(tarifa_customizada) > 0:
        tarifa1 = float(tarifa_customizada)
        tipo_linha = f"Personalizada/Metropolitana (R$ {tarifa1:.2f})"
    else:
        tarifa1 = TARIFA_CIRCULAR_VILA_BH if e_linha_circular_ou_suplementar(sn1) else TARIFA_CONVENCIONAL_BH
        tipo_linha = "Circular/Vila" if tarifa1 == 6.00 else "Convencional/Move"

    if is_transbordo:
        if tarifa_customizada and float(tarifa_customizada) > 0:
            tarifa2 = float(tarifa_customizada)
        else:
            tarifa2 = TARIFA_CIRCULAR_VILA_BH if e_linha_circular_ou_suplementar(sn2) else TARIFA_CONVENCIONAL_BH
            
        # Verifica se o transbordo ocorre dentro de uma Estação/Terminal MOVE fechado
        e_estacao_fechada = any(kw in ponto_troca for kw in ["estação", "estacao", "terminal", "move", "plataforma"]) or caminhada_troca_m == 0

        if e_estacao_fechada:
            # 100% GRATUITO na 2ª passagem dentro de Estações Fechadas / MOVE
            tarifa_total = tarifa1
            sem_desconto = tarifa1 + tarifa2
            economia_baldeacao = tarifa2
            descricao_tarifa = f"💳 Integração Gratuita em Estação MOVE (R$ {tarifa1:.2f} + R$ 0,00 no 2º ônibus)"
        else:
            # 50% de desconto na 2ª passagem em transbordos de rua (BHBUS em 90 min)
            tarifa_total = tarifa1 + (tarifa2 * 0.50)
            sem_desconto = tarifa1 + tarifa2
            economia_baldeacao = sem_desconto - tarifa_total
            descricao_tarifa = f"💳 Passagem Integrada BHBUS na Rua (R$ {tarifa1:.2f} + R$ {tarifa2*0.50:.2f} com 50% desc.)"
    else:
        tarifa_total = tarifa1
        economia_baldeacao = 0.0
        descricao_tarifa = f"💳 Tarifa Única ({tipo_linha}): R$ {tarifa1:.2f}"
        economia_baldeacao = 0.0
        descricao_tarifa = f"💳 Tarifa Única ({tipo_linha}): R$ {tarifa1:.2f}"

    # Estimativas comparativas
    tempo_min = rota.get('tempo_total_estimado_min', 30)
    dist_km_est = max(3.0, tempo_min * 0.35)
    
    custo_est_uber = round(8.00 + (dist_km_est * 2.20) + (tempo_min * 0.40), 2)
    custo_est_carro = round((dist_km_est / 9.5 * 5.80) + 12.00, 2) # Combustível + Estacionamento proporção BH

    economia_uber = max(0.0, custo_est_uber - tarifa_total)
    economia_carro = max(0.0, custo_est_carro - tarifa_total)

    return {
        'tarifa_total': round(tarifa_total, 2),
        'tarifa_total_str': f"R$ {tarifa_total:.2f}".replace('.', ','),
        'descricao_tarifa': descricao_tarifa,
        'economia_baldeacao': round(economia_baldeacao, 2),
        'custo_est_uber': custo_est_uber,
        'custo_est_carro': custo_est_carro,
        'economia_uber': round(economia_uber, 2),
        'economia_carro': round(economia_carro, 2)
    }


# ==============================================================================
# CAMINHADA LAST-MILE E SAÚDE
# ==============================================================================
def calcular_caminhada_e_saude(rota):
    """Calcula metros de caminhada total, tempo estimado a pé e queima de calorias"""
    dist_o = rota.get('dist_caminhada_origem_m', 0)
    dist_d = rota.get('dist_caminhada_destino_m', 0)
    dist_t = rota.get('caminhada_transbordo_m', 0)
    
    total_m = dist_o + dist_d + dist_t
    tempo_pe_min = int(total_m / 75.0) # ~4.5 km/h (75 m/min)
    calorias_kcal = round((total_m / 1000.0) * 60, 1) # ~60 kcal por km caminhado

    return {
        'total_m': total_m,
        'tempo_pe_min': tempo_pe_min,
        'calorias_kcal': calorias_kcal
    }


# ==============================================================================
# FILTRAGEM E ORDENAÇÃO PERSONALIZADA DE ROTAS
# ==============================================================================
def filtrar_e_ordenar_rotas(rotas, filtro_escolhido="⚡ Mais Rápida"):
    """
    Filtra e ordena a lista de rotas encontradas com base no filtro do usuário:
    - '⚡ Mais Rápida': Ordena pelo tempo total previsto
    - '🚶 Menos Caminhada': Ordena pela menor distância total a pé
    - '🚏 Linhas Diretas': Retorna apenas rotas sem transbordo (se existirem)
    - '♿ Acessível (PCD)': Prioriza ônibus diretos e estações com elevador/corredores Move
    """
    if not rotas:
        return []

    rotas_copia = list(rotas)

    if "Direta" in filtro_escolhido or "Diretas" in filtro_escolhido:
        diretas = [r for r in rotas_copia if r.get('transbordos', 0) == 0]
        if diretas:
            return sorted(diretas, key=lambda x: x.get('tempo_total_estimado_min', 999))

    if "Caminhada" in filtro_escolhido:
        return sorted(rotas_copia, key=lambda x: (
            x.get('dist_caminhada_origem_m', 0) + x.get('dist_caminhada_destino_m', 0) + x.get('caminhada_transbordo_m', 0)
        ))

    if "Acessível" in filtro_escolhido or "PCD" in filtro_escolhido:
        return sorted(rotas_copia, key=lambda x: (
            x.get('transbordos', 0),
            x.get('dist_caminhada_origem_m', 0) + x.get('dist_caminhada_destino_m', 0),
            x.get('tempo_total_estimado_min', 999)
        ))

    # Padrão: Mais rápida
    return sorted(rotas_copia, key=lambda x: (x.get('transbordos', 0), x.get('tempo_total_estimado_min', 999)))



