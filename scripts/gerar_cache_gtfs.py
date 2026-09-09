"""
Script de Geração de Cache Otimizado do GTFS de Belo Horizonte.
Gera um arquivo compacto data/gtfs_bh_cache.pkl indexado em memória para carregamento instantâneo.
"""

import os
import zipfile
import pickle
import time
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GTFS_ZIP = os.path.join(BASE_DIR, 'data', 'gtfs_bh_convencional.zip')
CACHE_OUT = os.path.join(BASE_DIR, 'data', 'gtfs_bh_cache.pkl')


def gerar_cache():
    print(f"Iniciando indexação ultra-rápida do GTFS: {GTFS_ZIP}")
    t0 = time.time()

    with zipfile.ZipFile(GTFS_ZIP, 'r') as z:
        routes = pd.read_csv(z.open('routes.txt'))
        stops = pd.read_csv(z.open('stops.txt'))
        trips = pd.read_csv(z.open('trips.txt'))

        shapes = None
        if 'shapes.txt' in z.namelist():
            shapes = pd.read_csv(z.open('shapes.txt'))

        calendar = None
        if 'calendar.txt' in z.namelist():
            calendar = pd.read_csv(z.open('calendar.txt'))

        calendar_dates = None
        if 'calendar_dates.txt' in z.namelist():
            calendar_dates = pd.read_csv(z.open('calendar_dates.txt'))

        # Padroniza tipos
        routes['route_id'] = routes['route_id'].astype(str)
        routes['route_short_name'] = routes['route_short_name'].astype(str)
        trips['route_id'] = trips['route_id'].astype(str)
        trips['trip_id'] = trips['trip_id'].astype(str)
        stops['stop_id'] = stops['stop_id'].astype(str)

        # Seleciona viagens representativas por rota/itinerário
        trips_rep = trips.drop_duplicates(subset=['route_id', 'shape_id']).copy()
        rep_trip_ids = set(trips_rep['trip_id'].values)

        # Carrega apenas stop_times correspondentes às viagens representativas
        st_chunks = []
        cols = ['trip_id', 'stop_id', 'stop_sequence']
        for chunk in pd.read_csv(z.open('stop_times.txt'), usecols=cols, chunksize=300000):
            chunk['trip_id'] = chunk['trip_id'].astype(str)
            matched = chunk[chunk['trip_id'].isin(rep_trip_ids)]
            if not matched.empty:
                st_chunks.append(matched)

        st_rep = pd.concat(st_chunks, ignore_index=True)
        st_rep['stop_id'] = st_rep['stop_id'].astype(str)

        # Associa stop_times com rotas
        trip_to_route = dict(zip(trips_rep['trip_id'], trips_rep['route_id']))
        st_rep['route_id'] = st_rep['trip_id'].map(trip_to_route)

        route_to_short = dict(zip(routes['route_id'], routes['route_short_name']))
        st_rep['route_short_name'] = st_rep['route_id'].map(route_to_short)

        # Cria mapas de indexação O(1)
        mapa_ponto_rotas = {}
        mapa_ponto_linhas = {}
        for sid, group in st_rep.groupby('stop_id'):
            mapa_ponto_rotas[sid] = set(group['route_id'].unique())
            mapa_ponto_linhas[sid] = sorted(list(set(group['route_short_name'].dropna().unique())))

        # Mapa de sequência de paradas por rota
        mapa_rota_pontos = {}
        for rid, group in st_rep.sort_values('stop_sequence').groupby('route_id'):
            mapa_rota_pontos[rid] = list(zip(group['stop_id'].values, group['stop_sequence'].values))

        # Dicionário rápido de paradas
        stops_dict = {}
        for _, s in stops.iterrows():
            stops_dict[str(s['stop_id'])] = {
                'name': str(s['stop_name']),
                'lat': float(s['stop_lat']),
                'lon': float(s['stop_lon'])
            }

        # Dicionário rápido de rotas
        routes_dict = {}
        for _, r in routes.iterrows():
            rid = str(r['route_id'])
            t_match = trips_rep[trips_rep['route_id'] == rid]
            shape_id = t_match.iloc[0]['shape_id'] if not t_match.empty and 'shape_id' in t_match.columns else None
            routes_dict[rid] = {
                'route_id': rid,
                'short_name': str(r['route_short_name']),
                'long_name': str(r['route_long_name']),
                'shape_id': shape_id
            }

        cache_data = {
            'routes': routes,
            'stops': stops,
            'trips': trips_rep,
            'shapes': shapes,
            'calendar': calendar,
            'calendar_dates': calendar_dates,
            'stop_times_rep': st_rep,
            'mapa_ponto_rotas': mapa_ponto_rotas,
            'mapa_ponto_linhas': mapa_ponto_linhas,
            'mapa_rota_pontos': mapa_rota_pontos,
            'stops_dict': stops_dict,
            'routes_dict': routes_dict
        }

        with open(CACHE_OUT, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        t_total = time.time() - t0
        cache_size_mb = os.path.getsize(CACHE_OUT) / (1024 * 1024)
        print(f"Cache GTFS gerado com sucesso em {t_total:.2f}s!")
        print(f"Arquivo salvo: {CACHE_OUT} ({cache_size_mb:.2f} MB)")


if __name__ == '__main__':
    gerar_cache()
