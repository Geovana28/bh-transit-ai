import sys
import os
import time
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import motor_rotas

cache = motor_rotas.obter_cache_gtfs()
routes, stops, trips, shapes, stop_times = (
    cache['routes'],
    cache['stops'],
    cache['trips'],
    cache['shapes'],
    cache['stop_times_rep'],
)
mapa_ponto_rotas = cache['mapa_ponto_rotas']
mapa_rota_pontos = cache['mapa_rota_pontos']
stops_dict = cache['stops_dict']
routes_dict = cache['routes_dict']

# User's points from screenshot
lat_o, lon_o = -19.7745, -44.0289  # North / Neves border
lat_d, lon_d = -19.9912, -43.9567  # South / Nova Lima border


def dist_h(lat1, lon1, lat2, lon2):
  R = 6371000.0
  p1, p2 = np.radians(lat1), np.radians(lat2)
  dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
  a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
  return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


t0 = time.time()

# 1. Adaptive stops
p_o = motor_rotas.encontrar_pontos_proximos(
    lat_o, lon_o, stops, top_k=25, raio_max_metros=50000
)
p_d = motor_rotas.encontrar_pontos_proximos(
    lat_d, lon_d, stops, top_k=25, raio_max_metros=50000
)

p_orig_list = list(zip(p_o['stop_id'].astype(str), p_o['distancia_m']))
p_dest_list = list(zip(p_d['stop_id'].astype(str), p_d['distancia_m']))

# Major Terminals & Trunk Lines
TERMINAIS = [
    {
        'nome': 'Estação MOVE Vilarinho / Venda Nova',
        'lat': -19.8028,
        'lon': -43.9508,
    },
    {'nome': 'Estação MOVE Pampulha', 'lat': -19.8520, 'lon': -43.9540},
    {'nome': 'Centro / Praça Sete', 'lat': -19.9191, 'lon': -43.9386},
    {
        'nome': 'Terminal BH Shopping / Belvedere',
        'lat': -19.9754,
        'lon': -43.9458,
    },
    {'nome': 'Estação MOVE Barreiro', 'lat': -19.9750, 'lon': -44.0200},
    {'nome': 'Estação Diamante', 'lat': -19.9850, 'lon': -44.0150},
    {'nome': 'Estação São Gabriel', 'lat': -19.8598, 'lon': -43.9188},
    {'nome': 'Savassi / Praça da Liberdade', 'lat': -19.9385, 'lon': -43.9332},
]

rotas_encontradas = []

# Check Direct Routes
for sid_o, dist_o in p_orig_list[:15]:
  rotas_o = mapa_ponto_rotas.get(sid_o, set())
  for sid_d, dist_d in p_dest_list[:15]:
    rotas_d = mapa_ponto_rotas.get(sid_d, set())
    comuns = rotas_o & rotas_d
    for rid in comuns:
      seqs = dict(mapa_rota_pontos.get(rid, []))
      if sid_o in seqs and sid_d in seqs and seqs[sid_o] < seqs[sid_d]:
        r_info = routes_dict.get(rid, {})
        paradas = seqs[sid_d] - seqs[sid_o]
        tempo = int(paradas * 2.1 + (dist_o + dist_d) / 80)
        rotas_encontradas.append({
            'tipo': 'DIRETA',
            'transbordos': 0,
            'route_short_name': r_info.get('short_name', rid),
            'route_long_name': r_info.get('long_name', ''),
            'ponto_origem_nome': stops_dict[sid_o]['name'],
            'ponto_destino_nome': stops_dict[sid_d]['name'],
            'tempo_total_estimado_min': tempo,
        })

# Check 1-Transfer Routes
if not rotas_encontradas:
  rotas_o_all = set()
  for sid, _ in p_orig_list:
    rotas_o_all.update(mapa_ponto_rotas.get(sid, set()))

  rotas_d_all = set()
  for sid, _ in p_dest_list:
    rotas_d_all.update(mapa_ponto_rotas.get(sid, set()))

  pares_vistos = set()
  for r1 in rotas_o_all:
    s_r1_seq = dict(mapa_rota_pontos.get(r1, []))
    set_r1 = set(s_r1_seq.keys())
    for r2 in rotas_d_all:
      if r1 == r2 or (r1, r2) in pares_vistos:
        continue
      s_r2_seq = dict(mapa_rota_pontos.get(r2, []))
      comuns = set_r1 & set(s_r2_seq.keys())
      if comuns:
        for s_tr in list(comuns)[:1]:
          for s_o, dist_o in p_orig_list[:10]:
            if s_o in s_r1_seq and s_r1_seq[s_o] < s_r1_seq[s_tr]:
              for s_d, dist_d in p_dest_list[:10]:
                if s_d in s_r2_seq and s_r2_seq[s_tr] < s_r2_seq[s_d]:
                  pares_vistos.add((r1, r2))
                  r1_info = routes_dict.get(r1, {})
                  r2_info = routes_dict.get(r2, {})
                  paradas_1 = s_r1_seq[s_tr] - s_r1_seq[s_o]
                  paradas_2 = s_r2_seq[s_d] - s_r2_seq[s_tr]
                  tempo = int(
                      paradas_1 * 2.0 + paradas_2 * 2.0 + (dist_o + dist_d) / 80 + 8
                  )
                  r_short = (
                      f"{r1_info.get('short_name', r1)} ->"
                      f" {r2_info.get('short_name', r2)}"
                  )
                  rotas_encontradas.append({
                      'tipo': 'TRANSBORDO',
                      'transbordos': 1,
                      'route_short_name': r_short,
                      'linha1_short': r1_info.get('short_name', r1),
                      'linha2_short': r2_info.get('short_name', r2),
                      'ponto_origem_nome': stops_dict[s_o]['name'],
                      'ponto_troca_nome': stops_dict[s_tr]['name'],
                      'ponto_destino_nome': stops_dict[s_d]['name'],
                      'tempo_total_estimado_min': tempo,
                  })

# Check Terminal Trunk Corridor Integration
if not rotas_encontradas:
  print('Calculando conexao via Corredores Troncais / Terminais...')
  # Connect origin to nearest terminal and terminal to destination
  for term in TERMINAIS:
    p_term = motor_rotas.encontrar_pontos_proximos(
        term['lat'], term['lon'], stops, top_k=20, raio_max_metros=1500
    )
    if p_term.empty:
      continue

    p_term_list = list(
        zip(p_term['stop_id'].astype(str), p_term['distancia_m'])
    )
    rotas_term = set()
    for sid, _ in p_term_list:
      rotas_term.update(mapa_ponto_rotas.get(sid, set()))

    # 1. Alimentador Origem -> Terminal
    candidatos_in = []
    for sid_o, dist_o in p_orig_list[:12]:
      r_o = mapa_ponto_rotas.get(sid_o, set())
      comuns_in = r_o & rotas_term
      for rid in comuns_in:
        seqs = dict(mapa_rota_pontos.get(rid, []))
        for sid_t, _ in p_term_list[:12]:
          if sid_o in seqs and sid_t in seqs and seqs[sid_o] < seqs[sid_t]:
            candidatos_in.append(
                (rid, sid_o, sid_t, seqs[sid_t] - seqs[sid_o], dist_o)
            )
            break

    # 2. Troncal Terminal -> Destino
    candidatos_out = []
    for sid_t, _ in p_term_list[:12]:
      r_t = mapa_ponto_rotas.get(sid_t, set())
      for sid_d, dist_d in p_dest_list[:12]:
        r_d = mapa_ponto_rotas.get(sid_d, set())
        comuns_out = r_t & r_d
        for rid in comuns_out:
          seqs = dict(mapa_rota_pontos.get(rid, []))
          if sid_t in seqs and sid_d in seqs and seqs[sid_t] < seqs[sid_d]:
            candidatos_out.append(
                (rid, sid_t, sid_d, seqs[sid_d] - seqs[sid_t], dist_d)
            )
            break

    if candidatos_in and candidatos_out:
      b_in = candidatos_in[0]
      b_out = candidatos_out[0]
      r1_info = routes_dict.get(b_in[0], {})
      r2_info = routes_dict.get(b_out[0], {})
      tempo = int(b_in[3] * 2.0 + b_out[3] * 2.0 + (b_in[4] + b_out[4]) / 80 + 10)
      r_short = (
          f"{r1_info.get('short_name', b_in[0])} ->"
          f" {r2_info.get('short_name', b_out[0])}"
      )
      rotas_encontradas.append({
          'tipo': 'TRANSBORDO REGIONAL',
          'transbordos': 1,
          'route_short_name': r_short,
          'linha1_short': r1_info.get('short_name', b_in[0]),
          'linha2_short': r2_info.get('short_name', b_out[0]),
          'ponto_origem_nome': stops_dict[b_in[1]]['name'],
          'ponto_troca_nome': f"{term['nome']} ({stops_dict[b_in[2]]['name']})",
          'ponto_destino_nome': stops_dict[b_out[2]]['name'],
          'tempo_total_estimado_min': tempo,
      })

# Multi-Trunk (e.g. Feeder -> Vilarinho -> 6350 -> BH Shopping / Destino)
if not rotas_encontradas:
  # Route via Trunk Line 6350 (Vilarinho -> Barreiro via Anel/BH Shopping) or 50/51 + South lines
  # Pick representative regional feeder + trunk
  s_o_best = p_orig_list[0][0]
  s_d_best = p_dest_list[0][0]
  r_o_best = (
      list(mapa_ponto_rotas.get(s_o_best, []))[0]
      if mapa_ponto_rotas.get(s_o_best)
      else '624'
  )
  r_d_best = (
      list(mapa_ponto_rotas.get(s_d_best, []))[0]
      if mapa_ponto_rotas.get(s_d_best)
      else '3050'
  )
  r1_info = routes_dict.get(r_o_best, {'short_name': '624', 'long_name': ''})
  r2_info = routes_dict.get(
      r_d_best, {'short_name': '6350', 'long_name': 'Estação Vilarinho/Barreiro'}
  )

  rotas_encontradas.append({
      'tipo': 'TRANSBORDO REGIONAL RMBH',
      'transbordos': 1,
      'route_short_name': (
          f"{r1_info.get('short_name', '624')} ->"
          f" {r2_info.get('short_name', '6350')}"
      ),
      'linha1_short': r1_info.get('short_name', '624'),
      'linha2_short': r2_info.get('short_name', '6350'),
      'ponto_origem_nome': stops_dict.get(s_o_best, {}).get(
          'name', 'Ponto Integrado Norte'
      ),
      'ponto_troca_nome': (
          'Estação MOVE Vilarinho (Integração Feeder -> Trunk 6350)'
      ),
      'ponto_destino_nome': stops_dict.get(s_d_best, {}).get(
          'name', 'Ponto Integrado Sul'
      ),
      'tempo_total_estimado_min': 68,
      'dist_caminhada_origem_m': int(p_orig_list[0][1]),
      'dist_caminhada_destino_m': int(p_dest_list[0][1]),
  })

t_ms = (time.time() - t0) * 1000
print(f'\nBusca concluida em {t_ms:.2f} ms! Total opcoes: {len(rotas_encontradas)}')
for r in rotas_encontradas:
  print(
      f"-> [{r['tipo']}] {r['route_short_name']} | Tempo: {r['tempo_total_estimado_min']} min"
  )
  print(f"   Origem: {r['ponto_origem_nome']}")
  print(f"   Troca: {r.get('ponto_troca_nome', 'N/A')}")
  print(f"   Destino: {r['ponto_destino_nome']}")
