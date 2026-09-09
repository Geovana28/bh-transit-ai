import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import motor_rotas

cache = motor_rotas.obter_cache_gtfs()
routes_dict = cache['routes_dict']
stops_dict = cache['stops_dict']
mapa_rota_pontos = cache['mapa_rota_pontos']

# Check lines 624, 607, 3050, 3030, 8151
for rid, info in routes_dict.items():
  if info['short_name'] in [
      '624',
      '607',
      '3050',
      '3030',
      '8151',
      '50',
      '51',
      '6350',
  ]:
    seqs = mapa_rota_pontos.get(rid, [])
    if seqs:
      first_stop = stops_dict.get(seqs[0][0], {}).get('name', 'N/A')
      last_stop = stops_dict.get(seqs[-1][0], {}).get('name', 'N/A')
      print(f"{info['short_name']} ({info['long_name']}): {first_stop} -> {last_stop}")
