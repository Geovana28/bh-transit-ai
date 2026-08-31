"""
Testes unitários para o módulo de roteamento e busca geográfica (src/motor_rotas.py).
"""

import unittest
import numpy as np
import pandas as pd
from src.motor_rotas import (
    calcular_distancia_haversine,
    converter_tempo_para_minutos,
    converter_minutos_para_tempo,
    encontrar_pontos_proximos
)

class TestMotorRotas(unittest.TestCase):

    def test_calcular_distancia_haversine_mesmo_ponto(self):
        """Mão dupla na mesma coordenada deve retornar distância 0."""
        lat, lon = -19.9167, -43.9345
        dist = calcular_distancia_haversine(lat, lon, lat, lon)
        self.assertAlmostEqual(dist, 0.0, places=2)

    def test_calcular_distancia_haversine_conhecida(self):
        """Distância entre Praça da Liberdade (-19.9317, -43.9381) e Rodoviária de BH (-19.9161, -43.9407)."""
        lat1, lon1 = -19.9317, -43.9381
        lat2, lon2 = -19.9161, -43.9407
        dist = calcular_distancia_haversine(lat1, lon1, lat2, lon2)
        # Distância aproximada em linha reta é de ~1.7 a 1.8 km (1700m - 1800m)
        self.assertTrue(1500.0 <= dist <= 2000.0)

    def test_converter_tempo_para_minutos(self):
        """Testa conversão de string de horário para minutos do dia."""
        self.assertEqual(converter_tempo_para_minutos("08:30"), 510)
        self.assertEqual(converter_tempo_para_minutos("17:45:00"), 1065)
        self.assertIsNone(converter_tempo_para_minutos("invalido"))

    def test_converter_minutos_para_tempo(self):
        """Testa conversão de minutos para formato HH:MM."""
        self.assertEqual(converter_minutos_para_tempo(510), "08:30")
        self.assertEqual(converter_minutos_para_tempo(1065), "17:45")
        self.assertEqual(converter_minutos_para_tempo(None), "--:--")

    def test_encontrar_pontos_proximos(self):
        """Testa busca espacial dos top_k pontos de ônibus."""
        df_stops = pd.DataFrame({
            'stop_id': ['P1', 'P2', 'P3'],
            'stop_name': ['Parada A', 'Parada B', 'Parada C'],
            'stop_lat': [-19.9167, -19.9200, -20.0000],
            'stop_lon': [-43.9345, -43.9350, -44.0000]
        })
        lat_ref, lon_ref = -19.9165, -43.9340
        proximos = encontrar_pontos_proximos(lat_ref, lon_ref, df_stops, top_k=2, raio_max_metros=5000)
        self.assertEqual(len(proximos), 2)
        self.assertEqual(proximos.iloc[0]['stop_id'], 'P1')

if __name__ == '__main__':
    unittest.main()
