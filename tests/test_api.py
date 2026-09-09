"""
Testes unitários para a REST API FastAPI (src/api.py).
"""

import unittest

try:
    from fastapi.testclient import TestClient
    from src.api import app
    # Verifica se TestClient consegue inicializar sem erro de httpx
    client = TestClient(app)
    HAS_FASTAPI_CLIENT = True
except (ImportError, RuntimeError, Exception):
    HAS_FASTAPI_CLIENT = False
    app = None
    client = None


class TestFastAPI(unittest.TestCase):

    def setUp(self):
        if not HAS_FASTAPI_CLIENT:
            self.skipTest("FastAPI TestClient (ou httpx) não disponível no ambiente")
        self.client = client

    def test_root_endpoint(self):
        """Testa endpoint raiz /."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("docs_url", data)
        self.assertIn("health_check", data)

    def test_health_endpoint(self):
        """Testa endpoint /health."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["version"], "2.0.0")

    def test_weather_endpoint(self):
        """Testa endpoint /api/v1/weather/current."""
        response = self.client.get("/api/v1/weather/current")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("temperatura_c", data)
        self.assertIn("fator_chuva", data)

    def test_predict_delay_endpoint(self):
        """Testa POST /api/v1/predict/delay com payload válido."""
        payload = {
            "horario_str": "18:00",
            "tipo_dia_str": "Dia Útil",
            "tempo_viagem_teorico_min": 35.0,
            "distancia_estimada_km": 12.0,
            "condicao_climatica": "🌧️ Chuva Moderada",
            "precipitacao_mm": 5.5
        }
        response = self.client.post("/api/v1/predict/delay", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("tempo_ajustado_min", data)
        self.assertIn("atraso_min", data)
        self.assertIn("fator_transito", data)
        self.assertGreaterEqual(data["tempo_ajustado_min"], 35)

    def test_analytics_metrics_endpoint(self):
        """Testa GET /api/v1/analytics/metrics."""
        response = self.client.get("/api/v1/analytics/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("regressor_r2", data)
        self.assertIn("feature_importance", data)

    def test_analytics_tri_endpoint(self):
        """Testa POST /api/v1/analytics/tri."""
        payload = {
            "tempo_medio_viagem_min": 30.0,
            "desvio_padrao_min": 5.0,
            "fator_chuva": 1.1
        }
        response = self.client.post("/api/v1/analytics/tri", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("tri_valor", data)
        self.assertIn("categoria", data)


if __name__ == '__main__':
    unittest.main()
