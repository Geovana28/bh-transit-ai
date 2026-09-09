"""
Testes unitários para o serviço de clima e integração Open-Meteo (src/servico_clima.py).
"""

import unittest
from unittest.mock import patch, MagicMock
from src.servico_clima import obter_clima_atual_bh, WMO_WEATHER_CODES, LAT_BH, LON_BH


class TestServicoClima(unittest.TestCase):

    def test_constantes_coordenadas_bh(self):
        """Valida que as coordenadas de Belo Horizonte estão corretas."""
        self.assertAlmostEqual(LAT_BH, -19.9167, places=3)
        self.assertAlmostEqual(LON_BH, -43.9345, places=3)

    def test_wmo_codes_estrutura(self):
        """Verifica se os códigos WMO possuem os campos essenciais."""
        for code, info in WMO_WEATHER_CODES.items():
            self.assertIn("descricao", info)
            self.assertIn("emoji", info)
            self.assertIn("fator_chuva", info)
            self.assertIn("alerta", info)
            self.assertGreaterEqual(info["fator_chuva"], 1.0)

    def test_obter_clima_atual_integracao(self):
        """Testa a execução de obter_clima_atual_bh() retornando estrutura válida."""
        clima = obter_clima_atual_bh()
        self.assertIsInstance(clima, dict)
        self.assertIn("temperatura_c", clima)
        self.assertIn("precipitacao_mm", clima)
        self.assertIn("fator_chuva", clima)
        self.assertIn("emoji", clima)
        self.assertIn("descricao", clima)
        self.assertGreaterEqual(clima["fator_chuva"], 1.0)


if __name__ == '__main__':
    unittest.main()
