"""
Testes unitários para os modelos de Machine Learning & MLOps (src/motor_ia_ml.py).
"""

import unittest
import pandas as pd
from src.motor_ia_ml import (
    gerar_dados_treinamento_sinteticos_bh,
    prever_impacto_transito_e_lotacao,
    avaliar_desempenho_modelos
)

class TestMotorIAML(unittest.TestCase):

    def test_gerar_dados_treinamento_sinteticos(self):
        """Testa integridade estrutural do dataset sintético."""
        df = gerar_dados_treinamento_sinteticos_bh()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 500)
        self.assertIn('minutos_dia', df.columns)
        self.assertIn('fator_transito', df.columns)
        self.assertIn('lotacao', df.columns)

    def test_prever_impacto_transito_e_lotacao_pico(self):
        """Testa inferência de modelo para horário de pico da tarde com chuva."""
        res = prever_impacto_transito_e_lotacao(
            horario_str="18:00",
            tipo_dia_str="Dia Útil",
            tempo_viagem_teorico_min=30.0,
            distancia_estimada_km=10.0,
            condicao_climatica="🌧️ Chuva Moderada"
        )
        self.assertIn('tempo_ajustado_min', res)
        self.assertIn('atraso_min', res)
        self.assertIn('fator_transito', res)
        self.assertGreaterEqual(res['fator_transito'], 1.0)
        self.assertGreaterEqual(res['tempo_ajustado_min'], 30)

    def test_avaliar_desempenho_modelos(self):
        """Testa o cálculo MLOps de métricas R² e F1-score."""
        metricas = avaliar_desempenho_modelos()
        self.assertIn('regressor_r2', metricas)
        self.assertIn('classifier_f1_score', metricas)
        self.assertGreater(metricas['regressor_r2'], 0.5)

if __name__ == '__main__':
    unittest.main()
