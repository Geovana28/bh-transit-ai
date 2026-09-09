"""
Testes unitários para os modelos de Machine Learning & MLOps (src/motor_ia_ml.py).
"""

import unittest
import pandas as pd
from src.motor_ia_ml import (
    gerar_dados_treinamento_sinteticos_bh,
    prever_impacto_transito_e_lotacao,
    avaliar_desempenho_modelos,
    obter_importancia_features,
    FEATURE_NAMES
)


class TestMotorIAML(unittest.TestCase):

    def test_gerar_dados_treinamento_sinteticos(self):
        """Testa integridade estrutural do dataset e presença das features de clima."""
        df = gerar_dados_treinamento_sinteticos_bh()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 500)
        for col in FEATURE_NAMES:
            self.assertIn(col, df.columns)
        self.assertIn('fator_transito', df.columns)
        self.assertIn('lotacao', df.columns)
        self.assertIn('precipitacao_mm', df.columns)

    def test_prever_impacto_transito_e_lotacao_pico(self):
        """Testa inferência de modelo para horário de pico da tarde com chuva numérica."""
        res = prever_impacto_transito_e_lotacao(
            horario_str="18:00",
            tipo_dia_str="Dia Útil",
            tempo_viagem_teorico_min=30.0,
            distancia_estimada_km=10.0,
            condicao_climatica="🌧️ Chuva Moderada",
            precipitacao_mm=6.0
        )
        self.assertIn('tempo_ajustado_min', res)
        self.assertIn('atraso_min', res)
        self.assertIn('fator_transito', res)
        self.assertIn('precipitacao_mm', res)
        self.assertEqual(res['precipitacao_mm'], 6.0)
        self.assertGreaterEqual(res['fator_transito'], 1.0)
        self.assertGreaterEqual(res['tempo_ajustado_min'], 30)

    def test_obter_importancia_features(self):
        """Testa o cálculo de interpretabilidade das features (XAI)."""
        importancias = obter_importancia_features()
        self.assertIn('regressor', importancias)
        self.assertIn('classificador', importancias)
        self.assertIn('Precipitação (Chuva mm/h)', importancias['regressor'])
        self.assertIn('Horário do Dia (Minutos)', importancias['regressor'])

    def test_avaliar_desempenho_modelos(self):
        """Testa o cálculo MLOps de métricas R² e F1-score."""
        metricas = avaliar_desempenho_modelos()
        self.assertIn('regressor_r2', metricas)
        self.assertIn('classifier_f1_score', metricas)
        self.assertIn('feature_importance', metricas)
        self.assertGreater(metricas['regressor_r2'], 0.5)


if __name__ == '__main__':
    unittest.main()
