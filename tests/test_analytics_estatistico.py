"""
Testes unitários para a suíte estatística acadêmica (src/analytics_estatistico.py).
"""

import unittest
from src.analytics_estatistico import (
    executar_teste_hipotese_atraso,
    calcular_defasagem_spatiotemporal_clima,
    calcular_indice_confiabilidade_tri,
    gerar_citacao_bibtex
)

class TestAnalyticsEstatistico(unittest.TestCase):

    def test_executar_teste_hipotese_atraso(self):
        """Testa o procedimento de teste de hipótese estatístico (p-value, t-test, Cohen's d)."""
        res = executar_teste_hipotese_atraso()
        self.assertIn('p_value_ttest', res)
        self.assertIn('cohens_d', res)
        self.assertIn('reject_h0', res)
        self.assertTrue(res['reject_h0'])
        self.assertLess(res['p_value_ttest'], 0.05)

    def test_calcular_defasagem_spatiotemporal_clima(self):
        """Testa a análise de correlação cruzada temporal (Lag Analysis)."""
        lag_res = calcular_defasagem_spatiotemporal_clima(max_lag=6)
        self.assertEqual(len(lag_res['lags']), 7)
        self.assertEqual(len(lag_res['pearson_correlations']), 7)
        self.assertIn('optimal_lag_tau', lag_res)

    def test_calcular_indice_confiabilidade_tri(self):
        """Testa a fórmula matemática do TRI (Transit Reliability Index)."""
        tri_val, cat = calcular_indice_confiabilidade_tri(10.0, 2.0, fator_chuva=1.0)
        self.assertIsInstance(tri_val, float)
        self.assertIn("Alta Confiabilidade", cat)

    def test_gerar_citacao_bibtex(self):
        """Testa a geração da citação em formato BibTeX."""
        bibtex = gerar_citacao_bibtex()
        self.assertIn("@techreport", bibtex)
        self.assertIn("BH Transit AI", bibtex)

if __name__ == '__main__':
    unittest.main()
