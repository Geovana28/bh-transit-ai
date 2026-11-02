"""
Testes unitários do Agente de IA com Tool Calling e Guardrails Anti-Alucinação (src/agente_llm.py).
Mede a acurácia de seleção de ferramentas (Tool Selection Accuracy) e o bloqueio de perguntas fora de escopo.
"""

import unittest
from src.agente_llm import execute_agent_query, AGENT_TOOLS_SPEC


class TestAgenteLLM(unittest.TestCase):

    def test_agent_tools_specification_structure(self):
        """Verifica que todas as ferramentas declaradas possuem nome, descrição e lista de parâmetros."""
        self.assertEqual(len(AGENT_TOOLS_SPEC), 4)
        for tool in AGENT_TOOLS_SPEC:
            self.assertIn("name", tool)
            self.assertIn("description", tool)
            self.assertIn("parameters", tool)

    def test_tool_selection_eta_and_crowding(self):
        """Perguntas sobre atraso, horário de pico e chuva em rotas devem acionar predict_eta_and_crowding."""
        res = execute_agent_query("Qual o atraso previsto para pegar ônibus na Savassi às 18h com chuva forte?")
        self.assertEqual(res["selected_tool"], "predict_eta_and_crowding")
        self.assertEqual(res["tool_arguments"]["horario_str"], "18:00")
        self.assertGreater(res["tool_arguments"]["precipitacao_mm"], 0.0)
        self.assertIn("conformal_interval_90", res["tool_output"])

    def test_tool_selection_weather(self):
        """Perguntas diretas sobre o clima em BH devem acionar get_realtime_weather_bh."""
        res = execute_agent_query("Como está o tempo e a temperatura em BH agora?")
        self.assertEqual(res["selected_tool"], "get_realtime_weather_bh")
        self.assertIn("temperatura_c", res["tool_output"])

    def test_tool_selection_tri_reliability(self):
        """Perguntas sobre índice TRI ou confiabilidade operacional devem acionar compute_route_reliability_tri."""
        res = execute_agent_query("Calcule o índice de confiabilidade TRI da linha em dia de chuva.")
        self.assertEqual(res["selected_tool"], "compute_route_reliability_tri")
        self.assertIn("tri_valor", res["tool_output"])

    def test_out_of_scope_guardrail_blocks_hallucination(self):
        """Perguntas fora do domínio de mobilidade urbana de BH devem ser recusadas pelo guardrail."""
        queries_fora = [
            "Me passe uma receita de bolo de chocolate.",
            "Qual a cotação do Bitcoin hoje?",
            "Quem ganhou o jogo de futebol ontem?",
        ]
        for q in queries_fora:
            res = execute_agent_query(q)
            self.assertEqual(res["selected_tool"], "out_of_scope_guardrail")
            self.assertTrue(res["tool_output"]["blocked"])


if __name__ == "__main__":
    unittest.main()
