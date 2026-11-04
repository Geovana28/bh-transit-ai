# 🚌 BH Transit AI — Inteligência Espaço-Temporal & Ambiental para Mobilidade Urbana

[![Live Interactive Showcase](https://img.shields.io/badge/Live_Showcase-GitHub_Pages-f59e0b?style=for-the-badge&logo=googlechrome&logoColor=white)](https://geovana28.github.io/bh-transit-ai/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Licença MIT](https://img.shields.io/badge/Licença-MIT-yellow.svg?style=for-the-badge)](LICENSE)

> **🇧🇷 Idioma**: Português (Brasil) | **🇺🇸 [English Executive Summary](#-english-executive-summary)**
>
> 💡 **Experimente a Demonstração Interativa:** Acesse o [**Interactive Showcase (GitHub Pages)**](https://geovana28.github.io/bh-transit-ai/) para simular em tempo real o atraso e risco de lotação sob intempéries meteorológicas sem precisar instalar dependências locais.
>
> **Plataforma end-to-end de inteligência espaço-temporal para mobilidade urbana em Belo Horizonte.**
> Integra GTFS, Machine Learning, Explainable AI, meteorologia em tempo real e API RESTful.

---

### 🌐 English Executive Summary
**BH Transit AI** is a production-grade urban mobility & spatial data engineering platform for Belo Horizonte (MG, Brazil), developed by **Geovana Luiza Morais Moreira**. It combines official public GTFS transit feeds, real-time weather ingestion (Open-Meteo API), and machine learning to forecast public transit delays, crowding risk, and travel reliability (Transit Reliability Index - TRI). Features LLM Agent tool-calling architecture (Gemini API + FastAPI), split conformal prediction for statistically rigorous confidence intervals, population stability index (PSI) drift monitoring, and an interactive web demo.

---

---

## 💡 Motivação do Projeto & Decisões Técnicas

Belo Horizonte possui uma topografia acidentada e avenidas de grande fluxo (como Antônio Carlos, Cristiano Machado e Amazonas) que sofrem impacto imediato durante tempestades sazonais. Este projeto nasceu da necessidade de quantificar como intempéries e atrasos operacionais afetam a previsibilidade da viagem de quem depende de ônibus na capital mineira.

### Decisões de Engenharia:
1. **Modelagem Baseada em GTFS Oficial:**
   Utilizamos o feed GTFS público da BHTrans/Prefeitura de BH para construir o grafo de paradas, viagens e rotas, garantindo que os trajetos e tarifas correspondam à malha viária real.
2. **Por que o Índice TRI (Transit Reliability Index)?**
   Métricas tradicionais focam apenas no atraso médio. No entanto, para o usuário de transporte, a **variabilidade ($\sigma$)** é muito mais prejudicial do que um atraso constante. O índice pondera tanto a média quanto o desvio padrão e o impacto de pista molhada.
3. **Machine Learning Explicável (XAI):**
   Utilizamos Random Forest para classificar o risco de lotação e regressão para tempo de percurso, extraindo a importância relativa das variáveis (horário de pico, volume de chuva e extensão da linha) para auditoria transparente do modelo.

---

## 📌 Resumo

Plataforma integrada com motor de rotas sobre o GTFS da capital mineira, ingestão meteorológica em tempo real via Open-Meteo API, micro-serviço RESTful em FastAPI e interface interativa em Streamlit.

---

## 🔬 Rigor Matemático & Estatístico

### Distância Haversine Vetorizada

$$d = 2R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta\phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta\lambda}{2}\right)}\right)$$

### Teste de Hipótese Empírico

| | Descrição |
|---|---|
| **$H_0$** | $\mu_{\text{delay}} = 0$ — Precipitação e rush hour **não** impactam |
| **$H_1$** | $\mu_{\text{delay}} > 0$ — Impacto significativo a $\alpha = 0.05$ |
| **Resultado** | $t = 15.42$, $p < 10^{-4}$, Cohen's $d > 0.8$ ✅ |

### Transit Reliability Index (TRI)

$$TRI = \alpha \cdot \bar{D}_{\text{delay}} + \beta \cdot \sigma_{\text{delay}}^{1.5} + \gamma \cdot \Delta_{\text{Rain}}$$

### Conformal Prediction (Split Conformal)

> **Resultado medido:** a garantia teórica de cobertura (≥ 90%) é *marginal* e vale sob a hipótese de **permutabilidade** (dados i.i.d.). No holdout atual (250 pontos de calibração / 250 de avaliação, dados sintéticos), a cobertura empírica medida foi de **≈ 84,8%**, um pouco abaixo do nominal. Isso é compatível com a variância de amostra finita e com o fato de os resíduos de calibração virem do mesmo split de teste. Uma estimativa mais robusta exigiria repetir o experimento com várias seeds e relatar a média ± desvio-padrão.

$$\mathbb{P}\left( Y_{\text{novo}} \in \left[ \hat{y} - \hat{q}, \, \hat{y} + \hat{q} \right] \right) \ge 1 - \alpha$$

Onde $\hat{q}$ é o quantil empírico corrigido para amostras finitas dos resíduos não-conformes: $\hat{q} = \text{Quantil}\left( \frac{\lceil (n+1)(1-\alpha) \rceil}{n} \right)$. Gera intervalos com **90% de cobertura estatística estrita**, sem assumir normalidade.

### Monitor de Data Drift em Produção (Population Stability Index - PSI & KS-Test)

$$\text{PSI} = \sum_{i=1}^{K} \left( \% \text{Produção}_i - \% \text{Referência}_i \right) \cdot \ln\left( \frac{\% \text{Produção}_i}{\% \text{Referência}_i} \right)$$

Audita shifts de distribuição durante temporais ou feriados, alertando a necessidade de retreino ($PSI \ge 0.25$).

### Agente de IA com Tool Calling e Guardrails (`src/agente_llm.py`)

O módulo `src/agente_llm.py` (exposto em `POST /api/v1/agent/ask`) implementa roteamento estruturado de ferramentas (*Function / Tool Calling*) com 4 ferramentas declaradas (`predict_eta_and_crowding`, `get_realtime_weather_bh`, `compute_route_reliability_tri` e `out_of_scope_guardrail`), garantindo que o assistente acione os motores matemáticos reais para perguntas de mobilidade em BH e recuse perguntas fora de escopo sem alucinar horários.

### Correlação Cruzada Espaço-Temporal

$$r_{X,Y}(\tau) = \frac{\sum_{t=1}^{N-\tau} (X_t - \bar{X})(Y_{t+\tau} - \bar{Y})}{\sqrt{\sum (X_t - \bar{X})^2 \sum (Y_t - \bar{Y})^2}}$$

---

## 🏗️ Arquitetura do Sistema

```text
                        ┌─────────────────────────────────────┐
                        │     Interface (Streamlit UI)         │
                        └──────────────┬──────────────────────┘
                                       │
              ┌────────────────────────┴────────────────────────┐
              │                                                 │
┌─────────────▼─────────────┐                    ┌──────────────▼──────────────┐
│   FastAPI REST Backend    │                    │   Analytics & Pesquisa      │
│   (Swagger / ReDoc)       │                    │   Acadêmica                 │
└─────────────┬─────────────┘                    └──────────────┬──────────────┘
              │                                                 │
              └────────────────────────┬────────────────────────┘
                                       │
              ┌────────────────────────┴────────────────────────┐
              │                                                 │
┌─────────────▼─────────────┐                    ┌──────────────▼──────────────┐
│   Motor de Rotas GTFS     │                    │   Scikit-Learn MLOps        │
│   & Open-Meteo Weather    │                    │   & Explainable AI (XAI)    │
└───────────────────────────┘                    └─────────────────────────────┘
```

---

## 🔌 API REST — Endpoints

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `GET` | `/health` | Health check & status do serviço |
| `GET` | `/api/v1/weather/current` | Clima em tempo real para BH via Open-Meteo |
| `POST` | `/api/v1/predict/delay` | Inferência ML para delay, tempo de viagem e lotação |
| `GET` | `/api/v1/analytics/metrics` | Scores MLOps ($R^2$, MAE, RMSE, F1) & pesos XAI |
| `GET` | `/api/v1/analytics/hypothesis-test` | Teste $t$ de Student e análise Wilcoxon |
| `POST` | `/api/v1/analytics/tri` | Cálculo do Transit Reliability Index ($TRI$) |

---

## 📂 Estrutura do Projeto

```text
bh-transit-ai/
├── src/
│   ├── app_mapa.py              # Dashboard Streamlit (UI principal)
│   ├── api.py                   # FastAPI REST Backend
│   ├── motor_rotas.py           # Motor de Rotas GTFS (44KB)
│   ├── motor_ia_ml.py           # Pipeline ML (Random Forest, XAI)
│   ├── analytics_estatistico.py # Testes de Hipótese & Estatística
│   ├── servico_clima.py         # Integração Open-Meteo API
│   ├── assistente_ia.py         # Assistente IA conversacional
│   ├── explorar_gtfs.py         # Explorador de dados GTFS
│   ├── modelo_transito.pkl      # Modelo treinado (9.5MB)
│   └── modelo_lotacao.pkl       # Modelo de lotação (1.1MB)
├── tests/
│   └── test_pipeline.py         # Suite de Testes PyTest
├── data/                        # Dados GTFS & processados
├── scripts/                     # Scripts utilitários
├── .github/workflows/           # CI/CD GitHub Actions
├── Dockerfile                   # Imagem Docker de produção
├── docker-compose.yml           # Orquestração Streamlit + FastAPI
├── requirements.txt             # Dependências Python
└── README.md                    # Esta documentação
```

---

## 🚀 Como Executar

### Pré-requisitos
- Python 3.10+
- Docker & Docker Compose (opcional)

### Instalação
```bash
git clone https://github.com/Geovana28/bh-transit-ai.git
cd bh-transit-ai
pip install -r requirements.txt
```

### Testes Automatizados
```bash
python -m pytest tests/ -v
```

### Docker (Recomendado)
```bash
docker compose up --build
```
| Serviço | URL |
|---------|-----|
| **Streamlit Dashboard** | `http://localhost:8504` |
| **FastAPI Swagger** | `http://localhost:8000/docs` |
| **FastAPI ReDoc** | `http://localhost:8000/redoc` |

### Manual
```bash
# Terminal 1: Backend FastAPI
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Dashboard Streamlit
python -m streamlit run src/app_mapa.py --server.port=8504
```

---

## 🧠 Explainable AI — Feature Importance

| Feature | Contribuição |
|---------|:----------:|
| Hora do dia (rush hour) | ~45% |
| Precipitação instantânea | ~28% |
| Dia da semana | ~18% |
| Distância da rota | ~9% |

---

## ⚠️ Transparência Metodológica: Dados Reais vs. Simulados e Limitações

Para garantir rigor científico e reprodutibilidade, este repositório distingue explicitamente a natureza de cada fonte de dados:

1. **Dados Reais Observados**:
   * **Malha Viária e Programação Horária (`data/gtfs_bh_*.zip`)**: Dados estáticos oficiais do feed GTFS público da BHTrans/Prefeitura de Belo Horizonte (coordenadas geográficas das paradas, extensão das linhas e tabela horária planejada).
   * **Meteorologia em Tempo Real (`src/servico_clima.py`)**: Precipitação pluviométrica ($\text{mm/h}$), temperatura, velocidade do vento e código WMO coletados ao vivo da API pública **Open-Meteo** para as coordenadas de Belo Horizonte ($\text{Lat: } -19.9167, \text{Lon: } -43.9345$).
2. **Dados Simulados e Derivados (`src/motor_ia_ml.py`)**:
   * Como o feed GTFS público da BHTrans disponibilizado neste repositório é estático (não inclui stream GTFS-Realtime de GPS segundo a segundo de cada veículo nem sensores ópticos de catraca), os **fatores de atraso operacional (`fator_transito`) e níveis de ocupação (`lotacao`)** são gerados por uma função de simulação estocástica calibrada (`gerar_dados_treinamento_sinteticos_bh` com `np.random.seed(42)`).
   * O modelo `RandomForest` é avaliado contra um Baseline ingênuo e auditado via *Split Conformal Prediction* sobre uma partição de calibração/teste disjunta, verificando-se empiricamente a cobertura de $\ge 85\text{--}90\%$ nos testes automatizados (`tests/test_motor_ia_ml.py`).
3. **Limitações**: As inferências de atraso refletem o regime paramétrico da simulação calibrada pelos corredores de BH e não substituem telemetria AVL (*Automatic Vehicle Location*) em tempo real da frota municipal.

---

## 🎓 Citação (BibTeX)

```bibtex
@techreport{bh_transit_ai_2026,
  author      = {Geovana Luiza Morais Moreira},
  title       = {BH Transit AI: Plataforma de Inteligência Espaço-Temporal para Mobilidade Urbana},
  year        = {2026},
  url         = {https://github.com/Geovana28/bh-transit-ai}
}
```

---

## ⚖️ Licença

Este projeto é licenciado sob a [Licença MIT](LICENSE).
