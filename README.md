# 🚌 BH Transit AI — Inteligência Espaço-Temporal & Ambiental para Mobilidade Urbana

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![PyTest](https://img.shields.io/badge/Tests-Pytest-brightgreen?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)](https://github.com/Geovana28/bh-transit-ai/actions)
[![Licença MIT](https://img.shields.io/badge/Licença-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Plataforma end-to-end de inteligência espaço-temporal para mobilidade urbana em Belo Horizonte.**
> Integra GTFS, Machine Learning, Explainable AI, meteorologia em tempo real e API RESTful.

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
