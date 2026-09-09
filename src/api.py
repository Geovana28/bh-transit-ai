"""
FastAPI REST API para BH Transit AI.

Fornece endpoints modernos com validação via Pydantic e documentação OpenAPI Swagger (/docs)
para inferência de atraso, ocupação, clima em tempo real e analytics de mobilidade urbana.
"""

from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import sys
import os

# Adiciona o diretório atual ao path de execução
sys.path.append(os.path.dirname(__file__))

from servico_clima import obter_clima_atual_bh
from motor_ia_ml import prever_impacto_transito_e_lotacao, avaliar_desempenho_modelos, obter_importancia_features
from analytics_estatistico import executar_teste_hipotese_atraso, calcular_indice_confiabilidade_tri

app = FastAPI(
    title="🚌 BH Transit AI - REST API",
    description="API de Inteligência Espaço-Temporal e Mobilidade Urbana para Belo Horizonte (MG).",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuração de CORS para permitir acesso por qualquer frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- SCHEMAS PYDANTIC ---

class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "BH Transit AI Engine"
    version: str = "2.0.0"
    author: str = "Data Science & MLOps Portfolio"


class PredictRequest(BaseModel):
    horario_str: str = Field("08:00", description="Horário de partida no formato HH:MM (ex: '08:30', '18:15')", example="08:00")
    tipo_dia_str: str = Field("Dia Útil", description="Tipo de dia: 'Dia Útil', 'Sábado' ou 'Domingo'", example="Dia Útil")
    tempo_viagem_teorico_min: float = Field(30.0, description="Tempo teórico da viagem sem trânsito em minutos", ge=1.0, example=30.0)
    distancia_estimada_km: float = Field(8.0, description="Distância percorrida em quilômetros", ge=0.5, example=8.0)
    condicao_climatica: Optional[str] = Field("☀️ Ensolarado / Tempo Bom", description="Condição meteorológica", example="🌧️ Chuva Moderada")
    precipitacao_mm: Optional[float] = Field(None, description="Volume pluviométrico instantâneo em mm/h", ge=0.0, example=4.5)


class PredictResponse(BaseModel):
    tempo_ajustado_min: int
    atraso_min: int
    fator_transito: float
    precipitacao_mm: float
    lotacao_texto: str
    explicacao: str


class TRIRequest(BaseModel):
    tempo_medio_viagem_min: float = Field(35.0, description="Tempo médio de viagem em minutos", ge=1.0, example=35.0)
    desvio_padrao_min: float = Field(8.0, description="Desvio padrão dos tempos de viagem", ge=0.0, example=8.0)
    fator_chuva: float = Field(1.0, description="Fator multiplicador de chuva (1.0 = seco)", ge=1.0, example=1.2)


class TRIResponse(BaseModel):
    tri_valor: float
    categoria: str
    formula: str


# --- ROTAS DA API ---

@app.get("/", tags=["Geral"])
def root():
    """Redirecionamento informativo para documentação interativa."""
    return {
        "message": "Bem-vindo à API do BH Transit AI!",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "health_check": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Geral"])
def health_check():
    """Verifica a integridade operacional da API."""
    return HealthResponse()


@app.get("/api/v1/weather/current", tags=["Clima & Ambiente"])
def get_current_weather():
    """Obtém as condições meteorológicas em tempo real para Belo Horizonte via Open-Meteo API."""
    clima = obter_clima_atual_bh()
    return clima


@app.post("/api/v1/predict/delay", response_model=PredictResponse, tags=["Machine Learning"])
def predict_delay_and_crowding(payload: PredictRequest):
    """
    Executa a inferência dos modelos de Machine Learning (Random Forest)
    para prever tempo ajustado, atraso provável e lotação.
    """
    try:
        resultado = prever_impacto_transito_e_lotacao(
            horario_str=payload.horario_str,
            tipo_dia_str=payload.tipo_dia_str,
            tempo_viagem_teorico_min=payload.tempo_viagem_teorico_min,
            distancia_estimada_km=payload.distancia_estimada_km,
            condicao_climatica=payload.condicao_climatica or "☀️ Ensolarado / Tempo Bom",
            precipitacao_mm=payload.precipitacao_mm
        )
        return PredictResponse(**resultado)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro durante a inferência do modelo: {str(e)}")


@app.get("/api/v1/analytics/metrics", tags=["MLOps & Analytics"])
def get_ml_metrics():
    """Retorna as métricas de validação formal do modelo (R², MAE, RMSE, F1) e Feature Importance."""
    metricas = avaliar_desempenho_modelos()
    return metricas


@app.get("/api/v1/analytics/hypothesis-test", tags=["MLOps & Analytics"])
def get_hypothesis_test_results():
    """Executa o teste formal de hipóteses estatísticas (Student's t-test e Wilcoxon)."""
    resultado = executar_teste_hipotese_atraso()
    return resultado


@app.post("/api/v1/analytics/tri", response_model=TRIResponse, tags=["MLOps & Analytics"])
def calculate_tri_index(payload: TRIRequest):
    """Calcula o Transit Reliability Index (TRI) para a rota especificada."""
    tri_val, categoria = calcular_indice_confiabilidade_tri(
        tempo_medio_viagem_min=payload.tempo_medio_viagem_min,
        desvio_padrao_min=payload.desvio_padrao_min,
        fator_chuva=payload.fator_chuva
    )
    return TRIResponse(
        tri_valor=tri_val,
        categoria=categoria,
        formula="TRI = alpha * D_delay + beta * sigma^1.5 + gamma * Delta_Rain"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
