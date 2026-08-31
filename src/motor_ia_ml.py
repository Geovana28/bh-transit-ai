"""
Motor de Machine Learning & MLOps para Mobilidade Urbana.

Este módulo implementa os modelos estatísticos e algoritmos de Machine Learning
(RandomForestRegressor e RandomForestClassifier) com foco em previsão de atrasos no
transporte público e ocupação urbana em Belo Horizonte (MG).
"""

from typing import Dict, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
import os

try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, confusion_matrix
    from sklearn.model_selection import train_test_split
    import joblib
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

MODEL_TRANSITO_PATH: str = os.path.join(os.path.dirname(__file__), 'modelo_transito.pkl')
MODEL_LOTACAO_PATH: str = os.path.join(os.path.dirname(__file__), 'modelo_lotacao.pkl')


def gerar_dados_treinamento_sinteticos_bh() -> pd.DataFrame:
    """
    Gera um dataset sintético baseado no padrão histórico de tráfego de Belo Horizonte (MG).

    Padrões simulados:
        - Pico da Manhã (07:00 - 09:30): Trânsito Intenso (Fator 1.3 - 1.55), Lotação Alta.
        - Pico da Tarde (17:00 - 19:30): Trânsito Crítico em BH (Fator 1.35 - 1.65), Lotação Alta.
        - Entrepico (10:00 - 16:00): Trânsito Moderado (Fator 1.08 - 1.22), Lotação Média.
        - Noturno (21:00 - 06:00): Trânsito Livre (Fator 1.0 - 1.05), Lotação Baixa.
        - Fim de semana (Sábado/Domingo): Menor volume urbano.

    Returns:
        pd.DataFrame: DataFrame contendo colunas ['minutos_dia', 'dia_semana',
            'distancia_km', 'fator_transito', 'lotacao'].
    """
    np.random.seed(42)
    n_amostras = 2000

    minutos_dia = np.random.randint(0, 1440, n_amostras)
    dias_semana = np.random.randint(0, 7, n_amostras)  # 0=Segunda, 6=Domingo
    distancia_km = np.random.uniform(2.0, 25.0, n_amostras)

    fatores_transito = []
    lotacao_labels = []

    for m, d in zip(minutos_dia, dias_semana):
        hora = m / 60.0
        e_dia_util = d < 5
        
        if e_dia_util:
            if 7.0 <= hora <= 9.5:
                fator = np.random.uniform(1.3, 1.55)
                lotacao = 2
            elif 17.0 <= hora <= 19.5:
                fator = np.random.uniform(1.35, 1.65)
                lotacao = 2
            elif 11.5 <= hora <= 14.0:
                fator = np.random.uniform(1.15, 1.3)
                lotacao = 1
            elif 21.0 <= hora or hora <= 6.0:
                fator = np.random.uniform(1.0, 1.05)
                lotacao = 0
            else:
                fator = np.random.uniform(1.08, 1.22)
                lotacao = 1
        else:
            if 10.0 <= hora <= 18.0 and d == 5:
                fator = np.random.uniform(1.1, 1.25)
                lotacao = 1
            else:
                fator = np.random.uniform(1.0, 1.08)
                lotacao = 0

        fatores_transito.append(fator)
        lotacao_labels.append(lotacao)

    return pd.DataFrame({
        'minutos_dia': minutos_dia,
        'dia_semana': dias_semana,
        'distancia_km': distancia_km,
        'fator_transito': fatores_transito,
        'lotacao': lotacao_labels
    })


def treinar_e_salvar_modelos() -> Tuple[Optional[Any], Optional[Any]]:
    """
    Treina os modelos de Regressão (Atraso/Trânsito) e Classificação (Lotação) com Scikit-Learn.

    Returns:
        Tuple[Optional[RandomForestRegressor], Optional[RandomForestClassifier]]:
            Tupla contendo o regressor e o classificador treinados, ou (None, None) se scikit-learn não estiver disponível.
    """
    if not HAS_SKLEARN:
        return None, None
        
    df = gerar_dados_treinamento_sinteticos_bh()
    
    X = df[['minutos_dia', 'dia_semana', 'distancia_km']]
    y_transito = df['fator_transito']
    y_lotacao = df['lotacao']

    regressor = RandomForestRegressor(n_estimators=50, random_state=42)
    regressor.fit(X, y_transito)

    classificador = RandomForestClassifier(n_estimators=50, random_state=42)
    classificador.fit(X, y_lotacao)

    try:
        joblib.dump(regressor, MODEL_TRANSITO_PATH)
        joblib.dump(classificador, MODEL_LOTACAO_PATH)
    except Exception:
        pass

    return regressor, classificador


def carregar_modelos_ml() -> Tuple[Optional[Any], Optional[Any]]:
    """
    Carrega os modelos de Machine Learning salvos em disco ou executa o treinamento inicial.

    Returns:
        Tuple[Optional[Any], Optional[Any]]: Instâncias dos modelos treinados ou (None, None).
    """
    if not HAS_SKLEARN:
        return None, None

    if os.path.exists(MODEL_TRANSITO_PATH) and os.path.exists(MODEL_LOTACAO_PATH):
        try:
            regressor = joblib.load(MODEL_TRANSITO_PATH)
            classificador = joblib.load(MODEL_LOTACAO_PATH)
            return regressor, classificador
        except Exception:
            pass
            
    return treinar_e_salvar_modelos()


# Inicializa os modelos globalmente
regressor_transito, classificador_lotacao = carregar_modelos_ml()


def avaliar_desempenho_modelos() -> Dict[str, Any]:
    """
    Calcula métricas formais de avaliação de desempenho dos modelos de Machine Learning (MLOps).

    Returns:
        Dict[str, Any]: Dicionário contendo R², MAE, RMSE para regressão e F1-score ponderado para classificação.
    """
    if not HAS_SKLEARN:
        return {"status": "scikit-learn not installed"}

    df = gerar_dados_treinamento_sinteticos_bh()
    X = df[['minutos_dia', 'dia_semana', 'distancia_km']]
    y_trans = df['fator_transito']
    y_lot = df['lotacao']

    X_train, X_test, y_tr_train, y_tr_test, y_lot_train, y_lot_test = train_test_split(
        X, y_trans, y_lot, test_size=0.2, random_state=42
    )

    rf_reg = RandomForestRegressor(n_estimators=50, random_state=42)
    rf_reg.fit(X_train, y_tr_train)
    preds_reg = rf_reg.predict(X_test)

    rf_clf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_clf.fit(X_train, y_lot_train)
    preds_clf = rf_clf.predict(X_test)

    r2 = r2_score(y_tr_test, preds_reg)
    mae = mean_absolute_error(y_tr_test, preds_reg)
    rmse = np.sqrt(mean_squared_error(y_tr_test, preds_reg))
    f1 = f1_score(y_lot_test, preds_clf, average='weighted')

    return {
        "regressor_r2": float(round(r2, 4)),
        "regressor_mae": float(round(mae, 4)),
        "regressor_rmse": float(round(rmse, 4)),
        "classifier_f1_score": float(round(f1, 4)),
        "test_samples": len(X_test)
    }


def prever_impacto_transito_e_lotacao(
    horario_str: str,
    tipo_dia_str: str,
    tempo_viagem_teorico_min: float,
    distancia_estimada_km: float = 8.0,
    condicao_climatica: str = "☀️ Ensolarado / Tempo Bom"
) -> Dict[str, Any]:
    """
    Estima o tempo real ajustado, atraso previsto, fator de trânsito e nível de lotação urbana.

    Args:
        horario_str (str): Horário no formato 'HH:MM'.
        tipo_dia_str (str): Tipo de dia ('Dia Útil', 'Sábado', 'Domingo').
        tempo_viagem_teorico_min (float): Tempo teórico calculado pelo itinerário.
        distancia_estimada_km (float): Distância percorrida em quilômetros.
        condicao_climatica (str): Condição meteorológica atual.

    Returns:
        Dict[str, Any]: Dicionário com tempo ajustado, atraso em minutos, fator de trânsito,
            texto de lotação e alerta explicativo.
    """
    try:
        partes = horario_str.split(':')
        horas = int(partes[0])
        mins = int(partes[1])
        minutos_dia = horas * 60 + mins
    except Exception:
        minutos_dia = 8 * 60

    if "Sábado" in tipo_dia_str or "Sabado" in tipo_dia_str:
        dia_semana = 5
    elif "Domingo" in tipo_dia_str or "Feriado" in tipo_dia_str:
        dia_semana = 6
    else:
        dia_semana = 2

    X_pred = pd.DataFrame([[minutos_dia, dia_semana, distancia_estimada_km]],
                          columns=['minutos_dia', 'dia_semana', 'distancia_km'])

    if regressor_transito is not None and classificador_lotacao is not None:
        fator_pred = float(regressor_transito.predict(X_pred)[0])
        lotacao_pred = int(classificador_lotacao.predict(X_pred)[0])
    else:
        hora_float = minutos_dia / 60.0
        if dia_semana < 5 and ((7.0 <= hora_float <= 9.5) or (17.0 <= hora_float <= 19.5)):
            fator_pred = 1.40
            lotacao_pred = 2
        else:
            fator_pred = 1.10
            lotacao_pred = 1

    alerta_clima = ""
    if "Forte" in condicao_climatica or "Temporal" in condicao_climatica or "Alagamento" in condicao_climatica:
        fator_pred += 0.35
        alerta_clima = " ⛈️ **Alerta Climatológico Defesa Civil BH:** Temporal/Chuva Forte detectada! Redução de velocidade nos corredores e retenções em vias como Vilarinho, Prudente de Morais e Amazonas."
    elif "Chuva" in condicao_climatica or "Moderada" in condicao_climatica:
        fator_pred += 0.15
        alerta_clima = " 🌧️ **Impacto de Chuva:** Asfalto molhado e tráfego reduzido na capital."

    tempo_ajustado_min = int(tempo_viagem_teorico_min * fator_pred)
    atraso_min = max(0, tempo_ajustado_min - int(tempo_viagem_teorico_min))

    mapa_lotacao = {
        0: "🟢 Vazio (Muitos assentos disponíveis)",
        1: "🟡 Moderado (Passageiros sentados e alguns em pé)",
        2: "🔴 Lotado (Ônibus cheio / Horário de pico)"
    }
    lotacao_texto = mapa_lotacao.get(lotacao_pred, "🟡 Moderado")

    if atraso_min > 5:
        explicacao = f"⚠️ **Alerta de Trânsito Inteligente (ML):** Tráfego intenso previsto em BH (+{atraso_min} min de atraso estimado).{alerta_clima}"
    elif atraso_min > 0:
        explicacao = f"🟡 **Trânsito Moderado (ML):** Pequeno impacto de tráfego estimado (+{atraso_min} min no trajeto).{alerta_clima}"
    else:
        explicacao = f"🟢 **Trânsito Livre (ML):** Tráfego fluindo normalmente.{alerta_clima}"

    return {
        'tempo_ajustado_min': tempo_ajustado_min,
        'atraso_min': atraso_min,
        'fator_transito': round(fator_pred, 2),
        'lotacao_texto': lotacao_texto,
        'explicacao': explicacao
    }
