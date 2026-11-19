import importlib
import os
import re
import sys
import zipfile
from datetime import datetime, time
from urllib.parse import quote

import folium
from folium import plugins
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# Importa e recarrega dinamicamente o motor de rotas e assistente
sys.path.append(os.path.dirname(__file__))
import motor_rotas
importlib.reload(motor_rotas)
from motor_rotas import (
    TARIFA_CIRCULAR_VILA_BH,
    TARIFA_CONVENCIONAL_BH,
    buscar_coordenadas_endereco,
    buscar_rotas_com_transbordo,
    calcular_caminhada_e_saude,
    calcular_tarifa_bhbus,
    filtrar_e_ordenar_rotas,
    verificar_estacao_metro_proxima,
)

import motor_ia_ml
importlib.reload(motor_ia_ml)
from motor_ia_ml import (
    avaliar_desempenho_modelos,
    obter_importancia_features,
    prever_impacto_transito_e_lotacao,
)

import servico_clima
importlib.reload(servico_clima)
from servico_clima import LAT_BH, LON_BH, obter_clima_atual_bh

import assistente_ia
importlib.reload(assistente_ia)
from assistente_ia import processar_consulta_assistente_ia

import analytics_estatistico
importlib.reload(analytics_estatistico)
from analytics_estatistico import (
    calcular_defasagem_spatiotemporal_clima,
    calcular_indice_confiabilidade_tri,
    executar_teste_hipotese_atraso,
    gerar_citacao_bibtex,
)

ACCENT = "#f59e0b"

# --- CONFIGURAÇÃO DA PÁGINA STREAMLIT ---
st.set_page_config(
    page_title="BH Transit AI — Mobilidade Urbana Inteligente",
    layout="wide",
    initial_sidebar_state="expanded",
)


def strip_leading_symbols(text: str) -> str:
    """Remove leading emojis/non-alphanumeric glyphs returned by legacy helpers."""
    cleaned = re.sub(r"^[^\w\(\[\+\-]+", "", str(text or "")).strip()
    return cleaned or str(text or "")


def render_kpi_strip(items, cols=5):
    """Render unified monochrome KPI strip with zero truncation and zero pill arrows."""
    cls_map = {2: "kpi-strip kpi-strip-2", 3: "kpi-strip kpi-strip-3", 4: "kpi-strip kpi-strip-4", 5: "kpi-strip kpi-strip-5"}
    strip_cls = cls_map.get(cols, "kpi-strip kpi-strip-4")
    cells = []
    for label, value, meta, tone in items:
        num_cls = f"kpi-num {tone}".strip()
        cells.append(
            f'<div class="kpi-cell">'
            f'<div class="kpi-lbl">{label}</div>'
            f'<div class="{num_cls}">{value}</div>'
            f'<div class="kpi-meta">{meta}</div>'
            f'</div>'
        )
    st.markdown(f'<div class="{strip_cls}">{"".join(cells)}</div>', unsafe_allow_html=True)


# --- INICIALIZAÇÃO DE NAVEGAÇÃO E FAVORITOS EM SESSION STATE ---
if "favoritos" not in st.session_state:
    st.session_state["favoritos"] = [
        {"origem": "BH Shopping, Belo Horizonte", "destino": "Mercado Central, Belo Horizonte", "rotulo": "BH Shopping -> Mercado Central"},
        {"origem": "Savassi, Belo Horizonte", "destino": "Mineirão, Belo Horizonte", "rotulo": "Savassi -> Mineirão"},
        {"origem": "Estação Barreiro, Belo Horizonte", "destino": "Praça Sete, Belo Horizonte", "rotulo": "Estação Barreiro -> Praça Sete"},
    ]

# --- MONOCHROME STRUCTURAL ENGINEERING DESIGN SYSTEM (AMBER DOMAIN ACCENT) ---
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    .main .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }}
    .stApp {{
        background-color: #0b0b0d !important;
        color: #f4f4f6 !important;
    }}
    header[data-testid="stHeader"] {{
        background-color: #0b0b0d !important;
        border-bottom: 1px solid #18181c;
    }}
    section[data-testid="stSidebar"] {{
        background-color: #121215 !important;
        border-right: 1px solid #242429;
    }}
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li {{
        color: #a1a1aa !important;
        font-size: 0.82rem;
    }}
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {{
        color: #f4f4f6 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600 !important;
    }}
    h1, h2, h3, h4, h5 {{
        color: #f4f4f6 !important;
        letter-spacing: -0.02em;
    }}

    /* Inputs, Selectboxes, Radio, Time Input */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input {{
        background-color: #0b0b0d !important;
        border-color: #323238 !important;
        color: #f4f4f6 !important;
        border-radius: 4px !important;
    }}
    [data-testid="stRadio"] label p,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stCheckbox"] label p {{
        color: #d4d4d8 !important;
        font-size: 0.84rem !important;
    }}

    /* Architectural Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        border-bottom: 1px solid #242429;
        background: transparent !important;
        flex-wrap: wrap;
    }}
    button[data-baseweb="tab"] {{
        color: #71717a !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.77rem !important;
        font-weight: 500 !important;
        padding: 10px 14px !important;
        background: transparent !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {ACCENT} !important;
    }}
    div[data-baseweb="tab-highlight"] {{
        background-color: {ACCENT} !important;
    }}

    /* Buttons */
    .stButton > button {{
        background-color: #121215 !important;
        color: #e4e4e7 !important;
        border: 1px solid #323238 !important;
        border-radius: 4px !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        padding: 0.5rem 0.9rem !important;
    }}
    .stButton > button:hover {{
        background-color: #18181c !important;
        border-color: {ACCENT} !important;
        color: #f4f4f6 !important;
    }}
    .stButton > button[kind="primary"] {{
        background-color: {ACCENT} !important;
        color: #0b0b0d !important;
        border: 1px solid {ACCENT} !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em;
    }}
    .stButton > button[kind="primary"]:hover {{
        background-color: #fbbf24 !important;
        border-color: #fbbf24 !important;
        color: #0b0b0d !important;
    }}

    /* Page Header */
    .page-header {{
        border-bottom: 1px solid #242429;
        padding-bottom: 14px;
        margin-bottom: 20px;
    }}
    .page-eyebrow {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #71717a;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }}
    .page-eyebrow span {{
        color: {ACCENT};
    }}
    .page-title {{
        margin: 4px 0 6px 0;
        font-size: 1.75rem;
        font-weight: 600;
        color: #f4f4f6;
        letter-spacing: -0.02em;
    }}
    .page-subtitle {{
        color: #a1a1aa;
        font-size: 0.9rem;
        margin: 0;
    }}

    /* Unified Monochrome KPI Strips */
    .kpi-strip {{
        display: grid;
        background: #121215;
        border: 1px solid #242429;
        border-radius: 6px;
        margin-bottom: 18px;
        overflow: hidden;
    }}
    .kpi-strip-2 {{ grid-template-columns: repeat(2, 1fr); }}
    .kpi-strip-3 {{ grid-template-columns: repeat(3, 1fr); }}
    .kpi-strip-4 {{ grid-template-columns: repeat(4, 1fr); }}
    .kpi-strip-5 {{ grid-template-columns: repeat(5, 1fr); }}
    @media (max-width: 960px) {{
        .kpi-strip-4, .kpi-strip-5 {{ grid-template-columns: repeat(2, 1fr); }}
    }}
    .kpi-cell {{
        padding: 14px 16px;
        border-right: 1px solid #242429;
    }}
    .kpi-cell:last-child {{ border-right: none; }}
    .kpi-lbl {{
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #71717a;
        font-weight: 500;
    }}
    .kpi-num {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.28rem;
        font-weight: 600;
        color: #f4f4f6;
        margin: 5px 0 3px 0;
        white-space: nowrap;
    }}
    .kpi-num.accent {{ color: {ACCENT}; }}
    .kpi-num.positive {{ color: #22c55e; }}
    .kpi-num.alert {{ color: #f43f5e; }}
    .kpi-meta {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.69rem;
        color: #a1a1aa;
    }}

    /* Structural Panels */
    .panel-card {{
        background: #121215;
        border: 1px solid #242429;
        border-radius: 6px;
        padding: 18px 20px;
        margin-bottom: 16px;
    }}
    .econ-card {{
        background: #121215;
        border: 1px solid #242429;
        border-radius: 6px;
        padding: 20px;
    }}
    .econ-card.optimal {{
        border-color: {ACCENT};
    }}
    .econ-tag {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #71717a;
        margin-bottom: 8px;
    }}
    .econ-card.optimal .econ-tag {{
        color: {ACCENT};
    }}
    .econ-val {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.85rem;
        font-weight: 600;
        color: #f4f4f6;
        margin: 4px 0 8px 0;
    }}
    .econ-desc {{
        font-size: 0.82rem;
        color: #a1a1aa;
        margin-bottom: 12px;
    }}
    .econ-delta {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        padding-top: 10px;
        border-top: 1px solid #242429;
    }}
    .sidebar-stat {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        background: #0b0b0d;
        border: 1px solid #242429;
        border-radius: 4px;
        margin: 8px 0 10px 0;
        overflow: hidden;
    }}
    .sidebar-stat > div {{
        padding: 10px;
        border-right: 1px solid #242429;
    }}
    .sidebar-stat > div:last-child {{ border-right: none; }}
    .sidebar-stat .s-lbl {{
        font-size: 0.65rem;
        text-transform: uppercase;
        color: #71717a;
        letter-spacing: 0.05em;
    }}
    .sidebar-stat .s-val {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.02rem;
        font-weight: 600;
        color: #f4f4f6;
        margin-top: 2px;
    }}
    .status-line {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #a1a1aa;
        padding: 8px 10px;
        background: #0b0b0d;
        border: 1px solid #242429;
        border-radius: 4px;
        margin-bottom: 8px;
    }}
    .status-line b {{
        color: #f4f4f6;
    }}
    .wa-btn {{
        display: inline-block;
        background: #121215;
        color: #22c55e !important;
        border: 1px solid #22c55e;
        padding: 10px 18px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 600;
        text-decoration: none;
    }}
    .wa-btn:hover {{
        background: #18181c;
    }}
    div[data-testid="stExpander"] {{
        background-color: #121215;
        border: 1px solid #242429;
        border-radius: 6px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- SIDEBAR BARRA LATERAL ---
with st.sidebar:
    st.markdown("### 01 / BH Transit AI")
    st.caption("Motor multimodal de roteamento GTFS, inferência de atraso ML e telemetria climática para Belo Horizonte.")

    st.markdown("---")
    st.markdown("### 02 / Clima em Tempo Real")

    if st.button("Sincronizar Open-Meteo", use_container_width=True, key="btn_sync_weather"):
        st.session_state["clima_bh_atual"] = obter_clima_atual_bh()
        st.rerun()

    clima_atual = st.session_state.get("clima_bh_atual")
    if clima_atual is None:
        clima_atual = obter_clima_atual_bh()
        st.session_state["clima_bh_atual"] = clima_atual

    clima_desc = strip_leading_symbols(clima_atual.get("descricao", "Tempo Bom"))
    temp_c = clima_atual.get("temperatura_c", 24.0)
    precip_mm = clima_atual.get("precipitacao_mm", 0.0)

    st.markdown(
        f"""
        <div class="sidebar-stat">
            <div><div class="s-lbl">Temperatura</div><div class="s-val">{temp_c:.1f} °C</div></div>
            <div><div class="s-lbl">Chuva</div><div class="s-val">{precip_mm:.1f} mm/h</div></div>
        </div>
        <div class="status-line"><b>Condição:</b> {clima_desc}</div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Fonte: {clima_atual.get('fonte', 'Open-Meteo API')}")

    pista_txt = strip_leading_symbols(clima_atual.get("alerta", "Chuva detectada")) if precip_mm > 0.5 else "Pistas secas na capital"
    st.markdown(
        f"""
        <div class="status-line"><b>Via:</b> {pista_txt}</div>
        <div class="status-line"><b>GTFS BH:</b> Ativo (BHBUS & ÓTIMO)</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 03 / Stack & Autoria")
    st.caption("Geovana Moreira • Belo Horizonte, MG")
    st.caption("Python • GTFS • Scikit-Learn • Folium • FastAPI")


# --- CAMINHO DOS DADOS ---
GTFS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "gtfs_bh_convencional.zip")


@st.cache_data
def carregar_dados_gtfs():
    """Carrega os dados GTFS indexados em memória com carregamento instantâneo."""
    cache = motor_rotas.obter_cache_gtfs()
    if cache is not None:
        return (
            cache["routes"],
            cache["stops"],
            cache["trips"],
            cache["shapes"],
            cache["stop_times_rep"],
            cache["calendar"],
            cache["calendar_dates"],
        )

    with zipfile.ZipFile(GTFS_PATH, "r") as z:
        routes = pd.read_csv(z.open("routes.txt"))
        stops = pd.read_csv(z.open("stops.txt"))
        trips = pd.read_csv(z.open("trips.txt"))
        shapes = pd.read_csv(z.open("shapes.txt")) if "shapes.txt" in z.namelist() else None
        calendar = pd.read_csv(z.open("calendar.txt")) if "calendar.txt" in z.namelist() else None
        calendar_dates = pd.read_csv(z.open("calendar_dates.txt")) if "calendar_dates.txt" in z.namelist() else None
        stop_times = None

    return routes, stops, trips, shapes, stop_times, calendar, calendar_dates


routes, stops, trips, shapes, stop_times, calendar, calendar_dates = carregar_dados_gtfs()

PRESETS_BH = {
    "BH Shopping (Belvedere)": (-19.9754, -43.9452),
    "Mercado Central": (-19.9208, -43.9419),
    "Mineirão (Pampulha)": (-19.8659, -43.9710),
    "Praça Sete (Centro)": (-19.9191, -43.9386),
    "Savassi (Praça Diogo de Vasconcelos)": (-19.9387, -43.9332),
    "Hospital Felício Rocho": (-19.9231, -43.9569),
    "Estação Barreiro": (-19.9745, -44.0205),
    "Estação Venda Nova": (-19.8023, -43.9573),
}


def obter_coordenadas_trajeto_rua(shape_id, fallback_p1_lat, fallback_p1_lon, fallback_p2_lat, fallback_p2_lon):
    """Retorna a sequência exata de coordenadas que acompanham a geometria de cada rua em BH."""
    if shapes is not None and pd.notna(shape_id):
        f_shape = shapes[shapes["shape_id"] == shape_id].sort_values("shape_pt_sequence")
        if not f_shape.empty:
            return f_shape[["shape_pt_lat", "shape_pt_lon"]].values.tolist()

    return [[fallback_p1_lat, fallback_p1_lon], [fallback_p2_lat, fallback_p2_lon]]


def obter_linhas_do_ponto(stop_id, stop_times_df, trips_df, routes_df):
    """Retorna lista das linhas de ônibus de BH que passam por um determinado ponto."""
    try:
        trips_no_ponto = stop_times_df[stop_times_df["stop_id"].astype(str) == str(stop_id)]["trip_id"].values
        if len(trips_no_ponto) == 0:
            return []
        route_ids_no_ponto = trips_df[trips_df["trip_id"].isin(trips_no_ponto)]["route_id"].unique()
        linhas_shorts = routes_df[routes_df["route_id"].isin(route_ids_no_ponto)]["route_short_name"].astype(str).unique()
        return sorted(list(set(linhas_shorts)))
    except Exception:
        return []


def criar_mapa_dark(location, zoom_start=13):
    """Cria mapa Folium dark sem API key usando Esri World Dark Gray Canvas + Reference."""
    mapa = folium.Map(location=location, zoom_start=zoom_start, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri — World Dark Gray Canvas",
        name="Dark Gray Canvas",
        max_zoom=19,
        max_native_zoom=16,
    ).add_to(mapa)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}",
        attr="Esri Reference",
        name="Rótulos",
        overlay=True,
        control=False,
        max_zoom=19,
        max_native_zoom=16,
    ).add_to(mapa)
    return mapa


def renderizar_mapa_rota(melhor_rota, o_lat, o_lon, d_lat, d_lon, key_prefix, stops_df=None, trips_df=None, stop_times_df=None, routes_df=None):
    """Desenha o mapa da rota em tema dark (Esri Dark Gray Canvas, sem API key) acompanhando as ruas reais de BH."""
    centro_mapa_lat = (o_lat + d_lat) / 2.0
    centro_mapa_lon = (o_lon + d_lon) / 2.0

    mapa = criar_mapa_dark(location=[centro_mapa_lat, centro_mapa_lon], zoom_start=13)

    # Marcador da Origem
    folium.CircleMarker(
        location=[o_lat, o_lon],
        radius=8,
        popup="<b>Origem</b>",
        tooltip="Origem",
        color="#22c55e",
        fill=True,
        fill_color="#22c55e",
        fill_opacity=0.95,
    ).add_to(mapa)

    # Marcador do Destino
    folium.CircleMarker(
        location=[d_lat, d_lon],
        radius=8,
        popup="<b>Destino</b>",
        tooltip="Destino",
        color="#f43f5e",
        fill=True,
        fill_color="#f43f5e",
        fill_opacity=0.95,
    ).add_to(mapa)

    # Embarque Inicial
    folium.CircleMarker(
        location=[melhor_rota["ponto_origem_lat"], melhor_rota["ponto_origem_lon"]],
        radius=8,
        popup=f"<b>Embarque Inicial:</b> {melhor_rota['ponto_origem_nome']}",
        tooltip=f"Embarque — Linha {melhor_rota.get('linha1_short', melhor_rota['route_short_name'])}",
        color=ACCENT,
        fill=True,
        fill_color="#0b0b0d",
        fill_opacity=1.0,
    ).add_to(mapa)

    # Caminhada inicial
    folium.PolyLine(
        locations=[[o_lat, o_lon], [melhor_rota["ponto_origem_lat"], melhor_rota["ponto_origem_lon"]]],
        color="#a1a1aa",
        weight=2.5,
        dash_array="5, 8",
        tooltip="Caminhada até o embarque",
    ).add_to(mapa)

    # Rota com transbordo (2 ônibus)
    if melhor_rota.get("transbordos", 0) == 1:
        l1 = melhor_rota.get("linha1_short") or melhor_rota.get("r1_short") or "Ônibus 1"
        l2 = melhor_rota.get("linha2_short") or melhor_rota.get("r2_short") or "Ônibus 2"
        s1 = melhor_rota.get("shape_id_1") or melhor_rota.get("shape_id")
        s2 = melhor_rota.get("shape_id_2") or melhor_rota.get("shape2_id")

        folium.CircleMarker(
            location=[melhor_rota["ponto_troca_lat"], melhor_rota["ponto_troca_lon"]],
            radius=10,
            popup=f"<b>Baldeação:</b> {melhor_rota.get('ponto_troca_nome', 'Ponto de Conexão')}<br><i>(Troca {l1} -> {l2})</i>",
            tooltip=f"Baldeação em {melhor_rota.get('ponto_troca_nome', 'Conexão')}",
            color="#f4f4f6",
            fill=True,
            fill_color=ACCENT,
            fill_opacity=1.0,
        ).add_to(mapa)

        coords_rua_1 = obter_coordenadas_trajeto_rua(
            s1,
            melhor_rota["ponto_origem_lat"],
            melhor_rota["ponto_origem_lon"],
            melhor_rota["ponto_troca_lat"],
            melhor_rota["ponto_troca_lon"],
        )
        folium.PolyLine(
            locations=coords_rua_1,
            color=ACCENT,
            weight=5,
            opacity=0.9,
            tooltip=f"1º Ônibus (Linha {l1})",
        ).add_to(mapa)

        coords_rua_2 = obter_coordenadas_trajeto_rua(
            s2,
            melhor_rota["ponto_troca_lat"],
            melhor_rota["ponto_troca_lon"],
            melhor_rota["ponto_destino_lat"],
            melhor_rota["ponto_destino_lon"],
        )
        folium.PolyLine(
            locations=coords_rua_2,
            color="#38bdf8",
            weight=5,
            opacity=0.9,
            tooltip=f"2º Ônibus (Linha {l2})",
        ).add_to(mapa)
    else:
        coords_rua_direta = obter_coordenadas_trajeto_rua(
            melhor_rota.get("shape_id"),
            melhor_rota["ponto_origem_lat"],
            melhor_rota["ponto_origem_lon"],
            melhor_rota["ponto_destino_lat"],
            melhor_rota["ponto_destino_lon"],
        )
        folium.PolyLine(
            locations=coords_rua_direta,
            color=ACCENT,
            weight=5,
            opacity=0.9,
            tooltip=f"Ônibus Direto (Linha {melhor_rota.get('route_short_name', '')})",
        ).add_to(mapa)

    # Paradas intermediárias
    if stops_df is not None:
        try:
            route_id_alvo = melhor_rota.get("route_id")
            cache = motor_rotas.obter_cache_gtfs()
            if cache and route_id_alvo:
                seqs = cache["mapa_rota_pontos"].get(route_id_alvo, [])
                ids_ignorar = {str(melhor_rota["ponto_origem_id"]), str(melhor_rota["ponto_destino_id"])}
                if melhor_rota.get("ponto_troca_id"):
                    ids_ignorar.add(str(melhor_rota["ponto_troca_id"]))

                count = 0
                for sid, _ in seqs:
                    if sid not in ids_ignorar and sid in cache["stops_dict"]:
                        s_info = cache["stops_dict"][sid]
                        linhas_passantes = motor_rotas.obter_linhas_do_ponto(sid)
                        texto_linhas = ", ".join(linhas_passantes[:6]) if linhas_passantes else "N/D"
                        if len(linhas_passantes) > 6:
                            texto_linhas += f" (+{len(linhas_passantes)-6} outras)"

                        folium.CircleMarker(
                            location=[s_info["lat"], s_info["lon"]],
                            radius=3.5,
                            popup=f"<b>Parada:</b> {s_info['name']}<br><b>ID:</b> {sid}<br><b>Linhas:</b> {texto_linhas}",
                            tooltip=f"{s_info['name']} (clique para ver linhas)",
                            color="#a1a1aa",
                            fill=True,
                            fill_color="#18181c",
                            fill_opacity=0.9,
                        ).add_to(mapa)
                        count += 1
                        if count >= 30:
                            break
        except Exception:
            pass

    # Desembarque Final
    folium.CircleMarker(
        location=[melhor_rota["ponto_destino_lat"], melhor_rota["ponto_destino_lon"]],
        radius=8,
        popup=f"<b>Desembarque:</b> {melhor_rota['ponto_destino_nome']}",
        tooltip="Desembarque Final",
        color="#38bdf8",
        fill=True,
        fill_color="#0b0b0d",
        fill_opacity=1.0,
    ).add_to(mapa)

    # Caminhada final
    folium.PolyLine(
        locations=[[melhor_rota["ponto_destino_lat"], melhor_rota["ponto_destino_lon"]], [d_lat, d_lon]],
        color="#a1a1aa",
        weight=2.5,
        dash_array="5, 8",
        tooltip="Caminhada até o destino",
    ).add_to(mapa)

    st_folium(mapa, width="100%", height=500, key=f"mapa_ruas_renderizado_{key_prefix}")


# --- ARCHITECTURAL HEADER ---
st.markdown(
    """
    <div class="page-header">
        <div class="page-eyebrow"><span>BH-GTFS // RMBH</span> // ROTEAMENTO MULTIMODAL, ML DE ATRASO & TELEMETRIA URBANA</div>
        <h1 class="page-title">BH Transit AI — Mobilidade Urbana Inteligente</h1>
        <p class="page-subtitle">Roteamento rua a rua em Belo Horizonte com integração GTFS, previsão de atraso/lotação via Random Forest, clima Open-Meteo e análise estatística.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- NAVEGAÇÃO POR ABAS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "01 / Planejador",
        "02 / Assistente IA",
        "03 / Heatmap Urbano",
        "04 / Economia",
        "05 / Mapa Interativo",
        "06 / Research & MLOps",
    ]
)

# ==============================================================================
# ABA 1: PLANEJADOR DE ROTAS COM CLIMA, FILTROS, FAVORITOS E WHATSAPP/QR CODE
# ==============================================================================
with tab1:
    st.subheader("Planejador de Itinerários Curva a Curva (GTFS BH)")

    if st.session_state["favoritos"]:
        st.caption("Atalhos Rápidos & Favoritos")
        cols_fav = st.columns(len(st.session_state["favoritos"]))
        for i, fav in enumerate(st.session_state["favoritos"]):
            with cols_fav[i]:
                if st.button(fav["rotulo"], key=f"btn_fav_{i}", use_container_width=True):
                    st.session_state["p1_orig"] = fav["origem"]
                    st.session_state["p1_dest"] = fav["destino"]

    col_orig, col_dest = st.columns(2)
    with col_orig:
        orig_txt_tab1 = st.text_input(
            "Origem (Endereço ou Local de BH):",
            value=st.session_state.get("p1_orig", "BH Shopping, Belo Horizonte"),
            key="input_orig_tab1",
        )
    with col_dest:
        dest_txt_tab1 = st.text_input(
            "Destino (Endereço ou Local de BH):",
            value=st.session_state.get("p1_dest", "Mercado Central, Belo Horizonte"),
            key="input_dest_tab1",
        )

    c_h1, c_h2, c_h3, c_h4 = st.columns(4)
    with c_h1:
        horario_tab1 = st.time_input("Horário de Saída:", value=time(8, 0), key="horario_tab1")
        horario_str_tab1 = horario_tab1.strftime("%H:%M")

    with c_h2:
        tipo_dia_tab1 = st.selectbox(
            "Quadro de Horários:",
            options=["Dia Útil (Segunda a Sexta)", "Sábado", "Domingo / Feriado"],
            index=0,
            key="tipo_dia_tab1",
        )

    with c_h3:
        usar_clima_tempo_real = st.checkbox("Clima em Tempo Real BH", value=True, key="chk_clima_rt")
        if usar_clima_tempo_real and clima_atual:
            clima_tab1 = clima_desc
            st.caption(f"Precipitação atual: `{precip_mm:.1f} mm/h`")
            precip_mm_passar = precip_mm
        else:
            clima_tab1 = st.selectbox(
                "Condição Climatológica:",
                options=["Ensolarado / Tempo Bom", "Chuva Moderada", "Chuva Forte / Temporal"],
                index=0,
                key="clima_tab1",
            )
            precip_mm_passar = None

    with c_h4:
        filtro_tab1 = st.selectbox(
            "Preferência de Rota:",
            options=["Mais Rápida", "Menos Caminhada", "Linhas Diretas", "Acessível (PCD)"],
            index=0,
            key="filtro_tab1",
        )

    with st.expander("Configurar Tarifa Customizada / Ônibus Metropolitano (ex: RMBH R$ 8,45)"):
        c_tar1, c_tar2 = st.columns([1, 2])
        with c_tar1:
            usar_tarifa_custom_tab1 = st.checkbox("Usar Tarifa Metropolitana / Personalizada", value=False, key="chk_custom_tab1")
        with c_tar2:
            val_tarifa_custom_tab1 = st.number_input("Valor da Passagem Individual (R$):", value=8.45, min_value=1.0, max_value=50.0, step=0.05, key="num_custom_tab1")

    col_b1, col_b2 = st.columns([3, 1])
    with col_b1:
        btn_buscar_tab1 = st.button("CALCULAR MELHOR ROTA (GTFS + ML)", type="primary", use_container_width=True, key="btn_buscar_tab1")
    with col_b2:
        if st.button("Salvar Favorito", use_container_width=True, key="btn_add_fav"):
            st.session_state["favoritos"].append(
                {
                    "origem": orig_txt_tab1,
                    "destino": dest_txt_tab1,
                    "rotulo": f"{orig_txt_tab1.split(',')[0]} -> {dest_txt_tab1.split(',')[0]}",
                }
            )
            st.success("Rota salva nos favoritos.")

    if btn_buscar_tab1 or st.session_state.get("busca_tab1_realizada", False):
        if btn_buscar_tab1:
            st.session_state["busca_tab1_realizada"] = True
            st.session_state["tab1_orig"] = orig_txt_tab1
            st.session_state["tab1_dest"] = dest_txt_tab1

        orig_processar = st.session_state.get("tab1_orig", orig_txt_tab1)
        dest_processar = st.session_state.get("tab1_dest", dest_txt_tab1)

        with st.spinner("Geocodificando endereços e executando motor GTFS + Random Forest..."):
            lat_o, lon_o, end_o = buscar_coordenadas_endereco(orig_processar)
            lat_d, lon_d, end_d = buscar_coordenadas_endereco(dest_processar)

            if lat_o is None or lat_d is None:
                st.error("Não foi possível localizar um dos endereços informados em Belo Horizonte.")
            else:
                rotas_brutas = buscar_rotas_com_transbordo(
                    lat_o,
                    lon_o,
                    lat_d,
                    lon_d,
                    routes,
                    stops,
                    trips,
                    stop_times,
                    top_k_pontos=20,
                    horario_partida=horario_str_tab1,
                    calendar_df=calendar,
                    calendar_dates_df=calendar_dates,
                    tipo_dia=tipo_dia_tab1,
                )

                if not rotas_brutas:
                    st.warning(f"Nenhuma rota encontrada para {horario_str_tab1} ({tipo_dia_tab1}).")
                else:
                    rotas_filtradas = filtrar_e_ordenar_rotas(rotas_brutas, filtro_tab1)
                    melhor_rota = rotas_filtradas[0]

                    pred_ml = prever_impacto_transito_e_lotacao(
                        horario_str=horario_str_tab1,
                        tipo_dia_str=tipo_dia_tab1,
                        tempo_viagem_teorico_min=melhor_rota["tempo_total_estimado_min"],
                        condicao_climatica=clima_tab1,
                        precipitacao_mm=precip_mm_passar,
                    )

                    tarifa_val_passar = val_tarifa_custom_tab1 if usar_tarifa_custom_tab1 else None
                    dados_tarifa = calcular_tarifa_bhbus(melhor_rota, tarifa_customizada=tarifa_val_passar)
                    dados_saude = calcular_caminhada_e_saude(melhor_rota)
                    metro_orig = verificar_estacao_metro_proxima(lat_o, lon_o)
                    metro_dest = verificar_estacao_metro_proxima(lat_d, lon_d)

                    lotacao_clean = strip_leading_symbols(pred_ml["lotacao_texto"])
                    atraso_meta = f"+{pred_ml['atraso_min']} min trânsito/clima" if pred_ml["atraso_min"] > 0 else "Fluxo normal"
                    tarifa_meta = "50% desc. integração" if melhor_rota.get("transbordos", 0) > 0 else "Tarifa única"

                    st.markdown("---")
                    render_kpi_strip(
                        [
                            ("Linha Recomendada", str(melhor_rota["route_short_name"]), "Rede GTFS BH", "accent"),
                            ("Tempo Previsto (ML)", f"~{pred_ml['tempo_ajustado_min']} min", atraso_meta, ""),
                            ("Tarifa Estimada", dados_tarifa["tarifa_total_str"], tarifa_meta, ""),
                            ("Lotação Prevista", lotacao_clean.split("(")[0].strip(), "Classificador ML", ""),
                            ("Caminhada Total", f"{dados_saude['total_m']} m", f"~{dados_saude['calorias_kcal']} kcal", ""),
                        ],
                        cols=5,
                    )

                    st.markdown(
                        f'<div class="panel-card"><div style="font-family:JetBrains Mono,monospace;font-size:0.72rem;color:{ACCENT};text-transform:uppercase;margin-bottom:4px;">Inferência de Trânsito & Clima</div><div style="color:#d4d4d8;font-size:0.88rem;">{strip_leading_symbols(pred_ml["explicacao"])}</div></div>',
                        unsafe_allow_html=True,
                    )

                    if metro_orig or metro_dest:
                        metro_parts = []
                        if metro_orig:
                            metro_parts.append(f"Origem próxima à <b>{metro_orig['nome']}</b> ({metro_orig['distancia_m']} m)")
                        if metro_dest:
                            metro_parts.append(f"Destino próximo à <b>{metro_dest['nome']}</b> ({metro_dest['distancia_m']} m)")
                        st.markdown(
                            f'<div class="panel-card" style="border-color:#22c55e;"><div style="font-family:JetBrains Mono,monospace;font-size:0.72rem;color:#22c55e;text-transform:uppercase;margin-bottom:4px;">Integração Metrô BH (Linha 1)</div><div style="color:#d4d4d8;font-size:0.88rem;">{" • ".join(metro_parts)}</div></div>',
                            unsafe_allow_html=True,
                        )

                    txt_wa = f"BH Transit AI: Rota de {orig_processar.split(',')[0]} ate {dest_processar.split(',')[0]} com a Linha {melhor_rota['route_short_name']}. Tempo est: {pred_ml['tempo_ajustado_min']} min. Tarifa: {dados_tarifa['tarifa_total_str']}."
                    url_wa = f"https://api.whatsapp.com/send?text={quote(txt_wa)}"

                    c_wa, c_qr = st.columns([3, 1])
                    with c_wa:
                        st.markdown("##### Compartilhar Itinerário")
                        st.caption("Envie o resumo da rota diretamente para o celular via WhatsApp ou QR Code.")
                        st.markdown(f'<a href="{url_wa}" target="_blank" class="wa-btn">Abrir no WhatsApp -></a>', unsafe_allow_html=True)
                    with c_qr:
                        qr_url = f"https://api.qrserver.com/v1/create-qr-code/?size=130x130&data={quote(url_wa)}"
                        st.image(qr_url, caption="Escanear no celular", width=115)

                    st.markdown("---")
                    st.subheader(f"Trajeto nas Ruas — Linha {melhor_rota['route_short_name']}")
                    renderizar_mapa_rota(melhor_rota, lat_o, lon_o, lat_d, lon_d, "tab1_mapa", stops, trips, stop_times, routes)


# ==============================================================================
# ABA 2: ASSISTENTE IA CONVERSACIONAL (CHATBOT)
# ==============================================================================
with tab2:
    st.subheader("Assistente Conversacional de Mobilidade (NLP + ML)")
    st.caption("Faça perguntas em linguagem natural. O motor extrai origem, destino, horário e ajusta a predição para trânsito e chuva.")

    if "historico_chat" not in st.session_state:
        st.session_state["historico_chat"] = []

    for idx_msg, msg in enumerate(st.session_state["historico_chat"]):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "mapa_dados" in msg and msg["mapa_dados"]:
                md = msg["mapa_dados"]
                renderizar_mapa_rota(md["rota"], md["lat_o"], md["lon_o"], md["lat_d"], md["lon_d"], f"chat_hist_{idx_msg}", stops, trips, stop_times, routes)

    prompt_usuario = st.chat_input("Ex: Estou na Savassi e preciso ir pro Mineirão na chuva às 18h...")

    if prompt_usuario:
        st.session_state["historico_chat"].append({"role": "user", "content": prompt_usuario})
        with st.chat_message("user"):
            st.markdown(prompt_usuario)

        with st.chat_message("assistant"):
            with st.spinner("Processando NLP e executando modelos de ML de BH..."):
                resp_texto, mapa_dados, _ = processar_consulta_assistente_ia(
                    prompt_usuario, routes, stops, trips, stop_times, calendar, calendar_dates
                )
                st.markdown(resp_texto)
                if mapa_dados:
                    renderizar_mapa_rota(
                        mapa_dados["rota"],
                        mapa_dados["lat_o"],
                        mapa_dados["lon_o"],
                        mapa_dados["lat_d"],
                        mapa_dados["lon_d"],
                        f"chat_new_{len(st.session_state['historico_chat'])}",
                        stops,
                        trips,
                        stop_times,
                        routes,
                    )
                st.session_state["historico_chat"].append(
                    {"role": "assistant", "content": resp_texto, "mapa_dados": mapa_dados}
                )


# ==============================================================================
# ABA 3: ANALYTICS URBANO & MAPA DE CALOR (HEATMAP DE BH)
# ==============================================================================
with tab3:
    st.subheader("Analytics Urbano & Densidade da Malha GTFS")
    st.caption("Distribuição espacial das paradas catalogadas em Belo Horizonte e sobreposição com a Linha 1 do Metrô.")

    render_kpi_strip(
        [
            ("Paradas Catalogadas", f"{len(stops):,}".replace(",", "."), "stops.txt GTFS BH", "accent"),
            ("Linhas Operacionais", f"{len(routes):,}".replace(",", "."), "routes.txt ativas", ""),
            ("Viagens Mapeadas", f"{len(trips):,}".replace(",", "."), "trips.txt programadas", ""),
            ("Estações Metrô", f"{len(motor_rotas.ESTACOES_METRO_BH)}", "Linha 1 (Eldorado–Vilarinho)", "positive"),
        ],
        cols=4,
    )

    col_h_opt1, col_h_opt2 = st.columns([2, 2])
    with col_h_opt1:
        raio_heat = st.slider("Raio de Intensidade do Calor (px):", min_value=8, max_value=30, value=16, key="raio_heat_slider")
    with col_h_opt2:
        amostra_qtd = st.slider("Amostragem de Pontos de BH:", min_value=500, max_value=len(stops), value=min(4000, len(stops)), key="amostra_heat_slider")

    stops_validos = stops.dropna(subset=["stop_lat", "stop_lon"])
    sample_stops = stops_validos.sample(n=min(amostra_qtd, len(stops_validos)), random_state=42)
    heatmap_data = [[float(row["stop_lat"]), float(row["stop_lon"]), 1.0] for _, row in sample_stops.iterrows()]

    mapa_heat = criar_mapa_dark(location=[-19.9167, -43.9345], zoom_start=12)

    plugins.HeatMap(
        heatmap_data,
        radius=raio_heat,
        blur=14,
        max_zoom=14,
        min_opacity=0.28,
        gradient={0.2: "#1e293b", 0.45: "#b45309", 0.72: "#f59e0b", 1.0: "#f43f5e"},
    ).add_to(mapa_heat)

    for est in motor_rotas.ESTACOES_METRO_BH:
        folium.CircleMarker(
            location=[est["lat"], est["lon"]],
            radius=6,
            popup=f"<b>Metrô BH — {est['nome']}</b>",
            tooltip=f"Metrô: {est['nome']}",
            color="#22c55e",
            fill=True,
            fill_color="#0b0b0d",
            fill_opacity=1.0,
        ).add_to(mapa_heat)

    st_folium(mapa_heat, width="100%", height=540, key="heatmap_bh_render_v2", returned_objects=[])


# ==============================================================================
# ABA 4: CALCULADORA DE ECONOMIA
# ==============================================================================
with tab4:
    st.subheader("Simulação de Gastos & Economia Financeira")
    st.caption("Compare o custo mensal real de deslocamento em Transporte Público (BHBUS / ÓTIMO) contra Carro Próprio e Aplicativos.")

    modo_calculo = st.radio(
        "Tipo de deslocamento:",
        options=[
            "Baldeação Mista (1 Metropolitano R$ 8,45 + 1 Municipal R$ 6,25)",
            "Custo Total Somado por Trajeto (Soma da Ida)",
            "Linha Única (Sem Baldeação)",
        ],
        index=0,
        horizontal=False,
    )

    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        dias_trabalho = st.slider("Dias de deslocamento por mês (Ida e Volta):", min_value=1, max_value=30, value=22)

        if "Baldeação Mista" in modo_calculo:
            val_bus1 = st.number_input("1º Ônibus (ex: Metropolitano ÓTIMO R$):", value=8.45, min_value=1.0, max_value=50.0, step=0.05, format="%.2f")
            val_bus2 = st.number_input("2º Ônibus (ex: Municipal BHBUS R$):", value=6.25, min_value=1.0, max_value=50.0, step=0.05, format="%.2f")
            pct_desc = st.slider("Desconto de Integração no 2º ônibus (%):", min_value=0, max_value=100, value=50, step=5)
            custo_ida = val_bus1 + (val_bus2 * (100 - pct_desc) / 100.0)
            custo_diario_bus = custo_ida * 2.0
        elif "Custo Total Somado" in modo_calculo:
            custo_ida = st.number_input("Valor Total Gasto em UMA IDA (R$):", value=14.70, min_value=1.0, max_value=100.0, step=0.10, format="%.2f")
            custo_diario_bus = custo_ida * 2.0
        else:
            val_unico = st.number_input("Tarifa da Passagem (R$):", value=8.45, min_value=1.0, max_value=50.0, step=0.05, format="%.2f")
            custo_ida = val_unico
            custo_diario_bus = custo_ida * 2.0

    with col_sim2:
        dist_media_km = st.slider("Distância média do trajeto (km):", min_value=2.0, max_value=60.0, value=18.0)
        valor_estacionamento_diario = st.number_input("Estacionamento diário em BH/Região (R$):", value=15.0)

    custo_mensal_bus = custo_diario_bus * dias_trabalho
    custo_diario_uber = (8.00 + (dist_media_km * 2.20)) * 2
    custo_mensal_uber = custo_diario_uber * dias_trabalho
    custo_diario_carro = ((dist_media_km * 2 / 9.5) * 5.80) + valor_estacionamento_diario
    custo_mensal_carro = custo_diario_carro * dias_trabalho

    st.markdown("---")
    st.markdown("##### Comparativo Financeiro Mensal")

    def brl(v: float) -> str:
        return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    col_res1, col_res2, col_res3 = st.columns(3)

    with col_res1:
        st.markdown(
            f"""
            <div class="econ-card optimal">
                <div class="econ-tag">01 / Transporte Público • Base Otimizada</div>
                <div class="econ-val">{brl(custo_mensal_bus)}</div>
                <div class="econ-desc">{brl(custo_ida)} por ida • {dias_trabalho} dias úteis/mês</div>
                <div class="econ-delta" style="color:{ACCENT};">Referência de menor custo operacional</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_res2:
        diff_carro = custo_mensal_carro - custo_mensal_bus
        txt_diff_carro = f"+{brl(abs(diff_carro))} / mês vs ônibus" if diff_carro > 0 else f"-{brl(abs(diff_carro))} / mês economia"
        cor_diff_carro = "#f43f5e" if diff_carro > 0 else "#22c55e"
        st.markdown(
            f"""
            <div class="econ-card">
                <div class="econ-tag">02 / Carro Próprio</div>
                <div class="econ-val">{brl(custo_mensal_carro)}</div>
                <div class="econ-desc">Combustível (9,5 km/L) + Estacionamento ({brl(valor_estacionamento_diario)}/dia)</div>
                <div class="econ-delta" style="color:{cor_diff_carro};">{txt_diff_carro}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_res3:
        diff_uber = custo_mensal_uber - custo_mensal_bus
        txt_diff_uber = f"+{brl(abs(diff_uber))} / mês vs ônibus" if diff_uber > 0 else f"-{brl(abs(diff_uber))} / mês economia"
        cor_diff_uber = "#f43f5e" if diff_uber > 0 else "#22c55e"
        st.markdown(
            f"""
            <div class="econ-card">
                <div class="econ-tag">03 / Aplicativos (Uber / 99)</div>
                <div class="econ-val">{brl(custo_mensal_uber)}</div>
                <div class="econ-desc">Estimativa média de tarifa dinâmica em Belo Horizonte</div>
                <div class="econ-delta" style="color:{cor_diff_uber};">{txt_diff_uber}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    co2_mensal_poupado = round((dist_media_km * 2 * dias_trabalho * 0.095), 1)
    arvores_equiv = round(co2_mensal_poupado / 15.0, 1)

    st.markdown(
        f"""
        <div class="panel-card" style="margin-top:16px; border-color:#22c55e;">
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#22c55e; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:4px;">
                Impacto Ambiental (ESG)
            </div>
            <div style="color:#d4d4d8; font-size:0.88rem;">
                Optar pelo transporte coletivo nesta rotina evita a emissão de <b>~{co2_mensal_poupado} kg de CO₂/mês</b> (equivalente à absorção anual de <b>~{arvores_equiv} árvores</b>).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# ABA 5: CLIQUE DIRETO NO MAPA (FLUXO UNIFICADO)
# ==============================================================================
with tab5:
    st.subheader("Seleção Direta no Mapa (Origem -> Destino)")
    st.caption("Clique diretamente nas vias de Belo Horizonte para definir origem e destino e traçar o itinerário automaticamente.")

    if "clique_orig_lat" not in st.session_state:
        st.session_state["clique_orig_lat"] = None
        st.session_state["clique_orig_lon"] = None
    if "clique_dest_lat" not in st.session_state:
        st.session_state["clique_dest_lat"] = None
        st.session_state["clique_dest_lon"] = None
    if "clique_rotas_salvas" not in st.session_state:
        st.session_state["clique_rotas_salvas"] = None

    c_status1, c_status2 = st.columns([3, 1])
    with c_status1:
        if st.session_state["clique_orig_lat"] is None:
            st.markdown('<div class="status-line"><b>Passo 01:</b> Clique no mapa para marcar o ponto de <b>Origem</b>.</div>', unsafe_allow_html=True)
        elif st.session_state["clique_dest_lat"] is None:
            st.markdown(f'<div class="status-line" style="border-color:{ACCENT};"><b>Passo 02:</b> Origem definida. Agora clique no mapa para marcar o <b>Destino</b>.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-line" style="border-color:#22c55e;"><b>Rota Pronta:</b> Origem e destino definidos no mapa abaixo.</div>', unsafe_allow_html=True)

    with c_status2:
        if st.button("Limpar Pontos", use_container_width=True, key="btn_limpar_clique"):
            st.session_state["clique_orig_lat"] = None
            st.session_state["clique_orig_lon"] = None
            st.session_state["clique_dest_lat"] = None
            st.session_state["clique_dest_lon"] = None
            st.session_state["clique_rotas_salvas"] = None
            st.rerun()

    opcoes_modo = ["Marcar Origem", "Marcar Destino"]
    idx_modo_padrao = 1 if (st.session_state["clique_orig_lat"] is not None and st.session_state["clique_dest_lat"] is None) else 0

    col_m1, col_m2 = st.columns([2, 2])
    with col_m1:
        modo_clique = st.radio(
            "Modo do próximo clique:",
            options=opcoes_modo,
            index=idx_modo_padrao,
            horizontal=True,
            key="radio_modo_clique",
        )
    with col_m2:
        c_p1 = f"{st.session_state['clique_orig_lat']:.4f}, {st.session_state['clique_orig_lon']:.4f}" if st.session_state["clique_orig_lat"] else "Não definida"
        c_p2 = f"{st.session_state['clique_dest_lat']:.4f}, {st.session_state['clique_dest_lon']:.4f}" if st.session_state["clique_dest_lat"] else "Não definida"
        st.caption(f"Origem: `{c_p1}` | Destino: `{c_p2}`")

    if st.session_state["clique_orig_lat"] and st.session_state["clique_dest_lat"]:
        lat_mapa = (st.session_state["clique_orig_lat"] + st.session_state["clique_dest_lat"]) / 2.0
        lon_mapa = (st.session_state["clique_orig_lon"] + st.session_state["clique_dest_lon"]) / 2.0
        zoom_mapa = 13
    elif st.session_state["clique_orig_lat"]:
        lat_mapa = st.session_state["clique_orig_lat"]
        lon_mapa = st.session_state["clique_orig_lon"]
        zoom_mapa = 14
    else:
        lat_mapa = -19.9167
        lon_mapa = -43.9345
        zoom_mapa = 13

    mapa_unificado = criar_mapa_dark(location=[lat_mapa, lon_mapa], zoom_start=zoom_mapa)

    if st.session_state["clique_orig_lat"]:
        folium.CircleMarker(
            [st.session_state["clique_orig_lat"], st.session_state["clique_orig_lon"]],
            radius=8,
            popup="<b>Origem Marcada</b>",
            tooltip="Origem",
            color="#22c55e",
            fill=True,
            fill_color="#22c55e",
            fill_opacity=0.95,
        ).add_to(mapa_unificado)

    if st.session_state["clique_dest_lat"]:
        folium.CircleMarker(
            [st.session_state["clique_dest_lat"], st.session_state["clique_dest_lon"]],
            radius=8,
            popup="<b>Destino Marcado</b>",
            tooltip="Destino",
            color="#f43f5e",
            fill=True,
            fill_color="#f43f5e",
            fill_opacity=0.95,
        ).add_to(mapa_unificado)

    rotas_exibir = st.session_state.get("clique_rotas_salvas")
    if rotas_exibir:
        melhor_r = rotas_exibir[0]
        o_lat = st.session_state["clique_orig_lat"]
        o_lon = st.session_state["clique_orig_lon"]
        d_lat = st.session_state["clique_dest_lat"]
        d_lon = st.session_state["clique_dest_lon"]

        folium.CircleMarker(
            location=[melhor_r["ponto_origem_lat"], melhor_r["ponto_origem_lon"]],
            radius=8,
            popup=f"<b>Embarque:</b> {melhor_r['ponto_origem_nome']}",
            tooltip=f"Embarque — Linha {melhor_r['route_short_name']}",
            color=ACCENT,
            fill=True,
            fill_color="#0b0b0d",
            fill_opacity=1.0,
        ).add_to(mapa_unificado)

        folium.PolyLine(
            locations=[[o_lat, o_lon], [melhor_r["ponto_origem_lat"], melhor_r["ponto_origem_lon"]]],
            color="#a1a1aa",
            weight=2.5,
            dash_array="5, 8",
            tooltip="Caminhada até o embarque",
        ).add_to(mapa_unificado)

        if melhor_r.get("transbordos", 0) == 1:
            l1 = melhor_r.get("linha1_short", "Ônibus 1")
            l2 = melhor_r.get("linha2_short", "Ônibus 2")
            s1 = melhor_r.get("shape_id_1") or melhor_r.get("shape_id")
            s2 = melhor_r.get("shape_id_2") or melhor_r.get("shape2_id")

            folium.CircleMarker(
                location=[melhor_r["ponto_troca_lat"], melhor_r["ponto_troca_lon"]],
                radius=9,
                popup=f"<b>Baldeação:</b> {melhor_r['ponto_troca_nome']}",
                tooltip=f"Baldeação ({l1} -> {l2})",
                color="#f4f4f6",
                fill=True,
                fill_color=ACCENT,
                fill_opacity=1.0,
            ).add_to(mapa_unificado)

            coords_1 = obter_coordenadas_trajeto_rua(s1, melhor_r["ponto_origem_lat"], melhor_r["ponto_origem_lon"], melhor_r["ponto_troca_lat"], melhor_r["ponto_troca_lon"])
            folium.PolyLine(locations=coords_1, color=ACCENT, weight=5, opacity=0.9, tooltip=f"1º Ônibus ({l1})").add_to(mapa_unificado)

            coords_2 = obter_coordenadas_trajeto_rua(s2, melhor_r["ponto_troca_lat"], melhor_r["ponto_troca_lon"], melhor_r["ponto_destino_lat"], melhor_r["ponto_destino_lon"])
            folium.PolyLine(locations=coords_2, color="#38bdf8", weight=5, opacity=0.9, tooltip=f"2º Ônibus ({l2})").add_to(mapa_unificado)
        else:
            coords_dir = obter_coordenadas_trajeto_rua(melhor_r.get("shape_id"), melhor_r["ponto_origem_lat"], melhor_r["ponto_origem_lon"], melhor_r["ponto_destino_lat"], melhor_r["ponto_destino_lon"])
            folium.PolyLine(locations=coords_dir, color=ACCENT, weight=5, opacity=0.9, tooltip=f"Ônibus Direto ({melhor_r['route_short_name']})").add_to(mapa_unificado)

        folium.CircleMarker(
            location=[melhor_r["ponto_destino_lat"], melhor_r["ponto_destino_lon"]],
            radius=8,
            popup=f"<b>Desembarque:</b> {melhor_r['ponto_destino_nome']}",
            tooltip="Desembarque Final",
            color="#38bdf8",
            fill=True,
            fill_color="#0b0b0d",
            fill_opacity=1.0,
        ).add_to(mapa_unificado)

        folium.PolyLine(
            locations=[[melhor_r["ponto_destino_lat"], melhor_r["ponto_destino_lon"]], [d_lat, d_lon]],
            color="#a1a1aa",
            weight=2.5,
            dash_array="5, 8",
            tooltip="Caminhada até o destino",
        ).add_to(mapa_unificado)

    mapa_click_data = st_folium(mapa_unificado, width="100%", height=480, key="mapa_clique_unificado_v3")

    if mapa_click_data and mapa_click_data.get("last_clicked"):
        lat_c = float(mapa_click_data["last_clicked"]["lat"])
        lon_c = float(mapa_click_data["last_clicked"]["lng"])

        if "Origem" in modo_clique or st.session_state["clique_orig_lat"] is None:
            if st.session_state["clique_orig_lat"] != lat_c or st.session_state["clique_orig_lon"] != lon_c:
                st.session_state["clique_orig_lat"] = lat_c
                st.session_state["clique_orig_lon"] = lon_c
                st.session_state["clique_rotas_salvas"] = None
                st.rerun()
        else:
            if st.session_state["clique_dest_lat"] != lat_c or st.session_state["clique_dest_lon"] != lon_c:
                st.session_state["clique_dest_lat"] = lat_c
                st.session_state["clique_dest_lon"] = lon_c
                with st.spinner("Calculando melhor itinerário..."):
                    rotas_calculadas = buscar_rotas_com_transbordo(
                        st.session_state["clique_orig_lat"],
                        st.session_state["clique_orig_lon"],
                        lat_c,
                        lon_c,
                        routes,
                        stops,
                        trips,
                        stop_times,
                        top_k_pontos=15,
                        horario_partida="08:00",
                        calendar_df=calendar,
                        calendar_dates_df=calendar_dates,
                        tipo_dia="Dia Útil (Segunda a Sexta)",
                    )
                    st.session_state["clique_rotas_salvas"] = rotas_calculadas
                st.rerun()

    if st.session_state["clique_orig_lat"] and st.session_state["clique_dest_lat"]:
        if st.button("RECALCULAR ROTA COM IA & CLIMA", type="primary", use_container_width=True, key="btn_recalc_aba5"):
            with st.spinner("Localizando rotas nas ruas de BH..."):
                rotas_calculadas = buscar_rotas_com_transbordo(
                    st.session_state["clique_orig_lat"],
                    st.session_state["clique_orig_lon"],
                    st.session_state["clique_dest_lat"],
                    st.session_state["clique_dest_lon"],
                    routes,
                    stops,
                    trips,
                    stop_times,
                    top_k_pontos=15,
                    horario_partida="08:00",
                    calendar_df=calendar,
                    calendar_dates_df=calendar_dates,
                    tipo_dia="Dia Útil (Segunda a Sexta)",
                )
                st.session_state["clique_rotas_salvas"] = rotas_calculadas
                st.rerun()

    if rotas_exibir:
        st.markdown("---")
        melhor_r = rotas_exibir[0]
        pred_ml_aba5 = prever_impacto_transito_e_lotacao(
            horario_str="08:00",
            tipo_dia_str="Dia Útil",
            tempo_viagem_teorico_min=melhor_r["tempo_total_estimado_min"],
            condicao_climatica=clima_desc,
            precipitacao_mm=precip_mm,
        )

        dados_tarifa_aba5 = calcular_tarifa_bhbus(melhor_r)
        dados_saude_aba5 = calcular_caminhada_e_saude(melhor_r)
        lotacao_aba5 = strip_leading_symbols(pred_ml_aba5["lotacao_texto"])

        render_kpi_strip(
            [
                ("Linha Recomendada", str(melhor_r["route_short_name"]), f"{len(rotas_exibir)} opções encontradas", "accent"),
                ("Tempo Previsto (ML)", f"~{pred_ml_aba5['tempo_ajustado_min']} min", f"+{pred_ml_aba5['atraso_min']} min atraso" if pred_ml_aba5["atraso_min"] > 0 else "Sem atrasos", ""),
                ("Tarifa Estimada", dados_tarifa_aba5["tarifa_total_str"], "Tarifa única" if melhor_r.get("transbordos", 0) == 0 else "Integração BHBUS", ""),
                ("Lotação Prevista", lotacao_aba5.split("(")[0].strip(), "Classificador ML", ""),
                ("Caminhada Total", f"{dados_saude_aba5['total_m']} m", f"~{dados_saude_aba5['calorias_kcal']} kcal", ""),
            ],
            cols=5,
        )

        st.markdown(
            f'<div class="panel-card"><div style="color:#d4d4d8;font-size:0.88rem;">{strip_leading_symbols(pred_ml_aba5["explicacao"])}</div></div>',
            unsafe_allow_html=True,
        )

        with st.expander("Instruções Passo a Passo do Itinerário", expanded=True):
            st.markdown(f"1. Caminhe **{melhor_r['dist_caminhada_origem_m']} m** até o ponto **{melhor_r['ponto_origem_nome']}**.")
            if melhor_r.get("transbordos", 0) == 1:
                st.markdown(f"2. Embarque no primeiro ônibus (**Linha {melhor_r.get('linha1_short', '1')}**).")
                st.markdown(f"3. Desembarque no ponto de baldeação: **{melhor_r['ponto_troca_nome']}**.")
                st.markdown(f"4. Embarque no segundo ônibus (**Linha {melhor_r.get('linha2_short', '2')}**).")
                st.markdown(f"5. Desembarque no ponto **{melhor_r['ponto_destino_nome']}**.")
                st.markdown(f"6. Caminhe **{melhor_r['dist_caminhada_destino_m']} m** até o seu destino final.")
            else:
                st.markdown(f"2. Embarque no ônibus direto **Linha {melhor_r['route_short_name']}** ({melhor_r['route_long_name']}).")
                st.markdown(f"3. Desembarque no ponto **{melhor_r['ponto_destino_nome']}**.")
                st.markdown(f"4. Caminhe **{melhor_r['dist_caminhada_destino_m']} m** até o seu destino final.")
    elif st.session_state["clique_orig_lat"] and st.session_state["clique_dest_lat"]:
        st.warning("Nenhuma linha direta ou com 1 baldeação encontrada para essas coordenadas. Tente clicar próximo a corredores principais.")


# ==============================================================================
# ABA 6: ACADEMIC & RESEARCH ANALYTICS
# ==============================================================================
with tab6:
    st.subheader("Research Analytics, Hypothesis Testing & MLOps")
    st.caption("Mathematical formulation, Student t-test verification, spatio-temporal lag cross-correlation, XAI feature importance, and FastAPI microservice contract.")

    st.markdown("##### 01 / Mathematical Formulation")
    col_math1, col_math2 = st.columns(2)
    with col_math1:
        with st.container(border=True):
            st.markdown("**Haversine Distance Vectorization**")
            st.latex(
                r"""
                d = 2 R \arcsin \left( \sqrt{ \sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right) } \right)
                """
            )
            st.caption(r"Where $R = 6371000m$, $\phi$ is latitude and $\lambda$ is longitude in radians.")

    with col_math2:
        with st.container(border=True):
            st.markdown("**Transit Reliability Index (TRI)**")
            st.latex(
                r"""
                TRI = \alpha \cdot \bar{D}_{delay} + \beta \cdot \sigma_{delay}^{1.5} + \gamma \cdot \Delta_{\text{Rain}}
                """
            )
            st.caption("Composite risk metric combining expected delay, variance hazard, and climate impact multiplier.")

    with st.container(border=True):
        st.markdown("**Spatio-Temporal Cross-Correlation Lag & Hypothesis Test**")
        col_latex_a, col_latex_b = st.columns(2)
        with col_latex_a:
            st.latex(
                r"""
                r_{X,Y}(\tau) = \frac{\sum_{t=1}^{N-\tau} (X_t - \bar{X})(Y_{t+\tau} - \bar{Y})}{\sqrt{\sum_{t=1}^{N} (X_t - \bar{X})^2 \sum_{t=1}^{N} (Y_t - \bar{Y})^2}}
                """
            )
        with col_latex_b:
            st.latex(
                r"""
                t = \frac{\bar{d}_{delay} - \mu_0}{s_d / \sqrt{N}}, \quad p = P(T \ge t \mid H_0)
                """
            )

    st.markdown("---")
    st.markdown("##### 02 / Empirical Hypothesis Testing")
    res_hyp = executar_teste_hipotese_atraso()

    render_kpi_strip(
        [
            ("Sample Size (N)", f"{res_hyp['n_samples']}", "Paired observations", ""),
            ("Student t-Stat", f"{res_hyp['t_statistic']:.3f}", "One-tailed test", "accent"),
            ("p-value", f"{res_hyp['p_value_ttest']:.4e}", f"alpha = {res_hyp['alpha']}", "positive" if res_hyp["reject_h0"] else ""),
            ("Cohen's d", f"{res_hyp['cohens_d']:.2f}", "Effect size", ""),
        ],
        cols=4,
    )

    hyp_border = "#22c55e" if res_hyp["reject_h0"] else ACCENT
    st.markdown(
        f'<div class="panel-card" style="border-color:{hyp_border};"><div style="color:#d4d4d8;font-size:0.88rem;">{res_hyp["conclusion"]}</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("##### 03 / Spatio-Temporal Lag Correlation (Weather vs Delay)")

    lag_data = calcular_defasagem_spatiotemporal_clima(max_lag=6)
    df_lags = pd.DataFrame(
        {
            "Lag Tau (hours)": lag_data["lags"],
            "Pearson (r)": lag_data["pearson_correlations"],
            "Spearman (rho)": lag_data["spearman_correlations"],
        }
    )

    col_chart, col_info = st.columns([2, 1])
    with col_chart:
        st.line_chart(df_lags.set_index("Lag Tau (hours)"), color=[ACCENT, "#a1a1aa"])
    with col_info:
        st.markdown(
            f"""
            <div class="panel-card">
                <div class="kpi-lbl">Optimal Lag</div>
                <div class="kpi-num accent">{lag_data['optimal_lag_tau']} periods</div>
                <div class="kpi-lbl" style="margin-top:12px;">Peak Pearson r</div>
                <div class="kpi-num">{lag_data['max_pearson_r']:.3f}</div>
                <p style="color:#a1a1aa;font-size:0.82rem;margin-top:10px;margin-bottom:0;">
                    Precipitation peaks drive maximum corridor delay spikes with a 2-hour temporal lag across BH arterial routes.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("##### 04 / MLOps Performance Metrics (Scikit-Learn)")
    mlops_metrics = avaliar_desempenho_modelos()

    render_kpi_strip(
        [
            ("Regressor R²", f"{mlops_metrics.get('regressor_r2', 'N/A')}", "Delay duration model", "accent"),
            ("Regressor MAE", f"{mlops_metrics.get('regressor_mae', 'N/A')}", "Mean absolute error", ""),
            ("Regressor RMSE", f"{mlops_metrics.get('regressor_rmse', 'N/A')}", "Root mean sq. error", ""),
            ("Classifier F1", f"{mlops_metrics.get('classifier_f1_score', 'N/A')}", "Occupancy classification", ""),
        ],
        cols=4,
    )

    st.markdown("---")
    st.markdown("##### 05 / Explainable AI (XAI) & Feature Importance")
    importancias = obter_importancia_features()
    df_imp = pd.DataFrame(list(importancias["regressor"].items()), columns=["Feature", "Importância Relativa"]).sort_values("Importância Relativa", ascending=True)

    col_xai1, col_xai2 = st.columns([2, 1])
    with col_xai1:
        st.bar_chart(df_imp.set_index("Feature"), color=ACCENT)
    with col_xai2:
        items_html = "".join(
            f'<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #242429;font-size:0.82rem;">'
            f'<span style="color:#d4d4d8;">{feat}</span>'
            f'<span style="font-family:JetBrains Mono,monospace;color:{ACCENT};">{val * 100:.1f}%</span></div>'
            for feat, val in importancias["regressor"].items()
        )
        st.markdown(
            f'<div class="panel-card"><div class="kpi-lbl" style="margin-bottom:8px;">Pesos Relativos (Random Forest)</div>{items_html}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("##### 06 / Production REST API (FastAPI + OpenAPI 3.0)")

    col_api1, col_api2 = st.columns([1, 1])
    with col_api1:
        with st.container(border=True):
            st.markdown("**Endpoints Disponíveis**")
            st.markdown(
                """
                - Swagger UI: `http://localhost:8000/docs`
                - ReDoc: `http://localhost:8000/redoc`
                - `GET /health` — Health Check & Integridade
                - `GET /api/v1/weather/current` — Clima BH em Tempo Real
                - `POST /api/v1/predict/delay` — Inferência de Atraso e Lotação
                - `GET /api/v1/analytics/metrics` — Métricas MLOps & XAI
                - `POST /api/v1/analytics/tri` — Índice de Confiabilidade TRI
                """
            )
    with col_api2:
        with st.container(border=True):
            st.markdown("**Exemplo de Chamada (Python Requests)**")
            st.code(
                """import requests

payload = {
    "horario_str": "18:00",
    "tipo_dia_str": "Dia Útil",
    "tempo_viagem_teorico_min": 35.0,
    "distancia_estimada_km": 12.0,
    "precipitacao_mm": 5.5
}
resp = requests.post(
    "http://localhost:8000/api/v1/predict/delay",
    json=payload
)
print(resp.json())""",
                language="python",
            )

    st.markdown("---")
    st.markdown("##### 07 / BibTeX Citation")
    bibtex_str = gerar_citacao_bibtex()
    st.code(bibtex_str, language="bibtex")
