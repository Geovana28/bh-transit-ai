import zipfile
import pandas as pd
import numpy as np
import os
import streamlit as st
import folium
from folium import plugins
from streamlit_folium import st_folium
import sys
import importlib
from datetime import datetime, time
from urllib.parse import quote

# Importa e recarrega dinamicamente o motor de rotas e assistente
sys.path.append(os.path.dirname(__file__))
import motor_rotas
importlib.reload(motor_rotas)
from motor_rotas import (
    buscar_rotas_com_transbordo, buscar_coordenadas_endereco,
    calcular_tarifa_bhbus, calcular_caminhada_e_saude,
    verificar_estacao_metro_proxima, filtrar_e_ordenar_rotas,
    TARIFA_CONVENCIONAL_BH, TARIFA_CIRCULAR_VILA_BH
)

import motor_ia_ml
importlib.reload(motor_ia_ml)
from motor_ia_ml import prever_impacto_transito_e_lotacao, avaliar_desempenho_modelos, obter_importancia_features

import servico_clima
importlib.reload(servico_clima)
from servico_clima import obter_clima_atual_bh, LAT_BH, LON_BH

import assistente_ia
importlib.reload(assistente_ia)
from assistente_ia import processar_consulta_assistente_ia

import analytics_estatistico
importlib.reload(analytics_estatistico)
from analytics_estatistico import (
    executar_teste_hipotese_atraso,
    calcular_defasagem_spatiotemporal_clima,
    calcular_indice_confiabilidade_tri,
    gerar_citacao_bibtex
)

# --- CONFIGURAÇÃO DA PÁGINA STREAMLIT ---
st.set_page_config(
    page_title="BH Transit AI - Mobilidade Urbana Inteligente",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- INICIALIZAÇÃO DE NAVEGAÇÃO E FAVORITOS EM SESSION STATE ---
if "favoritos" not in st.session_state:
    st.session_state["favoritos"] = [
        {"origem": "BH Shopping, Belo Horizonte", "destino": "Mercado Central, Belo Horizonte", "rotulo": "🏠 Casa ➔ 🏢 Trabalho"},
        {"origem": "Savassi, Belo Horizonte", "destino": "Mineirão, Belo Horizonte", "rotulo": "🎉 Savassi ➔ 🏟️ Mineirão"},
        {"origem": "Estação Barreiro, Belo Horizonte", "destino": "Praça Sete, Belo Horizonte", "rotulo": "🚏 Barreiro ➔ 📍 Centro"}
    ]

# --- DESIGN SYSTEM & CSS PERSONALIZADO (TEMA CLARO E PROFISSIONAL) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    /* =======================================================
       CORREÇÃO GERAL DE CORES E CONTRASTE (LIGHT THEME PURA)
       ======================================================= */
    html, body, [class*="css"], .stMarkdown, p, div {
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
        color: #1E293B;
    }

    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1350px;
    }

    .stApp {
        background-color: #F8FAFC !important;
    }

    /* Títulos e Cabeçalhos */
    h1, h2, h3, h4, h5, h6, 
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6,
    [data-testid="stHeader"] {
        color: #0F172A !important;
        font-weight: 700 !important;
    }

    /* Sidebar limpa e legível */
    [data-testid="stSidebar"] {
        background-color: #0F172A !important;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] div {
        color: #F1F5F9 !important;
    }

    /* Radio button options - texto e background limpos */
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] [data-testid="stWidgetLabel"] p,
    [data-testid="stRadio"] div[role="radiogroup"] label {
        color: #0F172A !important;
        background: transparent !important;
    }
    [data-testid="stRadio"] div[role="radiogroup"] label div {
        background: transparent !important;
        color: #0F172A !important;
    }
    [data-testid="stRadio"] div[role="radiogroup"] label p,
    [data-testid="stRadio"] div[role="radiogroup"] label span {
        color: #0F172A !important;
    }

    /* Cards Limpos */
    .glass-card {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
        margin-bottom: 20px;
    }

    /* Inputs e Steppers */
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input,
    div[data-baseweb="input"] {
        background-color: #FFFFFF !important;
        border-color: #CBD5E1 !important;
        color: #0F172A !important;
    }
    [data-testid="stNumberInput"] button {
        background-color: #F1F5F9 !important;
        color: #0F172A !important;
        border-color: #CBD5E1 !important;
    }

    /* Sliders */
    .stSlider label, .stSlider [data-testid="stWidgetLabel"] p {
        color: #0F172A !important;
        font-weight: 600 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #FFFFFF !important;
        padding: 6px;
        border-radius: 14px;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }

    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.92rem;
        color: #475569 !important;
        padding: 0 18px;
        border: 1px solid transparent;
    }

    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        border: 1px solid #1D4ED8 !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25) !important;
    }
    .stTabs [aria-selected="true"] p, .stTabs [aria-selected="true"] span {
        color: #FFFFFF !important;
    }

    /* Botões */
    .stButton > button {
        background: #2563EB !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 10px 24px !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
    }
    .stButton > button:hover {
        background: #1D4ED8 !important;
        color: #FFFFFF !important;
    }

    /* Badges */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        padding: 5px 12px;
        border-radius: 50px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    .badge-purple { background: rgba(126, 34, 206, 0.25); border: 1px solid #A855F7; color: #E9D5FF !important; }
    .badge-cyan { background: rgba(3, 105, 161, 0.25); border: 1px solid #38BDF8; color: #BAE6FD !important; }
    .badge-emerald { background: rgba(4, 120, 87, 0.25); border: 1px solid #34D399; color: #A7F3D0 !important; }
    .badge-amber { background: rgba(180, 83, 9, 0.25); border: 1px solid #FBBF24; color: #FDE68A !important; }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR BARRA LATERAL ---
with st.sidebar:
    st.markdown("### 🚌 **BH Transit AI**")
    st.markdown("Plataforma Inteligente de Mobilidade Urbana com IA, ML & Intermodalidade")
    
    st.markdown("""
        <div style="margin-top:10px; margin-bottom:15px;">
            <span class="badge-pill badge-purple">Python 3.10+</span>
            <span class="badge-pill badge-cyan">GTFS BH</span>
            <span class="badge-pill badge-emerald">Scikit-Learn</span>
            <span class="badge-pill badge-amber">Metrô BH</span>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🌤️ **Clima em Tempo Real (BH)**")
    
    col_w_btn, col_w_status = st.columns([2, 1])
    with col_w_btn:
        if st.button("🔄 Sincronizar", use_container_width=True, key="btn_sync_weather"):
            st.session_state["clima_bh_atual"] = obter_clima_atual_bh()
            st.rerun()

    clima_atual = st.session_state.get("clima_bh_atual")
    if clima_atual is None:
        clima_atual = obter_clima_atual_bh()
        st.session_state["clima_bh_atual"] = clima_atual

    st.markdown(f"**{clima_atual.get('emoji', '☀️')} {clima_atual.get('descricao', 'Tempo Bom')}**")
    
    col_cw1, col_cw2 = st.columns(2)
    with col_cw1:
        st.metric("Temperatura", f"{clima_atual.get('temperatura_c', 24.0):.1f} °C")
    with col_cw2:
        st.metric("Precipitação", f"{clima_atual.get('precipitacao_mm', 0.0):.1f} mm/h")

    st.caption(f"📡 *{clima_atual.get('fonte', 'Open-Meteo API')}*")
    
    if clima_atual.get('precipitacao_mm', 0.0) > 0.5:
        st.warning(clima_atual.get('alerta', 'Chuva detectada'))
    else:
        st.success("🟢 Pistas secas na capital")

    st.markdown("---")
    st.success("🟢 **Status:** Dados GTFS BH Ativos")
    st.info("💳 **Tarifas:** BHBUS & ÓTIMO Flexível")
    st.markdown("---")
    st.markdown("### 👩‍💻 Desenvolvido por:")
    st.markdown("**Fullstack & AI Developer**")
    st.markdown("📍 Belo Horizonte, MG")
    st.markdown("---")
    st.caption("v3.5 • Powered by GTFS, Open-Meteo, FastAPI & ML")


# --- CAMINHO DOS DADOS ---
GTFS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'gtfs_bh_convencional.zip')

@st.cache_data
def carregar_dados_gtfs():
    """Carrega os dados GTFS indexados em memória com carregamento instantâneo."""
    cache = motor_rotas.obter_cache_gtfs()
    if cache is not None:
        return (
            cache['routes'],
            cache['stops'],
            cache['trips'],
            cache['shapes'],
            cache['stop_times_rep'],
            cache['calendar'],
            cache['calendar_dates']
        )
    
    # Fallback se o cache binário não existir
    with zipfile.ZipFile(GTFS_PATH, 'r') as z:
        routes = pd.read_csv(z.open('routes.txt'))
        stops = pd.read_csv(z.open('stops.txt'))
        trips = pd.read_csv(z.open('trips.txt'))
        shapes = pd.read_csv(z.open('shapes.txt')) if 'shapes.txt' in z.namelist() else None
        calendar = pd.read_csv(z.open('calendar.txt')) if 'calendar.txt' in z.namelist() else None
        calendar_dates = pd.read_csv(z.open('calendar_dates.txt')) if 'calendar_dates.txt' in z.namelist() else None
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
    "Estação Venda Nova": (-19.8023, -43.9573)
}

def obter_coordenadas_trajeto_rua(shape_id, fallback_p1_lat, fallback_p1_lon, fallback_p2_lat, fallback_p2_lon):
    """Retorna a sequência exata de coordenadas que acompanham a geometria de cada rua em BH."""
    if shapes is not None and pd.notna(shape_id):
        f_shape = shapes[shapes['shape_id'] == shape_id].sort_values('shape_pt_sequence')
        if not f_shape.empty:
            return f_shape[['shape_pt_lat', 'shape_pt_lon']].values.tolist()
            
    return [[fallback_p1_lat, fallback_p1_lon], [fallback_p2_lat, fallback_p2_lon]]

def obter_linhas_do_ponto(stop_id, stop_times_df, trips_df, routes_df):
    """Retorna lista das linhas de ônibus de BH que passam por um determinado ponto"""
    try:
        trips_no_ponto = stop_times_df[stop_times_df['stop_id'].astype(str) == str(stop_id)]['trip_id'].values
        if len(trips_no_ponto) == 0:
            return []
        route_ids_no_ponto = trips_df[trips_df['trip_id'].isin(trips_no_ponto)]['route_id'].unique()
        linhas_shorts = routes_df[routes_df['route_id'].isin(route_ids_no_ponto)]['route_short_name'].astype(str).unique()
        return sorted(list(set(linhas_shorts)))
    except Exception:
        return []

def renderizar_mapa_rota(melhor_rota, o_lat, o_lon, d_lat, d_lon, key_prefix, stops_df=None, trips_df=None, stop_times_df=None, routes_df=None):
    """Desenha o mapa da rota acompanhando as ruas reais de BH (dados de shapes.txt) e exibe as paradas intermediárias"""
    centro_mapa_lat = (o_lat + d_lat) / 2.0
    centro_mapa_lon = (o_lon + d_lon) / 2.0
    
    mapa = folium.Map(location=[centro_mapa_lat, centro_mapa_lon], zoom_start=13, tiles="OpenStreetMap")

    # 🟢 Marcador da Origem
    folium.Marker(
        location=[o_lat, o_lon],
        popup="<b>Sua Origem</b>",
        tooltip="🟢 Origem",
        icon=folium.Icon(color="green", icon="user", prefix="fa")
    ).add_to(mapa)

    # 🔴 Marcador do Destino
    folium.Marker(
        location=[d_lat, d_lon],
        popup="<b>Seu Destino</b>",
        tooltip="🔴 Destino",
        icon=folium.Icon(color="red", icon="flag", prefix="fa")
    ).add_to(mapa)

    # 🚏 Embarque Inicial
    folium.CircleMarker(
        location=[melhor_rota['ponto_origem_lat'], melhor_rota['ponto_origem_lon']],
        radius=9,
        popup=f"<b>Embarque Inicial:</b> {melhor_rota['ponto_origem_nome']}",
        tooltip=f"🚏 Embarque no 1º Ônibus ({melhor_rota.get('linha1_short', melhor_rota['route_short_name'])})",
        color="#0288D1",
        fill=True,
        fill_color="#29B6F6",
        fill_opacity=0.9
    ).add_to(mapa)

    # 🚶 Caminhada inicial
    folium.PolyLine(
        locations=[[o_lat, o_lon], [melhor_rota['ponto_origem_lat'], melhor_rota['ponto_origem_lon']]],
        color="green", weight=3, dash_array="5, 10", tooltip="Caminhada até o embarque"
    ).add_to(mapa)

    # ROTA COM TRANSBORDO (2 ÔNIBUS SEGUINDO AS RUAS!)
    if melhor_rota.get('transbordos', 0) == 1:
        l1 = melhor_rota.get('linha1_short') or melhor_rota.get('r1_short') or 'Ônibus 1'
        l2 = melhor_rota.get('linha2_short') or melhor_rota.get('r2_short') or 'Ônibus 2'
        s1 = melhor_rota.get('shape_id_1') or melhor_rota.get('shape_id')
        s2 = melhor_rota.get('shape_id_2') or melhor_rota.get('shape2_id')

        folium.CircleMarker(
            location=[melhor_rota['ponto_troca_lat'], melhor_rota['ponto_troca_lon']],
            radius=12,
            popup=f"<b>🔀 Baldeação:</b> {melhor_rota.get('ponto_troca_nome', 'Ponto de Conexão')}<br><i>(Troca do Ônibus {l1} para {l2})</i>",
            tooltip=f"🔀 BALDEAÇÃO em {melhor_rota.get('ponto_troca_nome', 'Ponto de Conexão')}",
            color="#D84315",
            fill=True,
            fill_color="#FFB300",
            fill_opacity=1.0
        ).add_to(mapa)

        coords_rua_1 = obter_coordenadas_trajeto_rua(
            s1, 
            melhor_rota['ponto_origem_lat'], melhor_rota['ponto_origem_lon'],
            melhor_rota['ponto_troca_lat'], melhor_rota['ponto_troca_lon']
        )
        folium.PolyLine(
            locations=coords_rua_1,
            color="#1E88E5", weight=6, opacity=0.85,
            tooltip=f"1º Ônibus (Linha {l1})"
        ).add_to(mapa)

        coords_rua_2 = obter_coordenadas_trajeto_rua(
            s2, 
            melhor_rota['ponto_troca_lat'], melhor_rota['ponto_troca_lon'],
            melhor_rota['ponto_destino_lat'], melhor_rota['ponto_destino_lon']
        )
        folium.PolyLine(
            locations=coords_rua_2,
            color="#FF6D00", weight=6, opacity=0.85,
            tooltip=f"2º Ônibus (Linha {l2})"
        ).add_to(mapa)

    else:
        coords_rua_direta = obter_coordenadas_trajeto_rua(
            melhor_rota.get('shape_id'), 
            melhor_rota['ponto_origem_lat'], melhor_rota['ponto_origem_lon'],
            melhor_rota['ponto_destino_lat'], melhor_rota['ponto_destino_lon']
        )
        folium.PolyLine(
            locations=coords_rua_direta,
            color="#1E88E5", weight=6, opacity=0.85,
            tooltip=f"Ônibus Direto (Linha {melhor_rota.get('route_short_name', '')})"
        ).add_to(mapa)

    # --- DESENHAR PARADAS INTERMEDIÁRIAS (INDEXAÇÃO INSTANTÂNEA O(1)) ---
    if stops_df is not None:
        try:
            route_id_alvo = melhor_rota.get('route_id')
            cache = motor_rotas.obter_cache_gtfs()
            if cache and route_id_alvo:
                seqs = cache['mapa_rota_pontos'].get(route_id_alvo, [])
                ids_ignorar = {str(melhor_rota['ponto_origem_id']), str(melhor_rota['ponto_destino_id'])}
                if melhor_rota.get('ponto_troca_id'):
                    ids_ignorar.add(str(melhor_rota['ponto_troca_id']))
                
                count = 0
                for sid, _ in seqs:
                    if sid not in ids_ignorar and sid in cache['stops_dict']:
                        s_info = cache['stops_dict'][sid]
                        linhas_passantes = motor_rotas.obter_linhas_do_ponto(sid)
                        texto_linhas = ", ".join(linhas_passantes[:6]) if linhas_passantes else "N/D"
                        if len(linhas_passantes) > 6:
                            texto_linhas += f" (+{len(linhas_passantes)-6} outras)"

                        folium.CircleMarker(
                            location=[s_info['lat'], s_info['lon']],
                            radius=4,
                            popup=f"<b>🚏 Parada:</b> {s_info['name']}<br><b>ID:</b> {sid}<br><b>🚌 Ônibus que passam aqui:</b> {texto_linhas}",
                            tooltip=f"🚏 {s_info['name']} (Clique para ver os ônibus)",
                            color="#455A64",
                            fill=True,
                            fill_color="#78909C",
                            fill_opacity=0.8
                        ).add_to(mapa)
                        count += 1
                        if count >= 30:
                            break
        except Exception as e:
            pass

    # 🚏 Desembarque Final
    folium.CircleMarker(
        location=[melhor_rota['ponto_destino_lat'], melhor_rota['ponto_destino_lon']],
        radius=9,
        popup=f"<b>Desembarque:</b> {melhor_rota['ponto_destino_nome']}",
        tooltip=f"🚏 Desembarque Final",
        color="#7B1FA2",
        fill=True,
        fill_color="#AB47BC",
        fill_opacity=0.9
    ).add_to(mapa)

    # 🚶 Caminhada final
    folium.PolyLine(
        locations=[[melhor_rota['ponto_destino_lat'], melhor_rota['ponto_destino_lon']], [d_lat, d_lon]],
        color="purple", weight=3, dash_array="5, 10", tooltip="Caminhada até o destino"
    ).add_to(mapa)

    st_folium(mapa, width="100%", height=520, key=f"mapa_ruas_renderizado_{key_prefix}")


# --- HERO HEADER HEADER DESIGN ---
st.markdown("""
    <div>
        <h1 class="titulo-hero">🚌 BH Transit AI</h1>
        <p class="subtitulo-hero">Plataforma Inteligente de Mobilidade Urbana para Belo Horizonte e Região Metropolitana (GTFS, IA & ML)</p>
    </div>
""", unsafe_allow_html=True)

# --- NAVEGAÇÃO POR ABAS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️ Planejador de Rotas",
    "💬 Assistente Conversacional IA",
    "📊 Analytics & Heatmap Urbano",
    "💳 Calculadora de Economia",
    "📍 Clique Direto no Mapa",
    "🔬 Academic & Research Analytics"
])

# ==============================================================================
# ABA 1: PLANEJADOR DE ROTAS COM CLIMA, FILTROS, FAVORITOS E WHATSAPP/QR CODE
# ==============================================================================
with tab1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🗺️ Planejador de Itinerários Curva a Curva nas Ruas")

    if st.session_state["favoritos"]:
        st.markdown("##### ⭐ Atalhos Rápidos & Favoritos:")
        cols_fav = st.columns(len(st.session_state["favoritos"]))
        for i, fav in enumerate(st.session_state["favoritos"]):
            with cols_fav[i]:
                if st.button(fav["rotulo"], key=f"btn_fav_{i}", use_container_width=True):
                    st.session_state["p1_orig"] = fav["origem"]
                    st.session_state["p1_dest"] = fav["destino"]

    col_orig, col_dest = st.columns(2)

    with col_orig:
        orig_txt_tab1 = st.text_input(
            "🟢 Origem (Endereço ou Local de BH):", 
            value=st.session_state.get("p1_orig", "BH Shopping, Belo Horizonte"),
            key="input_orig_tab1"
        )

    with col_dest:
        dest_txt_tab1 = st.text_input(
            "🔴 Destino (Endereço ou Local de BH):", 
            value=st.session_state.get("p1_dest", "Mercado Central, Belo Horizonte"),
            key="input_dest_tab1"
        )

    st.markdown("### 🎛️ Filtros, Horários e Clima")
    c_h1, c_h2, c_h3, c_h4 = st.columns(4)

    with c_h1:
        horario_tab1 = st.time_input("Horário de Saída:", value=time(8, 0), key="horario_tab1")
        horario_str_tab1 = horario_tab1.strftime("%H:%M")

    with c_h2:
        tipo_dia_tab1 = st.selectbox(
            "Quadro de Horários:",
            options=["📅 Dia Útil (Segunda a Sexta)", "🟡 Sábado", "🔴 Domingo / Feriado"],
            index=0,
            key="tipo_dia_tab1"
        )

    with c_h3:
        usar_clima_tempo_real = st.checkbox("📡 Clima em Tempo Real BH", value=True, key="chk_clima_rt")
        if usar_clima_tempo_real and clima_atual:
            clima_tab1 = f"{clima_atual.get('emoji', '☀️')} {clima_atual.get('descricao', 'Tempo Bom')}"
            st.caption(f"🌧️ Chuva: `{clima_atual.get('precipitacao_mm', 0.0):.1f} mm/h`")
            precip_mm_passar = clima_atual.get('precipitacao_mm', 0.0)
        else:
            clima_tab1 = st.selectbox(
                "⛈️ Condição Climatológica:",
                options=["☀️ Ensolarado / Tempo Bom", "🌧️ Chuva Moderada", "⛈️ Chuva Forte / Temporal"],
                index=0,
                key="clima_tab1"
            )
            precip_mm_passar = None

    with c_h4:
        filtro_tab1 = st.selectbox(
            "🎯 Preferência de Rota:",
            options=["⚡ Mais Rápida", "🚶 Menos Caminhada", "🚏 Linhas Diretas", "♿ Acessível (PCD)"],
            index=0,
            key="filtro_tab1"
        )

    with st.expander("💳 Configurar Tarifa Customizada / Ônibus Metropolitano (ex: RMBH R$ 8,45)"):
        c_tar1, c_tar2 = st.columns([1, 2])
        with c_tar1:
            usar_tarifa_custom_tab1 = st.checkbox("Usar Tarifa Metropolitiana / Personalizada", value=False, key="chk_custom_tab1")
        with c_tar2:
            val_tarifa_custom_tab1 = st.number_input("Valor da Passagem Individual (R$):", value=8.45, min_value=1.0, max_value=50.0, step=0.05, key="num_custom_tab1")

    col_b1, col_b2 = st.columns([3, 1])
    with col_b1:
        btn_buscar_tab1 = st.button("🚀 CALCULAR MELHOR ROTA COM IA & ML", type="primary", use_container_width=True, key="btn_buscar_tab1")
    with col_b2:
        if st.button("⭐ Salvar Favorito", use_container_width=True, key="btn_add_fav"):
            st.session_state["favoritos"].append({
                "origem": orig_txt_tab1,
                "destino": dest_txt_tab1,
                "rotulo": f"📍 {orig_txt_tab1.split(',')[0]} ➔ {dest_txt_tab1.split(',')[0]}"
            })
            st.success("Salvo!")
    st.markdown("</div>", unsafe_allow_html=True)

    if btn_buscar_tab1 or st.session_state.get("busca_tab1_realizada", False):
        if btn_buscar_tab1:
            st.session_state["busca_tab1_realizada"] = True
            st.session_state["tab1_orig"] = orig_txt_tab1
            st.session_state["tab1_dest"] = dest_txt_tab1

        orig_processar = st.session_state.get("tab1_orig", orig_txt_tab1)
        dest_processar = st.session_state.get("tab1_dest", dest_txt_tab1)

        with st.spinner("🔍 Localizando no OpenStreetMap e executando modelo de IA/ML..."):
            lat_o, lon_o, end_o = buscar_coordenadas_endereco(orig_processar)
            lat_d, lon_d, end_d = buscar_coordenadas_endereco(dest_processar)

            if lat_o is None or lat_d is None:
                st.error("❌ Não foi possível encontrar um dos endereços informados em BH.")
            else:
                rotas_brutas = buscar_rotas_com_transbordo(
                    lat_o, lon_o, lat_d, lon_d,
                    routes, stops, trips, stop_times, top_k_pontos=20,
                    horario_partida=horario_str_tab1,
                    calendar_df=calendar, calendar_dates_df=calendar_dates,
                    tipo_dia=tipo_dia_tab1
                )

                if not rotas_brutas:
                    st.warning(f"⚠️ Nenhuma rota encontrada para às {horario_str_tab1} ({tipo_dia_tab1}).")
                else:
                    rotas_filtradas = filtrar_e_ordenar_rotas(rotas_brutas, filtro_tab1)
                    melhor_rota = rotas_filtradas[0]

                    pred_ml = prever_impacto_transito_e_lotacao(
                        horario_str=horario_str_tab1,
                        tipo_dia_str=tipo_dia_tab1,
                        tempo_viagem_teorico_min=melhor_rota['tempo_total_estimado_min'],
                        condicao_climatica=clima_tab1,
                        precipitacao_mm=precip_mm_passar
                    )

                    tarifa_val_passar = val_tarifa_custom_tab1 if usar_tarifa_custom_tab1 else None
                    dados_tarifa = calcular_tarifa_bhbus(melhor_rota, tarifa_customizada=tarifa_val_passar)
                    dados_saude = calcular_caminhada_e_saude(melhor_rota)
                    metro_orig = verificar_estacao_metro_proxima(lat_o, lon_o)
                    metro_dest = verificar_estacao_metro_proxima(lat_d, lon_d)

                    st.markdown("---")
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Linha(s) Recomendadas", melhor_rota['route_short_name'])
                    m2.metric("Tempo Previsto (ML)", f"⏱️ ~{pred_ml['tempo_ajustado_min']} min", delta=f"+{pred_ml['atraso_min']} min trânsito/clima" if pred_ml['atraso_min'] > 0 else "Sem atrasos")
                    m3.metric("Tarifa Estimada", dados_tarifa['tarifa_total_str'], delta="50% desc. 2ª tarifa" if melhor_rota.get('transbordos', 0) > 0 else "Tarifa Única")
                    m4.metric("Lotação Prevista", pred_ml['lotacao_texto'].split(' ')[0] + " " + pred_ml['lotacao_texto'].split(' ')[1])
                    m5.metric("Caminhada Total", f"🚶 {dados_saude['total_m']}m", delta=f"~{dados_saude['calorias_kcal']} kcal")

                    st.info(f"{pred_ml['explicacao']}")

                    if metro_orig or metro_dest:
                        metro_msg = "🚆 **Conexão com Metrô de BH:** "
                        if metro_orig:
                            metro_msg += f"Origem perto da **{metro_orig['nome']}** ({metro_orig['distancia_m']}m). "
                        if metro_dest:
                            metro_msg += f"Destino perto da **{metro_dest['nome']}** ({metro_dest['distancia_m']}m)."
                        st.success(metro_msg)

                    txt_wa = f"🚌 BH Transit AI: Rota de {orig_processar.split(',')[0]} ate {dest_processar.split(',')[0]} com a Linha {melhor_rota['route_short_name']}. Tempo est: {pred_ml['tempo_ajustado_min']} min. Tarifa: {dados_tarifa['tarifa_total_str']}."
                    url_wa = f"https://api.whatsapp.com/send?text={quote(txt_wa)}"
                    
                    c_wa, c_qr = st.columns([2, 1])
                    with c_wa:
                        st.markdown(f"### 📲 Compartilhar este Itinerário")
                        st.markdown(f"Envie os detalhes da rota diretamente no seu celular ou para um amigo:")
                        st.markdown(f'<a href="{url_wa}" target="_blank" style="text-decoration:none;"><button style="background:linear-gradient(135deg, #25D366 0%, #128C7E 100%) !important; color:white; border:none; padding:12px 24px; border-radius:12px; font-size:16px; font-weight:bold; cursor:pointer;">📱 Abrir no WhatsApp</button></a>', unsafe_allow_html=True)
                    
                    with c_qr:
                        qr_url = f"https://api.qrserver.com/v1/create-qr-code/?size=140x140&data={quote(url_wa)}"
                        st.image(qr_url, caption="Escanear Rota no Celular", width=120)

                    st.markdown("---")
                    st.subheader(f"🗺️ Trajeto nas Ruas ({melhor_rota['route_short_name']})")
                    renderizar_mapa_rota(melhor_rota, lat_o, lon_o, lat_d, lon_d, "tab1_mapa", stops, trips, stop_times, routes)


# ==============================================================================
# ABA 2: ASSISTENTE IA CONVERSACIONAL (CHATBOT)
# ==============================================================================
with tab2:
    st.subheader("💬 Converse com o Assistente de IA de Transporte de BH")
    st.markdown("Peça informações em linguagem natural. A IA extrai origem, destino, horário e ajusta a predição para trânsito e chuva!")

    if "historico_chat" not in st.session_state:
        st.session_state["historico_chat"] = []

    for msg in st.session_state["historico_chat"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "mapa_dados" in msg and msg["mapa_dados"]:
                md = msg["mapa_dados"]
                renderizar_mapa_rota(md['rota'], md['lat_o'], md['lon_o'], md['lat_d'], md['lon_d'], f"chat_hist_{len(st.session_state['historico_chat'])}", stops, trips, stop_times, routes)

    prompt_usuario = st.chat_input("Ex: Tô na Savassi e preciso ir pro Mineirão no temporal às 18h...")

    if prompt_usuario:
        st.session_state["historico_chat"].append({"role": "user", "content": prompt_usuario})
        
        with st.chat_message("user"):
            st.markdown(prompt_usuario)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Processando NLP e rodando modelos de ML de BH..."):
                resp_texto, mapa_dados, _ = processar_consulta_assistente_ia(
                    prompt_usuario, routes, stops, trips, stop_times, calendar, calendar_dates
                )
                st.markdown(resp_texto)
                if mapa_dados:
                    renderizar_mapa_rota(mapa_dados['rota'], mapa_dados['lat_o'], mapa_dados['lon_o'], mapa_dados['lat_d'], mapa_dados['lon_d'], f"chat_new_{len(st.session_state['historico_chat'])}", stops, trips, stop_times, routes)
                
                st.session_state["historico_chat"].append({
                    "role": "assistant",
                    "content": resp_texto,
                    "mapa_dados": mapa_dados
                })


# ==============================================================================
# ABA 3: ANALYTICS URBANO & MAPA DE CALOR (HEATMAP DE BH)
# ==============================================================================
with tab3:
    st.subheader("📊 Analytics Urbano & Mapa de Calor da Rede de BH")
    st.markdown("Visualização geoespacial da concentração e densidade das mais de **9.000 paradas de ônibus** catalogadas em Belo Horizonte.")

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Paradas Catalogadas", len(stops))
    a2.metric("Linhas Operacionais", len(routes))
    a3.metric("Viagens Diárias Mapeadas", len(trips))
    a4.metric("Estações do Metrô (Linha 1)", len(motor_rotas.ESTACOES_METRO_BH))

    st.markdown("---")
    st.markdown("### 🔥 Mapa de Calor (Heatmap) da Cobertura de Transporte em BH")
    
    col_h_opt1, col_h_opt2 = st.columns([2, 2])
    with col_h_opt1:
        raio_heat = st.slider("Raio de Intensidade do Calor (px):", min_value=8, max_value=30, value=16, key="raio_heat_slider")
    with col_h_opt2:
        amostra_qtd = st.slider("Amostragem de Pontos de BH:", min_value=500, max_value=len(stops), value=min(4000, len(stops)), key="amostra_heat_slider")

    stops_validos = stops.dropna(subset=['stop_lat', 'stop_lon'])
    sample_stops = stops_validos.sample(n=min(amostra_qtd, len(stops_validos)), random_state=42)
    
    heatmap_data = [[float(row['stop_lat']), float(row['stop_lon']), 1.0] for _, row in sample_stops.iterrows()]

    mapa_heat = folium.Map(
        location=[-19.9167, -43.9345], 
        zoom_start=12, 
        tiles="OpenStreetMap"
    )

    plugins.HeatMap(
        heatmap_data, 
        radius=raio_heat, 
        blur=14, 
        max_zoom=14,
        min_opacity=0.3,
        gradient={0.2: '#0000FF', 0.4: '#00FFFF', 0.6: '#00FF00', 0.8: '#FFFF00', 1.0: '#FF0000'}
    ).add_to(mapa_heat)

    for est in motor_rotas.ESTACOES_METRO_BH:
        folium.CircleMarker(
            location=[est['lat'], est['lon']],
            radius=8,
            popup=f"🚆 <b>{est['nome']}</b>",
            tooltip=f"🚆 Metrô: {est['nome']}",
            color="#00C853",
            fill=True,
            fill_color="#69F0AE",
            fill_opacity=1.0
        ).add_to(mapa_heat)

    st_folium(mapa_heat, width="100%", height=550, key="heatmap_bh_render_v2", returned_objects=[])


# ==============================================================================
# ABA 4: CALCULADORA DE ECONOMIA - DESIGN ULTRA MODERN GLASS
# ==============================================================================
with tab4:
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h2 style="font-size:1.8rem; font-weight:800; color:#0F172A; margin-bottom:4px;">💳 Simulação de Gastos & Economia Financeira</h2>
            <p style="color:#475569; font-size:1rem;">Compare o custo real do seu deslocamento mensal de <b>Ônibus (BHBUS / ÓTIMO)</b> contra <b>Carro Próprio</b> e <b>Uber/99</b>.</p>
        </div>
    """, unsafe_allow_html=True)

    # CARD DE CONFIGURAÇÃO DE ENTRADA
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### ⚙️ Como é a sua rotina de transporte?")
    
    modo_calculo = st.radio(
        "Selecione o tipo do seu deslocamento:",
        options=[
            "🔀 Baldeação Mista (1 Metropolitano R$ 8,45 + 1 Municipal R$ 6,25)",
            "💰 Custo Total Somado por Trajeto (Soma da Ida já pronta)",
            "🚌 Passageiro de 1 Linha Única (Sem Baldeação)"
        ],
        index=0,
        horizontal=False
    )

    col_sim1, col_sim2 = st.columns(2)

    with col_sim1:
        dias_trabalho = st.slider("🗓️ Dias de deslocamento por mês (Ida e Volta):", min_value=1, max_value=30, value=22)
        
        if "Baldeação Mista" in modo_calculo:
            st.info("💡 **Baldeação Mista:** Digite a tarifa do 1º e do 2º ônibus.")
            val_bus1 = st.number_input("1º Ônibus (ex: Metropolitano ÓTIMO R$):", value=8.45, min_value=1.0, max_value=50.0, step=0.05, format="%.2f")
            val_bus2 = st.number_input("2º Ônibus (ex: Municipal BHBUS R$):", value=6.25, min_value=1.0, max_value=50.0, step=0.05, format="%.2f")
            pct_desc = st.slider("Desconto de Integração no 2º ônibus (%):", min_value=0, max_value=100, value=50, step=5)
            
            custo_ida = val_bus1 + (val_bus2 * (100 - pct_desc) / 100.0)
            custo_diario_bus = custo_ida * 2.0
            info_resumo = f"R$ {val_bus1:.2f} + R$ {val_bus2 * (100 - pct_desc) / 100.0:.2f} (Desc {pct_desc}%) = R$ {custo_ida:.2f}/ida"

        elif "Custo Total Somado" in modo_calculo:
            st.info("💡 **Valor Fechado:** Digite a soma total cobrada em UMA IDA.")
            custo_ida = st.number_input("Valor Total Gasto em UMA IDA (R$):", value=14.70, min_value=1.0, max_value=100.0, step=0.10, format="%.2f")
            custo_diario_bus = custo_ida * 2.0
            info_resumo = f"R$ {custo_ida:.2f}/ida ➔ R$ {custo_diario_bus:.2f}/dia"

        else:
            st.info("💡 **Linha Única:** Digite o valor de 1 passagem.")
            val_unico = st.number_input("Tarifa da Passagem (R$):", value=8.45, min_value=1.0, max_value=50.0, step=0.05, format="%.2f")
            custo_ida = val_unico
            custo_diario_bus = custo_ida * 2.0
            info_resumo = f"R$ {custo_ida:.2f}/ida ➔ R$ {custo_diario_bus:.2f}/dia"

    with col_sim2:
        dist_media_km = st.slider("📏 Distância média do trajeto (km):", min_value=2.0, max_value=60.0, value=18.0)
        valor_estacionamento_diario = st.number_input("🅿️ Estacionamento diário em BH/Região (R$):", value=15.0)

    st.markdown("</div>", unsafe_allow_html=True)

    # Cálculo mensal
    custo_mensal_bus = custo_diario_bus * dias_trabalho
    custo_diario_uber = (8.00 + (dist_media_km * 2.20)) * 2
    custo_mensal_uber = custo_diario_uber * dias_trabalho
    custo_diario_carro = ((dist_media_km * 2 / 9.5) * 5.80) + valor_estacionamento_diario
    custo_mensal_carro = custo_diario_carro * dias_trabalho

    # --- CARDS VISUAIS DE RESULTADO ---
    st.markdown("<h3 style='font-weight:800; color:#0F172A; margin-top:10px;'>📊 Comparativo Financeiro Mensal</h3>", unsafe_allow_html=True)

    col_res1, col_res2, col_res3 = st.columns(3)

    with col_res1:
        st.markdown(f"""
            <div style="background: #F0FDF4; border: 1px solid #86EFAC; border-radius: 16px; padding: 22px; text-align: center; box-shadow: 0 4px 12px rgba(15,23,42,0.04);">
                <div style="font-size: 2.2rem; margin-bottom: 6px;">🚌</div>
                <div style="font-size: 0.95rem; font-weight: 700; color: #047857; text-transform: uppercase; letter-spacing: 0.05em;">Transporte Público</div>
                <div style="font-size: 2.1rem; font-weight: 800; color: #0F172A; margin: 8px 0;">R$ {custo_mensal_bus:.2f}</div>
                <div style="font-size: 0.85rem; color: #065F46; font-weight: 500;">Custo: R$ {custo_ida:.2f} por ida ({dias_trabalho} dias/mês)</div>
                <div style="margin-top: 12px; background: #D1FAE5; border-radius: 50px; padding: 6px 12px; font-size: 0.8rem; font-weight: 700; color: #047857;">✨ Opção Otimizada</div>
            </div>
        """.replace('.', ','), unsafe_allow_html=True)

    with col_res2:
        diff_carro = custo_mensal_carro - custo_mensal_bus
        txt_diff_carro = f"R$ {abs(diff_carro):.2f} mais caro" if diff_carro > 0 else f"R$ {abs(diff_carro):.2f} economia"
        cor_diff_carro = "#DC2626" if diff_carro > 0 else "#047857"
        bg_diff_carro = "#FEE2E2" if diff_carro > 0 else "#D1FAE5"
        st.markdown(f"""
            <div style="background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 16px; padding: 22px; text-align: center; box-shadow: 0 4px 12px rgba(15,23,42,0.04);">
                <div style="font-size: 2.2rem; margin-bottom: 6px;">🚗</div>
                <div style="font-size: 0.95rem; font-weight: 700; color: #B45309; text-transform: uppercase; letter-spacing: 0.05em;">Carro Próprio</div>
                <div style="font-size: 2.1rem; font-weight: 800; color: #0F172A; margin: 8px 0;">R$ {custo_mensal_carro:.2f}</div>
                <div style="font-size: 0.85rem; color: #78350F; font-weight: 500;">Gasolina + Estacionamento (R$ {valor_estacionamento_diario:.2f}/dia)</div>
                <div style="margin-top: 12px; background: {bg_diff_carro}; border-radius: 50px; padding: 6px 12px; font-size: 0.8rem; font-weight: 700; color: {cor_diff_carro};">{txt_diff_carro}</div>
            </div>
        """.replace('.', ','), unsafe_allow_html=True)

    with col_res3:
        diff_uber = custo_mensal_uber - custo_mensal_bus
        txt_diff_uber = f"R$ {abs(diff_uber):.2f} mais caro" if diff_uber > 0 else f"R$ {abs(diff_uber):.2f} economia"
        cor_diff_uber = "#DC2626" if diff_uber > 0 else "#047857"
        bg_diff_uber = "#FEE2E2" if diff_uber > 0 else "#D1FAE5"
        st.markdown(f"""
            <div style="background: #F3E8FF; border: 1px solid #D8B4FE; border-radius: 16px; padding: 22px; text-align: center; box-shadow: 0 4px 12px rgba(15,23,42,0.04);">
                <div style="font-size: 2.2rem; margin-bottom: 6px;">📱</div>
                <div style="font-size: 0.95rem; font-weight: 700; color: #6B21A8; text-transform: uppercase; letter-spacing: 0.05em;">Uber / 99</div>
                <div style="font-size: 2.1rem; font-weight: 800; color: #0F172A; margin: 8px 0;">R$ {custo_mensal_uber:.2f}</div>
                <div style="font-size: 0.85rem; color: #581C87; font-weight: 500;">Tarifa dinâmica média estimada em BH</div>
                <div style="margin-top: 12px; background: {bg_diff_uber}; border-radius: 50px; padding: 6px 12px; font-size: 0.8rem; font-weight: 700; color: {cor_diff_uber};">{txt_diff_uber}</div>
            </div>
        """.replace('.', ','), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ESG SUSTENTABILIDADE CARD
    co2_mensal_poupado = round((dist_media_km * 2 * dias_trabalho * 0.095), 1)
    arvores_equiv = round(co2_mensal_poupado / 15.0, 1)

    st.markdown(f"""
        <div style="background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 16px; padding: 18px 24px; display: flex; align-items: center; gap: 16px;">
            <div style="font-size: 2rem;">🌱</div>
            <div>
                <div style="font-weight: 700; color: #047857; font-size: 1.05rem;">Impacto Ambiental ESG Positivo</div>
                <div style="color: #065F46; font-size: 0.95rem; font-weight: 500;">Sua escolha pelo transporte público nesta rotina evita a emissão de <b>~{co2_mensal_poupado} kg de CO₂</b> por mês na atmosfera (equivalente a <b>~{arvores_equiv} árvores plantadas/ano</b>)! 🌍✨</div>
            </div>
        </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# ABA 5: CLIQUE DIRETO NO MAPA (FLUXO UNIFICADO & SEM TELA BRANCA)
# ==============================================================================
with tab5:
    st.subheader("📍 Marque no Mapa onde você está (Origem) e para onde vai (Destino)")
    st.caption("Clique diretamente nas ruas ou avenidas de BH para traçar o itinerário automaticamente.")

    # Inicialização de variáveis de sessão da Aba 5
    if "clique_orig_lat" not in st.session_state:
        st.session_state["clique_orig_lat"] = None
        st.session_state["clique_orig_lon"] = None

    if "clique_dest_lat" not in st.session_state:
        st.session_state["clique_dest_lat"] = None
        st.session_state["clique_dest_lon"] = None

    if "clique_rotas_salvas" not in st.session_state:
        st.session_state["clique_rotas_salvas"] = None

    # Banner de Orientação e Status Dinâmico
    c_status1, c_status2 = st.columns([3, 1])
    with c_status1:
        if st.session_state["clique_orig_lat"] is None:
            st.info("🟢 **Passo 1:** Clique no mapa abaixo para marcar o seu ponto de **Origem** (Embarque).")
        elif st.session_state["clique_dest_lat"] is None:
            st.warning("🔴 **Passo 2:** Origem marcada! Agora clique no mapa para marcar o seu **Destino** (Desembarque).")
        else:
            st.success("🎉 **Origem e Destino Marcados!** Confira a rota traçada no mapa e os detalhes abaixo.")

    with c_status2:
        if st.button("🔄 Limpar Pontos", use_container_width=True, key="btn_limpar_clique"):
            st.session_state["clique_orig_lat"] = None
            st.session_state["clique_orig_lon"] = None
            st.session_state["clique_dest_lat"] = None
            st.session_state["clique_dest_lon"] = None
            st.session_state["clique_rotas_salvas"] = None
            st.rerun()

    # Controles de Modo de Marcação
    opcoes_modo = ["🟢 Marcar Origem", "🔴 Marcar Destino"]
    idx_modo_padrao = 1 if (st.session_state["clique_orig_lat"] is not None and st.session_state["clique_dest_lat"] is None) else 0

    col_m1, col_m2 = st.columns([2, 2])
    with col_m1:
        modo_clique = st.radio(
            "O que você deseja marcar ao clicar agora:",
            options=opcoes_modo,
            index=idx_modo_padrao,
            horizontal=True,
            key="radio_modo_clique"
        )
    with col_m2:
        c_p1 = f"Lat: {st.session_state['clique_orig_lat']:.4f}, Lon: {st.session_state['clique_orig_lon']:.4f}" if st.session_state['clique_orig_lat'] else "Não definida"
        c_p2 = f"Lat: {st.session_state['clique_dest_lat']:.4f}, Lon: {st.session_state['clique_dest_lon']:.4f}" if st.session_state['clique_dest_lat'] else "Não definida"
        st.caption(f"🟢 **Origem:** `{c_p1}` | 🔴 **Destino:** `{c_p2}`")

    # Calcula centro dinâmico do mapa
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

    # Construção do Mapa Unificado
    mapa_unificado = folium.Map(location=[lat_mapa, lon_mapa], zoom_start=zoom_mapa, tiles="OpenStreetMap")

    # Adiciona Marcador da Origem
    if st.session_state["clique_orig_lat"]:
        folium.Marker(
            [st.session_state["clique_orig_lat"], st.session_state["clique_orig_lon"]],
            popup="<b>Sua Origem Marcada</b>",
            tooltip="🟢 Origem",
            icon=folium.Icon(color="green", icon="user", prefix="fa")
        ).add_to(mapa_unificado)

    # Adiciona Marcador do Destino
    if st.session_state["clique_dest_lat"]:
        folium.Marker(
            [st.session_state["clique_dest_lat"], st.session_state["clique_dest_lon"]],
            popup="<b>Seu Destino Marcado</b>",
            tooltip="🔴 Destino",
            icon=folium.Icon(color="red", icon="flag", prefix="fa")
        ).add_to(mapa_unificado)

    # Se uma rota já foi calculada, desenha o traçado diretamente no mapa unificado
    rotas_exibir = st.session_state.get("clique_rotas_salvas")
    if rotas_exibir:
        melhor_r = rotas_exibir[0]
        o_lat = st.session_state["clique_orig_lat"]
        o_lon = st.session_state["clique_orig_lon"]
        d_lat = st.session_state["clique_dest_lat"]
        d_lon = st.session_state["clique_dest_lon"]

        # 🚏 Embarque Inicial
        folium.CircleMarker(
            location=[melhor_r['ponto_origem_lat'], melhor_r['ponto_origem_lon']],
            radius=8,
            popup=f"<b>Embarque:</b> {melhor_r['ponto_origem_nome']}",
            tooltip=f"🚏 Embarque no Ônibus ({melhor_r['route_short_name']})",
            color="#0288D1",
            fill=True,
            fill_color="#29B6F6",
            fill_opacity=0.9
        ).add_to(mapa_unificado)

        # 🚶 Caminhada inicial
        folium.PolyLine(
            locations=[[o_lat, o_lon], [melhor_r['ponto_origem_lat'], melhor_r['ponto_origem_lon']]],
            color="green", weight=3, dash_array="5, 10", tooltip="Caminhada até o embarque"
        ).add_to(mapa_unificado)

        # Traçado do Ônibus
        if melhor_r.get('transbordos', 0) == 1:
            l1 = melhor_r.get('linha1_short', 'Ônibus 1')
            l2 = melhor_r.get('linha2_short', 'Ônibus 2')
            s1 = melhor_r.get('shape_id_1') or melhor_r.get('shape_id')
            s2 = melhor_r.get('shape_id_2') or melhor_r.get('shape2_id')

            folium.CircleMarker(
                location=[melhor_r['ponto_troca_lat'], melhor_r['ponto_troca_lon']],
                radius=10,
                popup=f"<b>🔀 Baldeação:</b> {melhor_r['ponto_troca_nome']}",
                tooltip=f"🔀 Baldeação ({l1} ➔ {l2})",
                color="#D84315",
                fill=True,
                fill_color="#FFB300",
                fill_opacity=1.0
            ).add_to(mapa_unificado)

            coords_1 = obter_coordenadas_trajeto_rua(s1, melhor_r['ponto_origem_lat'], melhor_r['ponto_origem_lon'], melhor_r['ponto_troca_lat'], melhor_r['ponto_troca_lon'])
            folium.PolyLine(locations=coords_1, color="#1E88E5", weight=6, opacity=0.85, tooltip=f"1º Ônibus ({l1})").add_to(mapa_unificado)

            coords_2 = obter_coordenadas_trajeto_rua(s2, melhor_r['ponto_troca_lat'], melhor_r['ponto_troca_lon'], melhor_r['ponto_destino_lat'], melhor_r['ponto_destino_lon'])
            folium.PolyLine(locations=coords_2, color="#FF6D00", weight=6, opacity=0.85, tooltip=f"2º Ônibus ({l2})").add_to(mapa_unificado)
        else:
            coords_dir = obter_coordenadas_trajeto_rua(melhor_r.get('shape_id'), melhor_r['ponto_origem_lat'], melhor_r['ponto_origem_lon'], melhor_r['ponto_destino_lat'], melhor_r['ponto_destino_lon'])
            folium.PolyLine(locations=coords_dir, color="#1E88E5", weight=6, opacity=0.85, tooltip=f"Ônibus Direto ({melhor_r['route_short_name']})").add_to(mapa_unificado)

        # 🚏 Desembarque Final
        folium.CircleMarker(
            location=[melhor_r['ponto_destino_lat'], melhor_r['ponto_destino_lon']],
            radius=8,
            popup=f"<b>Desembarque:</b> {melhor_r['ponto_destino_nome']}",
            tooltip="🚏 Desembarque Final",
            color="#7B1FA2",
            fill=True,
            fill_color="#AB47BC",
            fill_opacity=0.9
        ).add_to(mapa_unificado)

        # 🚶 Caminhada final
        folium.PolyLine(
            locations=[[melhor_r['ponto_destino_lat'], melhor_r['ponto_destino_lon']], [d_lat, d_lon]],
            color="purple", weight=3, dash_array="5, 10", tooltip="Caminhada até o destino"
        ).add_to(mapa_unificado)

    # Renderiza o mapa Folium único
    mapa_click_data = st_folium(mapa_unificado, width="100%", height=480, key="mapa_clique_unificado_v3")

    # Captura de cliques com auto-avanço inteligente
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
                # Calcula automaticamente a rota ao marcar o destino
                with st.spinner("🚀 Calculando melhor itinerário..."):
                    rotas_calculadas = buscar_rotas_com_transbordo(
                        st.session_state["clique_orig_lat"], st.session_state["clique_orig_lon"],
                        lat_c, lon_c,
                        routes, stops, trips, stop_times, top_k_pontos=15,
                        horario_partida="08:00",
                        calendar_df=calendar, calendar_dates_df=calendar_dates,
                        tipo_dia="📅 Dia Útil (Segunda a Sexta)"
                    )
                    st.session_state["clique_rotas_salvas"] = rotas_calculadas
                st.rerun()

    # Botão de cálculo manual
    if st.session_state["clique_orig_lat"] and st.session_state["clique_dest_lat"]:
        if st.button("🚀 RECALCULAR MELHOR ROTA COM IA & CLIMA", type="primary", use_container_width=True, key="btn_recalc_aba5"):
            with st.spinner("🔍 Localizando rotas nas ruas de BH..."):
                rotas_calculadas = buscar_rotas_com_transbordo(
                    st.session_state["clique_orig_lat"], st.session_state["clique_orig_lon"],
                    st.session_state["clique_dest_lat"], st.session_state["clique_dest_lon"],
                    routes, stops, trips, stop_times, top_k_pontos=15,
                    horario_partida="08:00",
                    calendar_df=calendar, calendar_dates_df=calendar_dates,
                    tipo_dia="📅 Dia Útil (Segunda a Sexta)"
                )
                st.session_state["clique_rotas_salvas"] = rotas_calculadas
                st.rerun()

    # Exibição dos Detalhes da Rota Encontrada
    if rotas_exibir:
        st.markdown("---")
        st.success(f"🎉 **{len(rotas_exibir)} Itinerários Encontrados em BH!**")

        melhor_r = rotas_exibir[0]
        pred_ml_aba5 = prever_impacto_transito_e_lotacao(
            horario_str="08:00",
            tipo_dia_str="Dia Útil",
            tempo_viagem_teorico_min=melhor_r['tempo_total_estimado_min'],
            condicao_climatica=clima_atual.get('descricao', '☀️ Ensolarado / Tempo Bom'),
            precipitacao_mm=clima_atual.get('precipitacao_mm', 0.0)
        )

        dados_tarifa_aba5 = calcular_tarifa_bhbus(melhor_r)
        dados_saude_aba5 = calcular_caminhada_e_saude(melhor_r)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Linha(s) Recomendadas", melhor_r['route_short_name'])
        m2.metric("Tempo Previsto (ML)", f"⏱️ ~{pred_ml_aba5['tempo_ajustado_min']} min", delta=f"+{pred_ml_aba5['atraso_min']} min atraso" if pred_ml_aba5['atraso_min'] > 0 else "Sem atrasos")
        m3.metric("Tarifa Estimada", dados_tarifa_aba5['tarifa_total_str'], delta="Tarifa Única" if melhor_r.get('transbordos', 0) == 0 else "Integração BHBUS")
        m4.metric("Lotação Prevista", pred_ml_aba5['lotacao_texto'].split(' ')[0] + " " + pred_ml_aba5['lotacao_texto'].split(' ')[1])
        m5.metric("Caminhada Total", f"🚶 {dados_saude_aba5['total_m']}m", delta=f"~{dados_saude_aba5['calorias_kcal']} kcal")

        st.info(f"{pred_ml_aba5['explicacao']}")

        # Detalhes passo a passo
        with st.expander("📋 Ver Instruções Passo a Passo do Itinerário", expanded=True):
            st.markdown(f"1. 🚶 Caminhe **{melhor_r['dist_caminhada_origem_m']}m** até o ponto **{melhor_r['ponto_origem_nome']}**.")
            if melhor_r.get('transbordos', 0) == 1:
                st.markdown(f"2. 🚌 Embarque no primeiro ônibus (**Linha {melhor_r.get('linha1_short', '1')}**).")
                st.markdown(f"3. 🔀 Desembarque no ponto de baldeação: **{melhor_r['ponto_troca_nome']}**.")
                st.markdown(f"4. 🚌 Embarque no segundo ônibus (**Linha {melhor_r.get('linha2_short', '2')}**).")
                st.markdown(f"5. 🚏 Desembarque no ponto **{melhor_r['ponto_destino_nome']}**.")
                st.markdown(f"6. 🚶 Caminhe **{melhor_r['dist_caminhada_destino_m']}m** até o seu destino final.")
            else:
                st.markdown(f"2. 🚌 Embarque no ônibus direto **Linha {melhor_r['route_short_name']}** ({melhor_r['route_long_name']}).")
                st.markdown(f"3. 🚏 Desembarque no ponto **{melhor_r['ponto_destino_nome']}**.")
                st.markdown(f"4. 🚶 Caminhe **{melhor_r['dist_caminhada_destino_m']}m** até o seu destino final.")
    elif st.session_state["clique_orig_lat"] and st.session_state["clique_dest_lat"]:
        st.warning("⚠️ Nenhuma linha de ônibus direta ou com 1 baldeação encontrada exatamente para essas duas coordenadas. Tente clicar em vias mais movimentadas ou consultar pelo nome do local na Aba 1.")


# ==============================================================================
# ABA 6: ACADEMIC & RESEARCH ANALYTICS (PAINEL CIENTÍFICO MESTRADO EUROPA)
# ==============================================================================
with tab6:
    st.subheader("🔬 Academic & Scientific Research Analytics")
    st.caption("This module presents the formal mathematical background, hypothesis testing framework, MLOps evaluation metrics, and spatio-temporal lag cross-correlation analysis for European Master's Application Portfolios (ETH Zürich, EPFL, TU Delft, Imperial College London, Karolinska Institutet).")

    # 1. MATHEMATICAL FORMULATION & LATEX EQUATIONS
    st.markdown("### 📐 1. Mathematical Formulation & Theoretical Rigor")
    
    col_math1, col_math2 = st.columns(2)
    with col_math1:
        with st.container(border=True):
            st.markdown("#### **Haversine Distance Vectorization**")
            st.latex(r"""
            d = 2 R \arcsin \left( \sqrt{ \sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right) } \right)
            """)
            st.caption(r"Where $R = 6371000m$, $\phi$ is latitude and $\lambda$ is longitude in radians.")

    with col_math2:
        with st.container(border=True):
            st.markdown("#### **Transit Reliability Index ($TRI$)**")
            st.latex(r"""
            TRI = \alpha \cdot \bar{D}_{delay} + \beta \cdot \sigma_{delay}^{1.5} + \gamma \cdot \Delta_{\text{Rain}}
            """)
            st.caption("Composite risk metric combining expected delay, variance hazard, and climate impact multiplier.")

    with st.container(border=True):
        st.markdown("#### **Spatio-Temporal Cross-Correlation Lag ($\tau$) & Hypothesis Test**")
        col_latex_a, col_latex_b = st.columns(2)
        with col_latex_a:
            st.latex(r"""
            r_{X,Y}(\tau) = \frac{\sum_{t=1}^{N-\tau} (X_t - \bar{X})(Y_{t+\tau} - \bar{Y})}{\sqrt{\sum_{t=1}^{N} (X_t - \bar{X})^2 \sum_{t=1}^{N} (Y_t - \bar{Y})^2}}
            """)
        with col_latex_b:
            st.latex(r"""
            t = \frac{\bar{d}_{delay} - \mu_0}{s_d / \sqrt{N}}, \quad p = P(T \ge t \mid H_0)
            """)

    # 2. HYPOTHESIS TESTING & STATISTICAL VERIFICATION
    st.markdown("---")
    st.markdown("### 🧪 2. Empirical Hypothesis Testing ($p$-value & Effect Size)")
    
    res_hyp = executar_teste_hipotese_atraso()
    
    c_stat1, c_stat2, c_stat3, c_stat4 = st.columns(4)
    with c_stat1:
        st.metric("Sample Size (N)", f"{res_hyp['n_samples']}")
    with c_stat2:
        st.metric("Student t-Stat", f"{res_hyp['t_statistic']:.3f}")
    with c_stat3:
        st.metric("p-value", f"{res_hyp['p_value_ttest']:.4e}")
    with c_stat4:
        st.metric("Cohen's d (Effect Size)", f"{res_hyp['cohens_d']:.2f}")

    if res_hyp['reject_h0']:
        st.success(f"✅ **Statistical Significance Confirmed ($\alpha = {res_hyp['alpha']}$):** {res_hyp['conclusion']}")
    else:
        st.warning(res_hyp['conclusion'])

    # 3. SPATIO-TEMPORAL LAG ANALYSIS
    st.markdown("---")
    st.markdown("### 📉 3. Spatio-Temporal Lag Correlation Analysis (Weather vs Delay)")
    
    lag_data = calcular_defasagem_spatiotemporal_clima(max_lag=6)
    df_lags = pd.DataFrame({
        'Lag Tau (hours)': lag_data['lags'],
        'Pearson Correlation (r)': lag_data['pearson_correlations'],
        'Spearman Rank Corr': lag_data['spearman_correlations']
    })
    
    col_chart, col_info = st.columns([2, 1])
    with col_chart:
        st.line_chart(df_lags.set_index('Lag Tau (hours)'))
    with col_info:
        with st.container(border=True):
            st.markdown(f"**Optimal Lag ($\tau^*$):** `{lag_data['optimal_lag_tau']} periods`")
            st.markdown(f"**Peak Pearson Correlation:** `{lag_data['max_pearson_r']:.3f}`")
            st.caption("Demonstrates that precipitation peak causes maximum traffic delay spikes with a temporal lag of 2 hours in BH transit corridors.")

    # 4. MLOPS & MODEL EVALUATION METRICS
    st.markdown("---")
    st.markdown("### ⚙️ 4. MLOps Performance Metrics (Scikit-Learn Evaluation)")
    
    mlops_metrics = avaliar_desempenho_modelos()
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Regressor R² Score", f"{mlops_metrics.get('regressor_r2', 'N/A')}")
    with m2:
        st.metric("Regressor MAE", f"{mlops_metrics.get('regressor_mae', 'N/A')}")
    with m3:
        st.metric("Regressor RMSE", f"{mlops_metrics.get('regressor_rmse', 'N/A')}")
    with m4:
        st.metric("Classifier F1-Score", f"{mlops_metrics.get('classifier_f1_score', 'N/A')}")

    # 5. EXPLAINABLE AI (XAI) & FEATURE IMPORTANCE
    st.markdown("---")
    st.markdown("### 🧠 5. Explainable AI (XAI) & Feature Importance")
    st.caption("Quantifies the mathematical influence of each feature on urban delay predictions (RandomForest Feature Importance).")
    
    importancias = obter_importancia_features()
    df_imp = pd.DataFrame(list(importancias["regressor"].items()), columns=["Feature", "Importância Relativa"]).sort_values("Importância Relativa", ascending=True)
    
    col_xai1, col_xai2 = st.columns([2, 1])
    with col_xai1:
        st.bar_chart(df_imp.set_index("Feature"))
    with col_xai2:
        with st.container(border=True):
            st.markdown("#### **Pesos Relativos (XAI)**")
            for feat, val in importancias["regressor"].items():
                st.markdown(f"- **{feat}:** `{val * 100:.1f}%`")
            st.caption("O horário do dia e o índice de precipitação pluviométrica em tempo real respondem por mais de 70% da variância de atrasos na malha viária de BH.")

    # 6. REST API & MICROSERVICES INTEGRATION
    st.markdown("---")
    st.markdown("### 🔌 6. Production REST API & OpenAPI Swagger Documentation")
    st.caption("A plataforma disponibiliza microsserviço RESTful em **FastAPI** com validação estrita via Pydantic e OpenAPI 3.0.")

    col_api1, col_api2 = st.columns([1, 1])
    with col_api1:
        with st.container(border=True):
            st.markdown("#### **Endpoints Disponíveis:**")
            st.markdown("""
            - 📄 **Interactive Swagger UI:** `http://localhost:8000/docs`
            - 📑 **ReDoc Documentation:** `http://localhost:8000/redoc`
            - 🟢 `GET /health` — Health Check & Integridade
            - 🌤️ `GET /api/v1/weather/current` — Clima BH em Tempo Real
            - 🤖 `POST /api/v1/predict/delay` — Inferência de Atraso e Lotação
            - 📊 `GET /api/v1/analytics/metrics` — Métricas MLOps & XAI
            - 📐 `POST /api/v1/analytics/tri` — Índice de Confiabilidade TRI
            """)
    with col_api2:
        with st.container(border=True):
            st.markdown("#### **Exemplo de Chamada (Python Requests):**")
            st.code("""import requests

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
print(resp.json())""", language="python")

    # 7. BIBTEX CITATION GENERATOR & CV INTEGRATION
    st.markdown("---")
    st.markdown("### 📜 7. Official BibTeX Citation Generator")
    st.markdown("Copy the BibTeX block below to cite this project in your **Statement of Purpose (SOP)**, **Europass CV**, or **Lattes Curriculum**:")
    
    bibtex_str = gerar_citacao_bibtex()
    st.code(bibtex_str, language="bibtex")


