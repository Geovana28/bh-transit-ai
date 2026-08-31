import re
from datetime import datetime, time
import sys
import os

sys.path.append(os.path.dirname(__file__))
from motor_rotas import (
    buscar_coordenadas_endereco, buscar_rotas_com_transbordo,
    calcular_tarifa_bhbus, calcular_caminhada_e_saude,
    verificar_estacao_metro_proxima, filtrar_e_ordenar_rotas
)
from motor_ia_ml import prever_impacto_transito_e_lotacao

LOCAIS_CONHECIDOS = {
    "bh shopping": "BH Shopping, Belo Horizonte",
    "belvedere": "BH Shopping, Belo Horizonte",
    "mercado central": "Mercado Central, Belo Horizonte",
    "mineirão": "Mineirão, Belo Horizonte",
    "mineirao": "Mineirão, Belo Horizonte",
    "pampulha": "Mineirão, Belo Horizonte",
    "praça sete": "Praça Sete, Belo Horizonte",
    "praca sete": "Praça Sete, Belo Horizonte",
    "centro": "Praça Sete, Belo Horizonte",
    "savassi": "Savassi, Belo Horizonte",
    "felício rocho": "Hospital Felício Rocho, Belo Horizonte",
    "felicio rocho": "Hospital Felício Rocho, Belo Horizonte",
    "estação barreiro": "Estação Barreiro, Belo Horizonte",
    "estacao barreiro": "Estação Barreiro, Belo Horizonte",
    "barreiro": "Estação Barreiro, Belo Horizonte",
    "estação venda nova": "Estação Venda Nova, Belo Horizonte",
    "estacao venda nova": "Estação Venda Nova, Belo Horizonte",
    "venda nova": "Estação Venda Nova, Belo Horizonte",
    "newton paiva": "Newton Paiva, Avenida Presidente Carlos Luz, Belo Horizonte",
    "newton paiva carlos luz": "Newton Paiva, Avenida Presidente Carlos Luz, Belo Horizonte",
    "newton paiva da carlos luz": "Newton Paiva, Avenida Presidente Carlos Luz, Belo Horizonte",
    "carlos luz": "Avenida Presidente Carlos Luz, Belo Horizonte",
    "santa ines": "Santa Inês, Belo Horizonte",
    "santa inês": "Santa Inês, Belo Horizonte",
    "ufmg": "UFMG - Universidade Federal de Minas Gerais, Belo Horizonte",
    "puc": "PUC Minas, Belo Horizonte",
    "gutierrez": "Gutierrez, Belo Horizonte",
    "buritis": "Buritis, Belo Horizonte",
    "castelo": "Castelo, Belo Horizonte",
    "padre eustáquio": "Padre Eustáquio, Belo Horizonte",
    "padre eustaquio": "Padre Eustáquio, Belo Horizonte",
    "sagrada família": "Sagrada Família, Belo Horizonte",
    "sagrada familia": "Sagrada Família, Belo Horizonte",
    "estação pampulha": "Estação Pampulha, Belo Horizonte",
    "estacao pampulha": "Estação Pampulha, Belo Horizonte",
    "estação vilarinho": "Estação Vilarinho, Belo Horizonte",
    "estacao vilarinho": "Estação Vilarinho, Belo Horizonte",
    "estação são gabriel": "Estação São Gabriel, Belo Horizonte",
    "estacao sao gabriel": "Estação São Gabriel, Belo Horizonte"
}

CORRECOES_TYPO = {
    "acrlos": "carlos",
    "satna": "santa",
    "ineis": "inês",
    "pvaiva": "paiva",
    "newto": "newton"
}

def corrigir_erros_digitacao(texto):
    words = texto.split()
    words_corrigidas = [CORRECOES_TYPO.get(w, w) for w in words]
    return " ".join(words_corrigidas)

def extrair_entidades_mensagem(texto_usuario):
    """
    Processamento de Linguagem Natural (NLP) avançado:
    Extrai Origem, Destino, Horário, Tipo de Dia, Clima e Preferência de Rota a partir do texto do usuário.
    """
    texto_limpo = corrigir_erros_digitacao(texto_usuario.lower())

    # 1. Extração de Clima
    clima = "☀️ Ensolarado / Tempo Bom"
    if any(c in texto_limpo for c in ["tempestade", "temporal", "chuva forte", "alagamento"]):
        clima = "⛈️ Chuva Forte / Temporal"
    elif any(c in texto_limpo for c in ["chuva", "chovendo", "garoa", "chuvisco"]):
        clima = "🌧️ Chuva Moderada"

    # 2. Extração de Preferência de Rota
    filtro = "⚡ Mais Rápida"
    if any(f in texto_limpo for f in ["menos caminhada", "pouca caminhada", "perto"]):
        filtro = "🚶 Menos Caminhada"
    elif any(f in texto_limpo for f in ["direto", "sem troca", "sem baldeação"]):
        filtro = "🚏 Linhas Diretas"
    elif any(f in texto_limpo for f in ["acessivel", "acessível", "cadeirante", "pcd"]):
        filtro = "♿ Acessível (PCD)"

    # 3. Extração de Horário
    now = datetime.now()
    if "agora" in texto_limpo or "neste momento" in texto_limpo:
        horario_str = now.strftime("%H:%M")
    else:
        horario_str = "08:00"
        match_hora = re.search(r'(\d{1,2})[h:]?(\d{2})?', texto_limpo)
        if match_hora:
            h = int(match_hora.group(1))
            m = int(match_hora.group(2)) if match_hora.group(2) else 0
            if ("tarde" in texto_limpo or "noite" in texto_limpo) and h < 12:
                h += 12
            horario_str = f"{h:02d}:{m:02d}"

    # 4. Extração do Tipo de Dia
    if "sábado" in texto_limpo or "sabado" in texto_limpo:
        tipo_dia = "🟡 Sábado"
    elif "domingo" in texto_limpo or "feriado" in texto_limpo:
        tipo_dia = "🔴 Domingo / Feriado"
    else:
        if "agora" in texto_limpo:
            w = now.weekday()
            tipo_dia = "🟡 Sábado" if w == 5 else ("🔴 Domingo / Feriado" if w == 6 else "📅 Dia Útil (Segunda a Sexta)")
        else:
            tipo_dia = "📅 Dia Útil (Segunda a Sexta)"

    # 5. Extração Inteligente de Origem e Destino
    origem_texto = None
    destino_texto = None

    locais_encontrados = []
    chaves_ordenadas = sorted(LOCAIS_CONHECIDOS.keys(), key=len, reverse=True)
    
    intervalos_ocupados = []
    for chave in chaves_ordenadas:
        pos = texto_limpo.find(chave)
        if pos != -1:
            fim = pos + len(chave)
            if not any(not (fim <= start or pos >= end) for start, end in intervalos_ocupados):
                intervalos_ocupados.append((pos, fim))
                locais_encontrados.append((pos, LOCAIS_CONHECIDOS[chave], chave))

    locais_encontrados.sort(key=lambda x: x[0])

    if len(locais_encontrados) >= 2:
        origem_texto = locais_encontrados[0][1]
        destino_texto = locais_encontrados[1][1]
    elif len(locais_encontrados) == 1:
        pos, nome_completo, chave = locais_encontrados[0]
        trecho_antes = texto_limpo[:pos]
        if any(p in trecho_antes for p in ["para", "pro", "pra", "até", "ate"]):
            destino_texto = nome_completo
        else:
            origem_texto = nome_completo

    if not origem_texto:
        match_orig = re.search(r'(?:to|tô|estou|estou na|estou no|to na|tô na|to no|tô no|saindo da|saindo do|embarque na|embarque no)\s+(.+?)(?:,|\.|\s+como|\s+e|\s+pra|\s+pro|\s+para|\s+vou|\s+preciso|$)', texto_limpo)
        if match_orig:
            cand_o = match_orig.group(1).strip()
            cand_o = re.sub(r'^(aqui|ali)\s+(na|no|em)\s+', '', cand_o)
            cand_o = re.sub(r'^(na|no|em)\s+', '', cand_o)
            origem_texto = LOCAIS_CONHECIDOS.get(cand_o, cand_o + ", Belo Horizonte")

    if not destino_texto:
        match_dest = re.search(r'(?:para|pro|pra|até|ate)\s+(?:o|a|os|as)?\s*(.+?)(?:\s+agora|\s+às|\s+as|\s+hoje|\?|\.|$)', texto_limpo)
        if match_dest:
            cand_d = match_dest.group(1).strip()
            cand_d = re.sub(r'^(ir|chegar)\s+(para|pro|pra|até|ate)\s+', '', cand_d)
            destino_texto = LOCAIS_CONHECIDOS.get(cand_d, cand_d + ", Belo Horizonte")

    if not origem_texto:
        origem_texto = "BH Shopping, Belo Horizonte"
    if not destino_texto:
        destino_texto = "Mercado Central, Belo Horizonte"

    return origem_texto, destino_texto, horario_str, tipo_dia, clima, filtro


def processar_consulta_assistente_ia(mensagem_usuario, routes, stops, trips, stop_times, calendar, calendar_dates):
    """
    Agente Conversacional Completo com suporte a Tarifa BHBUS (R$ 6,25 / R$ 6,00), Metrô, Clima e Saúde.
    """
    orig_txt, dest_txt, horario_str, tipo_dia, clima_txt, filtro_txt = extrair_entidades_mensagem(mensagem_usuario)

    lat_o, lon_o, end_o = buscar_coordenadas_endereco(orig_txt)
    lat_d, lon_d, end_d = buscar_coordenadas_endereco(dest_txt)

    if lat_o is None or lat_d is None:
        return (
            "🤖 **Assistente IA:** Desculpe, não consegui localizar os dois endereços exatos em BH. "
            "Tente digitar locais conhecidos como 'BH Shopping', 'Mercado Central', 'Savassi' ou 'Mineirão'.",
            None, None
        )

    rotas_brutas = buscar_rotas_com_transbordo(
        lat_o, lon_o, lat_d, lon_d,
        routes, stops, trips, stop_times, top_k_pontos=20,
        horario_partida=horario_str,
        calendar_df=calendar, calendar_dates_df=calendar_dates,
        tipo_dia=tipo_dia
    )

    if not rotas_brutas:
        return (
            f"🤖 **Assistente IA:** Analisei o quadro de horários do GTFS para às **{horario_str}** ({tipo_dia}), "
            f"mas não encontrei rotas diretas ou com 1 baldeação entre **{orig_txt}** e **{dest_txt}** nesse horário.",
            None, None
        )

    rotas = filtrar_e_ordenar_rotas(rotas_brutas, filtro_txt)
    melhor_rota = rotas[0]

    # --- INFERÊNCIA DE MACHINE LEARNING COM FATOR CLIMA ---
    pred_ml = prever_impacto_transito_e_lotacao(
        horario_str=horario_str,
        tipo_dia_str=tipo_dia,
        tempo_viagem_teorico_min=melhor_rota['tempo_total_estimado_min'],
        condicao_climatica=clima_txt
    )

    # --- TARIFA BHBUS E ECONOMIA ---
    dados_tarifa = calcular_tarifa_bhbus(melhor_rota)
    dados_saude = calcular_caminhada_e_saude(melhor_rota)

    # --- VERIFICAÇÃO DE CONEXÃO COM O METRÔ DE BH ---
    metro_orig = verificar_estacao_metro_proxima(lat_o, lon_o)
    metro_dest = verificar_estacao_metro_proxima(lat_d, lon_d)
    metro_info = ""
    if metro_orig or metro_dest:
        metro_info = "\n🚆 **Conexão com Metrô de BH:**\n"
        if metro_orig:
            metro_info += f" • Próximo da Origem: **{metro_orig['nome']}** (~{metro_orig['distancia_m']}m / {metro_orig['tempo_caminhada_min']} min a pé)\n"
        if metro_dest:
            metro_info += f" • Próximo do Destino: **{metro_dest['nome']}** (~{metro_dest['distancia_m']}m / {metro_dest['tempo_caminhada_min']} min a pé)\n"

    # --- SUSTENTABILIDADE (ESG) ---
    dist_km_est = max(2.5, melhor_rota['tempo_total_estimado_min'] * 0.35)
    co2_carro = dist_km_est * 0.130
    co2_onibus = dist_km_est * 0.035
    co2_poupad_kg = round(co2_carro - co2_onibus, 2)

    # Texto para compartilhamento no WhatsApp
    txt_whatsapp = f"🚌 Rota BH Transit AI: Ir de {orig_txt} ate {dest_txt} com a Linha {melhor_rota['route_short_name']}. Tempo est: {pred_ml['tempo_ajustado_min']} min. Tarifa BHBUS: {dados_tarifa['tarifa_total_str']}."
    from urllib.parse import quote
    url_whatsapp = f"https://api.whatsapp.com/send?text={quote(txt_whatsapp)}"

    resposta_texto = (
        f"🤖 **Assistente IA de Mobilidade:**\n\n"
        f"Entendi! Para ir de **{end_o or orig_txt}** até **{end_d or dest_txt}** às **{horario_str}**:\n\n"
        f"🚌 **Linha Recomendada:** `{melhor_rota['route_short_name']}` ({melhor_rota['tipo']})\n"
        f"🛫 **Embarque:** {melhor_rota['ponto_origem_nome']} (Saída prevista: **{melhor_rota.get('horario_saida', horario_str)}**)\n"
        f"🛬 **Desembarque:** {melhor_rota['ponto_destino_nome']}\n\n"
        f"💳 **Tarifa BHBUS:** **{dados_tarifa['tarifa_total_str']}** ({dados_tarifa['descricao_tarifa']})\n"
        f"⏱️ **Tempo Previsto com IA (ML):** ~{pred_ml['tempo_ajustado_min']} min (Tabela: ~{melhor_rota['tempo_total_estimado_min']} min)\n"
        f"👥 **Lotação Estimada:** {pred_ml['lotacao_texto']}\n"
        f"🚶 **Caminhada Total A Pé:** ~{dados_saude['total_m']}m ({dados_saude['tempo_pe_min']} min - queima est. ~{dados_saude['calorias_kcal']} kcal)\n"
        f"{metro_info}\n"
        f"{pred_ml['explicacao']}\n\n"
        f"💰 **Economia Nesta Viagem:** Salva **R$ {dados_tarifa['economia_uber']:.2f}** vs Uber e **R$ {dados_tarifa['economia_carro']:.2f}** vs Carro Próprio!\n"
        f"🌱 **Impacto ESG:** Evita **~{co2_poupad_kg} kg de CO₂** na atmosfera! 🌍✨\n\n"
        f"📱 [📲 **Compartilhar Rota no WhatsApp**]({url_whatsapp})\n\n"
        f"👇 *Confira o mapa interativo do trajeto abaixo:*"
    )

    dados_mapa = {
        'rota': melhor_rota,
        'lat_o': lat_o, 'lon_o': lon_o,
        'lat_d': lat_d, 'lon_d': lon_d
    }

    return resposta_texto, dados_mapa, rotas

