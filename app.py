"""
Dashboard Analítico Nheengatu-Uirapuru.

Interface gráfica avançada baseada em Inteligência Artificial Explicável (XAI).
Utiliza Streamlit com renderização nativa de estilos 'Mesh Gradient' e 
'Glass Morphism', servindo como ambiente visual para projeção hiperdimensional 
e mapeamento de similaridade de cosseno em tempo real.
"""

import os
import base64
import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from sklearn.decomposition import PCA
from src.inference_engine import SemanticExplorerEngine

# 1. Configuração de Página e Metadados
st.set_page_config(page_title="Nheengatu-Uirapuru", page_icon="🦜", layout="wide")

# Paleta de Cores Estrita da Identidade Uirapuru
COR_PRIMARIA = "#E65100"
COR_SECUNDARIA = "#00838F"

# 2. Motor de Renderização UI/UX (CSS Híbrido: Mesh Gradient & Glass Morphism)
st.markdown("""
<style>
    /* Fundo Mesh Gradient Vibrante (Estilo SaaS Moderno) */
    [data-testid="stAppViewContainer"] {
        background-color: #F0F4F8 !important;
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(230, 81, 0, 0.25) 0%, transparent 45%),
            radial-gradient(circle at 90% 30%, rgba(0, 131, 143, 0.25) 0%, transparent 45%),
            radial-gradient(circle at 50% 90%, rgba(255, 183, 77, 0.20) 0%, transparent 50%) !important;
        background-attachment: fixed !important;
    }
    
    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Placas de Vidro Fosco Universais */
    .glass-panel {
        background: rgba(255, 255, 255, 0.65) !important;
        backdrop-filter: blur(20px) saturate(150%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(150%) !important;
        border: 1px solid rgba(255, 255, 255, 0.8) !important;
        border-radius: 24px;
        padding: 32px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07) !important;
        margin-bottom: 24px;
    }

    /* Estilização do Hero Card (Top-1) */
    .glass-hero {
        text-align: center;
    }
    .hero-subtitle {
        color: #6c757d;
        font-size: 14px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }
    .hero-title {
        color: #E65100;
        font-size: 52px;
        font-weight: 900;
        margin: 0 0 16px 0;
        letter-spacing: -1px;
        text-shadow: 2px 2px 4px rgba(230, 81, 0, 0.15);
    }

    /* Minicartões (Top-2 ao Top-5) */
    .glass-minicard {
        background: rgba(255, 255, 255, 0.55) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.7) !important;
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 4px 16px 0 rgba(31, 38, 135, 0.05) !important;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-minicard:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 24px 0 rgba(31, 38, 135, 0.1) !important;
    }
    .card-header {
        color: #2B3674;
        font-size: 20px;
        font-weight: 800;
        border-bottom: 1px solid rgba(0,0,0,0.08);
        padding-bottom: 12px;
        margin-bottom: 16px;
    }
    .mini-metric-value {
        color: #2B3674;
        font-size: 26px;
        font-weight: 900;
        margin-bottom: 12px;
    }

    /* Botão Principal Estilizado */
    div[data-testid="stButton"] > button {
        background-color: #E65100 !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 10px 24px !important;
        font-weight: 700 !important;
        box-shadow: 0px 4px 16px rgba(230, 81, 0, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0px 6px 20px rgba(230, 81, 0, 0.4) !important;
    }
</style>
""", unsafe_allow_html=True)

def get_image_as_base64(path):
    """Lê um arquivo de imagem em formato binário e o encoda em Base64 para injeção HTML."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# 3. Alocação do Motor de Inferência (Preservação de Cache em RAM)
@st.cache_resource(show_spinner="Carregando matrizes de rotação e redes neurais...")
def carregar_motor():
    return SemanticExplorerEngine()

try:
    engine = carregar_motor()
except Exception as e:
    st.error(f"Erro fatal ao instanciar as redes neurais: {str(e)}")
    st.stop()

# 4. Cabeçalho UI/UX (Renderização Unificada com Injeção Dinâmica de Imagem B64)
try:
    img_path = os.path.join(os.path.dirname(__file__), "outputs", "22-UIRAPURU.jpg")
    img_b64 = get_image_as_base64(img_path)
    img_html = f'<img src="data:image/jpeg;base64,{img_b64}" style="width: 160px; border-radius: 16px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">'
except FileNotFoundError:
    img_html = '<div style="font-size: 80px; line-height: 1;">🦜</div>'

header_html = f"""
<div class="glass-panel" style="display: flex; align-items: center; gap: 30px; margin-top: 10px; padding: 24px 32px;">
    {img_html}
    <div>
        <h1 style='color: #2B3674; margin:0; font-weight: 900; font-size: 2.6rem; letter-spacing: -1px;'>Explorador Semântico Nheengatu-Português</h1>
        <p style='color: #00838F; font-weight: 600; font-size: 1.15rem; margin: 8px 0 0 0;'>Dashboard Analítico de Alinhamento Hiperdimensional | IA Explicável (XAI)</p>
    </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# 5. Pipeline de Interação de Entrada (Camada de Usuário)
col_dir, col_busca, col_btn = st.columns([2, 3, 1])
with col_dir:
    direcao_ui = st.radio("Fluxo de Alinhamento:", ["Português ➔ Nheengatu", "Nheengatu ➔ Português"], horizontal=True)
with col_busca:
    direcao_backend = "pt_to_yrl" if direcao_ui == "Português ➔ Nheengatu" else "yrl_to_pt"
    idioma = "Português" if direcao_backend == "pt_to_yrl" else "Nheengatu"
    termo_busca = st.text_input(f"Buscar termo em {idioma}:", placeholder="Ex: floresta, kaá...")
with col_btn:
    st.write("")
    st.write("")
    btn_buscar = st.button("Mapear Vetor", use_container_width=True)

st.write("")

# 6. Orquestração da Interface de Resultados e Explicabilidade
if btn_buscar and termo_busca:
    with st.spinner("Computando álgebra linear e rotacionando hiperplanos..."):
        try:
            relatorio = engine.search(query=termo_busca, direction=direcao_backend, top_k=5)
            resultados = relatorio["top_k_resultados"]
            top_1 = resultados[0]

            tab_geral, tab_topologia = st.tabs(["📊 Visão Geral", "🌌 Diagnóstico Topológico e XAI"])

            # ==========================================
            # ABA 1: COMPREENSÃO GERAL E ALERTAS XAI
            # ==========================================
            with tab_geral:
                st.write("")
                # Gatilho Funcional: Detecção de Zero-Shot Inference
                if top_1["score_cosseno"] < 0.45:
                    st.warning(f"⚠️ **Interpolação Zero-Shot:** A palavra '{termo_busca}' não integra a variedade topológica de treinamento. A coordenada de destino foi deduzida por gravidade semântica (vizinhança latente).")
                
                # Renderização: Melhor Correspondência Matemática
                st.markdown(f"""
                <div class="glass-panel glass-hero">
                    <div class="hero-subtitle">Tradução Alinhada de Alta Fidelidade</div>
                    <div class="hero-title">{top_1['palavra']}</div>
                    <div class="hero-metrics">
                        Similaridade de Cosseno: <span style="color: #00838F;">{top_1['score_cosseno']:.3f}</span> 
                        &nbsp;&nbsp;•&nbsp;&nbsp; 
                        Confiança CSLS: <span style="color: #00838F;">{top_1['score_csls']:.3f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Renderização: Cinturão Analítico da Vizinhança
                cols = st.columns(4)
                for i in range(1, 5):
                    with cols[i-1]:
                        res = resultados[i]
                        st.markdown(f"""
                        <div class="glass-minicard">
                            <div class="card-header">{i+1}º: {res['palavra']}</div>
                            <div>
                                <div style="font-size: 12px; color: #6c757d; text-transform: uppercase; font-weight: 800; letter-spacing: 1px;">Cosseno</div>
                                <div class="mini-metric-value">{res['score_cosseno']:.3f}</div>
                                <div style="font-size: 12px; color: #6c757d; text-transform: uppercase; font-weight: 800; letter-spacing: 1px;">CSLS</div>
                                <div class="mini-metric-value" style="color: {'#E65100' if res['score_csls'] < 0 else '#00838F'};">{res['score_csls']:.3f}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                st.write("") 
                # Gatilho Funcional: Detecção da falha de isolamento (Efeito Hubness)
                if any(r["palavra"] == "mu-kuara" for r in resultados) or (resultados[0]["score_cosseno"] - resultados[-1]["score_cosseno"] < 0.05):
                    st.info("💡 **Justificativa Metodológica (Hubness):** Notou deduções não-intuitivas convergindo? Este fenômeno chama-se *Hubness* (Efeito de Concentração). O modelo posicionou verbos específicos no centro absoluto do espaço vetorial, fazendo com que atuem como 'ímãs' para vetores não-mapeados.")

            # ==========================================
            # ABA 2: AVALIAÇÃO DIAGNÓSTICA E GEOMÉTRICA
            # ==========================================
            with tab_topologia:
                st.write("")
                # Cabeçalho Explicativo: PCA Local
                st.markdown("""
                <div class='glass-panel' style='padding: 24px 32px;'>
                    <h3 style='color: #2B3674; margin-top: 0; margin-bottom: 12px;'>Microestrutura Vetorial (PCA Dinâmico)</h3>
                    <p style='color: #555; margin-bottom: 0; font-size: 1.05rem;'>A redução espacial demonstra a geometria local gerada pela Matriz ortogonal W. Os pontos refletem o distanciamento exato entre as coordenadas hiperdimensionais.</p>
                </div>
                """, unsafe_allow_html=True)
                
                labels = [f"Origem: {termo_busca}"] + [f"Vizinho: {r['palavra']}" for r in resultados]
                tipos = ["Origem"] + ["Tradução"] * len(resultados)
                vetores = [relatorio["metadados"]["tensor_query"]] + [r["vetor_raw"] for r in resultados]
                
                # Conversão e Compressão Visual
                pca = PCA(n_components=2)
                componentes_2d = pca.fit_transform(vetores)
                df_pca = pd.DataFrame({
                    "Componente 1": componentes_2d[:, 0],
                    "Componente 2": componentes_2d[:, 1],
                    "Palavra": labels,
                    "Tipo": tipos
                })

                # Parametrização Gráfica via Plotly Engine
                fig_pca = px.scatter(
                    df_pca, x="Componente 1", y="Componente 2", text="Palavra", color="Tipo",
                    color_discrete_map={"Origem": COR_PRIMARIA, "Tradução": COR_SECUNDARIA}
                )
                fig_pca.update_traces(textposition='top center', marker=dict(size=16, line=dict(width=2, color='white')))
                
                # Sobreposição Dinâmica (Gráfico Invisível + Fundo da Dashboard)
                fig_pca.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=20, b=0),
                    xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.08)', zeroline=False),
                    yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.08)', zeroline=False)
                )
                
                st.plotly_chart(fig_pca, use_container_width=True)

                st.write("")

                # Cabeçalho Explicativo: t-SNE Global
                st.markdown("""
                <div class='glass-panel' style='padding: 24px 32px;'>
                    <h3 style='color: #2B3674; margin-top: 0; margin-bottom: 12px;'>Mapeamento Global (t-SNE)</h3>
                    <p style='color: #555; margin-bottom: 0; font-size: 1.05rem;'>Devido à incompressibilidade algorítmica do t-SNE (que não permite adição de novos pontos em tempo real sem retreinamento da <i>manifold</i>), a topologia abaixo atua como o mapa cartográfico referencial estático dos clusters pré-computados.</p>
                </div>
                """, unsafe_allow_html=True)
                
                try:
                    tsne_path = os.path.join(os.path.dirname(__file__), "outputs", "tsne_semantic_clusters.png")
                    # Espaçamento estético de leitura focal
                    col_vazia_esq, col_img_centro, col_vazia_dir = st.columns([1, 4, 1])
                    with col_img_centro:
                        st.image(Image.open(tsne_path), use_container_width=True)
                except FileNotFoundError:
                    st.warning("Imagem cartográfica estática não identificada na varredura local. [outputs/tsne_semantic_clusters.png]")

            # ==========================================
            # RODAPÉ TÉCNICO (Transparência Operacional)
            # ==========================================
            st.write("")
            with st.expander("⚙️ Ver cálculos em segundo plano (Auditoria Técnica)"):
                st.markdown(f"**Latência de Inferência:** `{relatorio['tempo_inferencia_ms']} ms`")
                st.markdown("**Atuação do Filtro CSLS:** A similaridade euclidiana mensura a proximidade pura. O CSLS aplica penalidades topológicas em áreas hiper-densas para garantir o isolamento semântico.")
                tensor_preview = relatorio["metadados"]["tensor_query"][:6]
                st.code(f"# Tensor Rotacionado Resultante (Amostra D[0:6] de 768)\n{tensor_preview} ...", language="python")

            # ==========================================
            # RODAPÉ (Créditos e Direitos)
            # ==========================================
            st.markdown("""
            <hr style="margin-top: 40px; margin-bottom: 20px; border: 0; border-top: 1px solid rgba(0,0,0,0.08);">
            <div style="text-align: center; color: #6c757d; font-size: 0.85rem; line-height: 1.5;">
                🦜 <b>Créditos da Identidade Visual:</b> Obra "Uirapuru" pertencente à <b>Associação Cultural Pintura Solidária</b>.<br>
                Conheça mais sobre a iniciativa em: <a href="https://pinturasolidaria.org.br/" target="_blank" style="color: #00838F; text-decoration: none;"><b>pinturasolidaria.org.br</b></a>
            </div>
            <br>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Erro fatal na topologia matricial: {str(e)}")