import streamlit as st
import librosa
import numpy as np
import yt_dlp
import os

# --- CONFIGURAÇÃO INICIAL ---
st.set_page_config(page_title="Detector de Tom", page_icon="🎸")

if 'fonte_ativa' not in st.session_state:
    st.session_state['fonte_ativa'] = None
if 'arquivo_atual' not in st.session_state:
    st.session_state['arquivo_atual'] = None

# --- 1. LÓGICA DE ANÁLISE ---
def identificar_tom_avancado(caminho_arquivo, filtrar_graves=False):
    # Carrega o áudio
    y, sr = librosa.load(caminho_arquivo, sr=22050, duration=60)
    
    # 1. LIMPEZA DE SILÊNCIO
    y, _ = librosa.effects.trim(y, top_db=25)
    if len(y) < sr: return None

    # 2. FILTRO DE PALCO
    fmin_val = librosa.note_to_hz('C2') # C2 = 65Hz (Padrão)
    if filtrar_graves:
        fmin_val = librosa.note_to_hz('C3') # C3 = 130Hz 

    # 3. SEPARAÇÃO HARMÔNICA (Tira a percussão)
    y_harmonic, _ = librosa.effects.hpss(y)
    
    # 4. CHROMAGRAMA COM FILTRO DE FREQUÊNCIA
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, fmin=fmin_val, n_octaves=5)
    
    chroma_vals = np.sum(chroma, axis=1)
    
    major_profile = [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 5.0, 2.0, 3.5, 1.5, 4.0]
    minor_profile = [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 5.0, 3.5, 2.0, 1.5, 4.0]
    
    notas = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    correlacoes = []
    
    for i in range(12):
        p_major = np.roll(major_profile, i)
        p_minor = np.roll(minor_profile, i)
        corr_major = np.corrcoef(chroma_vals, p_major)[0, 1]
        corr_minor = np.corrcoef(chroma_vals, p_minor)[0, 1]
        correlacoes.append({'nota': notas[i], 'modo': 'Major', 'score': corr_major})
        correlacoes.append({'nota': notas[i], 'modo': 'Minor', 'score': corr_minor})
    
    correlacoes.sort(key=lambda x: x['score'], reverse=True)
    return correlacoes[0]

# --- 2. DOWNLOADER ---
def baixar_audio_youtube(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'temp_yt.%(ext)s',
        'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'mp3','preferredquality': '192'}],
        'quiet': True,
        'nocheckcertificate': True,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        },
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'web'],
            }
        }
    }
    
    try:
        if os.path.exists("temp_yt.mp3"): 
            os.remove("temp_yt.mp3")
            
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            return "temp_yt.mp3"
            
    except Exception as e:
        st.error(f"Erro no download (YouTube bloqueou): {e}")
        return None

# --- 3. INTERFACE ---
st.title("🎸 Detector de Tom: Stage Mode")

modo_palco = st.checkbox("🎤 Modo Palco (Filtrar vazamento de som grave e bateria)", value=False)

tab1, tab2 = st.tabs(["YouTube", "Gravação/Arquivo"])

with tab1:
    url = st.text_input("Link do vídeo:")
    if st.button("Processar YouTube"):
        with st.spinner("Baixando..."):
            path = baixar_audio_youtube(url)
            if path:
                # Atualiza o estado explicitamente
                st.session_state['arquivo_atual'] = path
                st.session_state['fonte_ativa'] = 'youtube'
                st.rerun() # Recarrega a página para mostrar o resultado limpo

with tab2:
    gravacao = st.audio_input("Gravar Voz")
    uploaded = st.file_uploader("Upload MP3/WAV", type=['mp3','wav','ogg'])
    
    if gravacao:
        with open("temp_rec.wav", "wb") as f:
            f.write(gravacao.read())
        
        if st.button("Analisar Gravação"):
            st.session_state['arquivo_atual'] = "temp_rec.wav"
            st.session_state['fonte_ativa'] = 'rec'
            st.rerun()

    elif uploaded:
        with open("temp_up.mp3", "wb") as f:
            f.write(uploaded.getbuffer())
        if st.button("Analisar Arquivo"):
            st.session_state['arquivo_atual'] = "temp_up.mp3"
            st.session_state['fonte_ativa'] = 'upload'
            st.rerun()

if st.session_state['arquivo_atual'] and os.path.exists(st.session_state['arquivo_atual']):
    
    st.divider()
    st.write(f"📂 Fonte: **{st.session_state['fonte_ativa'].upper()}**")
    
    with st.spinner("Calculando harmonia..."):
        try:
            resultado = identificar_tom_avancado(
                st.session_state['arquivo_atual'], 
                filtrar_graves=modo_palco
            )
            
            if resultado:
                cor_fundo = "#d4edda" if resultado['score'] > 0.5 else "#fff3cd"
                cor_texto = "#155724" if resultado['score'] > 0.5 else "#856404"
                
                st.markdown(f"""
                <div style="background-color: {cor_fundo}; padding: 30px; border-radius: 15px; text-align: center; border: 2px solid {cor_texto}">
                    <h2 style="margin:0; color: #333; font-size: 20px">O tom provável é</h2>
                    <h1 style="margin:10px 0; color: {cor_texto}; font-size: 60px; font-weight: bold">{resultado['nota']} {resultado['modo']}</h1>
                    <p style="margin:0; color: #555">Certeza: {int(resultado['score']*100)}%</p>
                </div>
                """, unsafe_allow_html=True)

                if modo_palco:
                    st.info("🎙️ Modo Palco Ativo: Frequências graves ignoradas para focar na voz.")
                
            else:
                st.error("Áudio muito curto ou inválido.")
                
        except Exception as e:
            st.error(f"Erro na leitura: {e}")
