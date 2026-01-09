import streamlit as st
import os
from lofi_pipeline import LofiPipeline
import time
import traceback

st.set_page_config(
    page_title="Lofi Generator",
    page_icon="🎵",
    layout="centered"
)

st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.2em;
        padding: 0.75em;
        border-radius: 50px;
        border: none;
        font-weight: 600;
    }
    h1 {
        text-align: center;
        color: white;
        font-size: 3em;
    }
    .subtitle {
        text-align: center;
        color: white;
        opacity: 0.9;
        margin-bottom: 2em;
    }
    </style>
""", unsafe_allow_html=True)

if 'generated_file' not in st.session_state:
    st.session_state.generated_file = None
if 'generating' not in st.session_state:
    st.session_state.generating = False

@st.cache_resource
def load_pipeline():
    return LofiPipeline(
        weights_file='weights/full_model.hdf5',
        soundfont='sf/piano.sf2',
        vinyl_sfx='./sfx/vinyl_crackle_loop.wav',
        drums_sfx='./sfx/drums.wav',
        rain_sfx='./sfx/rain1.wav'
    )

pipeline = load_pipeline()


st.markdown("<h1>🎵 Lofi Generator</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-powered lofi hip-hop music</p>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    with st.expander("Advanced Settings"):
        length = st.slider("Track Length (notes)", 100, 500, 200, 50)
        top_k = st.slider("Creativity (top_k)", 3, 10, 6, 1)
        temperature = st.slider("Randomness (temperature)", 0.5, 1.5, 0.85, 0.05)
    
    st.write("")
    
    if st.button(" Generate Lofi Track", disabled=st.session_state.generating):
        st.session_state.generating = True
        st.session_state.generated_file = None
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            os.makedirs('outputs', exist_ok=True)
            output_file = f"outputs/lofi_{int(time.time())}.wav"
            
            status_text.text(" Step 1/5: Generating MIDI...")
            progress_bar.progress(20)
            midi_file = pipeline.generate_midi(length, top_k, temperature)
            
            status_text.text(" Step 2/5: Rendering to WAV...")
            progress_bar.progress(40)
            raw_wav = pipeline.midi_to_wav(midi_file)
            
            status_text.text(" Step 3/5: Softening audio...")
            progress_bar.progress(60)
            soft_wav = pipeline.soften_audio(raw_wav)
            
            status_text.text(" Step 4/5: Applying lofi effects...")
            progress_bar.progress(80)
            lofi_wav = pipeline.apply_lofi_fx(soft_wav)
            
            status_text.text(" Step 5/5: Mixing final track...")
            progress_bar.progress(90)
            final_file = pipeline.mix_final(lofi_wav, output_file)
            
            pipeline.cleanup_temp_files()
            
            progress_bar.progress(100)
            status_text.text("Generation complete!")
            
            st.session_state.generated_file = final_file
            st.success("Your lofi track is ready!")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.code(traceback.format_exc())
        
        finally:
            st.session_state.generating = False
            time.sleep(1)
            st.rerun()
    
    st.write("")
    
    if st.session_state.generated_file and os.path.exists(st.session_state.generated_file):
        st.markdown("---")
        st.markdown("### Your Track")
        
        with open(st.session_state.generated_file, 'rb') as audio_file:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav')
        
        st.download_button(
            label="Download Track",
            data=audio_bytes,
            file_name="lofi_track.wav",
            mime="audio/wav",
            use_container_width=True
        )
        
        if st.button("Generate Another", use_container_width=True):
            st.session_state.generated_file = None
            st.rerun()

