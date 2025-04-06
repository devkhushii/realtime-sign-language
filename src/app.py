import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
import tempfile
import threading
import queue

SAMPLE_RATE = 16000
DURATION = 4
CHUNK = int(SAMPLE_RATE * DURATION)

# List input devices
devices = sd.query_devices()
input_devices = {i: d['name'] for i, d in enumerate(devices) if d['max_input_channels'] > 0}

st.title("🎙️ Real-time Voice to Text")
st.markdown("Choose your microphone and start speaking!")

device_index = st.selectbox("🎤 Select your microphone:", list(input_devices.keys()), format_func=lambda x: input_devices[x])

# Load Whisper
@st.cache_resource
def load_model():
    return WhisperModel("base.en", device="cpu")

model = load_model()

# Audio recording thread
audio_queue = queue.Queue()

def record_audio(device):
    while True:
        audio = sd.rec(CHUNK, samplerate=SAMPLE_RATE, channels=1, dtype='int16', device=device)
        sd.wait()
        audio_queue.put(audio.copy())

if st.button("Start Listening"):
    if 'thread_started' not in st.session_state:
        threading.Thread(target=record_audio, args=(device_index,), daemon=True).start()
        st.session_state.thread_started = True

    placeholder = st.empty()
    full_transcription = ""

    while True:
        try:
            audio = audio_queue.get(timeout=15)
        except queue.Empty:
            st.warning("No audio captured!")
            break

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            write(f.name, SAMPLE_RATE, audio)
            segments, _ = model.transcribe(f.name)
            text = " ".join(segment.text for segment in segments).strip()

            if text:
                full_transcription += text + " "
                placeholder.markdown(f"**📝 Transcription:** {full_transcription}")






