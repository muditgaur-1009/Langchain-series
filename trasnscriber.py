import streamlit as st
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import threading
import queue

# Global variables
audio_queue = queue.Queue()
text_queue = queue.Queue()
stop_transcription = threading.Event()

# Set up the Whisper model
@st.cache_resource
def load_model():
    return WhisperModel("small", device="cpu", compute_type="int8")

model = load_model()

# Function to capture audio
def capture_audio(duration=5, sample_rate=16000):
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    return audio.flatten()

# Function to transcribe audio
def transcribe_audio():
    global stop_transcription
    while not stop_transcription.is_set():
        try:
            audio = audio_queue.get(timeout=1)
            segments, _ = model.transcribe(audio, beam_size=5)
            for segment in segments:
                text_queue.put(segment.text)
        except queue.Empty:
            continue

# Streamlit app
def main():
    st.title("Real-time Audio Transcription")

    if 'transcribing' not in st.session_state:
        st.session_state.transcribing = False
    if 'transcription_text' not in st.session_state:
        st.session_state.transcription_text = ""

    if st.button("Start Transcription" if not st.session_state.transcribing else "Stop Transcription"):
        st.session_state.transcribing = not st.session_state.transcribing
        if st.session_state.transcribing:
            stop_transcription.clear()
            # Start the transcription thread
            threading.Thread(target=transcribe_audio, daemon=True).start()
        else:
            stop_transcription.set()
            # Clear the transcription when stopping
            st.session_state.transcription_text = ""

    transcription = st.empty()

    # Capture and process audio
    if st.session_state.transcribing:
        audio = capture_audio()
        audio_queue.put(audio.astype(np.float32))
        
    # Update transcription text
    while not text_queue.empty():
        text = text_queue.get()
        st.session_state.transcription_text += " " + text
    
    # Display the current transcription
    transcription.markdown(st.session_state.transcription_text)

    # Add a placeholder to force refreshes
    st.empty()

if __name__ == "__main__":
    main()