import os
import io
import time
import tempfile
import queue
from dataclasses import dataclass

import av
import numpy as np
import soundfile as sf
import streamlit as st
import whisper
from streamlit_webrtc import webrtc_streamer, WebRtcMode

st.set_page_config(page_title="Whisper Transcriber (Mic + Upload)", page_icon="🎙️", layout="centered")
st.title("🎙️ Whisper Transcriber — Browser Mic + File Upload")
st.caption("Runs the official `whisper` package. Your original decoding flow, wrapped in Streamlit with in-browser recording.")

# -----------------------------
# Sidebar: model pick
# -----------------------------
with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox(
        "Model size",
        ["tiny", "base", "small", "medium", "large", "turbo"],
        index=5,
        help="'turbo' is fast; larger models may be more accurate but heavier.",
    )

@st.cache_resource(show_spinner=False)
def load_model(name: str):
    return whisper.load_model(name)

model = load_model(model_name)

# -----------------------------
# Audio buffer to capture mic frames
# -----------------------------
@dataclass
class AudioBuffer:
    sample_rate: int | None = None
    channels: int | None = None
    _chunks: list | None = None

    def __post_init__(self):
        self._chunks = []

    def add_frame(self, frame: av.AudioFrame):
        arr = frame.to_ndarray()
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        self.sample_rate = frame.sample_rate
        self.channels = arr.shape[0]
        # Ensure float32 for WAV writing later
        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float32) / 32768.0
        elif arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        self._chunks.append(arr)

    def to_wav_bytes(self) -> bytes:
        if not self._chunks:
            raise ValueError("No audio recorded")
        audio = np.concatenate(self._chunks, axis=1)  # (channels, samples)
        # Convert to mono
        if audio.shape[0] > 1:
            audio = audio.mean(axis=0, keepdims=False)
        else:
            audio = audio.squeeze(axis=0)
        buf = io.BytesIO()
        sr = self.sample_rate or 16000
        sf.write(buf, audio, sr, format="WAV")
        buf.seek(0)
        return buf.read()

# -----------------------------
# UI: Mic recording + Upload
# -----------------------------
st.subheader("1) Record via microphone or upload an audio file")
col1, col2 = st.columns(2)

recorded_wav_bytes: bytes | None = None
uploaded_file = None

with col1:
    st.markdown("**A. Record in browser**")
    st.caption("Click Start → speak → Stop. Then click ‘Use last recording’.")

    audio_buffer = AudioBuffer()

    class AudioProcessor:
        def __init__(self):
            self.q = queue.Queue()
        def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
            audio_buffer.add_frame(frame)
            return frame

    ctx = webrtc_streamer(
        key="mic",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

    if ctx and ctx.state.playing:
        st.info("Recording… speak now, then click Stop above.")

    if ctx and not ctx.state.playing and audio_buffer._chunks:
        if st.button("Use last recording"):
            try:
                recorded_wav_bytes = audio_buffer.to_wav_bytes()
                st.success("Captured recording.")
                st.audio(recorded_wav_bytes, format="audio/wav")
            except Exception as e:
                st.error(f"Could not finalize recording: {e}")

with col2:
    st.markdown("**B. Or upload an audio file**")
    uploaded_file = st.file_uploader(
        "Choose audio (wav/mp3/m4a/ogg/flac)",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
        accept_multiple_files=False,
    )
    if uploaded_file is not None:
        st.audio(uploaded_file)

# -----------------------------
# Transcribe button
# -----------------------------
st.subheader("2) Transcribe using your original Whisper flow")
if st.button("🔍 Transcribe now", use_container_width=True):
    # Choose input
    if recorded_wav_bytes is None and uploaded_file is None:
        st.warning("Record audio or upload a file first.")
        st.stop()

    # Persist to a temp file for whisper.load_audio
    suffix = ".wav" if recorded_wav_bytes is not None else os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        if recorded_wav_bytes is not None:
            tmp.write(recorded_wav_bytes)
        else:
            tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Transcribing… (first run may download model weights)"):
        # --- Your original script, unmodified in spirit ---
        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(tmp_path)
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

        # detect language
        _, probs = model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)

        # decode the audio
        options = whisper.DecodingOptions()
        result = whisper.decode(model, mel, options)

    # Display
    st.success("Done!")
    st.write(f"**Detected language:** {detected_lang}")
    st.text_area("Transcript", result.text, height=220)
    st.download_button("Download transcript (.txt)", result.text, file_name="transcript.txt")

    # Cleanup
    try:
        os.remove(tmp_path)
    except Exception:
        pass

st.markdown("---")
st.caption(
    "Notes: Install FFmpeg on your system for best compatibility. This app writes a temporary WAV and uses the official `whisper` package to decode."
)
