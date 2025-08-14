import io
import wave
from pathlib import Path
from typing import List, Union

import streamlit as st
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    AudioProcessorBase,
    RTCConfiguration,
)
import av  # required by streamlit-webrtc
from faster_whisper import WhisperModel


st.set_page_config(page_title="Whisper POC (Legacy Py)", page_icon="🎤", layout="centered")
st.title("🎤 Whisper POC (Mic → WAV → Transcribe) — Python 3.8/3.9")

st.markdown(
    "Click **Start** to begin recording, speak, then **Stop** and press **Transcribe**."
)

# ---- WebRTC config (public STUN helps P2P) ----
rtc_configuration = RTCConfiguration(
    {
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
        "iceCandidatePoolSize": 16,
    }
)

# ---- Audio Processor: buffers mono 16-bit PCM frames ----
class MicBuffer(AudioProcessorBase):
    def __init__(self):
        self.frames = []          # type: List[bytes]
        self.sample_rate = 48000  # default fallback
        self.channels = 1

    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        # Convert to mono, signed 16-bit PCM
        pcm = frame.to_ndarray(format="s16", layout="mono")
        self.sample_rate = frame.sample_rate or self.sample_rate
        self.frames.append(pcm.tobytes())
        return frame

# ---- WebRTC mic widget ----
ctx = webrtc_streamer(
    key="whisper-poc-legacy",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={"audio": True, "video": False},
    audio_receiver_size=256,
    audio_processor_factory=MicBuffer,
)

col1, col2 = st.columns(2)
with col1:
    clear_clicked = st.button("🧹 Clear recording")
with col2:
    transcribe_clicked = st.button("📝 Transcribe")

if clear_clicked and ctx and ctx.audio_processor:
    ctx.audio_processor.frames = []
    st.success("Cleared recorded audio buffer.")

def save_wav(frames: List[bytes], sr: int, path: Union[str, Path]) -> Path:
    path = Path(path)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)      # mono
        wf.setsampwidth(2)      # 16-bit
        wf.setframerate(sr)
        wf.writeframes(b"".join(frames))
    return path

if transcribe_clicked:
    if not ctx or not ctx.audio_processor:
        st.error("Mic is not initialized. Click Start and try again.")
    else:
        frames = ctx.audio_processor.frames
        sr = ctx.audio_processor.sample_rate or 48000

        if not frames:
            st.warning("No audio captured yet. Click Start, speak, then Stop.")
        else:
            st.info("Saving WAV and running transcription…")
            out_path = Path("recorded.wav")
            save_wav(frames, sr, out_path)

            # Choose a model size. "small" is a good balance for POC.
            model_size = "small"
            device = "cpu"          # change to "cuda" if you have NVIDIA CUDA
            compute_type = "int8"   # fast & accurate enough

            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            segments, info = model.transcribe(str(out_path), vad_filter=True)

            st.write("**Detected language:** {} (p={:.2f})".format(
                info.language, info.language_probability or 0.0
            ))

            transcript_io = io.StringIO()
            for seg in segments:
                transcript_io.write("[{:.2f} → {:.2f}] {}\n".format(seg.start, seg.end, seg.text))

            st.text_area("Transcript", transcript_io.getvalue(), height=250)
            st.success("Done! Use **Clear recording** to capture a fresh sample.")
