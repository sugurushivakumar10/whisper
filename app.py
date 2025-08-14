import io
import wave
from pathlib import Path

import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase, RTCConfiguration
import av  # required by streamlit-webrtc
from faster_whisper import WhisperModel


st.set_page_config(page_title="Whisper POC", page_icon="🎤", layout="centered")
st.title("🎤 Whisper POC: Mic → Transcribe (Streamlit)")

st.markdown(
    """
This is a simple mic recorder using WebRTC. Click **Start** to begin recording,
speak, then **Stop** and press **Transcribe**.
"""
)

# ---- Config for STUN (helps WebRTC connect reliably) ----
rtc_configuration = RTCConfiguration(
    {
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
        "iceCandidatePoolSize": 16,
    }
)


# ---- Audio Processor: collect raw PCM from mic ----
class MicBuffer(AudioProcessorBase):
    def __init__(self) -> None:
        self.frames = []        # list of bytes (int16 PCM)
        self.sample_rate = 48000  # default; will be updated from frames when available
        self.channels = 1

    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        # Convert to mono, signed 16-bit
        # layout="mono" ensures 1 channel; format "s16" is 16-bit PCM
        pcm = frame.to_ndarray(format="s16", layout="mono")
        # Remember the sample rate from this frame
        self.sample_rate = frame.sample_rate
        # Append raw bytes
        self.frames.append(pcm.tobytes())
        return frame


# ---- UI: WebRTC mic capture ----
ctx = webrtc_streamer(
    key="whisper-poc",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=256,  # small buffer for lower latency
    rtc_configuration=rtc_configuration,
    media_stream_constraints={"audio": True, "video": False},
    audio_processor_factory=MicBuffer,
)

col1, col2 = st.columns(2)
with col1:
    clear_clicked = st.button("🧹 Clear recording")
with col2:
    transcribe_clicked = st.button("📝 Transcribe")

# Reset buffer if requested
if clear_clicked and ctx and ctx.audio_processor:
    ctx.audio_processor.frames = []
    st.success("Cleared recorded audio buffer.")

# ---- Helper: save recorded PCM to a WAV file ----
def save_wav(frames: list[bytes], sr: int, path: Union[str, Path]) -> Path:
    path = Path(path)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)      # mono
        wf.setsampwidth(2)      # 16-bit
        wf.setframerate(sr)
        wf.writeframes(b"".join(frames))
    return path


# ---- Transcribe when clicked ----
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

            # Choose a model size. "small" is a good POC balance.
            # Use device="cuda" if you have an NVIDIA GPU.
            model_size = "small"
            device = "cpu"
            compute_type = "int8"  # fast & accurate enough for POC

            model = WhisperModel(model_size, device=device, compute_type=compute_type)

            # Transcribe. vad_filter=True helps trim silence.
            segments, info = model.transcribe(str(out_path), vad_filter=True)

            st.write(f"**Detected language:** {info.language} (p={info.language_probability:.2f})")

            # Show full transcript with timestamps
            transcript_io = io.StringIO()
            for seg in segments:
                line = f"[{seg.start:.2f} → {seg.end:.2f}] {seg.text}\n"
                transcript_io.write(line)

            st.text_area("Transcript", transcript_io.getvalue(), height=250)

            # Optional: also show a single-line concatenated transcript
            # (Re-run transcription to iterate segments again, or reassemble from above)
            st.success("Done!")
            st.caption("Tip: Click **Clear recording** to capture a fresh sample.")

