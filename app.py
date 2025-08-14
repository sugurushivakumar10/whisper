import streamlit as st
import whisper
import tempfile

# Allowed language codes
ALLOWED_LANGS = {"en": "English", "te": "Telugu", "kn": "Kannada"}

@st.cache_resource
def load_model():
    return whisper.load_model("tiny")  # 'tiny' for speed, can use 'base' for better accuracy

model = load_model()

st.title("🎙️ Whishper Testing...")
st.markdown("#🎙️ Voice to Text (English / Telugu / Kannada)")

# Record audio from mic
audio_data = st.audio_input("Record your voice")

if audio_data is not None:
    # Save audio to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(audio_data.getbuffer())
        audio_path = tmpfile.name

    st.write("✅ Audio recorded. Detecting language...")

    # Load and preprocess audio
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Detect language
    try:
        _, probs = model.detect_language(mel)  # Works for OpenAI Whisper
    except TypeError:
        # Fallback for environments that expect raw audio
        detected_lang_code = model.transcribe(audio_path)["language"]
        probs = {detected_lang_code: 1.0}

    # Only check allowed languages
    lang_probs = {lang: probs.get(lang, 0) for lang in ALLOWED_LANGS.keys()}
    best_lang = max(lang_probs, key=lang_probs.get)
    best_prob = lang_probs[best_lang]

    if best_prob > 0.2:  # Confidence threshold
        st.write(f"🌍 Detected Language: **{ALLOWED_LANGS[best_lang]}**")
        options = whisper.DecodingOptions(language=best_lang, without_timestamps=True)
        result = whisper.decode(model, mel, options)
        st.subheader("📝 Transcription")
        st.write(result.text)
    else:
        st.warning("⚠ Language not confidently detected as English, Telugu, or Kannada. Falling back to English translation...")
        result = model.transcribe(audio_path, task="translate")
        st.subheader("📝 English Translation")
        st.write(result["text"])
