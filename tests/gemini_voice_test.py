# gemini_mic_transcribe_env.py
import os
import tempfile
import sounddevice as sd
import soundfile as sf
import numpy as np
from google import genai
from dotenv import load_dotenv

# ----- CONFIG -----
DURATION = 20                 # seconds to record
SAMPLE_RATE = 16000           # whisper/Gemini friendly
CHANNELS = 1
MODEL = "gemini-2.5-flash"    # multimodal model for audio
# ------------------

def record_to_wav(path, duration=DURATION, samplerate=SAMPLE_RATE, channels=CHANNELS):
    print(f"Recording {duration} seconds (samplerate={samplerate})... Speak now.")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate,
                   channels=channels, dtype='float32')
    sd.wait()
    audio = np.squeeze(audio)
    sf.write(path, audio, samplerate, subtype='PCM_16')
    print(f"Saved recording to {path}")

def main():
    # load .env file
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in .env")
        return

    client = genai.Client(api_key=api_key)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    # record audio
    record_to_wav(wav_path)

    # upload file to Gemini Files API
    myfile = client.files.upload(file=wav_path)

    # ask Gemini to transcribe
    response = client.models.generate_content(
        model=MODEL,
        contents=["Transcribe this audio clip", myfile]
    )

    print("\n--- Gemini Transcript ---\n")
    print(response.text.strip())
    print("\n-------------------------\n")

if __name__ == "__main__":
    main()
