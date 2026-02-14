# mic_whisper_transcribe_no_ffmpeg.py
import whisper
import sounddevice as sd
import numpy as np
import sys
import tempfile
import soundfile as sf

# ----- CONFIG -----
DURATION = 10          # seconds to record
SAMPLE_RATE = 16000    # whisper expects 16k
MODEL_NAME = "base"    # tiny, base, small, medium, large
# ------------------

def record_audio(duration=DURATION, samplerate=SAMPLE_RATE, channels=1):
    print(f"Recording for {duration} seconds (samplerate={samplerate})... Speak now.")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='int16')
    sd.wait()
    recording = np.squeeze(recording).astype(np.float32) / 32768.0   # normalize int16 -> float32 [-1,1]
    return recording

def transcribe_array(model_name, audio_array, samplerate=SAMPLE_RATE):
    print(f"Loading Whisper model '{model_name}' (this may take a moment)...")
    model = whisper.load_model(model_name)
    # whisper.transcribe accepts a numpy array sampled at 16000 (same as SAMPLE_RATE)
    print("Transcribing...")
    # pass the numpy array directly
    result = model.transcribe(audio_array, fp16=False)   # fp16=False avoids CPU fp16 warnings/errors
    print("\n--- Transcript ---\n")
    print(result["text"].strip())
    print("\n------------------\n")
    return result

def main():
    duration = DURATION
    if len(sys.argv) > 1:
        try:
            duration = float(sys.argv[1])
        except Exception:
            pass

    audio = record_audio(duration=duration)
    # optional: write file for debugging
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, SAMPLE_RATE, subtype='PCM_16')
    print(f"Saved a copy for inspection at: {tmp.name}")
    transcribe_array(MODEL_NAME, audio)

if __name__ == "__main__":
    main()
