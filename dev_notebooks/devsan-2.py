"""
Integrated OceanCrew agentic system with optional voice input (Gemini transcription).

How it works:
- Presents a CLI choice: (1) voice input (record + transcribe) or (2) text input
- If voice: records audio, uploads to Gemini Files API, asks model to transcribe
- Uses the resulting text as `user_query` and kicks off the existing Crew pipeline

Notes:
- Requires: crewai package (your agent framework), google.genai (Gemini client), sounddevice, soundfile, python-dotenv
- Put GEMINI_API_KEY in a .env file or export it in the environment
- Adjust RECORD_DURATION or SAMPLE_RATE if needed

"""

import os
import tempfile
import argparse
import time
from dotenv import load_dotenv

# crew framework imports (from your example)
from crewai import LLM
from crewai import Agent, Task, Crew, Process

# Gemini client for audio transcription
from google import genai
import sounddevice as sd
import soundfile as sf
import numpy as np

# ---------------- CONFIG -----------------
RECORD_DURATION = 15       # seconds for recording (adjustable)
SAMPLE_RATE = 16000        # Gemini/Whisper-friendly
CHANNELS = 1
TRANSCRIBE_MODEL = "gemini-2.5-flash"
# -----------------------------------------


def record_to_wav(path, duration=RECORD_DURATION, samplerate=SAMPLE_RATE, channels=CHANNELS):
    """Record audio from default microphone and save to WAV (16-bit PCM)."""
    print(f"Recording {duration} seconds. Speak after the prompt...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate,
                   channels=channels, dtype='float32')
    sd.wait()
    audio = np.squeeze(audio)
    # normalize and convert to int16 for PCM_16
    max_abs = np.max(np.abs(audio))
    if max_abs > 0:
        audio = audio / max_abs
    sf.write(path, audio, samplerate, subtype='PCM_16')
    print(f"Saved recording to {path}")


def transcribe_with_gemini(wav_path, api_key=None, model=TRANSCRIBE_MODEL):
    """Upload WAV to Gemini and ask it to transcribe. Returns transcript string or raises.

    Note: Requires google-genai client installed and API key set.
    """
    load_dotenv()
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in environment or .env file")

    client = genai.Client(api_key=api_key)
    print("Uploading audio to Gemini Files API...")
    myfile = client.files.upload(file=wav_path)
    # ask Gemini to transcribe
    print("Requesting transcription from Gemini...")
    response = client.models.generate_content(
        model=model,
        contents=["Transcribe this audio clip", myfile]
    )

    # .text contains the model output
    transcript = response.text.strip()
    return transcript


# ---------------- Agent Crew Setup (same as your original example) ----------------

def build_crew(llm_model_name="gemini/gemini-2.5-flash", temperature=0.7):
    llm = LLM(
        model=llm_model_name,
        temperature=temperature,
    )

    prompt_guard = Agent(
        role="Prompt Guard Agent",
        goal="Check if the user input is safe and relevant to oceanographic queries.",
        backstory=(
            "You are a strict filter that decides if a query should be processed. "
            "If unsafe, you stop the pipeline by flagging it."
        ),
        llm=llm,
        verbose=True,
        memory=True,
    )

    query_processor = Agent(
        role="Query Processor Agent",
        goal="Interpret safe user queries and generate useful scientific responses.",
        backstory=(
            "You are an ocean data assistant who knows how to fetch, analyze, "
            "and summarize ARGO float data."
        ),
        llm=llm,
        verbose=True,
        memory=True,
    )

    output_formatter = Agent(
        role="Output Formatter Agent",
        goal="Sanitize and format the final response into clean, structured text.",
        backstory=(
            "You make the final response user-friendly, well-formatted, and safe "
            "for display in dashboards or chat."
        ),
        llm=llm,
        verbose=True,
        memory=True,
    )

    guard_task = Task(
        description=(
            "Check the input: {user_query}. "
            "If the query is unsafe or irrelevant, respond ONLY with 'UNSAFE PROMPT'. "
            "If safe, respond with 'SAFE PROMPT'."
        ),
        name="guardrails",
        expected_output="Either 'SAFE PROMPT' or 'UNSAFE PROMPT'.",
        agent=prompt_guard,
    )

    process_task = Task(
        description=(
            "If the guard output was 'SAFE PROMPT', process the user query: {user_query}. "
            "Return a mock processed output (e.g., salinity profile, trajectory, etc.). "
            "If guard output was 'UNSAFE PROMPT', just return 'BLOCKED'."
        ),
        name="processor",
        expected_output="A scientific summary or 'BLOCKED'.",
        agent=query_processor,
    )

    format_task = Task(
        description=(
            "Take the processor output and return a clean formatted message. "
            "If it was 'BLOCKED', say: '🚫 The input was unsafe and cannot be processed.' "
            "Otherwise, return the response as Markdown with sections."
        ),
        name="formatter",
        expected_output="A safe, user-friendly Markdown formatted answer.",
        agent=output_formatter,
    )

    crew = Crew(
        name="OceanCrew-turtle",
        agents=[prompt_guard, query_processor, output_formatter],
        tasks=[guard_task, process_task, format_task],
        process=Process.sequential,
        verbose=True,
        tracing=True,
    )

    return crew


# ---------------- Main CLI Integration ----------------

def run_interactive(allow_voice=True):
    """Run a simple CLI where user chooses voice or text input and then sends query to the Crew."""
    print("\n=== OceanCrew — Input Selector ===\n")
    print("Choose input mode:")
    print(" [1] Voice input (record & transcribe)")
    print(" [2] Text input (type your query)")
    choice = input("Select 1 or 2 (default 2): ").strip() or "2"

    user_query = None

    if choice == "1":
        if not allow_voice:
            print("Voice input disabled. Falling back to text input.")
            choice = "2"
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = tmp.name
            try:
                # record
                record_to_wav(wav_path)
                # transcribe
                transcript = transcribe_with_gemini(wav_path)
                print("\n--- Transcript ---\n")
                print(transcript)
                print("\n------------------\n")
                user_query = transcript
            except Exception as e:
                print("Error during voice flow:", e)
                print("Falling back to text input.")
                user_query = input("Type your query: ")
            finally:
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

    if choice != "1":
        user_query = input("Type your query: ")

    # build crew and kickoff
    crew = build_crew()
    print("\n=== Kicking off Crew with your query ===\n")
    # pass the user_query into the crew inputs
    result = crew.kickoff(inputs={"user_query": user_query})

    print("\n=== Final Output ===\n")
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OceanCrew with optional voice input")
    parser.add_argument("--no-voice", action="store_true", help="Disable voice input path")
    args = parser.parse_args()

    run_interactive(allow_voice=(not args.no_voice))
