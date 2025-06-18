import sys
import time
import logging
import os
import socket
import json
import subprocess
import openai
from openai import OpenAI
from llama_cpp import Llama
from elevenlabs import stream
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
from RealtimeSTT import AudioToTextRecorder


# Secrets
OPENAI_KEY = "YOUR_API_KEY e.g (sk_1234)"
openai_client = OpenAI(api_key=OPENAI_KEY)
ELEVENLABS_KEY = "YOU_API_KEY"



# ElevenLabs setup
client = ElevenLabs(api_key=ELEVENLABS_KEY)
# Voice IDs from ElevenLabs Library
VOICE_ID = "84Fal4DSXWfp7nJ8emqQ"
PAUL_ID = "5Q0t7uMcjvnagumLfvZi"
FIN_ID = "D38z5RcWu1voky8WS1ja"

client.voices.edit_settings(
    voice_id="5Q0t7uMcjvnagumLfvZi",
    request=VoiceSettings(
        stability=1,
        similarity_boost=1,
        style=0,
        speed=1.1,
    )
)

# Llama model setup
llm = Llama(model_path="C:\\YOUR\\MODEL\\GGUF\\PATH\\model.gguf") # Model Path from https://huggingface.co/Gwizzly/model500

onnx_path = r"Hellow_wolfy!.onnx" # ONNX file PATH
if not os.path.exists(onnx_path):
    print(f"Error: Onnx file not found at {onnx_path}. Exiting...")
    exit(1)


# Initialise MPV player and socket for command injection
def start_mpv_player():
    # Path to mpv.exe — adjust if needed
    mpv_path = r"C:\\Program Files\\mpv\\mpv.exe"  # Example: C:\Program Files\mpv\mpv.exe HAS TO BE ADDED TO PATH

    # Command arguments
    command = [
        mpv_path,
        "p_idle.mp4",
        "--loop",
        "--fullscreen",
        "--no-border",
        "--no-osd-bar",
        "--screen=2",
        "--input-ipc-server=\\\\.\\pipe\\mpvsocket"
    ]

    try:
        # Start mpv in background (do not wait for it to finish)
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("[MPV] Launched successfully.")
        return process
    except Exception as e:
        print(f"[MPV] Failed to launch: {e}")
        return None


MAX_TURNS = 3  # Number of user+assistant exchanges to keep
chat_history = ""


# Static prompt template
prompt_template = """You are a helpful AI assistant at the American University of Kuwait (AUK).
Answer the user's questions clearly and concisely based on AUK knowledge.
If user responds vaguely like "yes please", infer based on previous context.
"""

# For RealtimeSTT
def my_stop_callback():
    print("\nRecording Stopped!")


# Video Commander 
def trigger_video(mode):
    pipe_path = r'\\.\pipe\mpvsocket'  # Windows named pipe path
    try:
        with open(pipe_path, 'w+b', buffering=0) as pipe:
            if mode == "talking":
                command = {"command": ["loadfile", "p_talking.mp4", "replace"]}
            else:
                command = {"command": ["loadfile", "p_idle.mp4", "replace"]}

            payload = (json.dumps(command) + "\n").encode("utf-8")
            pipe.write(payload)

    except Exception as e:
        print(f"[Video Trigger] Failed to switch to '{mode}': {e}")


# Play Audio Response
def play_response(answer):
    trigger_video("talking")
    audio_stream = client.text_to_speech.convert_as_stream(
        text=answer,
        voice_id=PAUL_ID, # Specify the Voice ID from above
        model_id="eleven_multilingual_v2"
    )
    stream(audio_stream)
    time.sleep(5)
    trigger_video("idle")


# Acronym Catching and Fixing
import re
def fix_acronyms(text):
    # Match "A U K", "a.uk", "a u.k", "aUK", etc.
    auk_pattern = r'\b(?:a[\s\.\-]*)[uU][\s\.\-]*[kK]\b'
    
    # Match "O K", "O.K.", "o.k", etc. and replace with AUK (common mishearing)
    okay_pattern = r'\b[oO][\s\.\-]*[kK]\b'

    # Normalize both
    text = re.sub(auk_pattern, 'AUK', text, flags=re.IGNORECASE)
    text = re.sub(okay_pattern, 'AUK', text, flags=re.IGNORECASE)
    # Enforce consistent identifiers
    text = text.replace("Q8", "Kuwait")
    text = text.replace("measures", "Majors")
    text = text.replace("Qatar", "Kuwait")
    text = text.replace("Kurdistan", "Kuwait")
    text = re.sub(r"\bAUK\b", "American University of Kuwait", text)


    return text


# AVOIDING Token Overflow
def estimate_token_count(text):
    # Roughly 1 token per 4 characters for English text
    return len(text) / 4


MAX_TOKENS_TOTAL = 512
MAX_RESPONSE_TOKENS = 200
MAX_PROMPT_TOKENS = MAX_TOKENS_TOTAL - MAX_RESPONSE_TOKENS  # 312

def trim_chat_history_to_token_limit(chat_history, prompt_template, max_tokens=MAX_PROMPT_TOKENS):
    combined = chat_history.strip().split("\n")
    trimmed = []

    # Add lines from the end until we hit the token limit
    current_tokens = estimate_token_count(prompt_template)
    for line in reversed(combined):
        tokens = estimate_token_count(line + "\n")
        if current_tokens + tokens > max_tokens:
            break
        trimmed.insert(0, line)  # Prepend to maintain order
        current_tokens += tokens

    return "\n".join(trimmed)


# Web-Search Function for Local-LLM
from duckduckgo_search import DDGS
def search_web(query):
    query = f"{query} site:auk.edu.kw"
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=2)
        return "\n".join([r["body"] for r in results])


# Sensitive Data Detection 
def is_sensitive_or_factual(text):
    keywords = [
        "president", "dean", "director", "vice president", "email", "contact",
        "event", "address", "office", "who is", "location", "when", "where",
        "tuition", "fees", "schedule", "number", "phone", "website", "faculty", "majors"
    ]
    return any(k in text.lower() for k in keywords)


# Agent Chain For Sensitive Data
def query_chatgpt_fallback(question):
    response = openai_client.chat.completions.create(
        model="gpt-4.1",  # or "gpt-4" if you have access
        messages=[
            {"role": "system", "content": "You are a helpful assistant for the American University of Kuwait (AUK). Provide accurate and concise answers about AUK only."},
            {"role": "user", "content": question}
        ],
        temperature=0.2,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()


# Text Processing
def process_text(text):
    global chat_history

    if not text.strip():
        print("No speech detected. Skipping...")
        return

    text = fix_acronyms(text)
    print(f"Recorded text: {text}")

    # Add to history
    chat_history += f"User: {text}\n"
    chat_history = trim_chat_history_to_token_limit(chat_history, prompt_template)


    # Main model response
    prompt = f"{prompt_template}\n{chat_history}Assistant:"
    response = llm(prompt, stop=["User:", "</s>", "[INST]", "[/INST]","Assistant:"], max_tokens=200)
    assistant_reply = response["choices"][0]["text"].strip()

    print(f"\nAssistant says: {assistant_reply}")
    # For Web Fallback INSTEAD of AI Fallback
    # 🌐 Web Fallback for factual queries
    # if is_sensitive_or_factual(text):
    #     print("🌐 Web fallback triggered for better accuracy...")
    #     web_info = search_web(text)

    #     # Inject live info into retry prompt
    #     retry_prompt = f"""
    #     The user asked: "{text}"

    #     The following information was found online:
    #     {web_info}

    #     Based on this, provide a clear and factual response:
    #     """
    #     retry_response = llm(
    #         retry_prompt,
    #         stop=["User:", "</s>", "[INST]", "[/INST]"],
    #         max_tokens=180
    #     )
    #     assistant_reply = retry_response["choices"][0]["text"].strip()
    #     if any(term in assistant_reply.lower() for term in ["q8", "kurdistan", "qatar"]):
    #         assistant_reply = "I'm sorry, I couldn't find reliable current information about the AUK president."
    #     print(f"\n✅ Verified Assistant says: {assistant_reply}")

    if is_sensitive_or_factual(text):
        print("🤖 ChatGPT fallback triggered...")
        assistant_reply = query_chatgpt_fallback(text)
        print(f"✅ Verified Assistant says: {assistant_reply}")

    # Add assistant's final answer to history
    chat_history += f"Assistant: {assistant_reply}\n"
    # ✅ Force correct known institutional facts IF any are shown and repeated
    if "president" in text.lower() and "american university of kuwait" in text.lower():
        assistant_reply = "The president of the American University of Kuwait (AUK) is Dr. Rawda Awwad."

    play_response(assistant_reply)


# Wake Word Mode
def start_recorder():
    recorder = AudioToTextRecorder(
        wake_words="Hellow_wolfy!",  # MUST match your ONNX label
        openwakeword_model_paths=onnx_path,
        openwakeword_inference_framework="onnx",
        wakeword_backend="oww",
        on_wakeword_detected=lambda: print("[Wakeword] Detected!"),
        on_realtime_transcription_stabilized=process_text,
        on_recording_stop=my_stop_callback,
        post_speech_silence_duration=0.6,
        min_length_of_recording=2.0,
        wake_word_activation_delay=0.5,  # Wait after wakeword before listening
        debug_mode=True,
        use_microphone=True,
        device="cpu",
        level=logging.DEBUG
    )

    print('Say "Hello Wolfy" to begin.')
    recorder.start()


# Manual Mode (Retired) WITHOUT Fallbacks
def manual_recorder():
    mode = input("Type 'm' for manual text input, or press [Enter] to use microphone: ").lower()

    if mode == 'm':
        while True:
            user_input = input("Type your question: ")
            if user_input.strip().lower() == "exit":
                break
            process_text(user_input)
    else:
        recorder = AudioToTextRecorder(
            wake_words="",
            use_microphone=True,
            debug_mode=True,
            device="cpu",
            on_recording_stop=my_stop_callback,
            post_speech_silence_duration=0.4,
            min_length_of_recording=2.0
        )
        recorder.start()

        while True:
            input("\nPress [Enter] to start recording...")
            print("Recording... Speak now!")
            recorder.start_recording_event.set()

            time.sleep(1)

            input("Press [Enter] again to stop recording...")
            recorder.stop_recording_event.set()

            print("Processing...")
            text = recorder.text()
            process_text(text)


def gpt_recorder():
    mode = input("Type 'm' for manual text input, or press [Enter] to use microphone: ").lower()

    if mode == 'm':
        while True:
            user_input = input("Type your question: ")
            if user_input.strip().lower() == "exit":
                break
            process_text(user_input)
    else:
        recorder = AudioToTextRecorder(
            wake_words="",
            use_microphone=True,
            debug_mode=True,
            spinner=False,
            device="cpu",
            on_recording_stop=my_stop_callback,
            post_speech_silence_duration=0.4,
            min_length_of_recording=2.0
        )
        while True:
            input("\n🎤 Press [Enter] to start recording...")
            print("Recording... Speak now!")
            recorder.start()
            recorder.start_recording_event.set()
            time.sleep(0.5)

            input("⏹️  Press [Enter] again to stop recording...")
            recorder.stop_recording_event.set()

            print("🧠 Processing...")
            text = recorder.text()

            if not text.strip():
                print("⚠️ No speech detected. Try again.\n")
                continue

            process_text(text)


# Main
if __name__ == "__main__":
    mpv_process = start_mpv_player()
    try:
        gpt_recorder() # Call the modes here like manual_recorder()
    except KeyboardInterrupt:
        print("\nTerminated.")
        if mpv_process:
            mpv_process.terminate()
        
