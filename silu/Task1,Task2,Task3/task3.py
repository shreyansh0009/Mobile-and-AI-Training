import tkinter as tk

import os
from dotenv import load_dotenv

from sarvamai import SarvamAI
from sarvamai.play import play

import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

import winsound
import base64
import requests

SAMPLE_RATE = 44100
CHANNELS = 1


is_recording_started = False
recording = False
audio_data = []

def languageConverter():
    print('new Target Language', languages[selected_language.get()]);
    speech = convert_speech_to_text("recorded.wav")
    speech = convert_to_target_language(languages[selected_language.get()], speech)
    print(speech)
    convert_text_speech(speech, languages[selected_language.get()])

def convert_to_target_language(target_language, speech):
    # try:
    API_KEY = os.getenv('servam_ai_api_key')
    client = SarvamAI(
        api_subscription_key=API_KEY
    )
    response = client.text.translate(
        input=speech,
        source_language_code="auto",
        target_language_code=target_language,
        speaker_gender="Male"
    )
    return response.translated_text
    # except Exception as e:
    #     print("Translation error:", e)
    #     return "" 


def start_recording():
    global recording, audio_data
    recording = True
    audio_data = []
    
    def callback(indata, frames, time, status):
        if recording:
            audio_data.append(indata.copy())

    global stream
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        callback=callback
    )
    stream.start()

def stop_recording():
    global recording
    recording = False
    stream.stop()
    stream.close()

    
    audio = np.concatenate(audio_data, axis=0)
    write("recorded.wav", SAMPLE_RATE, audio)
    # speech = convert_speech_to_text("recorded.wav")
    # convert_text_speech(speech)



def convert_text_speech(speech, target_language):
    client = SarvamAI(
        api_subscription_key= os.getenv('servam_ai_api_key')
    )
    response = client.text_to_speech.convert(
        text=speech,
        target_language_code=target_language,
        speaker="anushka",
        enable_preprocessing=True,
    )
    
    audio_base64 = response.audios[0]

    #decode base64 → bytes
    audio_bytes = base64.b64decode(audio_base64)

    #save as WAV file
    audio_file = "output.wav"
    with open(audio_file, "wb") as f:
        f.write(audio_bytes)

     #play audio (Windows)
    winsound.PlaySound(audio_file, winsound.SND_FILENAME)


def convert_speech_to_text(audio_file):
    client = SarvamAI(
        api_subscription_key= os.getenv('servam_ai_api_key')
    )
    
    if audio_file_path:
        with open(audio_file_path, "rb") as audio_file:
            response = client.speech_to_text.transcribe(
                file=audio_file,
                model="saarika:v2.5"
            )
        print(response.transcript)
        return response.transcript
    else:
        return "No valid audio file found."








def recorder():
    global is_recording_started

    if not is_recording_started:
        is_recording_started = True
        status_label.config(text="🎙️ Recording started")
        start_recording()
        print("Recording started")
    else:
        is_recording_started = False
        status_label.config(text="⏹️ Recording stopped")
        stop_recording()
        print("Recording ended")


root = tk.Tk()
root.title("Language Translator")

languages = {
    "English": "en-IN",
    "Hindi": "hi-IN",
    "Tamil": "ta-IN",
    "Kannada": "kn-IN",
    "Malayalam": "ml-IN",
    "Marathi": "mr-IN",
    "Bengali": "bn-IN",
    "Odia": "od-IN"
}



tk.Button(root, text="🎤", font=("Arial", 30), command=recorder).pack(pady=20)

status_label = tk.Label(root, text="Click mic to start recording", font=("Arial", 12))
status_label.pack(pady=10)


selected_language = tk.StringVar()
selected_language.set("English")   # Default selection

language_dropdown = tk.OptionMenu(
    root,
    selected_language,
    *(languages.keys())
)

print('Selected Language is : ',selected_language.get())

language_dropdown.config(width=20)
language_dropdown.pack(pady=20)

tk.Button(root, text="Convert", font=("Arial", 20), command=languageConverter).pack(pady=20)

root.geometry("400x300+200+150")
root.mainloop()
