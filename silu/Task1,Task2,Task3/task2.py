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

def convert_text_speech(speech):
    client = SarvamAI(
        api_subscription_key= os.getenv('servam_ai_api_key')
    )
    response = client.text_to_speech.convert(
        text=speech,
        target_language_code="hi-IN",
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


def start_recording():
    global recording, audio_data
    recording = True
    audio_data = []


    # indata → audio data from mic
    # frames → number of audio frames
    # time → timing info
    # status → errors or warnings
    
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

    # Joins all small audio chunks into one array
    # axis=0 means join row-wise (time sequence)
    
    audio = np.concatenate(audio_data, axis=0)
    write("recorded.wav", SAMPLE_RATE, audio)
    speech = convert_speech_to_text("recorded.wav")
    convert_text_speech(speech)


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
root.title("Microphone Recorder")

tk.Button(root, text="🎤", font=("Arial", 30), command=recorder).pack(pady=20)

status_label = tk.Label(root, text="Click mic to start recording", font=("Arial", 12))
status_label.pack(pady=10)

root.geometry("400x300+200+150")
root.mainloop()
