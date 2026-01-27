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


def convertTextToSpeech():
    # 1.o means read from begeining of the string to end of the lind and strip is remove the white space from the begining and end
    string = text_input.get("1.0", tk.END).strip()
    if len(string) == 0:
        print('Please Enter any string')
        return
    api_key = os.getenv('servam_ai_api_key')
    client = SarvamAI(api_subscription_key=api_key)

    response = client.text_to_speech.convert(
        text=string,
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

    print(string)


root = tk.Tk()
root.title("Text to Speech - Sarvam")
tk.Label(root, text="Enter Text").pack()
text_input = tk.Text(root, height=8, width=40)
text_input.pack()
tk.Button(root, text="Convert to Voice", command=convertTextToSpeech).pack(pady=10)
root.geometry("400x300+200+150")
root.mainloop()