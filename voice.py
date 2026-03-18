from gtts import gTTS
from pydub import AudioSegment
import tempfile
import os
import speech_recognition as sr

def text_to_speech(text, lang='en', c=0):
    tts = gTTS(text=text, lang=lang)
    audio_file = f"output{c}.mp3"
    tts.save(audio_file)
    return audio_file

def transcribe(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        file_path = tmpfile.name
        tmpfile.write(uploaded_file.read())

    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(file_path, format="wav")

    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Could not understand the audio."
        except sr.RequestError:
            return "API error. Check internet connection."
    os.remove(file_path)
    return "Error"