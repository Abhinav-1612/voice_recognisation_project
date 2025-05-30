# Voice Authentication System Project
# 1. Text to Speech
# 2. Speech to Text
# 3. User Voice Recognition
import os
import numpy as np
import librosa
import soundfile as sf
import pyttsx3  
import sounddevice as sd 
import wave
import speech_recognition as sr
from scipy.spatial.distance import cosine
import torch
from speechbrain.inference import SpeakerRecognition
spk_recognizer = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",savedir="tmp_model")
speaker_db = {}

tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  #Adjust speed
tts_engine.setProperty('volume', 1.0)  #Adjust volume

def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

def record_audio(filename="voice.wav", duration=5, sr=16000):
    speak(f"Recording for {duration} seconds. Please speak now.")
    print(f"🎤 Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())
    print(f"✅ Recording saved as {filename}")
    speak("Recording saved successfully.")

def extract_embedding(audio_path):
    signal, sr = librosa.load(audio_path, sr=16000)
    sf.write("temp.wav", signal, sr)
    embedding = spk_recognizer.encode_batch(torch.tensor([signal]))
    return embedding.squeeze().cpu().numpy()

def enroll_speaker():
    speak("Please say your name after the beep.")
    print("🔊 Please say anything:")
    record_audio("enrolled_voice.wav", duration=5)
    name = input("Enter your name: ")
    embedding = extract_embedding("enrolled_voice.wav")
    speaker_db[name] = embedding
    print(f"✅ Speaker '{name}' enrolled successfully!")
    speak(f"Speaker {name} has been enrolled successfully.")

def recognize_speaker():
    speak("Please speak now for verification.")
    print("\n🔊 Please speak now:")
    record_audio("test_voice.wav", duration=5)
    test_embedding = extract_embedding("test_voice.wav")
    
    best_match = None
    best_score = float("inf")
    for name, emb in speaker_db.items():
        score = cosine(test_embedding, emb)
        print(f"🔍 Match Score with {name}: {score:.4f}")
        if score < best_score:
            best_score = score
            best_match = name
    
    if best_match and best_score < 0.6:
        print(f"🎉 Recognized as: {best_match}")
        speak(f"Welcome back, {best_match}!")
    else:
        print("⚠️ Speaker not recognized.")
        speak("Sorry, I could not recognize your voice.")

def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        speak("Listening for 10 seconds. Please speak.")
        print("🎙️ Listening...")
        try:
            audio = recognizer.record(source, duration=10)
            text = recognizer.recognize_google(audio)
            print(f"📝 Transcribed Text: {text}")
            speak("You said: " + text)
        except sr.UnknownValueError:
            print("⚠️ Could not understand the audio.")
            speak("I could not understand. Please try again.")
        except sr.RequestError:
            print("⚠️ Could not request results. Check your internet connection.")
            speak("Could not process speech. Please check your internet connection.")

def main():
    while True:
        print("\n🎙️ Voice Authentication System 🎙️")
        speak("Welcome to voice authentication system.")
        print("1️ Enroll a new speaker")
        print("2️ Recognize a speaker")
        print("3️ Exit")
        print("4️ Speech to Text convertion")
        speak("Press 1 to enroll a speaker, press 2 to recognize a speaker, press 3 to exit, or press 4 for speech to text.")
        
        choice = input("Choose an option (1/2/3/4): ")
        if choice == "1":
            enroll_speaker()
        elif choice == "2":
            recognize_speaker()
        elif choice == "3":
            print("👋 Exiting program.")
            speak("Goodbye. have a nice day")
            break
        elif choice == "4":
            speech_to_text()
        else:
            print("Invalid option. Please choose again.")
            speak("Invalid option, please try again.")

main()