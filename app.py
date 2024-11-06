import streamlit as st #to create the interactive web application
from transformers import pipeline 
import matplotlib.pyplot as plt 
import pandas as pd  
import numpy as np  
import tempfile 
import speech_recognition as sr
import librosa 
import soundfile as sf 
import seaborn as sns 
import os #For interacting with the operating system (file handling)
from ydata_profiling import ProfileReport #For automated EDA
import streamlit.components.v1 as components
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import ridge_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Load the models

import pickle #For loading serialized models saved in binary format

# Paths to your saved model files
politeness_model1 = 'politeness_model.sav'
abusiveness_model1 = 'abusiveness.sav'
threat_model1 = 'threat_level.sav'

# Load the models using pickle
with open(politeness_model1, 'rb') as file:
    model1 = pickle.load(file)

with open(abusiveness_model1, 'rb') as file:
    model2 = pickle.load(file)

with open(threat_model1, 'rb') as file:
    model3 = pickle.load(file)

politeness_model = pipeline("text-classification", model="Genius1237/xlm-roberta-large-tydip")
abusiveness_model = pipeline("text-classification", model="Hate-speech-CNERG/english-abusive-MuRIL")
threat_model = pipeline("text-classification", model="unitary/toxic-bert")

def analyze_speech(speech_text):
    # Analyze politeness
    # politeness_model  = pipeline(task="text-classification", model="Genius1237/xlm-roberta-large-tydip")
    politeness_result = politeness_model(speech_text)
    # Extract the label and score
    politeness_label = politeness_result[0]['label']
    politeness_score = politeness_result[0]['score']
    # politeness_score = politeness_result[0]['score']
    # if politeness_score > 0.6:
    #     politeness_label = "Low Politeness"
    # elif 0.3 <= politeness_score <= 0.59:
    #     politeness_label = "Moderate Politeness"
    # else:
    #     politeness_label = "High Politeness"

    # Analyze abusiveness
    abusiveness_result = abusiveness_model(speech_text)
    abusiveness_score = abusiveness_result[0]['score']
    abusiveness_label = abusiveness_result[0]['label']
    abusiveness_label = "Not Abusive" if abusiveness_result[0]['label'] =='LABEL_0'  else "Abusive"
    print(abusiveness_label)

    # Analyze threat level
    threat_result = threat_model(speech_text)
    threat_score = threat_result[0]['score']
    if threat_score > 0.6:
        threat_label = "High Threat"
    elif 0.3 <= threat_score <= 0.59:
        threat_label = "Moderate Threat"
    else:
        threat_label = "Low Threat"

    return politeness_score, abusiveness_score, threat_score, politeness_label, abusiveness_label, threat_label

def plot_results(politeness_score, abusiveness_score, threat_score):
    labels = ['Politeness', 'Abusiveness', 'Threat Level']
    scores = [politeness_score, abusiveness_score, threat_score]
    
    fig, ax = plt.subplots()
    ax.bar(labels, scores, color=['blue', 'red', 'orange'])
    ax.set_ylim([0, 1])
    ax.set_ylabel('Scores')
    ax.set_title('Speech Analysis Scores')
    
    st.pyplot(fig)

def transcribe_audio(audio_file_path):
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    y, sample_rate = librosa.load(audio_file_path, sr=None)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav_file:
        sf.write(tmp_wav_file.name, y, sample_rate)
        audio_wav_path = tmp_wav_file.name

    audio_file = sr.AudioFile(audio_wav_path)
    with audio_file as source:
        audio_data = recognizer.record(source)
    text = recognizer.recognize_google(audio_data)
    os.remove(audio_wav_path)  # Clean up temporary file
    return text

# Load the dataset
data = pd.read_csv('Finalized_data.csv')

# Streamlit app layout
st.title("Speech Analysis Application")
st.image("img.png", caption="Speech Analysis", use_column_width=True)

menu = st.sidebar.radio("Menu", ["Home", "Analyze Speech", "EDA"])

if menu == "Home":
    st.subheader("Home")
    st.write("Welcome to the Speech Analysis Application.")
    st.write("Navigate to the 'Analyze Speech' page to provide text or audio input and get analysis results.")
    st.write("Navigate to the 'EDA' page to perform exploratory data analysis on the provided dataset.")

elif menu == "Analyze Speech":
    st.subheader("Analyze Speech")
    
    input_type = st.radio("Select input type", ("Text", "Audio File"))

    if input_type == "Text":
        speech_text = st.text_area("Enter text for analysis")
        if st.button("Submit"):
            politeness_score, abusiveness_score, threat_score, politeness_label, abusiveness_label, threat_label = analyze_speech(speech_text)

            st.write(f"Politeness Score: {politeness_score:.2f} ({politeness_label})")
            st.write(f"Abusiveness Score: {abusiveness_score:.2f} ({abusiveness_label})")
            st.write(f"Threat Level Score: {threat_score:.2f} ({threat_label})")

            plot_results(politeness_score, abusiveness_score, threat_score)

    elif input_type == "Audio File":
        audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "ogg", "flac"])

        if audio_file is not None:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(audio_file.read())
                tmp_file_path = tmp_file.name

            st.audio(tmp_file_path, format='audio/wav')

            if st.button("Submit"):
                transcription = transcribe_audio(tmp_file_path)
                st.write(f"Transcription: {transcription}")
                
                politeness_score, abusiveness_score, threat_score, politeness_label, abusiveness_label, threat_label = analyze_speech(transcription)

                st.write(f"Politeness Score: {politeness_score:.2f} ({politeness_label})")
                st.write(f"Abusiveness Score: {abusiveness_score:.2f} ({abusiveness_label})")
                st.write(f"Threat Level Score: {threat_score:.2f} ({threat_label})")

                plot_results(politeness_score, abusiveness_score, threat_score)

elif menu == "EDA":
    st.subheader("Exploratory Data Analysis (EDA)")
    st.write("Performing EDA on the provided dataset using YData Profiling.")

    # Generate the profiling report
    profile = ProfileReport(data, title="YData Profiling Report", explorative=True)
    
    # Display the report in the Streamlit app
    st_profile_report = profile.to_html()
    components.html(st_profile_report, height=1000, scrolling=True)