import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import whisper

# Load Whisper model for transcription
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

# Load BART model for summarization
@st.cache_resource
def load_bart_model():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    return tokenizer, model

whisper_model = load_whisper_model()
tokenizer, model = load_bart_model()

def transcribe_audio(video_path):
    video = mp.VideoFileClip(video_path)
    audio_path = "extracted_audio.wav"
    video.audio.write_audiofile(audio_path, codec='pcm_s16le')
    
    result = whisper_model.transcribe(audio_path)
    return result['text']

def summarize_text(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs['input_ids'], max_length=130, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def process_video(video_file):
    video_path = "uploaded_video.mp4"
    with open(video_path, "wb") as f:
        f.write(video_file.read())
    
    transcription = transcribe_audio(video_path)
    summary = summarize_text(transcription)
    
    return transcription, summary

# Streamlit app layout
st.title("Video Summarization App")
st.write("Upload a video file to transcribe its audio and summarize the content.")

video_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if video_file is not None:
    try:
        transcription, summary = process_video(video_file)
        
        st.subheader("Transcribed Text")
        st.write(transcription)
        
        st.subheader("Summary")
        st.write(summary)
    except Exception as e:
        st.error(f"An error occurred: {e}")
