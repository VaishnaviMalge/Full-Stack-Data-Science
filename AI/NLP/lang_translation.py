import streamlit as st
from mtranslate import translate
import pandas as pd
import os
from gtts import gTTS
import base64

df = pd.read_csv(r"E:\Vaishnavi\practiced\vs code\AI\Multi-language translation\language.csv")
df.dropna(inplace = True)
langlist = tuple(df['name'].to_list())

st.title("Language Translator")
inputtext = st.text_area("Enter text to translate", height = 100)

language = st.selectbox("Select Language", langlist)


languages_dict = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "bs": "Bosnian",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "eo": "Esperanto",
    "es": "Spanish",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "gu": "Gujarati",
    "od" : "odia",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "jw": "Javanese",
    "km": "Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "la": "Latin",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "my": "Myanmar (Burmese)",
    "ne": "Nepali",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "si": "Sinhala",
    "sk": "Slovak",
    "sq": "Albanian",
    "sr": "Serbian",
    "su": "Sundanese",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Filipino",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh-CN": "Chinese"
}

lang_code = next((code for code, name in languages_dict.items() if name == language), None)

def file_downloader_html(bin_file, file_label = 'File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

c1, c2 = st.columns([4,3])

if len(inputtext) > 0 :
    try:
        output = translate(inputtext, lang_code)   
        with c1:
            st.text_area("Translated Text", output, height = 200)
        
        with c2:
            audio = gTTS(text = output, lang = lang_code, slow = False)
            audio.save("lang.mp3")
            audio_file = open("lang.mp3", "rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format = "audio/mp3")
            st.markdown(file_downloader_html("lang.mp3", "Audio File"), unsafe_allow_html = True)
    except Exception as e:
        st.error(e)