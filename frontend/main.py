import requests
import streamlit as st
from PIL import Image
import json

langs = {
    "hindi": "hindi",
    "bengali": "bengali"
}

# https://discuss.streamlit.io/t/version-0-64-0-deprecation-warning-for-st-file-uploader-decoding/4465
st.set_option("deprecation.showfileUploaderEncoding", False)

# defines an h1 header
st.title("Hindi/Bengali Hatespeech Detection")

# displays the select widget for the styles
lang = st.selectbox("Choose the style", [i for i in langs.keys()])

# displays text area for inputting text
text = st.text_area("Enter Text in {}".format(lang), 'शुभ दिवस! इस भावना विश्लेषण सॉफ्टवेअर का स्वागत है।')

## label mappings
sentiments = {1: 'Hate speech', 0: 'Non Hate speech'}
# displays a button
if st.button("Predict Sentiment"):
    if text is not None and lang is not None:
        data = {"text": text, "lang": lang}
        res_data = requests.post(f"http://backend:8080/predict/{lang}", json=data)
        response = res_data.json()
        
        ## prepare markdown for visualizing attention
        text_attention = ''
        for text, attention_weight in zip(response.get("clean_text").split(), response.get("attention_weights")):
            text_attention += '<span style="background: rgba(255, 0, 0, {});">'.format(attention_weight*2)+text+' </span>' 
       
        st.markdown('Predicted Sentiment: <strong>{}</strong>'.format(sentiments[response.get("prediction")]), unsafe_allow_html=True)
        st.markdown(text_attention, unsafe_allow_html=True)