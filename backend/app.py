# backend/main.py

import uuid

import uvicorn
from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware

from inference import InferSentiment


class SentimentText(BaseModel):
    lang: str
    text: str
    
langs = ['hindi', 'bengali']
app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict/{lang}")
def get_sentiment(sentiment_text: SentimentText):
   
    if sentiment_text.lang not in langs:
        return {"message": "invalid language"}

    infer_sentiment = InferSentiment(lang=sentiment_text.lang, text=sentiment_text.text)
    response = infer_sentiment.predict()
   
    json_response = JSONResponse(content=jsonable_encoder({
        "prediction": response['prediction'],
        "attention_weights": response['attention_weights'],
        "clean_text": response['clean_text']
        }))

    return json_response

@app.get("/")
def read_root():
    json_response = JSONResponse(content=jsonable_encoder({
        "message": "Hello from sentiment analyzer!"
        }))
    return json_response


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080)