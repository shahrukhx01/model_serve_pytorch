# Serving PyTorch Attention based Sentiment Classification Model with Streamlit, FastAPI and Docker

## How to run
1. Clone repository
```bash
git clone https://github.com/shahrukhx01/model_serve_pytorch.git
cd model_serve_pytorch
```
2. For ease of reproducibility and deployment build container images.
```bash
cd frontend
docker build -t hi_ben_attention_sentiment_frontend .
cd ../backend
docker build -t hi_ben_attention_sentiment_backend .
```
3. This will spin up both backend and frontend.
```bash
cd ..
docker-compose up
```
4. Access frontend/backend using the following url
![start_caQtDM_7id.sh](http://localhost:8501) 
<br/>
![Sentiment Analysis frontend](http://www.localhost:8501)
<br/>
[Sentiment Analysis backend](http://localhost:8080)
[I'm an inline-style link with title](https://localhost:8080 "Google's Homepage")

