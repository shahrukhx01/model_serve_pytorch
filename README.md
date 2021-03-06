# Serving PyTorch Attention based Sentiment Classification Model with Streamlit, FastAPI and Docker
![](https://github.com/shahrukhx01/model_serve_pytorch/blob/main/output.gif)
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

Sentiment Analysis frontend<br/>
`http://localhost:8501`
<br/>
Sentiment Analysis backend
<br/>
`http://localhost:8080`
