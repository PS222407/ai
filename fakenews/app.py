from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pipeline (vectorizer + model bundled together)
fakenews_pipeline = joblib.load("fakenews_pipeline_acc_98.54%.pkl")
category_pipeline = joblib.load("category_pipeline_acc_59.23%.pkl")
category_label_encoder = joblib.load('category_label_encoder.pkl')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

class NewsRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "API running"}

@app.post("/predict")
def predict(news: NewsRequest):
    cleaned = preprocess_text(news.text)
    fakenews_prediction = fakenews_pipeline.predict([cleaned])[0]
    fakenews_probability = fakenews_pipeline.predict_proba([cleaned])[0].max()
    category_prediction = category_pipeline.predict([cleaned])[0]
    category_prediction_label = category_label_encoder.inverse_transform([category_prediction])[0]
    category_probability = category_pipeline.predict_proba([cleaned])[0].max()

    label = "FAKE" if fakenews_prediction == 1 else "REAL"

    return {
        "prediction": label,
        "confidence": round(float(fakenews_probability), 4),
        "category_prediction": category_prediction_label,
        "category_confidence": round(float(category_probability), 4)
    }