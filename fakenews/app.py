from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import re
import torch
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, BertForSequenceClassification

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fake news classification ===================================================
## Regression model ==========================================================
# Load the pipeline (vectorizer + model bundled together)
fakenews_pipeline = joblib.load("fakenews_regr_pipeline_acc_98.54%.pkl")

# Emotion classification =====================================================
## Regression model ==========================================================
emotion_pipeline = joblib.load("emotion_regr_pipeline_acc_93.36%.pkl")
EMOTION_MAP = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}

# Category classification =====================================================
## BERT category model ========================================================
BERT_MODEL_PATH = "category_bert_model_acc_71.02%"
BERT_MAX_LEN = 64

category_label_encoder = joblib.load("category_bert_label_encoder.pkl")

bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
bert_model     = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
bert_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = bert_model.to(device)

print(f"BERT loaded on {device}")

## Regression model ============================================================
# category_pipeline = joblib.load("category_regr_pipeline_acc_59.23%.pkl")
# category_label_encoder = joblib.load('category_regr_label_encoder.pkl')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def predict_category_bert(text: str):
    """Run uncleaned text through BERT and return (label, confidence)."""
    encoding = bert_tokenizer(
        text,
        max_length=BERT_MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids      = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        logits = bert_model(input_ids=input_ids, attention_mask=attention_mask).logits

    probs      = torch.softmax(logits, dim=1)
    pred_idx   = probs.argmax(dim=1).item()
    confidence = probs[0, pred_idx].item()
    label      = category_label_encoder.inverse_transform([pred_idx])[0]
    return label, confidence

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

    # # Category regression model
    # category_prediction = category_pipeline.predict([cleaned])[0]
    # category_prediction_label = category_label_encoder.inverse_transform([category_prediction])[0]
    # category_probability = category_pipeline.predict_proba([cleaned])[0].max()

    # BERT uses the raw text — it does its own tokenization
    category_prediction_label, category_probability = predict_category_bert(news.text)

    emotion_prediction = emotion_pipeline.predict([cleaned])[0]
    emotion_probability = emotion_pipeline.predict_proba([cleaned])[0].max()
    emotion_prediction_label = EMOTION_MAP[emotion_prediction]

    label = "FAKE" if fakenews_prediction == 1 else "REAL"

    return {
        "prediction": label,
        "confidence": round(float(fakenews_probability), 4),
        "category_prediction": category_prediction_label,
        "category_confidence": round(float(category_probability), 4),
        "emotion_prediction": emotion_prediction_label,
        "emotion_confidence": round(float(emotion_probability), 4),
    }