Dataset used: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset  

# Commands used to create environment
```
python3 -m venv .venv
```
```
source .venv/bin/activate
```
```
pip freeze > requirements.txt
```

# Commands to run after cloning repo
## Prerequisites
```
sudo apt install python3-venv \
    && sudo apt install uvicorn
```
## setup local python environment
```
python3 -m venv .venv
```
```
source .venv/bin/activate
```
```
pip install -r requirements.txt
```
```
python -m uvicorn app:app --reload
```
Open index.html in your browser

# Orientation questions to research

Should we combine title + text or treat them separately ?  
Is removing stopwords helpful for transformer models ?  
Does subject feature improve performance ?  

Planning to try Logistic Regression, LSTM and BERT.  