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
apt install python3.13-venv
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

# Orientation questions to research

Should we combine title + text or treat them separately ?  
Is removing stopwords helpful for transformer models ?  
Does subject feature improve performance ?  

Planning to try Logistic Regression, LSTM and BERT.  