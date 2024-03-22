import os
import sys
import json
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB

data_dir = sys.argv[1]
save_dir = sys.argv[2]

train_data_path = os.path.join(data_dir, "train.json")
valid_data_path = os.path.join(data_dir, "valid.json")
valid_new_data_path = os.path.join(data_dir, "valid_new.json")

# read the data
with open(train_data_path, 'r', encoding='utf-8') as fp:
    train_data = json.load(fp)
train_df = pd.DataFrame(train_data)

with open(valid_data_path, 'r', encoding='utf-8') as fp:
    valid_data = json.load(fp)
valid_df = pd.DataFrame(valid_data)

with open(valid_new_data_path, 'r', encoding='utf-8') as fp:
    valid_new_data = json.load(fp)
valid_new_df = pd.DataFrame(valid_new_data)


# Split the data into training and validing sets
X_train, y_train = train_df['text'], train_df['langid']
X_valid, y_valid = valid_df['text'], valid_df['langid']
X_valid_new, y_valid_new = valid_new_df['text'], valid_new_df['langid']

X_master = pd.concat([X_train, X_valid, X_valid_new], ignore_index=True)
y_master = pd.concat([y_train, y_valid, y_valid_new], ignore_index=True)


# vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    sublinear_tf=True,
    token_pattern=r'(?u)\b\w+\b',
)

# train the model on combined data
model = make_pipeline(vectorizer, MultinomialNB(alpha=0.001))
model.fit(X_master, y_master)

# save the model
model_path = os.path.join(save_dir, "model.pkl")
joblib.dump(model, model_path)
print("Model saved at:", model_path)