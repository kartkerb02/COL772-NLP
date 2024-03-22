import sys
import numpy as np
import pandas as pd
import re

import json
import gensim.downloader as api

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

import nltk

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from collections import Counter
import datefinder
import pickle

from tqdm import tqdm 


test_file = sys.argv[1]
pred_file = sys.argv[2]

# print(test_file)
# print(pred_file)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

test_data = [json.loads(line.strip()) for line in open(test_file, 'r', encoding='utf-8')]
# test_data = sorted(test_data, key=lambda x: len(x['table']['rows']) * len(x['table']['rows'][0]))

# embeds = api.load("glove-wiki-gigaword-50")
# weights = torch.FloatTensor(embeds.vectors)
# with open("weights", 'rb') as f:
#     weights = pickle.load(f)

# embedding_dim = weights.size(1)

# with open("vocab_file", 'rb') as f:
#     vocab = pickle.load(f)

embeds = api.load("glove-wiki-gigaword-50")
weights = torch.FloatTensor(embeds.vectors)

embedding_dim = weights.size(1)

vocab = embeds.key_to_index 
vocab['<UNK>'] = len(vocab)
unk_embedding = torch.randn(1, embedding_dim)
unk_embedding = unk_embedding / torch.norm(unk_embedding) * torch.norm(weights.mean(dim=0))
weights = torch.cat([weights, unk_embedding], dim=0)

# def tokenize(text, vocab):
#     tokens = word_tokenize(text.lower())
#     token_indices = []
#     for token in tokens:
#         if token not in vocab:
#             vocab[token] = len(vocab)
#         token_indices.append(vocab[token])
#     return token_indices
def tokenize(text, vocab):
    tokens = word_tokenize(text.lower())
    return [vocab.get(token, vocab['<UNK>']) for token in tokens]

def token_text(text):
    tokens = word_tokenize(text.lower())
    # stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return set(tokens)

class QuestionColumnDataset(Dataset):
    def __init__(self, questions, columns, vocab, device):
        self.device = device
        self.vocab = vocab
        self.questions = questions
        self.columns = columns
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        return {
            'questions': self.questions[idx], 
            'columns': self.columns[idx],
        }
    

def custom_column_collate_fn(batch):
    questions = [torch.tensor(item['questions'], dtype=torch.long) for item in batch]
    num_cols = torch.tensor([len(item['columns']) for item in batch])
    
    columns_nested = [item['columns'] for item in batch] 
    questions_padded = pad_sequence(questions, batch_first=True, padding_value=0)

    max_num_cols = max(len(cols) for cols in columns_nested)
    max_col_len = max(max(len(col) for col in cols) for cols in columns_nested if cols)

    columns_padded = torch.zeros(len(batch), max_num_cols, max_col_len, dtype=torch.long)

    for i, cols in enumerate(columns_nested):
        for j, col in enumerate(cols):
            col_len = len(col)
            columns_padded[i, j, :col_len] = torch.tensor(col, dtype=torch.long)

    return {'questions': questions_padded, 'columns': columns_padded, 'num_cols': num_cols}


questions = [entry['question'] for entry in test_data]
columns = [entry['table']['cols'] for entry in test_data]  
qids = [entry['qid'] for entry in test_data]
questions = [tokenize(question, vocab) for question in questions]
columns = [[tokenize(col, vocab) for col in cols] for cols in columns]

dataset = QuestionColumnDataset(questions, columns, vocab, device)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=custom_column_collate_fn)

class ColumnPredictorModel(nn.Module):
    def __init__(self, pretrained_embedding_weights, hidden_dim):
        super(ColumnPredictorModel, self).__init__()
        self.embedding_dim = pretrained_embedding_weights.size(1)
        
        self.question_embedding = nn.Embedding.from_pretrained(pretrained_embedding_weights, freeze=False)
        self.question_lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        
        self.question_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.column_projection = nn.Linear(self.embedding_dim, hidden_dim)

    def forward(self, question, columns):
        q_embedded = self.question_embedding(question)
        _, (q_hidden, _) = self.question_lstm(q_embedded)
        q_hidden_concat = torch.cat((q_hidden[-2,:,:], q_hidden[-1,:,:]), dim=1)
        q_encoded = self.question_projection(q_hidden_concat)

        c_embedded = self.question_embedding(columns)
        c_encoded_sum = c_embedded.sum(dim=2)  
        c_encoded = self.column_projection(c_encoded_sum)  
        q_encoded = q_encoded.unsqueeze(1)  
        similarities = F.cosine_similarity(q_encoded, c_encoded, dim=2)

        return similarities

model = ColumnPredictorModel(pretrained_embedding_weights=weights, hidden_dim=256)
model_path = 'predictor.pth'

if torch.cuda.is_available():
    model.load_state_dict(torch.load(model_path))
    model.cuda()
else:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

def predict_columns(model, dataloader, device):
    model.eval()
    all_predicted_columns = []

    with torch.no_grad(): 
        for batch in dataloader:
            questions = batch['questions'].to(device)
            columns = batch['columns'].to(device)
            num_cols = batch['num_cols'].to(device)
            similarities = model(questions, columns)
            
            for i in range(similarities.size(0)): 
                if num_cols[i] < similarities.size(1):  
                    similarities[i, num_cols[i]:] = float('-inf')
                    
            predictions = similarities.argmax(dim=1)

            all_predicted_columns.append(predictions)
    return torch.cat(all_predicted_columns).tolist()

pred_cols = predict_columns(model, dataloader, device)

cols = []
for i, ind in enumerate(pred_cols):
    cols.append(test_data[i]['table']['cols'][ind])

def is_number(string):
    return len(extract_numbers(string)) > 0

def find_real_number(question):
    numbers = extract_numbers(question)
    if len(numbers) > 0:
        return numbers[0]
    return None

def extract_numbers(question):
    numbers = []
    # Adjust regex for improved number extraction
    number_pattern = r'\d+(?:,\d{3})*(?:\.\d+)?|\d{1,3}(?:,\d{3})*(?:\.\d+)?'
    for match in re.finditer(number_pattern, question.replace(' ', '')):
        num_str = match.group().replace(',', '')  # Convert to number, handling commas
        try:
            number = int(num_str)
            numbers.append(number)
        except ValueError:
            pass
    return numbers


def is_date(string):
    potential_dates = list(datefinder.find_dates(string, strict=True))
    return bool(potential_dates)

def extract_date_from_string(string):
    dates = list(datefinder.find_dates(string, strict=True))
    if dates:
        return dates[0]
    return None

def extract_dates_and_numbers(question):
    updated_question = question
    dates = []
    extracted_numbers = []

    matches = list(datefinder.find_dates(question, strict=True, index=True))
    matches.sort(key=lambda x: x[1][0], reverse=True)
    
    for match, (start_idx, end_idx) in matches:
        if end_idx - start_idx > 7:
            formatted_date = match.strftime('%Y-%m-%d')
            dates.append(match)  # Keep the datetime object for later comparison
            
            prefix = ' ' if start_idx > 0 and question[start_idx-1] != ' ' else ''
            suffix = ' ' if end_idx < len(question) and question[end_idx] not in [' ', '.', ','] else ''
            
            updated_question = updated_question[:start_idx] + prefix + formatted_date + suffix + updated_question[end_idx:]

    extracted_numbers = extract_numbers(updated_question)
    
    numbers = []
    for number in extracted_numbers:
        include_number = True
        for date in dates:
            year, month, day = date.year, date.month, date.day
            if int(number) in [year, month, day]:
                include_number = False
                break
        if include_number:
            numbers.append(number)
            
    updated_question = re.sub(r'(\d),(\d{3})', r'\1\2', updated_question)

    return updated_question, dates, numbers

def infer_column_types(df):
    column_types = []

    for col_name in df.columns:
        type_counts = Counter()

        for value in df[col_name]:
            value_str = str(value)
            if is_date(value_str):
                type_counts['date'] += 1
            elif is_number(value_str):
                type_counts['number'] += 1
            else:
                type_counts['text'] += 1

        if(type_counts['date'] > 0.9 * len(df)):
            column_types.append('date')
        elif (type_counts['number'] > 0.9 * len(df)):
            column_types.append('number')
        else:
            column_types.append('text')
    
    return column_types

def evaluate_and_predict_columns(model, dataloader, device):
    model.eval()
    all_predicted_columns = []

    with torch.no_grad(): 
        for batch in dataloader:
            questions = batch['questions'].to(device)
            columns = batch['columns'].to(device)
            num_cols = batch['num_cols'].to(device)
            similarities = model(questions, columns)
            
            for i in range(similarities.size(0)):  # Loop over batch size
                if num_cols[i] < similarities.size(1):  # Check if masking is needed
                    similarities[i, num_cols[i]:] = float('-inf')
                    
            predictions = similarities.argmax(dim=1)

            all_predicted_columns.append(predictions)
    return torch.cat(all_predicted_columns).tolist()

stemmer = PorterStemmer()

def get_stems(text):
    tokens = word_tokenize(text.lower())
    stems = set(stemmer.stem(token) for token in tokens)
    return stems

def get_cols(df, question, threshold=0.5):
    sf = 2

    question_stems = get_stems(question)
    
    columns_above_threshold = []
    for col in df.columns:
        col_stems = get_stems(col)
        col_len = sum(len(word)**sf for word in col_stems)
        
        intersecting_stems = question_stems.intersection(col_stems)
        intersecting_len = sum(len(stem)**sf for stem in intersecting_stems)

        overlap_ratio = intersecting_len / col_len if col_len > 0 else 0
        
        include = False
        if overlap_ratio > threshold:
            include = True
        else:
            for cell in df[col]:
                if str(cell).lower() in question.lower():
                    include = True
                    break
                    
        if include:
            columns_above_threshold.append(col)

    return columns_above_threshold

def date_comparison(x, ref_date):
    try:
        date = extract_date_from_string(x)
        
        if date > ref_date:
            out = f"after {ref_date}"
        elif date < ref_date:
            out = f"before {ref_date}"
        else:
            out = f"on {ref_date}"
    except:
        # print(f"catching date error {x}, {ref_date}")
        out = ""
    return out

def compare_values(target_value, column_values):
    comparison_results = [f'greater than {int(target_value)}' if value > target_value else f'less than {int(target_value)}' if value < target_value else f'equal is {target_value}' for value in column_values]
    return comparison_results

def handle_types(dfs, ques):
    new_dfs = []
    for i, df in enumerate(dfs):
        col_types = infer_column_types(df)
        ques[i], dates, numbers = extract_dates_and_numbers(ques[i])
        
        for col_index, col_name in enumerate(df.columns):
            if col_types[col_index] == 'date':
                for date in dates:
                    column_values = df[col_name].apply(lambda x: date_comparison(x, date))
                    df[f'Compare {col_name} {date}'] = column_values
            elif col_types[col_index] == 'number':
                for num in numbers:
                    column_values = df[col_name].apply(find_real_number)
                    mean = column_values.mean()
                    std = column_values.std()
                    if np.abs(mean - num) <= 3 * std:
                        comparison_results = compare_values(num, column_values)
                        df[f'Compare {col_name} {int(num)}'] = comparison_results
        
        new_dfs.append(df)
    return  new_dfs, ques

def process_data(data, model, dataloader, device):
    predicted_cols = evaluate_and_predict_columns(model, dataloader, device)
    dfs = []
    ques = []
    for i, item in enumerate(data):
        df = pd.DataFrame(item['table']['rows'], columns=item['table']['cols'])
        ques.append(item['question'])
        
        new_cols = get_cols(df, item['question'])
        
        pred_col = df.columns[predicted_cols[i]]
        if pred_col not in new_cols:
            new_cols.append(pred_col)
    
        df = df[new_cols]
        dfs.append(df)

    return dfs, ques

dfs, ques = process_data(test_data, model, dataloader, device)
dfs, ques = handle_types(dfs, ques)

def pred_rows(dfs, ques):
    highest_overlap_indices = []
    
    for i, df in enumerate(dfs):
        # Assuming token_text returns a set of tokens
        question_tokens = token_text(ques[i])

        highest_overlap = 0
        highest_overlap_index = 0

        for index, row in df.iterrows():
            row_content = ' '.join(str(cell) for cell in row)
            row_tokens = token_text(row_content)

            # Calculate overlap based on the length of overlapping tokens
            overlap = sum(len(token) for token in question_tokens if token in row_tokens)

            if overlap > highest_overlap:
                highest_overlap = overlap
                highest_overlap_index = index

        highest_overlap_indices.append(highest_overlap_index)
    return highest_overlap_indices

# from sklearn.feature_extraction.text import TfidfVectorizer

# def compute_tfidf(dfs, ques):
#     vectorizer = TfidfVectorizer(tokenizer=token_text, lowercase=True)
    
#     texts = []
#     for df, question in zip(dfs, ques):
#         texts.append(question)
#         for index, row in df.iterrows():
#             row_content = ' '.join(str(cell) for cell in row)
#             texts.append(row_content)
    
#     tfidf_matrix = vectorizer.fit_transform(texts)
#     return tfidf_matrix, vectorizer

# def pred_rows(dfs, ques):
#     tfidf_matrix = compute_tfidf(dfs, ques)
    
#     highest_overlap_indices = []
#     offset = 0 
    
#     for i, df in enumerate(dfs):
#         question_vec = tfidf_matrix[offset].toarray()
#         highest_similarity = 0
#         highest_overlap_index = 0
        
#         for index, _ in df.iterrows():
#             row_vec = tfidf_matrix[offset + index + 1].toarray() 
#             similarity = np.dot(question_vec, row_vec.T)
            
#             if similarity > highest_similarity:
#                 highest_similarity = similarity
#                 highest_overlap_index = index
        
#         highest_overlap_indices.append(highest_overlap_index)
#         offset += len(df) + 1
    
#     return highest_overlap_indices

rows = pred_rows(dfs, ques)


data_to_write = []
for i in range(len(qids)):
    data_to_write.append({
        "label_col": [cols[i]],
        "label_cell": [[rows[i], cols[i]]],
        "label_row": [rows[i]],
        "qid": qids[i]
    })

with open(pred_file, 'w') as outfile:
    for entry in data_to_write:
        json.dump(entry, outfile)
        outfile.write('\n')