import sys
import json
import gensim.downloader as api

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

import nltk
from nltk.tokenize import word_tokenize
import pickle

train_file = sys.argv[1]
val_file = sys.argv[2]

print(train_file)
print(val_file)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

train_data = [json.loads(line.strip()) for line in open(train_file, 'r', encoding='utf-8')]
val_data = [json.loads(line.strip()) for line in open(val_file, 'r', encoding='utf-8')]

train_data = train_data + val_data

train_data = sorted(train_data, key=lambda x: len(x['table']['rows']) * len(x['table']['rows'][0]))
# val_data = sorted(val_data, key=lambda x: len(x['table']['rows']) * len(x['table']['rows'][0]))

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

class QuestionColumnDataset(Dataset):
    def __init__(self, questions, columns, labels, vocab, device):
        self.device = device
        self.vocab = vocab
        self.questions = questions
        self.columns = columns
        self.labels = labels
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        return {
            'questions': self.questions[idx], 
            'columns': self.columns[idx],
            'labels': self.labels[idx]
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

    labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
    return {'questions': questions_padded, 'columns': columns_padded, 'labels': labels, 'num_cols': num_cols}


questions = [entry['question'] for entry in train_data]
columns = [entry['table']['cols'] for entry in train_data]  
labels = [entry['label_col'][0] for entry in train_data]
qids = [entry['qid'] for entry in train_data]
labels = [cols.index(label) if label in cols else -1 for cols, label in zip(columns, labels)]
questions = [tokenize(question, vocab) for question in questions]
columns = [[tokenize(col, vocab) for col in cols] for cols in columns]

dataset = QuestionColumnDataset(questions, columns, labels, vocab, device)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=custom_column_collate_fn)



# val_questions = [entry['question'] for entry in val_data]
# val_columns = [entry['table']['cols'] for entry in val_data]  
# val_labels = [entry['label_col'][0] for entry in val_data]
# val_qids = [entry['qid'] for entry in val_data]
# val_labels = [cols.index(label) if label in cols else -1 for cols, label in zip(val_columns, val_labels)]
# val_questions = [tokenize(question, vocab) for question in val_questions]
# val_columns = [[tokenize(col, vocab) for col in cols] for cols in val_columns]

# val_dataset = QuestionColumnDataset(val_questions, val_columns, val_labels, vocab, device)
# validation_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=custom_column_collate_fn)


# num_new_words = len(vocab) - len(weights)
# if num_new_words > 0:
#     new_embeddings = torch.randn(num_new_words, 50)
#     new_embeddings = new_embeddings / torch.norm(new_embeddings, dim=1, keepdim=True) * 4
#     weights = torch.cat((weights, new_embeddings), dim=0)


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

pretrained = False
if pretrained:
    model_path = 'column_predictor_model_2.pth'
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
        model.cuda()  # Move model to GPU if available
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
model.to(device)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    print("training")
    model.train()
    running_loss = 0.0
    
    for batch in dataloader:
        questions = batch['questions'].to(device)
        columns = batch['columns'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        similarities = model(questions, columns)
        loss = criterion(similarities, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataloader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    
#     accuracy = evaluate(model, validation_dataloader, device)
#     print(f'Validation Accuracy: {accuracy:.4f}')
    
#     accuracy = evaluate(model, dataloader, device)
#     print(f'Train Accuracy: {accuracy:.4f}')

def evaluate(model, dataloader, device):
    model.eval()  
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in dataloader:
            questions = batch['questions'].to(device)
            columns = batch['columns'].to(device)
            labels = batch['labels'].to(device)

            num_cols = batch['num_cols'].to(device)
            
            similarities = model(questions, columns)
            
            for i in range(similarities.size(0)): 
                if num_cols[i] < similarities.size(1):
                    similarities[i, num_cols[i]:] = float('-inf')

            predictions = similarities.argmax(dim=1)

            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
    
    accuracy = correct_predictions / total_predictions
    return accuracy

# model.to(device)

# accuracy = evaluate(model, dataloader, device)
# print(f'Training Accuracy: {accuracy:.4f}')


class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()
        self.feature_dim = feature_dim
        self.query_projection = nn.Linear(feature_dim, feature_dim)
        self.key_projection = nn.Linear(feature_dim, feature_dim)
        self.value_projection = nn.Linear(feature_dim, feature_dim)
        self.scale = torch.sqrt(torch.FloatTensor([feature_dim]))

    def forward(self, x):
        batch_size, num_columns, _ = x.size()
        
        queries = self.query_projection(x)
        keys = self.key_projection(x)
        values = self.value_projection(x)
        
        scores = torch.bmm(queries, keys.transpose(1, 2)) / self.scale.to(x.device)  
        attention_weights = F.softmax(scores, dim=-1)
        
        weighted_sum = torch.bmm(attention_weights, values) 
        return weighted_sum

class QuestionRowClassifierWithAttention(nn.Module):
    def __init__(self, embedding_dim, transformer_heads, transformer_layers, num_classes=2, num_columns=0):
        super(QuestionRowClassifierWithAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_columns = num_columns
        self.transformer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=transformer_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer, num_layers=transformer_layers)
        self.attention = Attention(embedding_dim)
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.column_name_embeddings = nn.Embedding(num_columns, embedding_dim)  

    def forward(self, question_embeddings, row_embeddings, column_indices):
        transformed_question = self.transformer_encoder(question_embeddings)
        question_repr = transformed_question.mean(dim=1)
        column_name_embeds = self.column_name_embeddings(column_indices)  # shape: (batch_size, num_columns, embedding_dim)
        batch_size, num_rows, num_columns, _ = row_embeddings.size()
        row_embeddings_reshaped = row_embeddings.view(batch_size * num_rows, num_columns, -1)  # Reshape for attention
        column_name_embeds_expanded = column_name_embeds.unsqueeze(1).repeat(1, num_rows, 1, 1).view(batch_size * num_rows, num_columns, -1)
        combined_embeddings = torch.cat((row_embeddings_reshaped, column_name_embeds_expanded), dim=-1)
        row_repr_with_attention = self.attention(combined_embeddings)
        row_repr = row_repr_with_attention.view(batch_size, num_rows, -1)  # Reshape back to separate rows

        similarities = torch.cosine_similarity(question_repr.unsqueeze(1), row_repr, dim=-1)
        logits = self.fc(similarities.unsqueeze(-1))

        return logits
    
class QuestionRowDataset(Dataset):
    def __init__(self, questions, columns, rows, labels, keep_indices):
        self.questions = questions
        self.columns = columns
        self.rows = rows
        self.keep_indices = keep_indices
        self.labels = labels
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        return {
            'questions': self.questions[idx],
            'rows': self.rows[idx],
            'labels': self.labels[idx]
        }
    
def one_hot_encode_labels(labels, rows_per_question_lengths):
    one_hot_labels = []
    for label, length in zip(labels, rows_per_question_lengths):
        label_vector = [0] * length
        for l in label:
            if l < length:
                label_vector[l] = 1
        one_hot_labels.append(label_vector)
    return one_hot_labels

def row_collate_fn(batch):
    # Pad questions
    questions_padded = pad_sequence([torch.tensor(item['questions'], dtype=torch.long) for item in batch], batch_first=True, padding_value=0)

    # Determine the maximums for rows, columns, and cell values
    max_rows = max(len(item['rows']) for item in batch)
    max_cols = max(max(len(row) for row in item['rows']) for item in batch)
    max_cell_value_len = max(max(max(len(value) for value in row) for row in item['rows']) for item in batch)

    print(max_rows, max_cols, max_cell_value_len)
    rows_padded_list = []
    labels_padded_list = []

    for item in batch:
        question_rows = item['rows']
        labels = item['labels']

        padded_rows_for_question = torch.zeros((max_rows, max_cols, max_cell_value_len), dtype=torch.long)

        for i, row in enumerate(question_rows):
            padded_row = torch.zeros((max_cols, max_cell_value_len), dtype=torch.long)
            for j, cell_value in enumerate(row):
                cell_value_tensor = torch.tensor(cell_value, dtype=torch.long)
                padded_row[j, :len(cell_value)] = cell_value_tensor[:max_cell_value_len]
            padded_rows_for_question[i] = padded_row

        rows_padded_list.append(padded_rows_for_question.unsqueeze(0)) 

        label_tensor = torch.zeros(max_rows, dtype=torch.float)
        for label in labels:
            if label < max_rows:
                label_tensor[label] = 1.0
        labels_padded_list.append(label_tensor.unsqueeze(0))

    rows_padded = torch.cat(rows_padded_list, dim=0)
    labels_padded = torch.cat(labels_padded_list, dim=0)

    return {'questions': questions_padded, 'rows': rows_padded, 'labels': labels_padded}


# Save the model checkpoint
model_path = 'predictor_model.pth'
torch.save(model.state_dict(), model_path)

# with open("vocab_file", 'wb') as f:
#     pickle.dump(vocab, f)

# with open("weights", 'wb') as f:
#     pickle.dump(weights, f)