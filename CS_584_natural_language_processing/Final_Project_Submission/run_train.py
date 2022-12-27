!pip install transformers

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import random
import torch
import spacy
import pickle
from bisect import bisect_left, bisect_right
from sklearn.model_selection import train_test_split
from torch import nn
from transformers import BertTokenizer, BertModel
from keras_preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from IPython.display import clear_output


data = pd.read_csv('/content/drive/MyDrive/nlp_project/data/tsd_train.csv')
# data.shape

# data.head()

data_test = pd.read_csv('/content/drive/MyDrive/nlp_project/data/tsd_test.csv')
# data_test.head()

# data_test.shape

def clean_spans(df):
    for index, sample in df.iterrows():
        spans = eval(sample['spans'])
        correct_spans = spans.copy()
        chars = list(sample['text'])
        for i, char in enumerate(chars):
            if i == 0:
                continue
            if (i in spans) and (i - 1 not in spans) and (chars[i - 1].isalnum()) and (char.isalnum()):
                correct_spans.append(i - 1)
            elif (i - 1 in spans) and (i not in spans) and (chars[i - 1].isalnum()) and (char.isalnum()):
                correct_spans.append(i)
        correct_spans.sort()
        sample['spans'] = correct_spans
    return df

data = clean_spans(data)

def get_toxic_tokens(df):
    df['toxic_tokens'] = [list() for x in range(len(df.index))]
    for _, sample in df.iterrows():
        toxic = ''
        for i, char in enumerate(list(sample["text"])):        
            if i in sample["spans"]:
                toxic += char
            elif len(toxic):
                sample['toxic_tokens'].append(toxic)
                toxic = ''
        if toxic:  # added to take care of the last toxic token in text
            sample['toxic_tokens'].append(toxic)
    
    return df
    

data = get_toxic_tokens(data)
# data.head()

nlp = spacy.blank("en")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))

def create_token_labels(x):
    text = nlp(x['text'])
    token_start = [token.idx for token in text]
    token_end = [token.idx + len(token) - 1 for token in text]
    toxic_ranges = ranges(x['spans'])
    l = len(x['text'])
    for range in toxic_ranges:
        start, end = range
        if end >= l:
            end = l - 1
        while start < l and x['text'][start] == ' ':
            start += 1
        while end >= 0 and x['text'][end] == ' ':
            end -= 1
        start = token_start[bisect_right(token_start, start) - 1]        
        end = token_end[bisect_left(token_end, end)]
        if start >= end:
            print('Error:', x['text'])
            continue
        token_span = text.char_span(start, end + 1)
        for token in token_span:
                token.ent_type_ = 'toxic'
    
    bert_tokens = []
    token_labels = []
    for token in text:
        bert_subtokens = tokenizer.tokenize(token.text)
        bert_tokens += bert_subtokens
        token_labels += [int(token.ent_type_ == 'toxic') for _ in bert_subtokens]

    return bert_tokens, token_labels

def get_bert_tokens(df):
    df['bert_tokens'] = [list() for x in range(len(df.index))]
    df['token_labels'] = [list() for x in range(len(df.index))]

    for _, sample in df.iterrows():
        sample['bert_tokens'], sample['token_labels'] = create_token_labels(sample)
    
    return df

data = get_bert_tokens(data)

# data.head()

data_test['bert_tokens'] = [list() for x in range(len(data_test.index))]

for i, sample in data_test.iterrows():
    text = nlp(sample['text'])
    for token in text:
        sample['bert_tokens'] += tokenizer.tokenize(token.text)

# data_test.head()


maxlen_train = max([len(x) for x in data['bert_tokens']])
# print(maxlen_train)

maxlen_test = max([len(x) for x in data_test['bert_tokens']])
# print(maxlen_test)

maxlen = max(maxlen_train, maxlen_test)

random.seed(77)
np.random.seed(77)
torch.manual_seed(77)
torch.cuda.manual_seed(77)

train_tokens = list(map(lambda t: ['[CLS]'] + t[:maxlen - 2] + ['[SEP]'], data['bert_tokens']))
test_tokens = list(map(lambda t: ['[CLS]'] + t[:maxlen - 2] + ['[SEP]'], data_test['bert_tokens']))

def pad_tokens(tokens, max_len=maxlen):
    tokens_len = len(tokens)
    pad_len = max(0, max_len - tokens_len)
    return (
        pad_sequences([tokens], maxlen=max_len, truncating="post", padding="post", dtype="int"),
        np.concatenate([np.ones(tokens_len, dtype="int"), np.zeros(pad_len, dtype="int")], axis=0)
    )


def get_token_ids_and_masks(tokens):
    token_ids, masks = [], []

    for x in tokens:
        token_id, mask = pad_tokens(tokenizer.convert_tokens_to_ids(x))
        token_ids.append(token_id[0])
        masks.append(mask)  

    token_ids = np.array(token_ids)
    masks = np.array(masks)
    
    return token_ids, masks

train_token_ids, train_masks = get_token_ids_and_masks(train_tokens)
test_token_ids, test_masks = get_token_ids_and_masks(test_tokens)


train_token_labels = list(map(lambda t: [0] + t[:maxlen - 2] + [0], data['token_labels']))

train_y = pad_sequences(train_token_labels, maxlen=maxlen, truncating="post", padding="post")[:, :, None]

class BertClassifier(nn.Module):
    
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.hidden = nn.Linear(bert.config.hidden_size, 64)
        self.hidden_activation = nn.LeakyReLU(0.1)
        self.output = nn.Linear(64, 1)
        self.output_activation = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask, labels=None):
        
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_output = outputs[0]
        cls_output = self.hidden(cls_output)
        cls_output = self.hidden_activation(cls_output)
        cls_output = self.output(cls_output)
        cls_output = self.output_activation(cls_output)
        criterion = nn.BCELoss()
        loss = 0
        if labels is not None:
            loss = criterion(cls_output, labels.float())
        return loss, cls_output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertClassifier(BertModel.from_pretrained('bert-base-uncased')).to(device)

BATCH_SIZE = 16

train_dataset = TensorDataset(
    torch.tensor(train_token_ids),
    torch.tensor(train_masks),
    torch.tensor(train_y))
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = TensorDataset(torch.tensor(test_token_ids), torch.tensor(test_masks))
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)



optimizer = Adam(model.parameters(), lr=3e-6)
torch.cuda.empty_cache()

EPOCHS = 3
loss = nn.BCELoss()
total_len = len(train_token_ids)
batch_losses = []

for epoch_num in range(EPOCHS):
    model.train()
    for step_num, batch_data in enumerate(tqdm(train_dataloader)):
        token_ids, masks, labels = tuple(t.to(device) for t in batch_data)

        loss, _ = model(input_ids=token_ids, attention_mask=masks, labels=labels)

        train_loss = loss.item()
        
        model.zero_grad()
        loss.backward()

        clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
        optimizer.step()
        
        batch_losses.append(train_loss)

torch.save({
            'epoch': EPOCHS,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, '/content/drive/MyDrive/NLP_project/checkpoint')



