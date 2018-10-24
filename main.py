import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from torchtext import data
from torchtext import datasets
from torchtext.data import Field

from config import *


def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x


torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

QUESTION = Field(sequential=True, tokenize='spacy')
ANSWER = Field(sequential=False, use_vocab=False)
fields = {'question': ('question', QUESTION), 'answer': ('answer', ANSWER)}

train_data, valid_data, test_data = data.TabularDataset.splits(
    path='data',
    train='train.json',
    validation='val.json',
    test='test.json',
    format='json',
    fields=fields
)

QUESTION.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device, sort=False)


class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        # x = [sent len, batch size]
        embedded = self.embedding(x)
        # embedded = [sent len, batch size, emb dim]
        embedded = embedded.permute(1, 0, 2)
        # embedded = [batch size, sent len, emb dim]
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)
        # pooled = [batch size, embedding_dim]
        return F.log_softmax(self.fc(pooled))


INPUT_DIM = len(QUESTION.vocab)
model = FastText(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM)

pretrained_embeddings = QUESTION.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

import torch.optim as optim
optimizer = optim.Adam(model.parameters())

criterion = nn.NLLLoss()
model = model.to(device)
criterion = criterion.to(device)


def accuracy(preds, y):
    preds = preds.argmax(dim=1)
    correct = (preds == y).float() #convert into float for division
    acc = correct.sum()/len(correct)
    return acc


def mean_rank(preds, y):
    rank = [0.] * len(preds)
    for i in range(len(preds)):
        rank[i] = preds[i].gt(preds[i][y[i]]).sum() + 1
    mrank = np.mean(rank)
    return mrank


def train(model, iterator, optimizer, criterion):
    epoch_loss, epoch_acc, epoch_mean_rank = 0, 0, 0
    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.question)
        loss = criterion(predictions, batch.answer)
        acc = accuracy(predictions, batch.answer)
        rank = mean_rank(predictions, batch.answer)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_mean_rank += rank.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_mean_rank / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss, epoch_acc, epoch_mean_rank = 0, 0, 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.question)

            loss = criterion(predictions, batch.answer)
            acc = accuracy(predictions, batch.answer)
            rank = mean_rank(predictions, batch.answer)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_mean_rank += rank.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_mean_rank / len(iterator)


for epoch in range(N_EPOCHS):
    train_loss, train_acc, train_mean_rank = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc, valid_mean_rank = evaluate(model, valid_iterator, criterion)

    print(
        f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Train MR: {train_mean_rank:.2f}'
        f' | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% | Val. MR: {valid_mean_rank:.2f}')

test_loss, test_acc, test_mean_rank = evaluate(model, test_iterator, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% | Test MR: {test_mean_rank:.2f}')


