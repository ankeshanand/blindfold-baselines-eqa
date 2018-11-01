import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import csv
import click

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchtext import data
from torchtext import datasets
from torchtext.data import Field

from config import *
from data.idx2answer import idx2answer

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
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
    fields=fields,
)

QUESTION.build_vocab(train_data, max_size=1000, vectors="glove.6B.100d")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device, sort=False, shuffle=True)


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
optimizer = optim.Adam(model.parameters(), lr=5e-3)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=False)

criterion = nn.NLLLoss()
model = model.to(device)
criterion = criterion.to(device)


def accuracy(preds, y):
    preds = preds.argmax(dim=1)
    correct = (preds == y).float() #convert into float for division
    acc = correct.sum()/len(correct)
    return acc


def inspect_predictions(preds, batch):
    preds = preds.argmax(dim=1)
    correct = (preds == batch.answer).float()
    questions = []
    qs = batch.question.permute(1, 0)
    for i in range(len(batch)):
        q = qs[i]
        questions.append([QUESTION.vocab.itos[q[j]] for j in range(len(q))])

    with open('results/preds.csv', 'a') as f:
        writer = csv.writer(f, delimiter=',')
        for i in range(len(questions)):
            writer.writerow([' '.join(questions[i]), idx2answer[preds[i].item()],
                             idx2answer[batch.answer[i].item()], correct[i].item()])


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


def evaluate(model, iterator, criterion, inspect=False):
    epoch_loss, epoch_acc, epoch_mean_rank = 0, 0, 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.question)

            loss = criterion(predictions, batch.answer)
            acc = accuracy(predictions, batch.answer)
            rank = mean_rank(predictions, batch.answer)
            if inspect:
                inspect_predictions(predictions, batch)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_mean_rank += rank.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_mean_rank / len(iterator)


min_valid_loss = 100.
min_valid_loss_epoch = None
corr_test_acc = None
inspect = False

for epoch in range(N_EPOCHS):
    train_loss, train_acc, train_mean_rank = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc, valid_mean_rank = evaluate(model, valid_iterator, criterion)

    if valid_loss < min_valid_loss:
        min_valid_loss = valid_loss
        min_valid_loss_epoch = epoch
        inspect = True
        open('results/preds.csv', 'w').close()

    test_loss, test_acc, test_mean_rank = evaluate(model, test_iterator, criterion, inspect=inspect)
    scheduler.step(valid_loss)

    if inspect:
        corr_test_acc = test_acc

    print(
        f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Train MR: {train_mean_rank:.2f}'
        f' | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% | Val. MR: {valid_mean_rank:.2f}')
    print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% | Test MR: {test_mean_rank:.2f}')
    inspect = False

print(f'Got minimum valid loss: {min_valid_loss:.3f}, at Epoch: {min_valid_loss_epoch+1:02}')
print(f'Test accuracy at minimum valid loss checkpoint: {corr_test_acc:.6f}')