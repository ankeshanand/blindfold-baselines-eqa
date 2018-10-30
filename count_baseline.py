import numpy as np
from torchtext import data
from torchtext.data import Field
from config import *


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


#Train
hash_table = {}
for example in train_data.examples:
    question_key = tuple(example.question)
    if question_key not in hash_table:
        hash_table[question_key] = {}

    key_answer = example.answer
    if key_answer not in hash_table[question_key]:
        hash_table[question_key][key_answer] = 1
    else:
        hash_table[question_key][key_answer] += 1

def compute_pred(examples):
    y = [] #answers
    preds = []
    total_not_found = 0
    keys_all = hash_table.keys()
    for example in examples:
        if tuple(example.question) not in hash_table:
            quest_key = None
            best_count = -1
            for key in keys_all:
                count = len(set(tuple(example.question)) & set(key))
                if best_count < count:
                    quest_key = key
                    best_count = count
            total_not_found +=1
        else:
            quest_key = tuple(example.question)
        y.append(example.answer)
        question_tab = hash_table[quest_key]
        pred = 0
        best_count = 0
        for ans_key,val in question_tab.items():
            if best_count < val:
                pred = ans_key
                best_count = val
        preds.append(pred)

    return np.array(y),np.array(preds),total_not_found

def accuracy(preds, y):
    correct = np.array(preds == y,dtype='float64')
    acc = correct.sum()/len(correct)
    return acc

#train acc
y,pred, _ = compute_pred(train_data)
train_acc = accuracy(pred,y)

#Eval val
y,pred,total_not_found_val=compute_pred(valid_data)

val_acc = accuracy(pred,y)
#Eval test
y,pred, total_not_found_test = compute_pred(test_data)
test_acc = accuracy(pred,y)
print(f'|Train Acc: {train_acc*100:.2f}%')
print(f'|Test Acc: {test_acc*100:.2f}% Total not exact:{total_not_found_test:d}')
print(f'|Valid Acc: {val_acc*100:.2f}% Total not exact:{total_not_found_val:d}')
