import json
import jsonlines

with open('data/eqa_v1.json') as f:
    data = json.load(f)

train_ids, val_ids, test_ids = set(data['splits']['train']), set(data['splits']['val']), set(data['splits']['test'])
train_data, val_data, test_data = [], [], []
answer2idx = {}

for houseId in data['questions']:
    if houseId in train_ids:
        for ques in data['questions'][houseId]:
            if ques['answer'] not in answer2idx:
                answer2idx[ques['answer']] = len(answer2idx)
            train_data.append({"question": ques['question'], "answer": answer2idx[ques['answer']]})

    elif houseId in val_ids:
        for ques in data['questions'][houseId]:
            if ques['answer'] not in answer2idx:
                answer2idx[ques['answer']] = len(answer2idx)
            val_data.append({"question": ques['question'], 'answer': answer2idx[ques['answer']]})

    elif houseId in test_ids:
        for ques in data['questions'][houseId]:
            if ques['answer'] not in answer2idx:
                answer2idx[ques['answer']] = len(answer2idx)
            test_data.append({'question': ques['question'], 'answer': answer2idx[ques['answer']]})

with jsonlines.open('data/train.json', mode='w') as writer:
    writer.write_all(train_data)

with jsonlines.open('data/val.json', mode='w') as writer:
    writer.write_all(val_data)

with jsonlines.open('data/test.json', mode='w') as writer:
    writer.write_all(test_data)

print(answer2idx)
idx2answer = {}
for k, v in answer2idx.items():
    idx2answer[v] = k
print(idx2answer)
