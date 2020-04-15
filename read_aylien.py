import json

data = []
with open('aylien-covid-news.jsonl') as f:
    counter = 0
    for line in f.readlines():
        if counter > 100:
            break
        data.append(json.loads(line))
        counter += 1