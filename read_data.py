import jsonlines
import json

counter = 0
max_count = 3

with jsonlines.open('aylien-covid-news.jsonl') as f:
    for line in f.iter():
        if(counter  >= max_count):
            break
        print(counter)
        # d = json.loads(line)
        # print(line.keys())
        # print(line.items())
        # print(line["sentiment"])
        print(line['body'])
        counter += 1
