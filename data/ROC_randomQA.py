import csv
import os
import random
import json

file_path = "data/rocstories/ROCStories__spring2016 - ROCStories_spring2016.csv"
with open(file_path, 'r', encoding='utf-8') as f:
    data = csv.reader(f)
    data_list = [' '.join(item[2:-1]) for item in data][1:] # select first 4 sentences
with open(file_path, 'r', encoding='utf-8') as f: # have to load twich due to csv.reader behaviour
    data = csv.reader(f)
    gold_text_list = [item[-1] for item in data][1:] # select last sentences

# file_path = "data/rocstories/ROCStories_winter2017 - ROCStories_winter2017.csv"
# with open(file_path, 'r', encoding='utf-8') as f:
#     data = csv.reader(f)
#     data_list += [' '.join(item[2:-1]) for item in data][1:] # select first 4 sentences
# with open(file_path, 'r', encoding='utf-8') as f: # have to load twich due to csv.reader behaviour
#     data = csv.reader(f)
#     gold_text_list += [item[-1] for item in data][1:] # select last sentences


output_path = os.path.join("data/rocstories_qa_16random", "ROCStories_trn.jsonl")
qa_format_writer = open(output_path, 'w', encoding='utf-8')

neg_option_list1 = list(gold_text_list)
neg_option_list2 = list(gold_text_list)
random.shuffle(neg_option_list1)
random.shuffle(neg_option_list2)

for context, gold_text, neg_text_1, neg_text_2 in zip(data_list, gold_text_list, neg_option_list1, neg_option_list2):
    item_dict = {
        "context": context,
        "question": "",
        "answerA": "",
        "answerB": "",
        "answerC": "",
        "correct": ""
    }
    options = ["A", "B", "C"]
    item_dict["correct"] = random.choice(options)
    item_dict["answer"+item_dict["correct"]] = gold_text
    options.remove(item_dict["correct"])
    for option, answer in zip(options, [neg_text_1, neg_text_2]):
        item_dict["answer"+option] = answer
    qa_format_writer.write(json.dumps(item_dict)+"\n")
