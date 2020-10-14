import os
import json
from spellchecker import SpellChecker 
import nltk


split = 'trn'

spell = SpellChecker() 

def corrector(text):
    global correct_word_dict
    global filter_out_list
    global idx
    tokens = nltk.word_tokenize(text)
    misspelled = spell.unknown(tokens) 

    for m in misspelled:
        if m in filter_out_list:
            continue
        c = spell.correction(m)
        if m == c:
            filter_out_list.append(m)
            continue
        if m in correct_word_dict:
            correct_word_dict[m]["text"].append(text)
            try:
                print(m, c, text)
            except:
                pass
            text = text.replace(m, c) if m in text else text.replace(m.capitalize(), c)
 
        else:
            try:
                print(text)
                print(m, c)
            except:
                print(idx)
                print
            is_correct = input(f"input ' for yes:")
            if is_correct == "'":
                text = text.replace(m, c) if m in text else text.replace(m.capitalize(), c)
                correct_word_dict[m] = {}
                correct_word_dict[m]["corrected"] = c
                correct_word_dict[m]["text"] = [text]
            else:
                filter_out_list.append(m)

    return text

data_dir = "data/socialiqa/"
path = os.path.join(data_dir, f"socialIQa_v1.4_{split}.jsonl")
with open(path, 'r', encoding='utf-8') as f:
    data = f.readlines()

writer = open(f"corrected_words_{split}.txt",'w', encoding='utf-8')
new_data_file = open(f"data/socialiqa_cleaned/socialIQa_v1.4_{split}.jsonl",'w', encoding='utf-8')

correct_word_dict = {} 
filter_out_list = ['skylar', "'s" ]
for idx, line in enumerate(data):
    item = json.loads(line.strip())

    context = item["context"]
    question = item["question"]
    endings = [item["answerA"],item["answerB"],item["answerC"] ]
    #item["context"] = corrector(context)
    #item["question"] = corrector(question)


    # if corrected:
    #     print(context + " " + question, file=writer)
    #     print(misspelled, corrected, file=writer)
    #     print(" ", file=writer)

    for idx, end in zip(["A","B","C"], endings):
        item["answer"+idx] = corrector(end)
    new_data_file.write(json.dumps(item)+"\n")

json.dump(correct_word_dict, writer, indent=4)
writer.close()
new_data_file.close()
