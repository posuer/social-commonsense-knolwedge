import os
import json
import pickle

from tqdm import tqdm
from Connected_KB import Connected_KB
kb = Connected_KB()


def main():
    data_dir = "SocialIQa/"
    for file in ['dev', 'tst', 'trn']:
        filepath = os.path.join(data_dir, "socialIQa_v1.4_"+file+".jsonl")
        all_result = []
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                path_result = []
                item = json.loads(line.strip())

                context = item["context"]
                question = item["question"]
                answers_dict = {"answerA":item["answerA"], "answerB":item["answerB"],"answerC":item["answerC"]}

                for cq_word in tqdm(kb.get_keywords_from_text(context+question)):
                    for a_word in kb.get_keywords_from_text(' '.join(answers_dict.values())):
                        path_result.append(kb.search_path_byEntity(cq_word, a_word, 3))
                
                all_result.append(path_result)


        with open(os.path.join(data_dir, "socialIQa_v1.4_"+file+"_knowledgePath.pickle"), 'wb') as writer:
            pickle.dump(all_result, writer)

if __name__ == "__main__":
    main()