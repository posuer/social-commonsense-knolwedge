import os
import json
import pickle
import multiprocessing
#import numba as nb
from tqdm import tqdm
from Connected_KB import Connected_KB

import sys
import time
start_time = time.time()


#@nb.jit(nopython=True)
def multi_processing(idx, cq_word, a_word, max_hops):
    kb_sub = Connected_KB()
    return idx, cq_word, a_word, kb_sub.search_path_byEntity(cq_word, a_word, max_hops)

def multi_processing_all(query_list):#cq_word, a_word, max_hops
    kb_sub = Connected_KB()
    path_list = []
    #for cq_word, a_word, paths in pool.starmap(multi_processing, query_list):
    tag = 0
    for idx, cq_word, a_word, max_hops in query_list:
        path = kb_sub.search_path_byEntity(cq_word, a_word, max_hops)
        if path:
            path_list.append(path)
        if idx != tag:
            tag = idx
            print(tag)
    return path_list

#@nb.jit()
def main(filename):

    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    print("Runing on", cores, "CPUs.")
    data_dir = "SocialIQa/"

    filepath = os.path.join(data_dir, "socialIQa_v1.4_"+filename+".jsonl")
    all_query_list = []
    all_query_dict = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        all_result = [list() for i in range(len(lines))]
        
        idx = 1
        for line in lines:
            path_result = []
            item = json.loads(line.strip())

            context = item["context"]
            question = item["question"]
            answers_dict = {"answerA":item["answerA"], "answerB":item["answerB"],"answerC":item["answerC"]}

            #query_list = []
            for cq_word in kb.get_keywords_from_text(context+' '+question):
                if idx not in all_query_dict.keys(): all_query_dict[idx] = {"cq_word":set(), "a_word":set()}
                all_query_dict[idx]["cq_word"].add(cq_word)
                for a_word in kb.get_keywords_from_text(' '.join(answers_dict.values())):
                    all_query_list.append((idx, cq_word, a_word, 3))
                    all_query_dict[idx]["a_word"].add(a_word)
            print(idx, all_query_dict[idx])

            idx += 1
            
            '''
            # multi processing for each QA pair
            path_lens = []
            for cq_word, a_word, paths in pool.starmap(multi_processing, query_list):
                if paths:
                    path_result.append(paths)
                    path_lens.append(len(paths))
                    print(len(paths), "paths for this query:", cq_word, a_word,)
                else:
                    print("No path for this query:", cq_word, a_word,)
            
            # Vanila Algorithm
            #for cq_word, a_word, hops in tqdm(query_list):
            #    paths = kb.search_path_byEntity(cq_word, a_word, hops)
            #    path_result.append(paths)
            all_result.append(path_result)
            '''
            
            #all_query_list.append(query_list)
    # multi processing for all QA pair
    cnt = 1
    for idx, cq_word, a_word, paths in pool.starmap(multi_processing, all_query_list):
        if paths:
            all_result[idx-1].append(paths)
        #if cnt % 10 == 0:
        #sys.stdout.write('done %d/%d, Time %s\r' % (cnt, idx, (time.time() - start_time)/60))
        
        if cnt % 300 == 0:
            pickle.dump(all_result, open(os.path.join(data_dir, "socialIQa_v1.4_"+filename+"_knowledgePath.pickle"), 'wb'))
        cnt += 1

    pickle.dump(all_result, open(os.path.join(data_dir, "socialIQa_v1.4_"+filename+"_knowledgePath.pickle"), 'wb'))
    
    pickle.dump(all_query_dict, open(os.path.join(data_dir, "socialIQa_v1.4_"+filename+"_keywords.pickle"), 'wb'))


if __name__ == "__main__":
    print("Loading KB ...")
    kb = Connected_KB()
    filename = sys.argv[1]
    print("Start searching for", filename)
    main(filename)