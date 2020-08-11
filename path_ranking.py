import os
import pickle

import numpy as np
from scipy.spatial import distance

def get_embeddings(word_dict, vec_file, emb_size):
	word_vectors = np.random.uniform(-0.1, 0.1, (len(word_dict), emb_size))	
	f = open(vec_file, 'r', encoding='utf-8')
	vec = {}
	for line in f:
		line = line.split()
		vec[line[0]] = np.array([float(x) for x in line[-emb_size:]])
	f.close()  
	for key in word_dict:
		low = key.lower()
		if low in vec:
			word_vectors[word_dict[key]] = vec[low]
	print("word embedding loaded")
	return word_vectors
	
def main():
    file_list = ["dev","tst","trn_s0","trn_s1","trn_s2","trn_s3","trn_s4","trn_s5","trn_s6","trn_s7","trn_s8","trn_s9","trn_s10","trn_s11"]
    data_dir = "SocialIQa/"
    for filename in file_list:
        all_paths = pickle.load(open(os.path.join(data_dir, "socialIQa_v1.4_"+filename+"_keywords.pickle"), 'rb'))
        for qa_pair in all_paths:
            for query in qa_pair:
                

    
    distance.cosine([1, 1, 0], [0, 1, 0])

if __name__ == "__main__":
    main()