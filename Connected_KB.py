import os
import csv
import pickle
import nltk
#import networkx as nx
#import numba as nb
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer 

class Connected_KB(object):
    def __init__(self):
        #self.ATOMIC = {}
        self.ConceptNet = {}
        self.ATOMIC_Events = {}
        self.ATOMIC_Infers = {}
        #self.G = nx.MultiGraph()

        self.lemmatizer = WordNetLemmatizer() 
        try:
            self.lemmatizer.lemmatize("visits")
            nltk.pos_tag(nltk.word_tokenize("PersonX abuses PersonX's power".lower()))
        except:
            nltk.download('wordnet')
            nltk.download('punkt')
            nltk.download('averaged_perceptron_tagger')

        self.kb_load()


    def __format_atomic_infer(self, inf):
        inf = inf.lower().replace("person x", "personx").replace("person y", "persony").replace("person z", "personz") \
                        .replace(" x ", " personx ").replace(" y ", " persony ").replace(" z ", " personz ") \
                        .strip()
        if inf.endswith(" x"):
            inf = inf.replace(" x", " personx") 
        if inf.endswith(" y"):
            inf = inf.replace(" y", " persony")
        if inf.endswith(" z"):
            inf = inf.replace(" z", " personz")
            
        if inf.startswith("x "):
            inf = inf.replace("x ", "personx ") 
        if inf.startswith("y "):
            inf = inf.replace("y ", "persony ")
        if inf.startswith("z "):
            inf = inf.replace("z ", "personz ")
        return inf

    def __get_keywords_from_text(self, text): #for linking KBs
        stopwords = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']
        persons = ["personx", "persony","personz", "___"]
        keywords = [word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(text.lower())) if pos[:2] in ['NN', 'JJ', 'VB', "RB"] and word not in stopwords+persons ]
        keywords_lem = [self.lemmatizer.lemmatize(word) for word in keywords] 
        return keywords_lem

    def get_keywords_from_text(self, text): #for QA text
        stopwords = ["n't",'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']
        keywords = [word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(text.lower())) if pos[:2] in ['NN', 'JJ', 'VB', "RB"] and word not in stopwords ]
        keywords_lem = set([self.lemmatizer.lemmatize(word) for word in keywords])
        return keywords_lem

    def __lemmatize_word(self, word):

        word_lem = self.lemmatizer.lemmatize(word)
        if word_lem in self.ConceptNet.keys():
            return word_lem

        if word_lem not in self.ConceptNet.keys() and word_lem == word:  #fix failed lemmatization. eg: enjoys
            for pos_tag in ['a','v','n']:
                word_lem = self.lemmatizer.lemmatize(word, pos=pos_tag)
                if word_lem in self.ConceptNet.keys():
                    return word_lem
        
        return None
   
    def search_path_byEntity1(self, start_word, end_word, max_hops): 
        start_word = self.__lemmatize_word(start_word)
        end_word = self.__lemmatize_word(end_word)
        if not (start_word and end_word):
            return None

        paths = nx.all_simple_paths(self.G, source=('conceptnet', start_word), target=('conceptnet',end_word), cutoff=max_hops)
        return paths
    
    
    def search_path_byEntity(self, start_word, end_word, max_hops): 
        start_word = self.__lemmatize_word(start_word)
        end_word = self.__lemmatize_word(end_word)
        if not (start_word and end_word):
            return None

        connected_KB ={
            'conceptnet':self.ConceptNet,
            'atomic_event':self.ATOMIC_Events,
            'atomic_infer':self.ATOMIC_Infers
        }
        visited = set()
        all_path = []
        #@nb.jit()  
        def search_all_path(path, this_kb_type, key, rel_w_parent, direction):
            visited.add((this_kb_type, key))
            path.append({
                    "kb_type": this_kb_type,
                    "key": key,
                    "rel_w_parent": rel_w_parent,
                    "rel_direction": direction,
                    })
            
            if this_kb_type == 'conceptnet' and key == end_word: #if found destination in conceptnet node
                all_path.append(list(path))
            elif end_word in connected_KB[this_kb_type][key]['conceptnet']: #if found destination in event or infer node
                if len(path)>=2 and path[-2]["kb_type"] != 'conceptnet': #avoid last two hop be "concept - event/infer", when less than maxhops
                    all_path.append(list(path))
            elif len(path) <= max_hops and len(all_path) <= 1000:
                this_node = connected_KB[this_kb_type][key]

                for kb_type, relations in this_node.items():
                    if this_kb_type == 'conceptnet':
                        if kb_type == 'conceptnet':
                            for neighbor in relations:
                                if (kb_type, neighbor["tail"]) not in visited:
                                    search_all_path(path, kb_type, neighbor["tail"], neighbor["rel"], neighbor["direction"])

                        elif kb_type == 'atomic_event' or kb_type == 'atomic_infer' and len(path)!=max_hops: #avoid last two hop be "concept - event/infer", purne searching space
                            for neighbor in relations:
                                if (kb_type, neighbor) not in visited:
                                    search_all_path(path, kb_type, neighbor, None, None)
                            
                    elif this_kb_type == 'atomic_event' or this_kb_type == 'atomic_infer':
                        if kb_type == 'conceptnet':
                            if len(path)>=2 and path[-2]["kb_type"] == 'conceptnet':
                                continue #avoid path like "conceptnet - event/infer - conceptnet"
                            for neighbor in relations:
                                if (kb_type, neighbor) not in visited:
                                    search_all_path(path, kb_type, neighbor, None, None)
                            
                        elif kb_type == 'atomic_event':
                            for neighbor in relations:
                                if (kb_type, neighbor["tail"]) not in visited:
                                    search_all_path(path, kb_type, neighbor["tail"], neighbor["rel"], None)
                            
                        elif kb_type == 'atomic_infer':
                            for infer_type, neighbor_list in relations.items():
                                if infer_type == 'prefix': continue
                                for neighbor in neighbor_list:
                                    if (kb_type, neighbor) not in visited:
                                        search_all_path(path, kb_type, neighbor, infer_type, None)
            
            path.pop()
            visited.remove((this_kb_type, key))
        
        path = []
        search_all_path(path, 'conceptnet', start_word, None, None)
        return all_path
            


    def retrieve_neighbor_atomic(self):
        pass
    
    def retrieve_neighbor_conceptnet(self):
        pass

    def kb_load(self):
        if os.path.exists("ConceptNet_en_QA.pickle") and os.path.exists("ATOMIC_v4_Events.pickle") and os.path.exists("ATOMIC_v4_Infers.pickle"):# and os.path.exists("ATOMIC_ConceptNet_wholeGraph.pickle"):
            self.ConceptNet = pickle.load(open( "ConceptNet_en_QA.pickle", "rb" ))
            self.ATOMIC_Events = pickle.load(open( "ATOMIC_v4_Events.pickle", "rb" ))
            self.ATOMIC_Infers = pickle.load(open( "ATOMIC_v4_Infers.pickle", "rb" ))
            #self.G = pickle.load(open( "ATOMIC_ConceptNet_wholeGraph.pickle", "rb" ) )

        
        else:
            #ConceptNet
            discarded_rel = ["hascontext", "relatedto", "synonym", "antonym", "derivedfrom", "formof", "etymologicallyderivedfrom", "etymologicallyrelatedto", "externalurl"]
            with open("../conceptnet-assertions-5.7.0.csv",'r',encoding='utf-8') as reader:
                csv_reader = csv.reader(reader, delimiter='\t')
                for line in tqdm(csv_reader):
                    head = line[2].split('/')
                    tail = line[3].split('/')
                    if head[2] == "en" and tail[2] == "en":
                        relation = line[1].split('/')[2].lower()
                        if relation in discarded_rel: continue #pass discarded_rel

                        head_key = head[3].lower()
                        tail_key = tail[3].lower()
                        if head_key not in self.ConceptNet.keys(): self.ConceptNet[head_key] = {"atomic_event":[], "conceptnet":[], "atomic_infer":set()}
                        self.ConceptNet[head_key]["conceptnet"].append({"rel": relation, "tail": tail_key, 'direction':0 })
                        if tail_key not in self.ConceptNet.keys(): self.ConceptNet[tail_key] = {"atomic_event":[], "conceptnet":[], "atomic_infer":set()}
                        self.ConceptNet[tail_key]["conceptnet"].append({"rel": relation, "tail": head_key, 'direction':1 })
                        
                        #self.G.add_edge(('conceptnet',head_key), ('conceptnet',tail_key), rel= relation, head=head_key, tail=tail_key  )
                        #self.G.add_edge(('conceptnet',tail_key), ('conceptnet',head_key), {'rel': relation, 'head':head_key, 'tail':tail_key } )


            #ATOMIC
            relations = ["oEffect","oReact","oWant","xAttr","xEffect","xIntent","xNeed","xReact","xWant","prefix"]
            num = 0
            with open("atomic_data/v4_atomic_all_agg.csv") as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in tqdm(csv_reader):
                    if num == 0:
                        num +=1 
                        continue
                    event = row[0]
                    infers = [row[i].strip('][').replace('"', "").split(', ') for i in range(1,11)]
                    infers_clear = []
                    for i in infers:
                        infers_clear.append([self.__format_atomic_infer(x) for x in i if x != 'none'])   
                    
                    self.ATOMIC_Events[event]= {"atomic_infer": dict( [(rel, set(infer)) for rel, infer in zip(relations, infers_clear)]), "conceptnet":set()}
                    
                    #for rel, infers in zip(relations, infers_clear):
                        #self.G.add_node(('atomic', event, infer, rel) )
                        #for infer in infers:
                        #    self.G.add_edge(('atomic_event', event), ("atomic_infer", infer), rel=rel) 

            for key, value in tqdm(self.ATOMIC_Events.items()):
                for prefix in value["atomic_infer"]["prefix"]:
                    word_lem = self.__lemmatize_word(prefix)
                    if word_lem:
                        self.ConceptNet[word_lem]["atomic_event"].append(key)
                        self.ATOMIC_Events[key]["conceptnet"].add(word_lem)
                        
                        #self.G.add_edge(('atomic_event', key), ("conceptnet", word_lem)) 

                for inf_key, inf_value in value["atomic_infer"].items():
                    if inf_key != "prefix":
                        for infer_phrase in inf_value:
                            #self.G.add_edge(('conceptnet', word_lem), ('atomic', key, infer_phrase, inf_key) )

                            if infer_phrase not in self.ATOMIC_Infers.keys():  self.ATOMIC_Infers[infer_phrase] = {"conceptnet":set(), "atomic_event":[]}
                            self.ATOMIC_Infers[infer_phrase]["atomic_event"].append({"rel":inf_key,"tail":key})

                            for conceptnet_key in self.__get_keywords_from_text(infer_phrase):
                                word_lem = self.__lemmatize_word(conceptnet_key)
                                if word_lem:
                                    self.ATOMIC_Infers[infer_phrase]["conceptnet"].add(word_lem)
                                    self.ConceptNet[word_lem]['atomic_infer'].add(infer_phrase)

                                    #self.G.add_edge(('conceptnet', word_lem), ('atomic_infer', infer_phrase) )


            pickle.dump(self.ConceptNet, open( "ConceptNet_en_QA.pickle", "wb" ) )
            pickle.dump(self.ATOMIC_Events, open( "ATOMIC_v4_Events.pickle", "wb" ) )
            pickle.dump(self.ATOMIC_Infers, open( "ATOMIC_v4_Infers.pickle", "wb" ) )
            #pickle.dump(self.G, open( "ATOMIC_ConceptNet_wholeGraph.pickle", "wb" ) )

            
               
       
if __name__ == "__main__":
    connected_KB = Connected_KB()
    

    