#!/usr/bin/env python
# coding: utf-8

# In[2]:


from proj_funcs import *


# In[3]:


# get multi-parallel corpus
print("reading corpus...")
path = "./europarl"
files = os.listdir(path)

corpus = {}
for file in files:
    lang = file.split(".")[1]
    corpus[lang] = {}
    src = []
    trg = []
    if lang == 'en':
        language = 'english'
    elif lang == 'es':
        language = 'spanish'
    elif lang == 'fr':
        language = 'french'
    with open(os.path.join(path, file), "r") as f:
        for line in f:
            s, t = preprocess(line, language = language, both = True)
            src.append(s)
            trg.append(t)
        corpus[lang]['source'] = src
        corpus[lang]['target'] = trg


# In[ ]:


# train the baseline model
it_n = int(input("enter n of interations: "))
corp_size = 100000
print('training an ibm model EN --> FR for', it_n, 'iterations')
print("corpus size: ", corp_size)
en_fr_p_table = ibm_training(corpus['en']['source'][0:corp_size], corpus['fr']['target'][0:corp_size], iteration_n = it_n)
en_fr_p_dict = save_model(en_fr_p_table, corpus['fr']['target'][0:corp_size])
print('training completed.')


# In[ ]:


# train the piv models
print('training an pivot model EN --> ES', corp_size, 'sentences for', it_n, 'iterations')
print("corpus size: ", corp_size)
en_es_p_table = ibm_training(corpus['en']['source'][0:corp_size], corpus['es']['target'][0:corp_size], iteration_n = it_n)
en_es_p_dict = save_model(en_es_p_table, corpus['es']['target'][0:corp_size])
print('training completed.')

print('training an pivot model ES --> FR', corp_size, 'sentences for', it_n, 'iterations')
print("corpus size: ", corp_size)
es_fr_p_table = ibm_training(corpus['es']['source'][0:corp_size], corpus['fr']['target'][0:corp_size], iteration_n = it_n)
es_fr_p_dict = save_model(es_fr_p_table, corpus['fr']['target'][0:corp_size])
print('training completed.')


# In[13]:


# train the reversed baseline model
print('training an ibm model FR --> EN for', it_n, 'iterations')
print("corpus size: ", corp_size)
fr_en_p_table = ibm_training(corpus['fr']['source'][0:corp_size], corpus['en']['target'][0:corp_size], iteration_n = it_n)
fr_en_p_dict = save_model(fr_en_p_table, corpus['en']['target'][0:corp_size])
print('training completed.')


# In[15]:


#save alignments to file

path = "./"

src_corpus = corpus['en']['source'][100:200]
trg_corpus = corpus['fr']['target'][100:200]
piv_corpus = corpus['es']['source'][100:200]

src_trg_prob_dict = en_fr_p_dict

src_piv_prob_dict = en_es_p_dict
piv_trg_prob_dict = es_fr_p_dict

trg_src_prob_dict = fr_en_p_dict

# get ibm model
print("saving ibm model to file...")
filename = input("enter filename: ")
align_to_file(path, filename, src_corpus, trg_corpus, src_trg_prob_dict = src_trg_prob_dict, 
                  piv = False, piv_corpus = piv_corpus, src_piv_prob_dict = src_piv_prob_dict, piv_trg_prob_dict = piv_trg_prob_dict, 
                  sym = False, trg_src_prob_dict = trg_src_prob_dict)


# get piv model
print("saving piv model to file...")
filename = input("enter filename: ")
align_to_file(path, filename, src_corpus, trg_corpus, src_trg_prob_dict = src_trg_prob_dict, 
                  piv = True, piv_corpus = piv_corpus, src_piv_prob_dict = src_piv_prob_dict, piv_trg_prob_dict = piv_trg_prob_dict, 
                  sym = False, trg_src_prob_dict = trg_src_prob_dict)


# get ibm + sym model
print("saving ibm + sym model to file...")
filename = input("enter filename: ")
align_to_file(path, filename, src_corpus, trg_corpus, src_trg_prob_dict = src_trg_prob_dict, 
                  piv = False, piv_corpus = piv_corpus, src_piv_prob_dict = src_piv_prob_dict, piv_trg_prob_dict = piv_trg_prob_dict, 
                  sym = True, trg_src_prob_dict = trg_src_prob_dict)


# get piv + sym model
print("saving piv + sym model to file...")
filename = input("enter filename: ")
align_to_file(path, filename, src_corpus, trg_corpus, src_trg_prob_dict = src_trg_prob_dict, 
                  piv = True, piv_corpus = piv_corpus, src_piv_prob_dict = src_piv_prob_dict, piv_trg_prob_dict = piv_trg_prob_dict, 
                  sym = True, trg_src_prob_dict = trg_src_prob_dict)



