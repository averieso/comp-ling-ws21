#!/usr/bin/env python
# coding: utf-8

# In[67]:


import os
from nltk.tokenize import wordpunct_tokenize, word_tokenize
import itertools
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import copy


# # Preprocessing

# In[19]:


def preprocess(sentence, language = 'english', both = False, target = False):
    '''preprocesses a sentence by tokenizing into a list and lowercasing.
       add NULL token if target = False'''

    sentence = sentence.lower()
    s = word_tokenize(sentence, language) 
    # add NULL token for source data
    if both == True:
        s_s = ["NULL"] + s
        s_t = s
        return s_s, s_t
    elif target == True:
        return s
    elif target == False:
        s = ["NULL"] + s
        return s


# # IBM 1

# In[51]:


def get_corpus(list_of_sents):
    ''' get set of all words in a corpus'''
    cor = [w for sen in list_of_sents for w in set(sen)]
    return set(cor)


# In[53]:


def ibm_prob(src_corpus, trg_corpus, prob_table = None):
    '''ibm 1 training'''
    fr_cor = get_corpus(trg_corpus)
    
    #initialize
    if prob_table == None: 
        p = defaultdict(lambda: 1 / len(fr_cor))
        
    else:
        p = prob_table

    total = defaultdict(int)
    count = defaultdict(int)
    s_total = defaultdict(int)
    
    #compute normalization
    for n in range(len(src_corpus)):
        for f in trg_corpus[n]:
            s_total[f] = 0
            for e in src_corpus[n]:
                s_total[f] += p[(f, e)]

        #collect counts  
        for f in trg_corpus[n]:
            for e in src_corpus[n]:
                c = p[(f, e)] / s_total[f]
                count[(f, e)] += c
                total[e] += c

    #estimate probabilities
    for (f, e) in count:
        p[(f, e)] = count[(f, e)] / total[e]
    return p


# In[55]:


def save_model(p_table, trg_corpus):
    '''saves the probability table into a readable dictionary'''
    # copy the translation probabilities and save by key = trg word in a new dict
    prob_dict = {}

    for word in get_corpus(trg_corpus):
        prob_dict[word] = {}

    for key, value in p_table.items():
        prob_dict[key[0]][key[1]] = value


    return prob_dict


# In[54]:


def ibm_training(src_corpus, trg_corpus, p_table = None, iteration_n = 1):
    '''trains ibm model 1 with n number of iterations. p_table can be specified to
       train a previously trained model'''
    x = 0
    while x < iteration_n:
        print("iteration", x+1)
        new_prob_table = ibm_prob(src_corpus, trg_corpus, p_table)
        p_table = new_prob_table
        x+=1
    return p_table


# In[57]:


def align(src_sent, trg_sent, prob_dict):
    '''ibm 1 decoding. prob_dict refers to the probability table in dictionary form'''
    alignment = []
    for j, f in enumerate(trg_sent):
        best_prob = 0
        best_i = 1
        count = Counter(prob_dict[f])
        for i, e in enumerate(src_sent):
            # since the evaluation script doesn't have NULL, the alignment generated here would also eliminate any NULL assignment (where i = 0)
            if count[e] > best_prob and i > 0:
                best_prob = count[e]
                if Counter(src_sent)[e] > 1:
                    list_of_ind = [n for n, w in enumerate(src_sent) if w == e]
                    best_i = min(list_of_ind, key = lambda x:abs(x - j))
                else:
                    best_i = i
        align = "%s-%s"%(j, best_i-1) # following the format required by the evaluation script
        alignment.append(align)
    formatted_alignment = " ".join(alignment)
    return formatted_alignment


# # Pivot

# In[56]:


def pivot(src_sent, piv_sent, trg_sent, src_piv_prob_dict, piv_trg_prob_dict):
    '''create a new prob dict based on two pivot dictionaries from source to pivot
       from pivot to target'''
    # src = e, trg = f, piv = g
    src_trg_prob_dict = defaultdict()
    param = 0.5
    for j, f in enumerate(trg_sent):
        src_trg_prob_dict[f] = defaultdict(int)
        for k, g in enumerate(piv_sent):
            p_f_g = piv_trg_prob_dict[f][g]
            for i, e in enumerate(src_sent):
                if k == 0 and i == 0:
                    p_g_e = param
                elif k == 0 and i > 0:
                    p_g_e = (1-param) / len(src_sent)
                else:
                    p_g_e = src_piv_prob_dict[g][e]
                src_trg_prob_dict[f][e] += p_f_g * p_g_e
    return src_trg_prob_dict


# # Symmetrization

# In[58]:


def neighbours(src_ind, trg_ind, len_src, len_trg):
    '''part of the grow_diag_final implementation, gets neighbours of
       a particular alignment point'''
    neighbour_list = []
    for a,b in itertools.product([-1, 0, 1], [-1, 0, 1]):
        if [a, b] != [0,0]:
            if 0 <= src_ind + a <= len_src and 0 <= trg_ind + b <= len_trg:
                neighbour_list.append([src_ind + a, trg_ind + b])
    
    return neighbour_list


# In[60]:


def grow_diag_final(e2f, f2e):
    '''the grow diag final heuristic given e2f = alignment from source to target, and
       f2e, the alignment from target to source. assumed to be strings'''
    # formatted alignment 
    e2f = np.array([alignment.split("-") for alignment in e2f.split(" ")])
    f2e = np.array([alignment.split("-") for alignment in f2e.split(" ") ])
    
    e2f = e2f.astype(int) # in terms of f (f-e)
    f2e = f2e.astype(int) # in terms of e (e-f)
    
    # create matrix
    align_matrix = np.zeros([len(f2e), len(e2f)])
    # row = len e, column = len f
    # matrix is indexed in terms of e [with e as rows]
    
    
    for f_e in e2f:
        for e_f in f2e:
            #intersect tag
            if e_f[0] == f_e[1] and e_f[1] == f_e[0]:
                align_matrix[e_f[0], e_f[1]] = 1
            #union tag
            elif align_matrix[e_f[0], e_f[1]] == 0:
                align_matrix[e_f[0], e_f[1]] = 2
            elif align_matrix[f_e[1], f_e[0]] == 0:  
                align_matrix[f_e[1], f_e[0]] = 2
            
    
    
    # grow diag
    union_arr = np.where(align_matrix == 2)
    union_pos = list(zip(union_arr[0], union_arr[1]))
    new_points_added = True
    while new_points_added:
        new_points_added= False
        for i_e, i_f in np.ndindex(align_matrix.shape):
            for e_new, f_new in neighbours(i_e, i_f, len(f2e)-1, len(e2f)-1):
                if 1 not in align_matrix[e_new, :] or 1 not in align_matrix[:, f_new]:
                    if (e_new, f_new) in union_pos:
                        align_matrix[e_new, f_new] = 1
                        new_points_added = True
                        
                        
    # final
    for i_e, i_f in np.ndindex(align_matrix.shape):
        if 1 not in align_matrix[i_e, :] or 1 not in align_matrix[:, i_f]:
            if (i_e, i_f) in union_pos:
                align_matrix[i_e, i_f] = 1   
                
    # parse
    e2f_line = []
    e2f_matrix = align_matrix.T
    for f_e, value in np.ndenumerate(e2f_matrix):
        if value == 1:
            f = f_e[0]
            e = f_e[1]
            e2f_line.append(str(f) + "-" + str(e))
    e2f_line = " ".join(e2f_line)
    
    
    f2e_line = []
    f2e_matrix = align_matrix
    for e_f, value in np.ndenumerate(f2e_matrix):
        if value == 1:
            e = e_f[0]
            f = e_f[1]
            f2e_line.append(str(e) + "-" + str(f))
    f2e_line = " ".join(f2e_line)
            
    return e2f_line, f2e_line


# In[68]:


def align_to_file(path, filename, src_corpus, trg_corpus, src_trg_prob_dict = None, 
                  piv = False, piv_corpus = None, src_piv_prob_dict = None, piv_trg_prob_dict = None, 
                  sym = False, trg_src_prob_dict = None,
                  piv_sym = False, trg_piv_prob_dict = None, piv_src_prob_dict = None):
    '''saves alignments to file with customizable method.
        if piv = True, pivot function is applied, else a direct model
        if sym = True, symmetrization is applied'''
    with open(os.path.join(path, filename), 'a') as file:
        for n in range(len(src_corpus)):
            e2f = align(src_corpus[n], trg_corpus[n], src_trg_prob_dict)

            if piv == True: 
                src_trg_piv_dict = pivot(src_corpus[n], piv_corpus[n], trg_corpus[n], src_piv_prob_dict, piv_trg_prob_dict)
                e2f = align(src_corpus[n], trg_corpus[n], src_trg_piv_dict)

                if sym == True: # (bridge + reversed baseline) + sym
                    trg_plus_null = copy.deepcopy(trg_corpus[n])
                    trg_plus_null.insert(0, "NULL")
                    src_rem_null = src_corpus[n][1:]
                    if piv_sym == True: # (bridge + reversed bridge) + sym
                        piv_rem_null = piv_corpus[n][1:]
                        tgt_src_piv_dict = pivot(trg_plus_null, src_rem_null, piv_rem_null, trg_piv_prob_dict, piv_src_prob_dict)
                        trg_src_prob_dict = tgt_src_piv_dict
                    f2e = align(trg_plus_null, src_rem_null, trg_src_prob_dict)
                    e2f_line, f2e_line = grow_diag_final(e2f, f2e)
                    alignment = e2f_line
                
                else: # bridge alone
                    alignment = e2f
                    
            elif sym == True: # ibm + sym
                trg_plus_null = copy.deepcopy(trg_corpus[n])
                trg_plus_null.insert(0, "NULL")
                src_rem_null = src_corpus[n][1:]
                f2e = align(trg_plus_null, src_rem_null, trg_src_prob_dict)
                e2f_line, f2e_line = grow_diag_final(e2f, f2e)
                alignment = e2f_line
            
            else: # baseline e2f
                alignment = align(src_corpus[n], trg_corpus[n], src_trg_prob_dict)

            
            file.write(alignment)
            file.write('\n')


# # Phrase extraction

# In[69]:


def read_aligned_file(path_to_file):
    '''reads alignments from files and parse into a list'''
    alignments = []
    with open(path_to_file, "r") as f:
        for line in f: 
            alignments.append(line.split("\n")[0])
    return alignments


# In[64]:


def phrase_extract(e_sent, f_sent, alignment, max_phrase_length = 7):
    '''get phrases extracted from source sentence (e_sent) and target sentence (f_sent)
    given an alignment'''
    #read alignment as a string (formatted) and turn it into a np array
    alignment = [f_e.split("-") for f_e in alignment.split(" ")]
    alignment = np.array(alignment)
    alignment[:,[0, 1]] = alignment[:,[1, 0]]
    alignment = alignment.astype(int)
    
    def extract(f_start, f_end, e_start, e_end):
        if e_end - e_start > max_phrase_length:
            return {}
        if f_end == 0:
            return {}
        for e,f in alignment:
            if (f_start <= f <= f_end) and (e < e_start or e > e_end):
                return {}
        #add phrase pairs
        E = set()
        fs = f_start
        #repeat
        while True:
            fe = f_end
            #repeat
            while True:
                e_phrase = [e_sent[i] for i in range(e_start, e_end)]
                f_phrase = [f_sent[j] for j in range(fs, fe)]
                paired = " ".join(e_phrase) + " / " + " ".join(f_phrase)
                E.add(paired)
                fe += 1
                #until fe aligned
                if fe in alignment[:,1] or fe == len(f_sent):
                    break
            fs -= 1
            #until fs aligned
            if fs in alignment[:,1] or fs < 0:
                break
        return E
    
    bp = set()
    for e_start in range(len(e_sent)):
        for e_end in range(e_start, len(e_sent)):
            #find minimally matching foreign phrase
            f_start, f_end = len(f_sent)-1, -1
            for e,f in alignment:
                if e_start <= e <= e_end:
                    f_start = min(f, f_start)
                    f_end = max(f, f_end)
            phrase = extract(f_start, f_end, e_start, e_end)
            bp.update(phrase)
    
    return bp

