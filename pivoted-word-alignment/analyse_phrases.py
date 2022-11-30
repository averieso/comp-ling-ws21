#!/usr/bin/env python
# coding: utf-8

# In[2]:


from proj_funcs import *


# In[17]:


# read alignment files
models = dict()
   

path = "./jhu-evaluation/"

#file names
models['baseline'] = "ibm100k20it.a"
models['baselinesym'] = "ibm100k20it_gdf.a"
models['piv'] = "piv_100k20it.a"
models['pivsym'] = "piv_100k20it_gdf.a"
models['gold'] = "hansards.a"


#all models
print("reading alignment files...")
for modelname, modelpath in models.items():
    file_path = path + modelpath
    models[modelname] = read_aligned_file(file_path)


# In[16]:


# read and preprocess sentences from test set 
print('reading test set...')
path = "./jhu-evaluation/data"
with open(os.path.join(path, "hansards.e")) as f:
    en_hansards = f.readlines()

with open(os.path.join(path, "hansards.f")) as f:
    fr_hansards = f.readlines()

# tokenization
for i, sen in enumerate(en_hansards):
    en_hansards[i] = preprocess(sen, language = "english", target = False)

for i, sen in enumerate(fr_hansards):
    fr_hansards[i] = preprocess(sen, language = "french", target = True)
    


# In[30]:


# extract phrases from each model 
print("extracting phrases...")
model_phrases = defaultdict()

for n in range(len(en_hansards)):
    model_phrases[n] = {}
    e_sent = en_hansards[n]
    f_sent = fr_hansards[n]
    for modelname, alignments in models.items():
        a = alignments[n]
        ph = phrase_extract(en_hansards[n][1:], fr_hansards[n], a)
        #print(ph)
        model_phrases[n][modelname] = ph



# get phrase stats
print("**********")
print("getting phrase stats / Table 2: ")
print("**********")
phrase_set = defaultdict()
for n in range(len(en_hansards)):
    for model in model_phrases[n]:
        if n == 0:
            phrase_set[model] = set()
        ph = model_phrases[n][model]
        phrase_set[model].update(ph)
        
ph_stats = defaultdict(int)
ph_stats['intersect with gold'] = defaultdict(int)
ph_stats['overall'] = defaultdict(int)

gold_ph = phrase_set['gold']
    
for model in phrase_set.keys():
    model_set = phrase_set[model]
    in_withgold = gold_ph.intersection(model_set)
    ph_stats['intersect with gold'][model] += len(in_withgold)
    ph_stats['overall'][model] += len(model_set)

for column, model in ph_stats.items():
    print(column)
    for mod in model:
        print(mod, ": ", ph_stats[column][mod])


# get unique phrases
print("**********")
print('getting unique phrases / Table 3')
print("**********")
b_ps = copy.deepcopy(phrase_set['baseline'])
bs_ps = copy.deepcopy(phrase_set['baselinesym'])
p_ps = copy.deepcopy(phrase_set['piv'])
ps_ps = copy.deepcopy(phrase_set['pivsym'])

b_ps.update(bs_ps) #all baseline

p_ps.update(ps_ps) #all bridge

sym_phrases = b_ps ^ p_ps #symmetric difference 
p_unique = sym_phrases.intersection(p_ps) #unique bridge
b_unique = sym_phrases.intersection(b_ps) #unique baseline
print("number of unique phrases:")
print("pivot: ", len(p_unique), "baseline: ", len(b_unique))

pg = p_unique.intersection(phrase_set['gold']) #unique bridge intersect gold
bg = b_unique.intersection(phrase_set['gold']) #unique baseline intersect gold
g = sym_phrases.intersection(phrase_set['gold']) #check if numbers add up 
print("unique phrases intersect with gold")
print("pivot: ", len(pg), "baseline: ", len(bg), "total: ", len(g))
print("")

intersected_phrases = b_ps.intersection(p_ps)
ig = intersected_phrases.intersection(phrase_set['gold'])
print('both:', len(intersected_phrases), "intersect with gold:", len(ig))

all_phrases = copy.deepcopy(b_ps)
all_phrases.update(p_ps)
ag = all_phrases.intersection(phrase_set['gold'])
print("overall: ", len(all_phrases), "intersect with gold: ", len(ag))







# In[33]:


# get a set of phrases extracted for each model for each sentence
phrase_set = defaultdict()
for n in range(len(en_hansards)):
    for model in model_phrases[n]:
        if n == 0:
            phrase_set[model] = set()
        ph = model_phrases[n][model]
        phrase_set[model].update(ph)


# In[41]:


# compare extracted phrases for one line in the test set
n = input("choose line number to analyse (0-99):")
if isinstance(n, int):
    n = int(n)
else:
    n = 15
print("comparison of baseline and pivot model for line ", n, ": ")
p = model_phrases[n]['piv']
b = model_phrases[n]['baseline']

sym_phrases = b ^ p #symmetric difference 
p_unique = sym_phrases.intersection(p) #unique bridge
b_unique = sym_phrases.intersection(b) #unique baseline

print("en sentence: ")
print(" ".join(en_hansards[n][1:]))
print("fr sentence: ")
print(" ".join(fr_hansards[n]))
print("")

print("phrases unique to pivot model: ")
print("---------")
for p in p_unique:
    split_p = p.split(" / ")
    if split_p[0] == "" or split_p[1] == "":
        continue
    print(p)
print("")
print("phrases unique to baseline model: ")
print("---------")
for b in b_unique:
    split_b = b.split(" / ")
    if split_b[0] == "" or split_b[1] == "":
        continue
    print(b)


# In[ ]:




