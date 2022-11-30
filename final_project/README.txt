The README


* Files submitted:
   * The Effects of Pivot Language on Word Alignment Quality.pdf: report
   * proj_funcs.py: contains all functions to run the scripts
   * training.py; runs a training of all the models mentioned in the paper and saves the alignment results to files
   * analyse_phrases.py: reads alignment files from various models and outputs unique phrases from baseline vs pivot model for a chosen sentence
   * README.txt
   * jhu_evaluation: a folder taken directly from jhu mt class
	contains modified files of hansards.a, hansards.e and hansards.f 
	contains trained models mentioned in the paper, so they can be evaluated immediately
		baseline = ibm100k20it
		baseline + sym = ibm100k20it_gdf
		piv = piv_100k20it
		piv = piv_100k20it_gdf
		gold = hansards.a
   * Europarl: a folder of the multi-parallel corpus
	contains data in English, French and Spanish; 100k sentences
* Python version: 3.7.5
* Any additional packages used: 
   * numpy, os, nltk.tokenize, itertools, pandas, collections, copy
* how to run the code in order to reproduce the results: 
   * 'python training.py' to get the models; 
	- you will be prompted to input the desired number of iterations
	- the alignment files will be saved automatically to the current directory
	- you will be prompted to input the desired filename for each model
   * 'python analyse_phrases.py' to get phrase analysis of a particular sentence
	- Table 2 from the paper will be printed
	- Table 3 from the paper will be printed
	- you will be prompted to choose which sentence from 0-100 to analyse [equivalent to line 100-200 of the europarl corpus]
   * to get AER, enter the jhu_evaluation directory. Same procedure as in mt-class. 
* runtime: 
   * for 1 iteration with a corpus of 100k sentences: 142.08 s