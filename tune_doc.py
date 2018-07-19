# -*- coding: utf-8 -*-
'''
    Start Date: June 12th 2018
    A) Extractive Text Summarization
        1) Tuning the docs/sum pairs for neural netwrok training ie
        -turning the sentences in the docs into labels of 0 or 1 
        indicative of their membership to the summary    
'''
import nltk
import numpy as np
from Helper._data_generator import read_text_file
from data_generator import count_word_overlap
from metrics import write_to_file

TRAIN_DOC = "A_Training/Sum_Corpus/doc.txt"
TRAIN_SUM = "A_Training/Sum_Corpus/sum.txt"
TRAIN_TUNED_DOC = "A_Training/Sum_Corpus/tuned_doc.txt"

VAL_DOC = "B_Validating/Sum_Corpus/doc.txt"
VAL_SUM = "B_Validating/Sum_Corpus/sum.txt"
VAL_TUNED_DOC = "B_Validating/Sum_Corpus/tuned_doc.txt"

TEST_DOC = "C_Testing/Sum_Corpus/doc.txt"
TEST_SUM = "C_Testing/Sum_Corpus/sum.txt"
TEST_TUNED_DOC = "C_Testing/Sum_Corpus/tuned_doc.txt"

def main():
    #wrapper_doc_tuning()    
  
#wrapper to function for doc tuning 
def wrapper_doc_tuning():
   run_tuning(TRAIN_DOC,TRAIN_SUM,TRAIN_TUNED_DOC,"Train")
   run_tuning(VAL_DOC,VAL_SUM,VAL_TUNED_DOC,"Validation")
   run_tuning(TEST_DOC,TEST_SUM,TEST_TUNED_DOC,"Test")
       
    
def run_tuning(source_doc,source_summ,dest,desc):  
    doc = read_text_file(source_doc) 
    summ = read_text_file(source_summ)       
    print("Tuning " + desc + " doc")
    tuned_doc=""
    for i in range(len(doc)):             
        art_sents = nltk.sent_tokenize(doc[i])                 
        abs_sents = summ[i]        
        tuned_doc +=summ_sents_extractor(art_sents,abs_sents) + "\n"                                        
        if(i%500 == 0):                
            prog = int(i/len(doc) * 100)
            print(prog,"% ...") 
        
    write_to_file(dest,tuned_doc,"w")                          
   
#labels sentences from the article
#based on the word_overlap between the article n abstract               
def summ_sents_extractor(article,abstract):
    n = len(article)
    tuned_doc= np.zeros(n,dtype=int)
    word_overlap_counts=[]
    for sent in range(n):         
        _,count = count_word_overlap(abstract,article[sent])
        word_overlap_counts.append(count)
    #word_overlap_count = np.flip(np.sort(word_overlap_counts),0)[:5]
    
    #take topmost 5 sentences with highest word-overlap with abstract
    word_overlap_indices= np.flip(np.argsort(word_overlap_counts),0)[:5]    
    word_overlap_indices= np.sort(word_overlap_indices) #preserve order
    #print(word_overlap_indices)
    np.put(tuned_doc,word_overlap_indices,np.ones(5))
    tuned_doc = map(str,tuned_doc)
    tuned_doc = ' '.join(tuned_doc)    
    '''
    summ=""
    for i in word_overlap_indices:
        summ += article[i]
    print("->"+ summ + "\n\n=" + abstract + "\n\n\n")
    '''
    return tuned_doc

if __name__ == "__main__":
    main()