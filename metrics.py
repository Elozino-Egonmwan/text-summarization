import nltk
from nltk.translate.bleu_score import SmoothingFunction
from Helper._data_generator import read_text_file
import numpy as np

'''
TRAIN_REF = read_text_file("C:/Users/elozi/Desktop/Evaluate/Fusion/reference/fusionTrain_reference.txt")
TRAIN_SYS = read_text_file("C:/Users/elozi/Desktop/Evaluate/Fusion/system/fusionTrain_system.txt") 

VAL_REF = read_text_file("C:/Users/elozi/Desktop/Evaluate/Fusion/reference/fusionVal_reference.txt") 
VAL_SYS = read_text_file("C:/Users/elozi/Desktop/Evaluate/Fusion/system/fusionVal_system.txt") 

TEST_REF = read_text_file("C:/Users/elozi/Desktop/Evaluate/Fusion/reference/fusionTest_reference.txt")
TEST_SYS = read_text_file("C:/Users/elozi/Desktop/Evaluate/Fusion/system/fusionTest_system.txt")

INFER_REF= read_text_file("C:/Users/elozi/Desktop/Evaluate/Fusion/reference/fusionInferences_reference.txt")
INFER_SYS= read_text_file("C:/Users/elozi/Desktop/Evaluate/Fusion/system/fusionInferences_system.txt")
'''
log = "LOG_DIR_300/Fusion/log.txt"
smoothing = SmoothingFunction()
def main():
    '''
    status = ""
    #print("Train Bleu Score: ", calcBleu(read_text_file(TRAIN_REF),read_text_file(TRAIN_SYS)))
    
    status+="Entire File"
    status+= "\nValidation Bleu Score: " + str(calcBleu(VAL_REF, VAL_SYS))
    status+= "\nInferences Bleu Score: " + str(calcBleu(INFER_REF,INFER_SYS))
    
    status+= "\n\nPer Sentence"
    status+= "\nValidation Bleu Score: " + str(bleuPerSent(VAL_REF,VAL_SYS))
    status+= "\nInferences Bleu Score: " + str(bleuPerSent(INFER_REF,INFER_SYS))
    
    write_to_file(log,status,"w")
    print(status)
    '''
    
#evaluate BLEU metric    
def calcBleu(ref,hyp):
    ref = (' '.join(ref)).split()
    hyp = (' '.join(hyp)).split()   
    
    bleuScore = nltk.translate.bleu_score.corpus_bleu([[ref]],[hyp],smoothing_function=smoothing.method5)
    #bleuScore = nltk.translate.bleu_score.sentence_bleu([ref],hyp,smoothing_function=smoothing.method5)
    return bleuScore 
 
#compute bleu per sent           
def bleuPerSent(ref,sys):
    bleu = 0.0
    bleuValues =[]
    n = len(ref)
    for i in range(n):        
        #val = calcBleu(ref[i],sys[i])
        val = nltk.translate.bleu_score.sentence_bleu([ref[i]],sys[i],smoothing_function=smoothing.method5) 		 
        bleu += val
        bleuValues.append(val)
    
    bleuValues = np.array(bleuValues)
    sortedBleu = np.flip(np.sort(bleuValues),0)
    sortedIndex = np.flip(np.argsort(bleuValues),0)
    
    print(sortedIndex[:10])
    print(sortedBleu[:10])    
    for i in range(10):
        print(sys[sortedIndex[i]]," -->")
        print(ref[sortedIndex[i]], "\n")
    return bleu/n

#writes to file
def write_to_file(file,txt,mode):    
    with open(file,mode) as f: 
        for line in txt:
            f.write(line)

            
if __name__ == "__main__":
    main()# -*- coding: utf-8 -*-    