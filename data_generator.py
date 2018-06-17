'''
    Author:         Elozino Egonmwan
    Date Started:   Nov.13.2017
    Description:    Extracts fusion pairs <Input sentences, fusion output>
                    from the CNN/Daily Mail Corpus which contains
                    over 287,000 pairs of articles and their highlights
                    using the following RULES:
                    
                    1) Word overlap count between an highlight sentence in the 
                    abstract and the sentences in the article must fall
                    within the stipulated bounds.
                    
                    2) Adjacent sentences in the extracted input 
                    must not be more than 5 sentences apart.
                    
                    3) The difference between the length of 
                        a. the union of all unique words in selected article sents
                        b. the abstract sent <after lemmatization and removal of stops
                     must not be greater than 3
                    
    Perks:          With slight modifications, it can also identify paraphrase
                    and compression pairs.

'''   

import nltk
import time
from Model._data_generator import preprocess,read_text_file,get_abs_art,write_to_file,reset_file,get_files_path
from nltk.stem.porter import PorterStemmer

START = time.time()
START_TIME = time.asctime( time.localtime(START))

OVERLAP_UPPER_LIMIT = 0.6 #60% of the abstract sent without stopwords
OVERLAP_LOWER_LIMIT = 0.4 #40% of the abstract sent without stopwords
ADJACENCY_DIST = 5 #maximum distance between adjacent sents
COVERAGE_DIFF = 3 #maximum difference between all article sents and abstract sent

#get paths to files
FILES,TESTING,STATUS_LOG,POSITIVES,TWOS_A,TWOS_B,TWOS_F,\
THREES_A,THREES_B,THREES_C,THREES_F,\
FOURS_A,FOURS_B,FOURS_C,FOURS_D,FOURS_F,\
FIVES_A,FIVES_B,FIVES_C,FIVES_D,FIVES_E,FIVES_F,\
NEGATIVES,_TWOS_A,_TWOS_B,_TWOS_F,\
_THREES_A,_THREES_B,_THREES_C,_THREES_F,\
_FOURS_A,_FOURS_B,_FOURS_C,_FOURS_D,_FOURS_F,\
_FIVES_A,_FIVES_B,_FIVES_C,_FIVES_D,_FIVES_E,_FIVES_F = get_files_path()
 
twos_count = threes_count = fours_count= fives_count= multi_count = nPos = total = 0
twos_count_neg = threes_count_neg = fours_count_neg = fives_count_neg = nNeg = 0

#initialize storage
positves,twos_a,twos_b,twos_f,threes_a,threes_b,threes_c,threes_f,\
fours_a,fours_b,fours_c,fours_d,fours_f,\
fives_a,fives_b,fives_c,fives_d,fives_e,fives_f,\
negatves,_twos_a,_twos_b,_twos_f,_threes_a,_threes_b,_threes_c,_threes_f,\
_fours_a,_fours_b,_fours_c,_fours_d,_fours_f,\
_fives_a,_fives_b,_fives_c,_fives_d,_fives_e,_fives_f=[],[],[],[],[],[],[],[],\
[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],\
[],[],[],[],[],[],[],[],[],[]


def main():
   lines = read_text_file(TESTING)   
   print("Processing...")
   error_log = []   
   num_of_processed_lines = 0
   reset_file(FILES) #reset file #temp commented
   for l in range(len(lines)):
       abstract, article = get_abs_art(lines[l])
       abs_sents = nltk.sent_tokenize(abstract)
       art_sents = nltk.sent_tokenize(article)                 
       try:           
           fusion_pairs_extractor(art_sents,abs_sents,str(l))                
       except BaseException as e:           
           print("Error in line: ",l, " ",str(e))           
           error_log.append(l)           
       else:
           num_of_processed_lines += 1
           if(num_of_processed_lines%500 == 0):
               print(num_of_processed_lines,"...")               
               
   save_info(num_of_processed_lines, error_log)            #temp comment
   writeToFile()
   
#extracts sentences from the article
#based on the word_overlap between the article n abstract               
def fusion_pairs_extractor(article,abstract,case_id):
    global total
    for a in range(len(abstract)): 
        total += 1
        positives, negatives = screen_one(abstract,a,article)
        positives = screen_two(positives)        
        screen_three(positives,article,abstract,a,case_id)        
        save_negative_example(negatives, article,abstract,a,case_id)       

#first screening
#word overlap between article sents and abstract must be within limit
def screen_one(abstract, a, article):
    global ignored
    upper_limit = round(len(preprocess(abstract[a]))*OVERLAP_UPPER_LIMIT)
    lower_limit = round(len(preprocess(abstract[a]))*OVERLAP_LOWER_LIMIT) 
    positives = []
    negatives = []
    for r in range(len(article)):                           
        word_overlap,count = count_word_overlap(abstract[a],article[r])
        if count > upper_limit:
            positives = []
            break
        if (count > lower_limit and count <=  upper_limit):                
            positives.append(r)
        elif (count > 0 and count <= lower_limit):
            negatives.append(r)        
    return positives, negatives

#second screening
#adjacent sentences must be within bounded distance
def screen_two(positives):
    if(len(positives) > 1):
        positives = adjacency_dist(positives)
    return positives

#third screening
#the difference between the length of the overlap of the
#   a. the union of all unique words in selected article sents
#   b. the abstract sent <after lemmatization and removal of stops
# must be within bound 
def screen_three(positives,article,abstract,a,case_id):  
    #porterStemmer = PorterStemmer()
    if(len(positives) > 1):        
        unions = ''
        for ids in positives:
            intersec, _ = count_word_overlap(abstract[a],article[ids])            
            unions += ' '.join(intersec)
        
        unions = ' '.join(set([u for u in unions.split()])) 
        
        '''
        unions = set([u for u in unions.split()])
        print(unions)        
        abst = set([porterStemmer.stem(w) for w in preprocess(abstract[a])])
        if abst.issubset(unions):
            save_positive_example(positives, article,abstract,a,case_id)       
        '''
        
        word_overlap, count = count_word_overlap(abstract[a],unions)
        diff = len(preprocess(abstract[a])) - count            
        if(diff <= COVERAGE_DIFF):
            save_positive_example(positives, article,abstract,a,case_id) 
                           

#saves all positive examples to file
def save_positive_example(positives, article,abstract,a,case_id):        
    if(len(positives) > 1):        
        stats(len(positives),positives,article,abstract[a])
        
        #TEMPORARILY commented        
        positives = formOutput(positives,article,case_id)
        line = "\n" + case_id + ",S," + str(a) + "," + abstract[a]        
        positves.append(line)
        for l in positives:            
            positves.append(l)
        

#save negative examples to file                   
def save_negative_example(track_negatives, article,abstract,a,case_id):      
    if(len(track_negatives) > 1):        
        stats_neg(len(track_negatives),track_negatives,article,abstract[a])
        if(len(track_negatives) > 5):
            track_negatives = track_negatives[:5]
            
        #Temporarily commented                            
        filter_negatives = formOutput(track_negatives,article,case_id)
        line = "\n" + case_id + ",S," + str(a) + "," + abstract[a]        
        negatves.append(line)
        for l in filter_negatives:           
            negatves.append(l)
                    
    
#counts word overlap between sents after stemming and removal of stopwords
def count_word_overlap(sent1,sent2):
    porterStemmer = PorterStemmer()
    sent1 = [porterStemmer.stem(w) for w in preprocess(sent1)]
    sent2 = [porterStemmer.stem(w) for w in preprocess(sent2)]
    n= set(sent1).intersection(set(sent2))
    return n,len(n)

#forms adjacency pairs and checks for validity
def adjacency_dist(source):    
    valid = []
    for i in range(len(source) - 1):        
        group = source[i:i+2]        
        if (isValid(group) and i==0):
            valid.append(group[0])
            valid.append(group[1])
        elif(isValid(group) and i != 0):
            valid.append(group[1])
        else:
            break
    return valid

#validate that adjacent sentences are within the distance limit
def isValid(pair):
    if((pair[1]-pair[0]) <= ADJACENCY_DIST):
        return True
    else:
        return False

def formOutput(ids, article, case_id):
    out = []
    for i in ids:
        line = ""
        line = line + case_id + "," + "Sub," + str(i) + "," + article[i]        
        out.append(line)
    return out

#computes stats of the number of input sentences fused for positive examples
def stats(n,ids,article,abst):    
    global twos_count, threes_count, fours_count, fives_count, multi_count           
    if (n == 2):
        twos_count += 1        
        twos_a.append(article[ids[0]])
        twos_b.append(article[ids[1]])
        twos_f.append(abst)
    
    if (n == 3):
        threes_count += 1
        threes_a.append(article[ids[0]])
        threes_b.append(article[ids[1]])
        threes_c.append(article[ids[2]])
        threes_f.append(abst)
        
    if (n == 4):
        fours_count += 1
        fours_a.append(article[ids[0]])
        fours_b.append(article[ids[1]])
        fours_c.append(article[ids[2]])
        fours_d.append(article[ids[3]])
        fours_f.append(abst)

    if (n == 5):
        fives_count += 1
        fives_a.append(article[ids[0]])
        fives_b.append(article[ids[1]])
        fives_c.append(article[ids[2]])
        fives_d.append(article[ids[3]])
        fives_e.append(article[ids[4]])
        fives_f.append(abst)
        
    elif (n > 5):
        multi_count += 1

#computes stats of the number of input sentences fused for negative examples
def stats_neg(n,ids,article,abst):    
    global twos_count_neg, threes_count_neg, fours_count_neg, fives_count_neg
    if (n == 2):
        twos_count_neg += 1
        _twos_a.append(article[ids[0]])
        _twos_b.append(article[ids[1]])
        _twos_f.append(abst)
       
    if (n == 3):
        threes_count_neg += 1
        _threes_a.append(article[ids[0]])
        _threes_b.append(article[ids[1]])
        _threes_c.append(article[ids[2]])
        _threes_f.append(abst)
        
    if (n == 4):
        fours_count_neg += 1
        _fours_a.append(article[ids[0]])
        _fours_b.append(article[ids[1]])
        _fours_c.append(article[ids[2]])
        _fours_d.append(article[ids[3]])
        _fours_f.append(abst)
        
    elif (n >= 5):
        fives_count_neg += 1
        _fives_a.append(article[ids[0]])
        _fives_b.append(article[ids[1]])
        _fives_c.append(article[ids[2]])
        _fives_d.append(article[ids[3]])
        _fives_e.append(article[ids[4]])
        _fives_f.append(abst)
    
#prints the computed stats for positive examples      
def printStats():
    global twos_count, threes_count, fours_count, fives_count, multi_count
    line = " ("
    if (twos_count > 0):
        line += "2's- " + str(twos_count)
    if (threes_count > 0):
        line += "; 3's- " + str(threes_count)
    if (fours_count > 0):
        line += "; 4's- " + str(fours_count)
    if (fives_count > 0):
        line += "; 5's- " + str(fives_count)
    if (multi_count > 0):
        line += "; >5's- " + str(multi_count)
    return line + ")" 

#prints the computed stats for negative examples      
def printStats_neg():
    global twos_count_neg, threes_count_neg, fours_count_neg, fives_count_neg
    line = " ("
    if (twos_count_neg > 0):
        line += "2's- " + str(twos_count_neg)
    if (threes_count_neg > 0):
        line += "; 3's- " + str(threes_count_neg)
    if (fours_count_neg > 0):
        line += "; 4's- " + str(fours_count_neg)
    if (fives_count_neg > 0):
        line += "; 5's- " + str(fives_count_neg)    
    return line + ")" 


#logs statistics to files
def save_info(nProc,errorLog): 
   global nPos,nNeg     
   nPos = twos_count + threes_count + fours_count + fives_count + multi_count
   nNeg = twos_count_neg + threes_count_neg + fours_count_neg + fives_count_neg
   status_msg = "<Abstract, Article> pairs\t: " + str(nProc)
   status_msg += "\n\nPositive examples\t\t\t\t: " + str(nPos)
   status_msg += printStats()
   status_msg += "\nNegative examples\t\t\t\t: " + str(nNeg)
   status_msg += printStats_neg()
   status_msg += "\nIgnored examples\t\t\t\t: " + str(total -nPos - nNeg)
   status_msg += "\nTotal\t\t\t\t\t\t\t\t: " + str(total)
   if(len(errorLog) > 0):
       status_msg += "\nError in line(s)\t\t\t\t: " + ' '.join(str(errorLog))
       
   status_msg += "\n\nProgram started on\t\t\t: " + START_TIME
   end = time.time()
   end_time = time.asctime( time.localtime(end))
   
   status_msg += "\nProgram ended on\t\t\t\t: " + end_time 
   duration = end - START
   status_msg += "\nDuration of Program\t\t\t: " + time.strftime('%H:%M:%S', time.gmtime(duration))
   
   write_to_file(STATUS_LOG, status_msg)    
        
def writeToFile():    
    print("Writing to file...")        
    if(nPos > 0):
        print("Writing positive examples to file..")
        write_to_file(POSITIVES,'\n'.join(str(u) for u in positves))
        if(len(twos_a) > 0):            
            write_to_file(TWOS_A,'\n'.join(str(u) for u in twos_a))
            write_to_file(TWOS_B,'\n'.join(str(u) for u in twos_b))
            write_to_file(TWOS_F,'\n'.join(str(u) for u in twos_f))
        if(len(threes_a) > 0):
            write_to_file(THREES_A,'\n'.join(str(u) for u in threes_a))
            write_to_file(THREES_B,'\n'.join(str(u) for u in threes_b))
            write_to_file(THREES_C,'\n'.join(str(u) for u in threes_c))
            write_to_file(THREES_F,'\n'.join(str(u) for u in threes_f))
        if(len(fours_a) > 0):
            write_to_file(FOURS_A,'\n'.join(str(u) for u in fours_a))
            write_to_file(FOURS_B,'\n'.join(str(u) for u in fours_b))
            write_to_file(FOURS_C,'\n'.join(str(u) for u in fours_c))
            write_to_file(FOURS_D,'\n'.join(str(u) for u in fours_d))
            write_to_file(FOURS_F,'\n'.join(str(u) for u in fours_f))
        if(len(fives_a) > 0):
            write_to_file(FIVES_A,'\n'.join(str(u) for u in fives_a))
            write_to_file(FIVES_B,'\n'.join(str(u) for u in fives_b))
            write_to_file(FIVES_C,'\n'.join(str(u) for u in fives_c))
            write_to_file(FIVES_D,'\n'.join(str(u) for u in fives_d))
            write_to_file(FIVES_E,'\n'.join(str(u) for u in fives_e))
            write_to_file(FIVES_F,'\n'.join(str(u) for u in fives_f))
    
    if(nNeg > 0):
        print("Writing negative examples to file..")
        write_to_file(NEGATIVES,'\n'.join(str(u) for u in negatves))
        if(len(_twos_a) > 0):
            write_to_file(_TWOS_A,'\n'.join(str(u) for u in _twos_a))
            write_to_file(_TWOS_B,'\n'.join(str(u) for u in _twos_b))
            write_to_file(_TWOS_F,'\n'.join(str(u) for u in _twos_f))
        if(len(_threes_a) > 0):
            write_to_file(_THREES_A,'\n'.join(str(u) for u in _threes_a))
            write_to_file(_THREES_B,'\n'.join(str(u) for u in _threes_b))
            write_to_file(_THREES_C,'\n'.join(str(u) for u in _threes_c))
            write_to_file(_THREES_F,'\n'.join(str(u) for u in _threes_f))
        if(len(_fours_a) > 0):
            write_to_file(_FOURS_A,'\n'.join(str(u) for u in _fours_a))
            write_to_file(_FOURS_B,'\n'.join(str(u) for u in _fours_b))
            write_to_file(_FOURS_C,'\n'.join(str(u) for u in _fours_c))
            write_to_file(_FOURS_D,'\n'.join(str(u) for u in _fours_d))
            write_to_file(_FOURS_F,'\n'.join(str(u) for u in _fours_f))
        if(len(fives_a) > 0):
            write_to_file(_FIVES_A,'\n'.join(str(u) for u in _fives_a))
            write_to_file(_FIVES_B,'\n'.join(str(u) for u in _fives_b))
            write_to_file(_FIVES_C,'\n'.join(str(u) for u in _fives_c))
            write_to_file(_FIVES_D,'\n'.join(str(u) for u in _fives_d))
            write_to_file(_FIVES_E,'\n'.join(str(u) for u in _fives_e))
            write_to_file(_FIVES_F,'\n'.join(str(u) for u in _fives_f))
    
if __name__ == "__main__":
    main()