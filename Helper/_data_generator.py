# -*- coding: utf-8 -*-

'''
    Author:         Elozino Egonmwan
    Date Started:   Nov.06.2017
    Description:    Contains helper functions    
'''

       
#reads from a text file   
def read_text_file(text_file,limit = None):  
  lines = []  
  with open(text_file, "r") as f:  
    print("Reading from ",text_file, "...")
    if(limit is None):
        for line in f:
            lines.append(line.strip())
    else:
        for line in f:             
            lines.append(line.strip())
            if(len(lines) == limit):
                break
  return lines
    
#Separate out article and abstract sentences
def get_abs_art(sent):  
  article_sents = []
  abstract_sents = []
  is_article = False
  for word in sent.split():
    if (word.startswith('article=')):
      is_article = True      
    if is_article:      
      article_sents.append(word)
    else:
      abstract_sents.append(word)
      
  # Make abstract into a single string and remove all unnecessary annotations
  abstract = ' '.join(abstract_sents)
  abstract = abstract.replace('abstract=b"','').replace("abstract=b'",'').replace('<s>','').replace('</s>"','').replace('</s>','.').replace('. .','.')

  # Make article into a single string and remove all unnecessary annotations
  article = ' '.join(article_sents)
  article = article.replace('article=b','').replace('"','')  

  return abstract, article


#make sentences into groups
def group_sent(sent,group_length):
    group = []
    for i in range(len(sent) - group_length + 1):
        group.append(sent[i:i+group_length])
    return group



#remove stopwords, punctuations
def preprocess(sents):
    
    #stoplist = set(stopwords.words('english'))
    stoplist = {'me', 'up', 'them', 'mightn', 'it', 'have', 'during', 'from', 'did', 
                      'she', 'were', 'can', 'about', 'over', 'in', 'ours', 'then', 'which', 'isn', 
                      'than', 'wasn', 'again', 'too', 'how', 'once', 'before', 'here', 'such', 
                      'more', 'until', 'needn', 'shouldn', 'll', 'having', 'does', 'with', 'of', 
                      'above', 'himself', 'very', 'wouldn', 'and', 'no', 'had', 've', 'ourselves', 
                      'if', 'he', 'theirs', 'd', 'as', 'do', 'i', 'when', 'but', 't', 'yourselves', 
                      'you', 'nor', 'ma', 'has', 'so', 'its', 'a', 'herself', 
                      'is', 'being', 'after', 'into', 'some', 'that', 'been', 'hasn', 'will', 'should',
                      'themselves', 'my', 'don', 'her', 'won', 'down', 'be', 'only', 'ain', 'or', 
                      'haven', 'what', 'was', 'not', 'who', 'him', 'didn', 'each', 'yours',
                      'o', 'both', 'an', 'this', 'under', 'for', 'why', 's', 'y', 'we', 'where', 'doing', 
                      'itself', 'most', 'weren', 'own', 'shan', 'at', 'those', 'couldn', 'to', 'his', 
                      'they', 'between', 'same', 'your', 'all', 'just', 'are', 're', 'there', 'm', 
                      'below', 'mustn', 'now', 'our', 'while', 'hadn', 'out', 'the', 'am', 'by', 
                      'hers', 'whom', 'off', 'aren', 'these', 'any', 'on', 'yourself', 'doesn', 
                      'other', 'few', 'myself', 'further', 'their',
                      'but','-lrb-','cnn','-rrb-','.'} #against, through, because were removed
    
    processed_sents = [word.lower() for word in sents.split() 
                            if word.lower() not in stoplist and len(word) > 2]
    return processed_sents


#writes to file
def write_to_file(file,line):
    with open(file,"a") as f:        
        f.write(line)
        f.write('\n')

#deletes initial file contents   
def reset_file(FILES):
    for f in FILES:
        with open(f, "w"):
            pass

def get_files_path():
    TESTING = "CNN_Daily_Mail/test.txt"
    STATUS_LOG = "C_Testing/Fusion_Corpus/log2.txt"
    
    POSITIVES = "C_Testing/Fusion_Corpus/Positives/zeefusioncorpus2.csv"
    TWOS_A = "C_Testing/Fusion_Corpus/Positives/Twos/first.txt"
    TWOS_B = "C_Testing/Fusion_Corpus/Positives/Twos/second.txt"
    TWOS_F = "C_Testing/Fusion_Corpus/Positives/Twos/fused.txt"
    
    THREES_A = "C_Testing/Fusion_Corpus/Positives/Threes/first.txt"
    THREES_B = "C_Testing/Fusion_Corpus/Positives/Threes/second.txt"
    THREES_C = "C_Testing/Fusion_Corpus/Positives/Threes/third.txt"
    THREES_F = "C_Testing/Fusion_Corpus/Positives/Threes/fused.txt"
    
    FOURS_A = "C_Testing/Fusion_Corpus/Positives/Fours/first.txt"
    FOURS_B = "C_Testing/Fusion_Corpus/Positives/Fours/second.txt"
    FOURS_C = "C_Testing/Fusion_Corpus/Positives/Fours/third.txt"
    FOURS_D = "C_Testing/Fusion_Corpus/Positives/Fours/fourth.txt"
    FOURS_F = "C_Testing/Fusion_Corpus/Positives/Fours/fused.txt"
    
    FIVES_A = "C_Testing/Fusion_Corpus/Positives/Fives/first.txt"
    FIVES_B = "C_Testing/Fusion_Corpus/Positives/Fives/second.txt"
    FIVES_C = "C_Testing/Fusion_Corpus/Positives/Fives/third.txt"
    FIVES_D = "C_Testing/Fusion_Corpus/Positives/Fives/fourth.txt"
    FIVES_E = "C_Testing/Fusion_Corpus/Positives/Fives/fifth.txt"
    FIVES_F = "C_Testing/Fusion_Corpus/Positives/Fives/fused.txt"
    
    NEGATIVES = "C_Testing/Fusion_Corpus/Negatives/negs_zeefusioncorpus2.csv"
    _TWOS_A = "C_Testing/Fusion_Corpus/Negatives/Twos/first.txt"
    _TWOS_B = "C_Testing/Fusion_Corpus/Negatives/Twos/second.txt"
    _TWOS_F = "C_Testing/Fusion_Corpus/Negatives/Twos/fused.txt"
    
    _THREES_A = "C_Testing/Fusion_Corpus/Negatives/Threes/first.txt"
    _THREES_B = "C_Testing/Fusion_Corpus/Negatives/Threes/second.txt"
    _THREES_C = "C_Testing/Fusion_Corpus/Negatives/Threes/third.txt"
    _THREES_F = "C_Testing/Fusion_Corpus/Negatives/Threes/fused.txt"
    
    _FOURS_A = "C_Testing/Fusion_Corpus/Negatives/Fours/first.txt"
    _FOURS_B = "C_Testing/Fusion_Corpus/Negatives/Fours/second.txt"
    _FOURS_C = "C_Testing/Fusion_Corpus/Negatives/Fours/third.txt"
    _FOURS_D = "C_Testing/Fusion_Corpus/Negatives/Fours/fourth.txt"
    _FOURS_F = "C_Testing/Fusion_Corpus/Negatives/Fours/fused.txt"
    
    _FIVES_A = "C_Testing/Fusion_Corpus/Negatives/Fives/first.txt"
    _FIVES_B = "C_Testing/Fusion_Corpus/Negatives/Fives/second.txt"
    _FIVES_C = "C_Testing/Fusion_Corpus/Negatives/Fives/third.txt"
    _FIVES_D = "C_Testing/Fusion_Corpus/Negatives/Fives/fourth.txt"
    _FIVES_E = "C_Testing/Fusion_Corpus/Negatives/Fives/fifth.txt"
    _FIVES_F = "C_Testing/Fusion_Corpus/Negatives/Fives/fused.txt"
    
    files = []
    files.extend([STATUS_LOG,POSITIVES,TWOS_A,TWOS_B,TWOS_F,\
            THREES_A,THREES_B,THREES_C,THREES_F,\
            FOURS_A,FOURS_B,FOURS_C,FOURS_D,FOURS_F,\
            FIVES_A,FIVES_B,FIVES_C,FIVES_D,FIVES_E,FIVES_F,\
            NEGATIVES,_TWOS_A,_TWOS_B,_TWOS_F,\
            _THREES_A,_THREES_B,_THREES_C,_THREES_F,\
            _FOURS_A,_FOURS_B,_FOURS_C,_FOURS_D,_FOURS_F,\
            _FIVES_A,_FIVES_B,_FIVES_C,_FIVES_D,_FIVES_E,_FIVES_F])
    
    return files,TESTING,STATUS_LOG,POSITIVES,TWOS_A,TWOS_B,TWOS_F,\
            THREES_A,THREES_B,THREES_C,THREES_F,\
            FOURS_A,FOURS_B,FOURS_C,FOURS_D,FOURS_F,\
            FIVES_A,FIVES_B,FIVES_C,FIVES_D,FIVES_E,FIVES_F,\
            NEGATIVES,_TWOS_A,_TWOS_B,_TWOS_F,\
            _THREES_A,_THREES_B,_THREES_C,_THREES_F,\
            _FOURS_A,_FOURS_B,_FOURS_C,_FOURS_D,_FOURS_F,\
            _FIVES_A,_FIVES_B,_FIVES_C,_FIVES_D,_FIVES_E,_FIVES_F