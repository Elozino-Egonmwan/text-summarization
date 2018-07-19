# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import nltk
from word2vec import load_processed_embeddings
from Helper._data_generator import read_text_file,write_to_file
from fuseability_model import get_sent_embeddings,batching,forward_prop,back_prop,positive, negative, contrastive_divergence,error,format_file,log_time
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from logistic_classifier import _classifier
import itertools

#train set
TRAIN_SOURCE_SENTS = "A_Training/Sum_Corpus/doc.txt"
TRAIN_LABELS = "A_Training/Sum_Corpus/tuned_doc.txt"

#validation set
VAL_SOURCE_SENTS = "B_Validating/Sum_Corpus/doc.txt"
VAL_LABELS = "B_Validating/Sum_Corpus/tuned_doc.txt"

#test set
TEST_SOURCE_SENTS = "C_Testing/Sum_Corpus/doc.txt"
TEST_LABELS = "C_Testing/Sum_Corpus/tuned_doc.txt"

#trained model
MODEL_PATH = 'LOG_DIR_300/Extractor/RBM/Sent/'
MODEL_PATH1 = 'LOG_DIR_300/RBM_model/Sent1/'
DIM_RED = 'LOG_DIR_300/Extractor/RBM/Dim_Red/'
DIM_RED1 = 'LOG_DIR_300/RBM_model/Dim_Red/'
CLASSIFIER = 'LOG_DIR_300/Extractor/Classifier/'
MODEL = 'LOG_DIR_300/Extractor/RBM/Evaluating/'
STATUS_LOG = 'LOG_DIR_300/Extractor/Evaluating/log.txt'

#NUM_EXAMPLES = 1000 #temporarily working with 1500 examples
BATCH_SIZE = 100
display_batch =  100
isDim = False

def main(): 
    #start = time.time()
    tf.reset_default_graph()        #start clean    
    #data = prepare_dataTrain(TRAIN_SOURCE_SENTS,TRAIN_LABELS,limit=10000)        
    #data = prepare_data(TRAIN_SOURCE_SENTS,TRAIN_LABELS,limit=20000)        
    data = prepare_data(VAL_SOURCE_SENTS,VAL_LABELS,limit=8000)        
    #data = prepare_data(TEST_SOURCE_SENTS,TEST_LABELS,limit=8000)        
    run_experiment("Train",data)   
    
    '''       
    #validation set    
    with tf.Session(graph = tf.Graph()) as sess: 
        validate_data = prepare_data(SOURCE_SENTS1,SOURCE_SENTS2,_SOURCE_SENTS1,_SOURCE_SENTS1)        
        run_experiment("Validation",validate_data,start)
          
    #test set    
    with tf.Session(graph = tf.Graph()) as sess: 
        test_data = prepare_data(TEST_SOURCE_SENTS,TEST_LABELS,limit=50)          
        run_experiment("Test",test_data)    
    '''
    
def prepare_data(source,source_labels,limit):    
    docs = read_text_file(source,limit)        
    docs = nltk.sent_tokenize(' '.join(docs))    
    docs = docs[:20000]
    
    labels = read_text_file(source_labels,limit)        
    labels = (' '.join(labels)).split()    
    labels = labels[:20000]
    labels = list(map(int,labels))    
    
    data = docs,labels
    return data

def prepare_dataTrain(source,source_labels,limit):    
    labels = read_text_file(source_labels,limit)        
    labels = (' '.join(labels)).split()    
    
    positive_labels_indices = [i for i, j in enumerate(labels) if j == '1']
    positive_labels_indices = positive_labels_indices[:20000]
    print("positives: ", len(positive_labels_indices))
    
    negative_labels_indices = [i for i, j in enumerate(labels) if j == '0']
    negative_labels_indices = negative_labels_indices[:20000]
    print("negatives: ", len(negative_labels_indices))
    
    '''    
    labels_indices= zip(positive_labels_indices,negative_labels_indices)
    labels_indices= list(itertools.chain(*labels_indices))
    labels = [labels[i] for i in labels_indices]
    labels = list(map(int,labels))    
    #print(labels[:100])
    '''
    
    labels_indices = positive_labels_indices + negative_labels_indices
    labels = [labels[i] for i in labels_indices]
    labels = list(map(int,labels))    
        
    docs = read_text_file(source,limit)        
    docs = nltk.sent_tokenize(' '.join(docs))       
    docs =[docs[i] for i in labels_indices]    
    print("num of data", len(docs))       
    
    data = form_pairs(docs,labels)  
    np.random.shuffle(data)     
    
    docs,labels = extract_labels_data(data)
    
    return docs,labels

def extract_labels_data(pairs):
    docs = []    
    labels = []
    
    for t in pairs:
        d,l = t
        labels.append(l)
        docs.append(d)        
    
    return docs,labels

def form_pairs(first,second):
    pairs =[]
    for i in range(len(second)):
        t =(first[i],second[i])
        pairs.append(t)
    return pairs

def run_experiment(desc,data):     
    global isDim
    doc,labels = data     
    doc_ = wrapper_model("Doc",MODEL_PATH,MODEL,doc)           
    
    #dimensionality reduction    
    isDim = True
    dim = wrapper_model("Dim_R",DIM_RED,MODEL,doc_) 
    '''
    num,batchSize,_,_ = dim.shape  
    n = num*batchSize
    labels =labels[:n]    
    '''
    '''
    #classifier
    for i in range(200):
        f_pos,f_neg = _classifier(desc,dim,labels,"Train",model_dir=CLASSIFIER)    
        f_pos,f_neg = _classifier(desc+str((i+1)*100),dim,labels,"Pred",model_dir=CLASSIFIER)         
        if f_pos <= 0.10 and f_neg <= 0.15:
            break
    '''
    #_classifier(desc,dim,labels,"Train",model_dir=CLASSIFIER)    
    _classifier(desc,dim,labels,"Pred",model_dir=CLASSIFIER)         
    
def wrapper_model(desc,source_dir,dest_dir,sents):
    if(not isDim):
        with tf.Session(graph = tf.Graph()) as sess:                  
            vocab, word_embeddings = load_processed_embeddings(sess)  
            #sents = batching(sents,len(sents),sess,dest_dir)              
            #sents=[get_sent_embeddings(sent,word_embeddings,vocab,sess,dest_dir) for sent in sents]
                                      
            sent_embed=get_sent_embeddings(sents,word_embeddings,vocab,sess,dest_dir)
            n_examples,n_visible,embd_dim = sent_embed.shape                 
            sents = batching(sent_embed,n_examples,sess,dest_dir)              
             
    processed_docs = run_model(desc,sents,source_dir,dest_dir)
    return processed_docs

def run_model(desc,batches,source_dir,dest_dir): 
    global DIMENSION     
    nBatch,n_visible,embd_dim = batches[0].shape     
    n = len(batches)    
    DIMENSION = embd_dim    
    w,bh,bv= getParameters(nBatch,source_dir)    
    processed = []        
    for batch_no in range(n):
        with tf.Session(graph = tf.Graph()) as sess:       
            visible = batches[batch_no]                        
            hidden_1,_ = forward_prop(visible,w,bh)                                
            processed.append(sess.run(hidden_1))               
            
    return np.asarray(processed)

def getParameters(nBatch,source_dir,logistic = None):    
    if logistic is None:
        w, bh,bv = load_model(source_dir)
        weights = []
        bias_hidden = []
        bias_visible = [] 
        for i in range(nBatch):
            weights.append(w)
            bias_hidden.append(bh)
            bias_visible.append(bv)    
        return np.asarray(weights),np.asarray(bias_hidden), np.asarray(bias_visible)
    
    else:
        w,b = load_model(source_dir,logistic)                   
        return np.asarray(w),np.asarray(b)               


def load_model(source_dir,logistic = None):    
    with tf.Session(graph = tf.Graph()) as sess:        
        model_path = source_dir + 'model.ckpt'
        saver = tf.train.import_meta_graph(model_path+ '.meta')   
        saver.restore(sess, model_path) 
        
        if logistic is None:
            w = "weights:0"
            b_h = "bias_hidden:0"
            b_v = "bias_visible:0"               
            
            weights = sess.run(w)[0]
            bias_hidden = sess.run(b_h)[0]
            bias_visible = sess.run(b_v)[0]            
            
            return weights, bias_hidden, bias_visible
        
        else:
            w = "weights:0"
            b = "biases:0"            
            
            weights = sess.run(w)
            bias = sess.run(b)
            print(weights)            
            return weights, bias                       

if __name__ == "__main__":
    main()