# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import time
from word2vec import load_processed_embeddings
from Model._data_generator import read_text_file,write_to_file
from fuseability_model import shuffle_data,form_triples,extract_labels_data,get_sents_embeddings,batching,\
forward_prop,back_prop,positive, negative, contrastive_divergence,error,format_file,log_time,convert_to_categorical
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
#from logistic_classifier2 import Logistic_Classifier
from DNN_classifier import dnn_classifier

#validation set
SOURCE_SENTS1 = "B_Validating/Fusion_Corpus/Positives/Twos/first.txt"
SOURCE_SENTS2 =  "B_Validating/Fusion_Corpus/Positives/Twos/second.txt"
_SOURCE_SENTS1 = "B_Validating/Fusion_Corpus/Negatives/Twos/first.txt"
_SOURCE_SENTS2 =  "B_Validating/Fusion_Corpus/Negatives/Twos/second.txt"

#test set
TEST_SOURCE_SENTS1 = "C_Testing/Fusion_Corpus/Positives/Twos/first.txt"
TEST_SOURCE_SENTS2 =  "C_Testing/Fusion_Corpus/Positives/Twos/second.txt"
_TEST_SOURCE_SENTS1 = "C_Testing/Fusion_Corpus/Negatives/Twos/first.txt"
_TEST_SOURCE_SENTS2 =  "C_Testing/Fusion_Corpus/Negatives/Twos/second.txt"

#train set
TRAIN_SOURCE_SENTS1 = "A_Training/Fusion_Corpus/Positives/Twos/first.txt"
TRAIN_SOURCE_SENTS2 =  "A_Training/Fusion_Corpus/Positives/Twos/second.txt"
_TRAIN_SOURCE_SENTS1 = "A_Training/Fusion_Corpus/Negatives/Twos/first.txt"
_TRAIN_SOURCE_SENTS2 =  "A_Training/Fusion_Corpus/Negatives/Twos/second.txt"

#trained model
MODEL_PATH1 = 'LOG_DIR_300/RBM_model/Sent1/'
MODEL_PATH2 = 'LOG_DIR_300/RBM_model/Sent2/'
MODEL = 'LOG_DIR_300/RBM_model/Evaluating/'
CONCAT = 'LOG_DIR_300/RBM_model/Concantenated/'
DIM_RED = 'LOG_DIR_300/RBM_model/Dim_Red/'
CLASSIFIER = 'LOG_DIR_300/RBM_model/Logistic_Classifier/'
STATUS_LOG = 'LOG_DIR_300/RBM_model/log.txt'

#NUM_EXAMPLES = 2000 #temporarily working with 1500 examples
BATCH_SIZE = 100
display_batch =  100
isConcat = False

def main(): 
    start = time.time()
    '''        
    #validation set    
    with tf.Session(graph = tf.Graph()) as sess: 
        validate_data = prepare_data(SOURCE_SENTS1,SOURCE_SENTS2,_SOURCE_SENTS1,_SOURCE_SENTS1)        
        run_experiment("Validation",validate_data,start)
    '''   
    
    #test set    
    with tf.Session(graph = tf.Graph()) as sess: 
        test_data = prepare_data(TEST_SOURCE_SENTS1,TEST_SOURCE_SENTS2,_TEST_SOURCE_SENTS1,_TEST_SOURCE_SENTS1)        
        run_experiment("Test",test_data,start)
    
    '''
    #train set    
    with tf.Session(graph = tf.Graph()) as sess: 
        train_data = prepare_data(TRAIN_SOURCE_SENTS1,TRAIN_SOURCE_SENTS2,_TRAIN_SOURCE_SENTS1,_TRAIN_SOURCE_SENTS1)        
        run_experiment("Train",train_data,start)
    '''
    
 
def run_experiment(desc,data,start):
    
    labels, sent1, sent2 = extract_labels_data(data)    
    
    #temporarily working with only a few datapoints    
    #labels = labels[0:NUM_EXAMPLES]
    '''
    cat_lab = convert_to_categorical(labels)    
    
    labels = np.asarray(batching(labels,len(labels),batchSize=BATCH_SIZE))    
    cat_lab = np.asarray(batching(cat_lab,len(cat_lab),batchSize=BATCH_SIZE))    
      
    n,nBatch = labels.shape  
    nData = n*nBatch      
    '''
    write_to_file(STATUS_LOG,desc) 
    
    
    sent1_ = wrapper_model("Sent1",MODEL_PATH1,MODEL,sent1)    
    sent2_ = wrapper_model("Sent2",MODEL_PATH2,MODEL,sent2)    
    
    conc = concat_hidden_states(sent1_,sent2_)    
    conc = wrapper_model("Conc",CONCAT,MODEL,conc)
    
    #dimensionality reduction
    dim = transposer(conc)     
    dim = transposer(wrapper_model("Dim_R",DIM_RED,MODEL,dim))            
    #log_time(start)
    #classifier
    '''
    weights, biases= getParameters(nBatch,CLASSIFIER,logistic=True)        
    classifier = Logistic_Classifier(nData,dim,cat_lab,labels,n,CLASSIFIER)
    classifier.test_classifier(desc,weights,biases)
    '''
    dnn_classifier(desc,dim,labels)
    
    #NUM_OF_HIDDEN_UNITS = 1
    #dest_file = wrapper_rbm("Dim Reduction",DIM_RED,NUM_EPOCHS)
    #print_tensors_in_checkpoint_file(file_name='LOG_DIR/RBM_model/Concantenated/model.ckpt',tensor_name='',all_tensors=False)      
    #print_tensors_in_checkpoint_file(file_name='LOG_DIR/RBM_model/Dim_Red/temp.ckpt',tensor_name='',all_tensors=False)          
    
    
def wrapper_model(desc,source_dir,dest_dir,sents):
    if(not isConcat):
        with tf.Session(graph = tf.Graph()) as sess:        
                vocab, word_embeddings = load_processed_embeddings(sess)            
                get_sents_embeddings(sents,word_embeddings,vocab,sess,dest_dir)
                
        with tf.Session(graph = tf.Graph()) as sess:         
                saver = tf.train.import_meta_graph(dest_dir+'temp.ckpt.meta')   
                saver.restore(sess,dest_dir+'temp.ckpt')
                
                sent_embed = sess.run("embeds:0")
                #sent_embed = tf.convert_to_tensor(sent_embed)         
                n_examples,n_visible,embd_dim = sent_embed.shape 
                #print(n_examples)
                #DIMENSION = int(embd_dim)            
                batched = batching(sent_embed,n_examples,sess,dest_dir)              
                #batching(sent_embed,n_examples,sess,dest_dir,batchSize=BATCH_SIZE)  
        '''    
        with tf.Session(graph = tf.Graph()) as sess:        
                saver = tf.train.import_meta_graph(dest_dir+'temp.ckpt.meta')   
                saver.restore(sess,dest_dir+'temp.ckpt')
                batched = sess.run("batches:0") 
        '''
    else:
        batched = sents
    
    format_file(STATUS_LOG,display_batch)
    processed_sent = run_model(desc,batched,source_dir,dest_dir) 
            
    return processed_sent
            
        
def run_model(desc,batches,source_dir,dest_dir): 
    global DIMENSION     
    nBatch,n_visible,embd_dim = batches[0].shape     
    n = len(batches)    
    DIMENSION = embd_dim    
    w,bh,bv= getParameters(nBatch,source_dir)
    cost = 0
    processed = []    
    status = '{0: <10}'.format(desc)    
    for batch_no in range(n):
        with tf.Session(graph = tf.Graph()) as sess:       
            visible = batches[batch_no]                        
            hidden_1,hidden_1_states = forward_prop(visible,w,bh)        
            pos = positive(hidden_1,visible)    
            
            #reconstruction
            visible_1 = back_prop(hidden_1_states,w,bv) 
            neg_hidden,neg_hidden_states = forward_prop(visible_1,w,bh)        
            neg = negative(neg_hidden,visible_1)         
            
            cd = sess.run(contrastive_divergence(pos,neg))
            err = sess.run(error(visible_1,visible))
            
            hidden_1 = tf.transpose(hidden_1,perm =[0,2,1])            
            cost += err  
            processed.append(sess.run(hidden_1))   
            
            if(batch_no % display_batch == 0):            
                cst = str("{:.5f}".format(err))
                status += '{0: <10}'.format(cst) 
                
    av_batch_cost = cost/n
    format_av_batch_cost = str("{:.5f}".format(av_batch_cost))
    status += '{0: <10}'.format(format_av_batch_cost)
    write_to_file(STATUS_LOG,status)
    return np.asarray(processed)

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
    
   
def prepare_data(source1,source2,_source1,_source2):
    #shuffle the ~2500 positive pairs and form triples with the label
    sents1 = read_text_file(source1)
    sents2 = read_text_file(source2)     
    sents1,sents2 = shuffle_data(sents1,sents2,2000) #2000p,3400n test
    data = form_triples(1,sents1,sents2)    
        
    #mix with the 4300 negative pairs
    _sents1 = read_text_file(_source1)
    _sents2 = read_text_file(_source2)     
    _sents1,_sents2 = shuffle_data(_sents1,_sents2,3400)    
    neg = form_triples(0,_sents1,_sents2)    
    
    data.extend(neg)   #combine positive and negative examples    
    np.random.shuffle(data)    #shuffle the mix   
    
    return data

def concat_hidden_states(source1,source2):
    global isConcat
    model = []
    for batch_no in range(len(source1)):         
        h1 = np.asarray(source1)[batch_no]
        h2 = np.asarray(source2)[batch_no]
        batch = np.concatenate((h1,h2),axis = 1)        
        model.append(batch)        
    model = np.array(model)    
    isConcat = True
    #save_m(model,MODEL)
    
    return model

def transposer(source):
    with tf.Session(graph = tf.Graph()) as sess: 
        data = []    
        for batch_no in range(len(source)): 
            h = np.asarray(source)[batch_no] 
            h = tf.transpose(h,perm= [0,2,1])
            data.append(sess.run(h))        
            
    data = np.array(data)
    #print(data.shape)
    #save_m(data,MODEL)
    return data


def save_m(model,dest_dir):
    sess = tf.Session()
    batches = tf.Variable(model,name='batches')
    sess.run(tf.global_variables_initializer())  
    #sess.run(batches.initializer)
    saver = tf.train.Saver()  
    saver.save(sess,dest_dir+'temp.ckpt')     
    
def get_conc_hidden_states(sent1,sent2):       
    sent1_ = wrapper_model("Sent1",MODEL_PATH1,MODEL,sent1)    
    sent2_ = wrapper_model("Sent2",MODEL_PATH2,MODEL,sent2)    
    
    conc = concat_hidden_states(sent1_,sent2_)    
    conc = wrapper_model("Conc",CONCAT,MODEL,conc)
    num_batches,batchSize,features,dimension = conc.shape
    unstacked_conc_h= tf.reshape(conc,[num_batches*batchSize,features,dimension])
    return unstacked_conc_h,conc

if __name__ == "__main__":
    main()
