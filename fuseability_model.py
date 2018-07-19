# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
'''
    Deep Belief Net:
        Two(2) Siamese RBMs (Restricted Boltzmann Machines)
        Outputs 2 hidden states (1 per sentence in pair)
        The hidden states are concatenated and serves as input to another layer of RBM
        The learned features are passed to a Logistic Regressor
        The model learns to decide if a pair of sentences is fuseable or not.
    
    Date Started: Dec 16, 2017
    Layer 1 (Sent1): Dec 24, 2017
    Noticed logical errors -  Fixed: Dec 27, 2017
    Layer 2 (Sent2): Dec 28th,2017
    Layer 3(Concatenation of learnt features from 1 and 2) : Dec 28th, 2017
    Documentation and maintenance:  Dec 29th, 2017
'''

import tensorflow as tf
import numpy as np
import random
import time
from Helper._data_generator import read_text_file,write_to_file
from Helper._word2vec import isValid
from word2vec import avg_Length, create_dict, load_processed_embeddings
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
#from logistic_classifier2 import Logistic_Classifier
from logistic_classifier import _classifier

#about 42,000 positive pairs
SOURCE_SENTS1 = "A_Training/Fusion_Corpus/Positives/Twos/first.txt"
SOURCE_SENTS1 = "A_Training/Fusion_Corpus/Positives/Twos/first.txt"
SOURCE_SENTS2 =  "A_Training/Fusion_Corpus/Positives/Twos/second.txt"
MODEL_PATH1 = 'LOG_DIR_300/RBM_model/Sent1/'
MODEL_PATH2 = 'LOG_DIR_300/RBM_model/Sent2/'
CONCAT = 'LOG_DIR_300/RBM_model/Concantenated/'
DIM_RED = 'LOG_DIR_300/RBM_model/Dim_Red/'
CLASSIFIER = 'LOG_DIR_300/RBM_model/Logistic_Classifier/'
STATUS_LOG = 'LOG_DIR_300/RBM_model/log.txt'
TEMP_LOG = 'LOG_DIR_300/RBM_model/temp.txt'

#about 83,000 negative pairs
_SOURCE_SENTS1 = "A_Training/Fusion_Corpus/Negatives/Twos/first.txt"
_SOURCE_SENTS2 =  "A_Training/Fusion_Corpus/Negatives/Twos/second.txt"


BATCH_SIZE = 50
NUM_EPOCHS = 1
_NUM_EPOCHS = 1 #more epochs when training the concatenated states
LEARNING_RATE = 0.0001
NUM_OF_VISIBLE_UNITS = 25
NUM_OF_HIDDEN_UNITS = 15
DIMENSION = 300 #will be derived and automatically updated
#NUM_EXAMPLES = 500 #temporarily working with 500 examples

standard_dev = 0.01
n = 1 #number of batches initialized to 1 (this should be returned from function batching)
isConcat = False #manually computes number of visible units if true(ie when concatenating the 2 sentences)
display_batch =  100
description = "Fusibility"

def main():     
    #print_tensors_in_checkpoint_file(file_name='LOG_DIR/RBM_model/Dim_Red/model.ckpt',tensor_name='',all_tensors=False)      
    start = time.time()
    training_data = prepare_data()    
    #get labels and sentence pairs from data
    labels, sent1, sent2 = extract_labels_data(training_data)
    #train the pairs.@returns path to dir containing last layer of model     
    train_model_dir = train_pair(sent1,sent2)      
    log_time(start)          
    features = np.asarray(get_logistic_features('LOG_DIR_300/RBM_model/Dim_Red/model.ckpt'))    
    print(features.shape)
    _classifier("Training",features,labels)    
    
    tf.reset_default_graph()
    
def convert_to_categorical(data):
    converted = []
    for d in data:
        if d == 1:
            converted.append([1,0])
        else:
            converted.append([0,1])
    return converted

def prepare_data():
    #shuffle the ~42,000 positive pairs and form triples with the label
    sents1 = read_text_file(SOURCE_SENTS1)
    sents2 = read_text_file(SOURCE_SENTS2)     
    sents1,sents2 = shuffle_data(sents1,sents2,20000)
    training_data = form_triples(1,sents1,sents2)    
        
    #randomly select only 30,000 of the negative pairs and shuffle
    _sents1 = read_text_file(_SOURCE_SENTS1)
    _sents2 = read_text_file(_SOURCE_SENTS2)     
    _sents1,_sents2 = shuffle_data(_sents1,_sents2,20000)    
    neg = form_triples(0,_sents1,_sents2)    
    
    training_data.extend(neg)   #combine positive and negative examples    
    np.random.shuffle(training_data)    #shuffle the mix   
    
    return training_data
    
#trains sentence pairs, their concatenatedd hidden states in returns directory to saved model parameters    
def train_pair(sents1, sents2):   
    global NUM_OF_HIDDEN_UNITS, description,NUM_OF_VISIBLE_UNITS 
    description = "Raw"
    dest_file1 = wrapper_rbm("Sent1",MODEL_PATH1,NUM_EPOCHS,sents1)     #train first sentence in pair
    dest_file2 = wrapper_rbm("Sent2",MODEL_PATH2,NUM_EPOCHS,sents2)     #train 2nd sentence in pair
    print("Trained parameters of RBM for:\nSent1: ", dest_file1, "\nSent2: ", dest_file2)  
    
    #dest_file1 = 'LOG_DIR/RBM_model/Sent1/model.ckpt'
    #dest_file2 = 'LOG_DIR/RBM_model/Sent2/model.ckpt'
    #concat_hidden_states(dest_file1,dest_file2) #concatenate the hidden states of sents 1 and 2
    #isConcat = True  
    description = "Conc"
    NUM_OF_VISIBLE_UNITS = 30
    dest_file3 = wrapper_rbm("Conc",CONCAT,_NUM_EPOCHS)   #train concatenated states    
        
    #dimensionality reduction
    #dimensionality_reduction(dest_file3)
    description = "Dim"
    NUM_OF_HIDDEN_UNITS = 1
    NUM_OF_VISIBLE_UNITS = DIMENSION
    dest_file = wrapper_rbm("Dim Reduction",DIM_RED,NUM_EPOCHS)
    return dest_file

   
#prepares the data for training RBM
#@dest_dir : path to directory where model weights should be stored
#@raw_sents: path to the directory containing raw sentences(not vectors) to be trained
#if None, it means they have been trained already, proceed to training the concatenation
#@returns path to trained model
def wrapper_rbm(desc,dest_dir,num_epochs,raw_sents=None):
    global DIMENSION
#preprocesses raw sentences, looks up their embeddings and groups them in batches
    if not raw_sents is None:
        with tf.Session(graph = tf.Graph()) as sess:        
            vocab, word_embeddings = load_processed_embeddings(sess)            
            get_sents_embeddings(raw_sents,word_embeddings,vocab,sess,dest_dir)                    
        
    train_writer = tf.summary.FileWriter(dest_dir)     
    write_to_file(STATUS_LOG,desc)    
    error = 0    
    
    for step in range(num_epochs):         
        with tf.Session(graph = tf.Graph()) as sess:             
            if(is_exists_saved_model(dest_dir)): 
                print("training from saved model")
                #model = restore_model(dest_dir)
                model = step_restore_model(dest_dir)
                cost,err = train_from_saved(model,dest_dir,train_writer,step)
                error += err                 
            else:  
                print("training from scratch")                                                                            
                cost,err = train_from_scratch(dest_dir,train_writer,step)                  
                error += err             
            print("Step ", step, " Cost: ", "{:.5f}".format(err))
    
    #writing to file 
    m = int(n / display_batch)       
    width = 7 + (10 * m)
    format_av_cost = '{0: >' + str(width) + '}'
    av_cost = format_av_cost.format('Av.Cost')
    #av_cost = '{0: <10}'.format('Av.Cost')
    av_err = error/num_epochs
    _status = str("{:.5f}".format(av_err))
    status = av_cost + '{0: >10}'.format(_status) 
    status += '\n'
    write_to_file(STATUS_LOG,status)
            
    path_to_trained_model = dest_dir + 'model.ckpt' 
    train_writer.close()
    #print_tensors_in_checkpoint_file(file_name=path_to_trained_model,tensor_name='',all_tensors=False)  
    #tf.reset_default_graph()
    
    return path_to_trained_model        

def step_restore_model(dest_dir):
    #with tf.Session(graph = tf.Graph()) as sess: 

    sess = tf.Session()
    model = []
    model_path = dest_dir + 'model.ckpt'
    saver = tf.train.import_meta_graph(model_path+ '.meta')   
    saver.restore(sess, model_path)        
    #print_tensors_in_checkpoint_file(file_name=model_path,tensor_name='',all_tensors=False)              
    #print(sess.run(tf.report_uninitialized_variables()))
    #set names for the Variables
    w = "weights:0"
    b_h = "bias_hidden:0"
    b_v = "bias_visible:0"            
    
    #load the tensors by name    
    graph = tf.get_default_graph()
    weights = graph.get_tensor_by_name(w)
    bias_hidden = graph.get_tensor_by_name(b_h)
    bias_visible = graph.get_tensor_by_name(b_v)
    
    
    model.append(sess.run(weights))
    model.append(sess.run(bias_hidden))
    model.append(sess.run(bias_visible))
    
    hidden = []
    
    for batch_no in range(n):
        h = "hidden"+str(batch_no)+":0"        
        h_ = graph.get_tensor_by_name(h) 
        hidden.append(h_)
        #hidden.append(sess.run(h))
    model.append(sess.run(hidden))
    return model

def restore_model(dest_dir,numBatch = None):
    with tf.Session(graph = tf.Graph()) as sess: 
        
        if numBatch is None:
            nBatch = n
        else:
            nBatch = numBatch
        #sess = tf.Session()
        model = []
        model_path = dest_dir + 'model.ckpt'
        saver = tf.train.import_meta_graph(model_path+ '.meta')   
        saver.restore(sess, model_path)        
        #print_tensors_in_checkpoint_file(file_name=model_path,tensor_name='',all_tensors=False)              
        #print(sess.run(tf.report_uninitialized_variables()))
        #set names for the Variables
        w = "weights:0"
        b_h = "bias_hidden:0"
        b_v = "bias_visible:0"        
        
        model.append(sess.run(w))
        model.append(sess.run(b_h))
        model.append(sess.run(b_v))
        hidden = []
        
        for batch_no in range(nBatch):
            h = "hidden"+str(batch_no)+":0"        
            #h_ = graph.get_tensor_by_name(h) 
            #hidden.append(h_)
            hidden.append(sess.run(h))
        model.append(hidden)
    return model
    
#initialize RBM parameters, save and continue training
def train_from_scratch(dest_dir,train_writer,step): 
    #global DIMENSION    
    #model_init =[] 
    
    sess = tf.Session() 
    #nBatch,n_visible,embd_dim = batches[0].shape 
    #DIMENSION = int(embd_dim)
    w,bh,bv,h = getParameters(NUM_OF_VISIBLE_UNITS,DIMENSION)                 
    parameters_init = _init_RBM(w,bh,bv,h,sess)
    
    #model_init.append(parameters_init)    
    #save_model(model_init,dest_dir)     
    cd, err = train_from_saved(parameters_init,dest_dir,train_writer,step)    
    return cd,err

#load parameters from previous step and continue training, returns cost of trainig a batch
def train_from_saved(updates,dest_dir,train_writer,step): 
    global DIMENSION
    cd = 0  #cd (Contrastive divergence)
    err = 0    
    str_step = 'Step ' + str(step)
    status = '{0: <10}'.format(str_step) 
    hidden = []
    #sess = tf.Session()
    print('batching')
    if(description == "Raw"):
        with tf.Session(graph = tf.Graph()) as sess:         
            saver = tf.train.import_meta_graph(dest_dir+'temp.ckpt.meta')   
            saver.restore(sess,dest_dir+'temp.ckpt')            
            sent1_embed = sess.run("embeds:0")            
            n_examples,n_visible,embd_dim = sent1_embed.shape             
            DIMENSION = int(embd_dim)            
            batches = batching(sent1_embed,n_examples,sess,dest_dir)              
            
    elif(description == "Conc"):
        src_file1 = 'LOG_DIR_300/RBM_model/Sent1/model.ckpt'
        src_file2 = 'LOG_DIR_300/RBM_model/Sent2/model.ckpt'
        batches = concat_hidden_states(src_file1,src_file2) #concatenate the hidden states of sents 1 and 2
        
    else:
        src_file = 'LOG_DIR_300/RBM_model/Concantenated/model.ckpt'
        batches = dimensionality_reduction(src_file)        
    print('done batching')
    
    format_file(STATUS_LOG)    
    for batch_no in range(n):
        with tf.Session(graph = tf.Graph()) as sess:         
            #batch = return_batch(dest_dir,batch_no) 
            batch = batches[batch_no] 
            batch_cd,er,updates = train_RBM(updates,batch,batch_no,dest_dir,train_writer,step)
            _,_,_,h = updates
            #if batch_no > 0:
                #h = sess.run(h)
            hidden.append(h)
            
            cd += batch_cd
            err+= er 
            temp_status = "Step " + str(step) + " Batch " + str(batch_no) +" Cost: " + "{:.5f}".format(er)
            write_to_file(TEMP_LOG,temp_status)
            print("Step ", step, " Batch ", batch_no," Cost: ", "{:.5f}".format(er))
            if(batch_no % display_batch == 0):            
                cost = str("{:.5f}".format(er))
                status += '{0: <10}'.format(cost) 
                
    av_batch_cost = err/n
    format_av_batch_cost = str("{:.5f}".format(av_batch_cost))
    status += '{0: <10}'.format(format_av_batch_cost)
    write_to_file(STATUS_LOG,status)
    
    #hiddn = tf.convert_to_tensor(hidden)
    #sess.run(tf.global_variables_initializer())            
    #hiddn = sess.run(hiddn)
    
    print("preparing model for saving")
    model =[]
    w,bh,bv,_ = updates
    model.append(w)
    model.append(bh)
    model.append(bv)
    model.append(hidden)
    print("saving model")    
    save_model(model,dest_dir)
    print("done saving")
    
    return cd/n, av_batch_cost
        
def return_batch(dest_dir,batch_no):
    with tf.Session(graph = tf.Graph()) as sess:         
        saver = tf.train.import_meta_graph(dest_dir+'temp.ckpt.meta')   
        saver.restore(sess,dest_dir+'temp.ckpt')
        batched = sess.run("batches:0")[batch_no]
    return batched
    
    
def format_file(file,display=None):
    status = '{0: <10}'.format('Batch')
    if not display is None:
        display_b = display
    else:
        display_b = display_batch
    for batch_no in range(n):
        if(batch_no % display_b == 0):
            status += '{0: <10}'.format(str(batch_no))
    status += '{0: <10}'.format('Avg.Batch Cost')    
    write_to_file(file,status)
    
#train each batch through a step of forward and backward pass, save the new parameters
def train_RBM(updates,batch,batch_no,dest_dir, write_to_summary,step):
    
    with tf.Session(graph = tf.Graph()) as sess:         
        visible = batch
        weights,bias_hidden,bias_visible,_ = updates
        
        #update for this batch        
        hidden_1,hidden_1_states = forward_prop(visible,weights,bias_hidden)        
        pos = positive(hidden_1,visible)    
        
        #reconstruction
        visible_1 = back_prop(hidden_1_states,weights,bias_visible) 
        neg_hidden,neg_hidden_states = forward_prop(visible_1,weights,bias_hidden)        
        neg = negative(neg_hidden,visible_1)
        
        #if batch_no == 0:            
            #sess.run(tf.global_variables_initializer())
        
        cd = sess.run(contrastive_divergence(pos,neg))
        err = sess.run(error(visible_1,visible))   
        
        weights_1,bias_hidden_1,bias_visible_1 = update_vars(weights,bias_hidden,bias_visible,cd)             
        
        #sess.run([weights_1, bias_hidden_1,bias_visible_1]) 
        #if batch_no == 0:       
            #weights       
            #weights_1,bias_hidden_1,bias_visible_1= sess.run([weights_1, bias_hidden_1,bias_visible_1]) 
            
        hidden_1 = sess.run(hidden_1)
        updates = [weights_1,bias_hidden_1,bias_visible_1,hidden_1]
        
    return cd,err,updates   

def initialize_tensorboard(cd,err,w,bh,bv,wu,bhu,bvu):
        #visualizing learning
        tf.summary.scalar("contrastive_divergence",cd)
        tf.summary.scalar("error",err)       
        
        #visualizing the parameters before update
        tf.summary.histogram("weights",w)
        tf.summary.histogram("bias_hidden",bh)
        tf.summary.histogram("bias_visible",bv)        
        
        #visualizing the updated parameters
        tf.summary.histogram("weights_updated",wu)
        
        tf.summary.histogram("bias_hidden_updated",bhu)
        tf.summary.histogram("bias_visible_updated",bvu)        
        tf.global_variables_initializer().run()
       
#save initial random weights for training     
def save_model(model,dest_dir):     
    
    with tf.Session(graph = tf.Graph()) as sess:         
        weights,bias_h,bias_v,hidden = model 
        
        wghts = tf.placeholder(tf.float32, weights.shape)    
        bh = tf.placeholder(tf.float32, bias_h.shape)    
        bv = tf.placeholder(tf.float32, bias_v.shape)
        h = tf.placeholder(tf.float32, hidden[0].shape)
            
        tf.Variable(sess.run(wghts, feed_dict={wghts: weights}),name = 'weights') 
        tf.Variable(sess.run(bh, feed_dict={bh: bias_h}),name = 'bias_hidden') 
        tf.Variable(sess.run(bv, feed_dict={bv: bias_v}),name = 'bias_visible')
        
        for batch_no in range(n):
            h_ = "hidden"+str(batch_no)            
            hddn = hidden[batch_no]
            tf.Variable(sess.run(h, feed_dict={h: hddn}),name = h_)
        
        sess.run(tf.global_variables_initializer())
        
        saver = tf.train.Saver()    
        saver.save(sess, dest_dir+ 'model.ckpt') 
    

#tries to load data for step, if it exists
def is_exists_saved_model(dest_dir):     
    try:        
        model_path = dest_dir + 'model.ckpt'        
        tf.train.import_meta_graph(model_path + '.meta')               
    except Exception as e:         
        print("Error: ", e)
        return False        
    else:
        return True
    
#given the visible layer, weights(visible to hidden) and bias for the hidden layer, computes the hidden layer weights   
def forward_prop(visible, weights, bias_hidden):
    
    hidden_1 = tf.matmul(tf.transpose(visible,perm =[0,2,1]),weights) + bias_hidden #activations
    hidden_prob = tf.nn.sigmoid(hidden_1)  #probs of the hidden units
    hidden_states = tf.nn.relu(tf.sign(hidden_prob - tf.random_uniform(tf.shape(hidden_prob))))    #sample h given x (gibbs sampling)
    return hidden_prob, hidden_states

#pre step for calcluating the cost - contrastive divergence
def positive(hidden_1,visible):
    pos = tf.matmul(visible,hidden_1)
    return pos

#given the hidden layer, weights(hidden to visible) and bias for the visible layer, computes the visible layer weights       
def back_prop(hidden,weights,bias_visible):
    visible_1 = tf.matmul(weights,tf.transpose(hidden,perm= [0,2,1])) + bias_visible
    visible_1 = tf.nn.sigmoid(visible_1)
    _,nVis,dim = visible_1.get_shape().as_list()
    visible_1 = tf.truncated_normal((1, nVis,dim), mean=visible_1, stddev=standard_dev)
    #visible_1 = tf.nn.relu(tf.sign(visible_1 - tf.random_uniform(tf.shape(visible_1))))    #sample v given h
    return visible_1

def negative(hidden,visible_1):
    neg = tf.matmul(visible_1,hidden)
    return neg

def contrastive_divergence(positive,negative):    
    cost = tf.reduce_mean(positive - negative)
    return cost

def error(v1,v0):
    err= tf.reduce_mean(tf.square(v0-v1))
    return err
#updates the parameters given the cost
def update_vars(weights,bias_hidden, bias_visible, cost):
    sess = tf.Session()
    weights_1 = weights + (LEARNING_RATE * cost)    
    bias_hidden_1 = bias_hidden + (LEARNING_RATE * cost)    
    bias_visible_1 = bias_visible + (LEARNING_RATE * cost)
    sess.run(tf.global_variables_initializer())    
    return weights_1,bias_hidden_1,bias_visible_1  

#initialize weights    
def _init_RBM(weights,bias_hidden,bias_visible,hidden,sess): 
    
    #bias_visible = tf.Variable(bias_visible, name = "bias_visible" + str(batch_no)) #25,15
    weights = tf.Variable(weights,name ="weights")    
    bias_visible = tf.Variable(bias_visible,name ="bias_visible")    
    bias_hidden = tf.Variable(bias_hidden,name ="bias_hidden") 
    
    _hidden = []
    for batch_no in range(n):
        h = tf.Variable(hidden, name ="hidden"+str(batch_no)) #25,15
        _hidden.append(h)
    
    sess.run(tf.global_variables_initializer())  
    
    return sess.run([weights,bias_hidden,bias_visible,_hidden])

def getParameters(nVis,dim):
    global NUM_OF_VISIBLE_UNITS
    with tf.Session(graph = tf.Graph()) as sess:        
        if(isConcat):
            #num_visible = nVis
            #print("Conc ",nVis)
            NUM_OF_VISIBLE_UNITS = nVis
            
        #else:
        num_visible = NUM_OF_VISIBLE_UNITS #experiment
        num_hidden = NUM_OF_HIDDEN_UNITS #experiment
        
        weights = []
        bias_hidden = []
        bias_visible = []
        hidden = []
        
        _weights = weight_variable([num_visible,num_hidden]) #25,15
        _bias_hidden = bias_variable([1,num_hidden])   
        _bias_visible = bias_variable([num_visible,1]) 
        _hidden = weight_variable([num_hidden,dim])
        #bias_visible = []
        sess.run(tf.global_variables_initializer())  
        
        w = sess.run(_weights)
        bh = sess.run(_bias_hidden)
        bv = sess.run(_bias_visible)
        h = sess.run(_hidden)
        
        for i in range(BATCH_SIZE):
            weights.append(w)
            bias_hidden.append(bh)
            bias_visible.append(bv)
            hidden.append(h)	
        
    #sess.run(tf.global_variables_initializer())  
    return np.asarray(weights),np.asarray(bias_hidden), np.asarray(bias_visible), np.asarray(hidden)
    #return np.asarray(weights),np.asarray(bias_hidden), np.asarray(bias_visible)

#groups the training data into batches of length- 'size' and saves them
def batching(data,size,sess=None,dest_dir=None, batchSize = None):
    global n
    if batchSize is None:
        batchSize = BATCH_SIZE
        
    n = int (size / batchSize)    
    batches = []
    start = 0
    end = batchSize
    for i in range(n):        
        batches.append(data[start:end])
        start = end
        end = start + batchSize    
    return batches

#takes the hidden states of the trained sentences and concantenates them
def concat_hidden_states(source1,source2):
    global isConcat,NUM_OF_VISIBLE_UNITS,DIMENSION
    model = []   
    with tf.Session(graph = tf.Graph()) as sess:         
        saver = tf.train.import_meta_graph(source1 +'.meta')
        saver.restore(sess,source1)                           
        graph = tf.get_default_graph()
        h1 = [tf.transpose(graph.get_tensor_by_name("hidden"+str(i) + ":0"),perm=[0,2,1]) for i in range(n)]        
        h1 = sess.run(h1)
            
    with tf.Session(graph = tf.Graph()) as sess:         
        saver = tf.train.import_meta_graph(source2 +'.meta')
        saver.restore(sess,source2)                           
        graph = tf.get_default_graph()
        h2 = [tf.transpose(graph.get_tensor_by_name("hidden"+str(i) + ":0"),perm=[0,2,1]) for i in range(n)]        
        h2 = sess.run(h2)    
    
    with tf.Session(graph = tf.Graph()) as sess:   
        model = np.concatenate((h1,h2),axis = 2)
        print(model.shape)
        #_,size,NUM_OF_VISIBLE_UNITS,DIMENSION = model.shape
        #model = np.array(model)
        isConcat = True        
    return model
    
    
def dimensionality_reduction(source):
    global NUM_OF_VISIBLE_UNITS,DIMENSION
    
    with tf.Session(graph = tf.Graph()) as sess:         
        saver = tf.train.import_meta_graph(source +'.meta')
        saver.restore(sess,source)                           
        graph = tf.get_default_graph()
        h = [graph.get_tensor_by_name("hidden"+str(i) + ":0") for i in range(n)]        
        #h = np.asarray(h)        
        data = sess.run(h)    
    
    with tf.Session(graph = tf.Graph()) as sess:
        
        data = np.asarray(data) 
        #print(data.shape)
        #_,size,NUM_OF_VISIBLE_UNITS,DIMENSION = data.shape       
    return data
        
#Weight Initialization
def weight_variable(shape,name=None):
  initial = tf.truncated_normal(shape, stddev=standard_dev)
  return tf.Variable(initial,name=name)

#bias initialization
def bias_variable(shape,name=None):
  initial = tf.constant(0.0, shape=shape)
  return tf.Variable(initial,name=name)

#labels initialization
def label_variable(shape,name=None):
  initial = tf.constant(1, shape=shape)
  return tf.Variable(initial,name=name)
#probabilities initialization
def probs_variable(shape,name):
  initial = tf.zeros(shape)
  return tf.Variable(initial,name=name)

#look up embeddings for input text 
def get_sents_embeddings(sents,word_embeddings,vocab,sess,dest_dir,sub_set = None):       
    if not sub_set is None:        
        sents = preprocess(sents[:sub_set])    #temporarily work with the first 1500 sents    
    else:
        sents = preprocess(sents)
    #sents = preprocess(sents)    #temporarily work with the first 1500 sents    
    avg_length = avg_Length(sents,sess) #returns the AVERAGE of all the sentence lenghts
    #count_unknown_words(vocab,sents)   
    avg_length = NUM_OF_VISIBLE_UNITS #set it to the num of visible units hard coded for maintenace
    
    vocab_dict = create_dict(vocab,avg_length)      
    ids = np.array(list(vocab_dict.transform(sents))) - 1 #transform inputs
    #print(ids)     
    embed = tf.nn.embedding_lookup(word_embeddings,ids)     
    embed = tf.Variable(embed,name="embeds")    
    
    #sess.run(embed.initializer) 
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()  
    saver.save(sess, dest_dir+'temp.ckpt')   

def get_sent_embeddings(sents,word_embeddings,vocab,sess,dest_dir,caller=None):       
    sents = preprocess(sents)    
    if caller is None:
        avg_length = NUM_OF_VISIBLE_UNITS #set it to the num of visible units hard coded for maintenace        
    vocab_dict = create_dict(vocab,avg_length)      
    ids = np.array(list(vocab_dict.transform(sents))) - 1  #transform inputs     
    embed = tf.nn.embedding_lookup(word_embeddings,ids) 
    return sess.run(embed)
    
#shuffles sents1 and 2 while keeping relationship within the pairs
def shuffle_data(data1, data2, limit=None):    
    data_tuple =[]
    for d in range(len(data1)):
        pair = (data1[d],data2[d])
        data_tuple.append(pair)
    np.random.shuffle(data_tuple)
    
    #randomly picks a group of 'limit(n) tuples
    if not limit is None:
        data_tuple = random.sample(data_tuple,limit)
    
    #reassemble data
    data1 = []
    data2 = []
    for data in data_tuple:
        d1, d2 = data
        data1.append(d1)
        data2.append(d2)
    return data1,data2

#extracts labels(1 0r 0, fuseable or not) and sentence pairs
def extract_labels_data(tripple):
    sent1 = []
    sent2 = []
    labels = []
    
    for t in tripple:
        l,s1,s2 = t
        labels.append(l)
        sent1.append(s1)
        sent2.append(s2)
    
    return labels,sent1,sent2

def form_triples(first,second,third):
    tripples =[]
    for i in range(len(second)):
        t =(first,second[i],third[i])
        tripples.append(t)
    return tripples

#logs duration of program to file
def log_time(start):    
    start_time = time.asctime( time.localtime(start))
    print(start_time)
    end = time.time()
    end_time = time.asctime( time.localtime(end))
    print(end_time)
    duration = time.strftime('%H:%M:%S', time.gmtime(end - start))
    status = '{0: <19}'.format('Program Started')
    status +='{0: <27}'.format(': ' + start_time)
    status +='{0: <20}'.format('\nProgram Ended')
    status +='{0: <27}'.format(': ' + end_time)
    status += '{0: <20}'.format('\nDuration')
    status +='{0: <27}'.format(': ' + duration)
    write_to_file(STATUS_LOG,status)

#retrieves saved parameters for the last layer from file   
def get_logistic_features(source):
    with tf.Session(graph = tf.Graph()) as sess: 
        #features = []
        #load data from source           
        saver = tf.train.import_meta_graph(source+ '.meta')   
        saver.restore(sess, source)  
        graph = tf.get_default_graph()
        h = [graph.get_tensor_by_name("hidden"+str(i) + ":0") for i in range(n)]  
        features = sess.run(h)        
    return features
   
#given an array of sents, removes words with punctuations and non-english words
def preprocess(sents):
    validSents = []
    for s in sents:
        sent = s.strip().split(' ')        
        word = [w.lower() for w in sent if isValid(str(w))]
        validSents.append(' '.join(word))
    return validSents
 
def stats(sent,avg,mx,leng): 
    words  = [s.split() for s in sent]
    length = [len(s) for s in words]  
    s1 = [l for l in length if l>=avg-10 and l<=avg+10]       
    s2 = [l for l in length if l<=avg]       
    s3 = [l for l in length if l>=avg]
    s4 = [l for l in length if l>=mx-5 and l<=mx+5]  
    s5 = [l for l in length if l>=leng-5 and l<=leng+5]  
    s6 = [l for l in length if l==leng]                                                                                                                    
    return len(s1),len(s2),len(s3), len(s4), len(s5), len(s6)

#conts number of unknown words in text, which are not in vocab
def count_unknown_words(vocab,txt):    
    text=[]
    text.append(' '.join(txt))
    for w in text:
        text = w.strip().split(' ')    
    text = list(set(text))
    unk= [word for word in text if not word in vocab]
    print(unk, len(unk))
   
if __name__ == "__main__":
    main()