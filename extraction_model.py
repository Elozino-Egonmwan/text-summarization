import tensorflow as tf
import numpy as np
import time
from Helper._data_generator import read_text_file,write_to_file
from Helper._word2vec import isValid
from word2vec import avg_Length, create_dict, load_processed_embeddings
from logistic_classifier import _classifier
import nltk
#train set
SOURCE_SENTS = "A_Training/Sum_Corpus/doc.txt"
LABELS = "A_Training/Sum_Corpus/tuned_doc.txt"
MODEL_PATH = 'LOG_DIR_300/Extractor/RBM/Sent/'
DIM_RED = 'LOG_DIR_300/Extractor/RBM/Dim_Red/'
STATUS_LOG = 'LOG_DIR_300/Extractor/RBM/log.txt'
TEMP_LOG = 'LOG_DIR_300/Extractor/RBM/temp.txt'
CLASSIFIER = 'LOG_DIR_300/Extractor/Classifier/'

BATCH_SIZE = 50
NUM_EPOCHS = 3
LEARNING_RATE = 0.0001
NUM_OF_VISIBLE_UNITS = 25
NUM_OF_HIDDEN_UNITS = 15
DIMENSION = 50 #will be derived and automatically updated
#NUM_EXAMPLES = 500 #temporarily working with 500 examples

standard_dev = 0.01
n = 1 #number of batches initialized to 1 (this should be returned from function batching)
display_batch =  100
description = "Extractor"

def main():         
    start = time.time()
    data,labels = prepare_data(SOURCE_SENTS,LABELS,20000)    
    
    train_sent(data)      
    log_time(start)                  
    
    '''
    features = np.asarray(get_logistic_features('LOG_DIR_300/Extractor/RBM/Dim_Red/model.ckpt'))    
    print(features.shape)    
    _classifier(description,features,labels,"Train",model_dir=CLASSIFIER)
    '''
    tf.reset_default_graph()
    

def prepare_data(source,source_labels,limit):    
    labels = read_text_file(source_labels,limit)        
    labels = (' '.join(labels)).split()    
    
    positive_labels_indices = [i for i, j in enumerate(labels) if j == '1']
    positive_labels_indices = positive_labels_indices[:20000]
    print("positives: ", len(positive_labels_indices))
    
    negative_labels_indices = [i for i, j in enumerate(labels) if j == '0']
    negative_labels_indices = negative_labels_indices[:20000]
    print("negatives: ", len(negative_labels_indices))
    
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

#trains sentences
def train_sent(sents):   
    global NUM_OF_HIDDEN_UNITS, description,NUM_OF_VISIBLE_UNITS 
    description = "Raw"
    wrapper_rbm("Sent",MODEL_PATH,NUM_EPOCHS,sents)
        
    #dimensionality reduction    
    description = "Dim"
    NUM_OF_HIDDEN_UNITS = 1
    NUM_OF_VISIBLE_UNITS = DIMENSION
    wrapper_rbm("Dim Reduction",DIM_RED,2) 

   
#prepares the data for training RBM
#@raw_sents: path to the directory containing raw sentences(not vectors) to be trained
#if None, it means they have been trained already, proceed to training the concatenation
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
    
    train_writer.close()
    
def step_restore_model(dest_dir):
    sess = tf.Session()
    model = []
    model_path = dest_dir + 'model.ckpt'
    saver = tf.train.import_meta_graph(model_path+ '.meta')   
    saver.restore(sess, model_path)        
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
    
#initialize RBM parameters, save and continue training
def train_from_scratch(dest_dir,train_writer,step):     
    sess = tf.Session()     
    w,bh,bv,h = getParameters(NUM_OF_VISIBLE_UNITS,DIMENSION)                 
    parameters_init = _init_RBM(w,bh,bv,h,sess)    
    
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
    
    else:        
        src_file = 'LOG_DIR_300/Extractor/RBM/Sent/model.ckpt'
        batches = dimensionality_reduction(src_file)        
    print('done batching')
    
    format_file(STATUS_LOG)    
    for batch_no in range(n):
        with tf.Session(graph = tf.Graph()) as sess:                     
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
        
        cd = sess.run(contrastive_divergence(pos,neg))
        err = sess.run(error(visible_1,visible))   
        
        weights_1,bias_hidden_1,bias_visible_1 = update_vars(weights,bias_hidden,bias_visible,cd)                     
            
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
    
    return np.asarray(weights),np.asarray(bias_hidden), np.asarray(bias_visible), np.asarray(hidden)    

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
def get_sents_embeddings(sents,word_embeddings,vocab,sess,dest_dir,sub_set = None,caller=None):       
    if not sub_set is None:        
        sents = preprocess(sents[:sub_set])    #temporarily work with the first 1500 sents    
    else:
        sents = preprocess(sents)
    #sents = preprocess(sents)    #temporarily work with the first 1500 sents    
    avg_length = avg_Length(sents,sess) #returns the AVERAGE of all the sentence lenghts
    #count_unknown_words(vocab,sents)     
    if caller is None:
        avg_length = NUM_OF_VISIBLE_UNITS #set it to the num of visible units hard coded for maintenace        
    vocab_dict = create_dict(vocab,avg_length)      
    ids = np.array(list(vocab_dict.transform(sents))) - 1 #transform inputs     
    embed = tf.nn.embedding_lookup(word_embeddings,ids)  
    embed = tf.Variable(embed,name="embeds")    
    
    sess.run(embed.initializer) 
    saver = tf.train.Saver([embed])  
    saver.save(sess, dest_dir+'temp.ckpt')   

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

   
#given an array of sents, removes words with punctuations and non-english words
def preprocess(sents):
    validSents = []
    for s in sents:
        sent = s.strip().split(' ')        
        word = [w.lower() for w in sent if isValid(str(w))]
        validSents.append(' '.join(word))
    return validSents 
   
def get_logistic_features(source):
    with tf.Session(graph = tf.Graph()) as sess:         
        saver = tf.train.import_meta_graph(source+ '.meta')   
        saver.restore(sess, source)  
        graph = tf.get_default_graph()
        h = [graph.get_tensor_by_name("hidden"+str(i) + ":0") for i in range(n)]  
        features = sess.run(h)        
    return features
    
if __name__ == "__main__":
    main()