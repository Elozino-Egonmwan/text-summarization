import tensorflow as tf
import numpy as np
import nltk
from Helper._data_generator import read_text_file
from fuseability_checker import get_sent_states
from tensorflow.contrib import rnn,layers
from logistic_classifier import get_metrics
#from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

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
MODEL_DIR = 'LOG_DIR_300/Extractor_RNN/Model/'
RBM_MODEL = 'LOG_DIR_300/Extractor/RBM/Sent/'

TRAIN_REF = "LOG_DIR_300/Extractor_RNN/reference/extTrain_reference.txt"
TRAIN_SYS = "LOG_DIR_300/Extractor_RNN/tunedExtTrain.txt"

VAL_REF = "LOG_DIR_300/Extractor_RNN/reference/extVal_reference.txt"
VAL_SYS = "LOG_DIR_300/Extractor_RNN/tunedExtVal.txt"

TEST_REF = "LOG_DIR_300/Extractor_RNN/reference/extTest_reference.txt"
TEST_SYS = "LOG_DIR_300/Extractor_RNN/tunedExtTest.txt"

BATCH_SIZE = 50 #20
NUM_UNITS = 200  #200
num_classes = 2

num_examples =100
state="Training"
count = 0
num_batches=1

params = {"batch_size": BATCH_SIZE}

def main():
    tf.reset_default_graph()        #start clean
    tf.logging.set_verbosity(tf.logging.INFO)
    
    #run model in 1 of 3 modes
    #train_model()  
    #eval_model()
    test_model()     

#prepares data to be parsed to model in different moods(states-eg Training..)
def model_wrapper(n_examples,mood,source,labels,sys,ref):    
    global num_examples,state,num_batches
    state=mood
    num_examples = n_examples
    
    if state=="Training":
        num_batches = int(num_examples/BATCH_SIZE)
        epochs = 38  #50
        steps = num_batches * epochs
        num_epochs = None                
    else:
        num_epochs = 1        
    
    sess = tf.InteractiveSession()
    
    #initialize estimator
    run_config = tf.estimator.RunConfig(save_summary_steps=num_batches)
    estimator = tf.estimator.Estimator(
        model_fn=rbmE_Class,
        model_dir=MODEL_DIR,
        config = run_config,
        params = params)      
    
    #get data
    doc,labels, ref_labels = prepData(source,labels,num_examples)               
    #write_to_file(ref,sumries,"w")  #write ref extracted sents to file    
    reset_file(sys)
    
    #get rbm pretrained states    
    pre_embd = sess.run(get_sent_states(doc,RBM_MODEL)) #tensor 500 *15*50  
    
    '''prepare for Training, eval or testing'''        
    if state=="Infering":                
        inp_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(pre_embd)},        
        batch_size = BATCH_SIZE,
        num_epochs=num_epochs,
        shuffle=False)
    else:
        inp_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(pre_embd)},
        y=np.array(labels),        
        batch_size = BATCH_SIZE,
        num_epochs=num_epochs,
        shuffle=False)
    
    # Set up logging for training
    # Log the values in the "predictions" tensor with label "pred"    
    tensors_to_log = {"pred": "predictions"}            
    print_predictions = tf.train.LoggingTensorHook(
            tensors_to_log,every_n_iter=1,
            formatter = logits2preds)    
    
    
    '''run model'''
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)    
    tf.reset_default_graph() #reset graph before importing saved checkpoints
    
    if state=="Training":           
        estimator.train(input_fn=inp_fn,hooks=[print_predictions],steps=steps)    
        #estimator.train(input_fn=inp_fn,steps=steps)    
        
    else:
        eval_results = estimator.evaluate(input_fn=inp_fn,hooks=[print_predictions])
        #eval_results = estimator.evaluate(input_fn=inp_fn)
        print(eval_results)
    
    coord.request_stop()
    coord.join(threads)  
    
    #metrics evaluation
    preds = read_text_file(sys)
    preds =list(map(int,preds))  
    get_metrics(preds,ref_labels,state,num_examples)    

def rbmE_Class(mode,features,labels,params):    
    inp = features["x"]                
    labels =tf.to_float(labels)
    
    #Encoder
    enc_cell = rnn.GRUCell(num_units=NUM_UNITS)
    enc_out, enc_state = tf.nn.dynamic_rnn(enc_cell,inp,time_major=False,dtype=tf.float32)  

    output = tf.transpose(enc_out, [1, 0, 2])
    x = tf.gather(output, int(output.get_shape()[0]) - 1)
    #x = enc_state
    # Softmax layer.
    weight, bias = weight_and_bias(NUM_UNITS, num_classes)
    logits = tf.nn.softmax(tf.matmul(x, weight)+ bias)    
    #get_metrics(preds,labels,desc,nExamples)
    tf.identity(logits, name="predictions")   
    loss = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(logits)))                
    
    #training mode           
    if mode == tf.estimator.ModeKeys.TRAIN:         
        train_op = layers.optimize_loss(
            loss, tf.train.get_global_step(),
            optimizer='Adam',            
            learning_rate = 0.0001,
            clip_gradients=5.0)       
        
        spec = tf.estimator.EstimatorSpec(mode=mode,predictions=logits,
                                          loss = loss,
                                          train_op=train_op)
    #evaluation mode
    elif mode == tf.estimator.ModeKeys.EVAL:
        spec = tf.estimator.EstimatorSpec(mode=mode,loss=loss,predictions=logits)
    else:
        spec = tf.estimator.EstimatorSpec(mode=mode,predictions=logits)
    return spec    

#coverts list of logits to predictions
def logits2preds(logits):         
    if state!="Infering":
        global count    
        logits = logits["pred"]        
        if((count*BATCH_SIZE)%num_examples==0):
            mode="w"            
        else:
            mode="a"
        count+=1   
    pred =[1 if pr[0] >= pr[1] else 0 for pr in logits]
    #pred_ids = [i for i,j in enumerate(pred) if j == 1]    
    pred = list(map(str,pred))    
    
    if state=="Training":
        file= TRAIN_SYS            
    elif state=="Evaluating":
        file= VAL_SYS
    elif state=="Testing":
        file=TEST_SYS        
    write_to_file(file,pred,mode)    
    return "Step "+str(count)    

def prepData(source,source_labels,limit):       
    if state=="Training":
        return get_data(source,source_labels,limit)
    else:
        return get_data(source,source_labels,limit)
    
def get_dataTrain(source,source_labels,limit):   
    n = int(limit/2)
    labels = read_text_file(source_labels,n)        
    labels = (' '.join(labels)).split()    
    
    positive_labels_indices = [i for i, j in enumerate(labels) if j == '1']
    positive_labels_indices = positive_labels_indices[:n]
    print("positives: ", len(positive_labels_indices))
    
    negative_labels_indices = [i for i, j in enumerate(labels) if j == '0']
    negative_labels_indices = negative_labels_indices[:n]
    print("negatives: ", len(negative_labels_indices))    
    
    labels_indices = positive_labels_indices + negative_labels_indices
    labels = [labels[i] for i in labels_indices]
    labels = list(map(int,labels))    
    cat_labels = convert_to_categorical(labels)    
        
    docs = read_text_file(source,n)        
    docs = nltk.sent_tokenize(' '.join(docs))       
    docs =[docs[i] for i in labels_indices]    
    #print("num of data", len(docs))       
    '''
    data = form_pairs(docs,labels)  
    np.random.shuffle(data)     
    
    docs,labels = extract_labels_data(data)
    #ref_labels_indices = [i for i, j in enumerate(labels) if j == [1.0, 0.0]]
    #ref_docs =[docs[i] for i in ref_labels_indices]        
    
    ref_labels =[1 if pr[0] >= pr[1] else 0 for pr in labels]
    '''
    return docs,cat_labels,labels

def get_data(source,source_labels,limit):  
    n = int(limit/2)
    docs = read_text_file(source,n)        
    docs = nltk.sent_tokenize(' '.join(docs))   
    docs = docs[:limit]    
    
    labels = read_text_file(source_labels,n)        
    labels = (' '.join(labels)).split()    
    labels = labels[:limit]
    labels = list(map(int,labels))
    cat_labels = convert_to_categorical(labels)            
    
    return docs,cat_labels,labels

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

def convert_to_categorical(data):
    cat = []
    for d in data:
        if d == 1:
            cat.append([1.0,0.0])
        else:
            cat.append([0.0,1.0])
    return cat
  
def weight_and_bias(in_size, out_size):
    weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
    bias = tf.constant(0.1, shape=[out_size])
    return tf.Variable(weight,name="weights"), tf.Variable(bias,name="bias")

#writes to file
def write_to_file(file,txt,mode):
    #print("writing to file")
    with open(file,mode) as f: 
        for line in txt:
            f.write(line)
            f.write('\n')

def reset_file(file):    
    with open(file, "w"):
        pass

def train_model():    
    n_examples = 40000 #41400
    mood= "Training"
    source = TRAIN_SOURCE_SENTS
    labels = TRAIN_LABELS    
    sys = TRAIN_SYS
    ref= TRAIN_REF
    model_wrapper(n_examples,mood,source,labels,sys,ref)
    
def eval_model():     
    n_examples = 10000
    mood= "Evaluating"
    source = VAL_SOURCE_SENTS
    labels = VAL_LABELS    
    sys = VAL_SYS
    ref= VAL_REF
    model_wrapper(n_examples,mood,source,labels,sys,ref)

def test_model():    
    n_examples = 40000
    mood= "Testing"
    source = TEST_SOURCE_SENTS
    labels = TEST_LABELS    
    sys = TEST_SYS
    ref= TEST_REF
    model_wrapper(n_examples,mood,source,labels,sys,ref)

if __name__ == "__main__":
    main()