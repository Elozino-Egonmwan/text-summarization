 # -*- coding: utf-8 -*-
'''
    Date started: April 11, 2018
    Helpful links:
        Tensorflow Dev Summit-RNN API -  https://www.youtube.com/watch?v=RIR_-Xlbp7s
        LSTM inputs formatting - https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/
        Sample RNN - https://gist.github.com/ilblackdragon/c92066d9d38b236a21d5a7b729a10f12

    Seq2Seq model for sentence fusion
    4 Models:
        1) RBM Encoder - GRU RNN Decoder
        2) GRU RNN Encoder - GRU RNN Decoder
        3) Bi-directional GRU RNN Encoder - Bi- directionalGRU RNN Decoder
        4) Add attention to the best of 1-3    
'''

import tensorflow as tf
import numpy as np
import random
from metrics import bleuPerSent
from Helper._data_generator import read_text_file
from fuseability_checker import get_conc_hidden_states
from fuseability_model import preprocess
from word2vec import load_processed_embeddings,create_dict
from tensorflow.contrib import seq2seq,rnn,layers
from tensorflow.python.layers import core as layers_core
from itertools import groupby
#import logging

#train set
TRAIN_SOURCE_SENTS1 = "A_Training/Fusion_Corpus/Positives/Twos/first.txt"
TRAIN_SOURCE_SENTS2 =  "A_Training/Fusion_Corpus/Positives/Twos/second.txt"
TRAIN_FUSED_SOURCE =  "A_Training/Fusion_Corpus/Positives/Twos/fused.txt"

#validation set
SOURCE_SENTS1 = "B_Validating/Fusion_Corpus/Positives/Twos/first.txt"
SOURCE_SENTS2 =  "B_Validating/Fusion_Corpus/Positives/Twos/second.txt"
FUSED_SOURCE =  "B_Validating/Fusion_Corpus/Positives/Twos/fused.txt"

#test set
TEST_SOURCE_SENTS1 = "C_Testing/Fusion_Corpus/Positives/Twos/first.txt"
TEST_SOURCE_SENTS2 =  "C_Testing/Fusion_Corpus/Positives/Twos/second.txt"
TEST_FUSED_SOURCE =  "C_Testing/Fusion_Corpus/Positives/Twos/fused.txt"

MODEL_DIR = "LOG_DIR_300/Fusion/Model/v3/"
TRAIN_REF = "LOG_DIR_300/Fusion/reference/fusionTrain_reference.txt"
TRAIN_SYS = "LOG_DIR_300/Fusion/system/fusionTrain_system.txt"

VAL_REF = "LOG_DIR_300/Fusion/reference/fusionVal_reference.txt"
VAL_SYS = "LOG_DIR_300/Fusion/system/fusionVal_system.txt"

TEST_REF = "LOG_DIR_300/Fusion/reference/fusionTest_reference.txt"
TEST_SYS = "LOG_DIR_300/Fusion/system/fusionTest_system.txt"

INFER_REF= "LOG_DIR_300/Fusion/reference/fusionInferences_reference.txt" 
INFER_SYS= "LOG_DIR_300/Fusion/system/fusionInferences_system.txt"

#trained RBM
MODEL_PATH1 = 'LOG_DIR_300/RBM_model/Sent1/'
#MODEL_PATH2 = 'LOG_DIR_300/RBM_model/Sent2/'
MODEL = 'LOG_DIR_300/RBM_model/Evaluating/'

BATCH_SIZE = 50 #50
DECODER = "LOG_DIR_300/Fusion/Ground_truth/"
NUM_UNITS = 200  #200

#initialize
GO = "sttt "     #"<s> "
START = 0
STOP= " stte"   #" </s>"
END = 1
vocab = read_text_file('LOG_DIR_300/embeddings/metadata.tsv')        
vocab_size = len(vocab)
UNK = -1 
embd_dim = 300
seq_len = 10
num_examples =100
state="Training"
count = 0
probs=0.0
num_batches=1

params = {"batch_size": BATCH_SIZE}

def main():
    tf.reset_default_graph()        #start clean
    #tf.logging._logger.setLevel(logging.INFO)               
    tf.logging.set_verbosity(tf.logging.INFO)
    
    #run model in 1 of 4 modes
    train_model()  
    #eval_model()
    #test_model() 
    #infer_model()      

#prepares data to be parsed to model in different moods(states-eg Training..)
def model_wrapper(n_examples,mood,source1,source2,sourceFused,sys,ref):
    #parameters
    global num_examples,state,seq_len,inc_prob,num_batches
    state=mood
    num_examples = n_examples
    
    if state=="Training":
        num_batches = int(num_examples/BATCH_SIZE)
        epochs = 1200  #1000
        steps = num_batches * epochs
        num_epochs = None        
        inc_prob = 1.0/steps        
    else:
        num_epochs = 1        
    
    sess = tf.InteractiveSession()
    
    #initialize estimator
    #run_config = tf.estimator.RunConfig(save_summary_steps=num_batches)
    run_config = tf.estimator.RunConfig(save_summary_steps=num_batches,save_checkpoints_steps=num_batches*3)
    estimator = tf.estimator.Estimator(
        model_fn=rbmE_gruD,
        model_dir=MODEL_DIR,
        config = run_config,
        params = params)       
    
    #get data
    s1,s2,fused = get_data(source1,source2,sourceFused)           
    
    #get rbm conc states    
    encoder_embd,_ = get_conc_hidden_states(s1,s2) #tensor 500 *15*50  
    sos = tf.constant(0.5,shape=[num_examples,1,embd_dim])    
    eos = tf.constant(1.0,shape=[num_examples,1,embd_dim])    
    encoder_embd = tf.concat([sos,encoder_embd,eos],axis=1)       
    encoder_embd = sess.run(encoder_embd) 
    #print(encoder_embd[:10])           
    
    write_to_file(ref,fused,"w")  #write ref fusion to file    
    reset_file(sys)   #reset system fusion for new predictions 

    '''prepare for Training, eval or testing'''
    if state != "Infering":
           
        #get ground truth vectors 500*seq_len*50  
        sos_fused = preProc(fused)                        
        sos_id,_ = lookUp_batch_embeddings(DECODER,sos_fused,extra_pad=True) 
                               
        sos_id_eos, ids = postProcDecoding(sos_id)                 
        dec_inp = ids2words(sos_id_eos)        
        _,decoder_embd = lookUp_batch_embeddings(DECODER,dec_inp)         
        #ids,decoder_embd = lookUp_batch_embeddings(DECODER,fused)        
        
        #mask padded or unk words
        weights = sess.run(tf.to_float(tf.not_equal(ids, -1)))         
        ids[ids==-1] = vocab_size-1                
    
    if state=="Infering":        
        seq_len=15
        inp_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(encoder_embd)},        
        batch_size = BATCH_SIZE,
        num_epochs=num_epochs,
        shuffle=False)
    else:
        inp_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(encoder_embd),"ids":np.array(ids),"weights":np.array(weights)},
        y=np.array(decoder_embd),        
        batch_size = BATCH_SIZE,
        num_epochs=num_epochs,
        shuffle=False)
    
    # Set up logging for predictions
    # Log the values in the "predictions" tensor with label "pred"
    tensors_to_log = {"pred": "predictions"}        
    lr={"learning_rate":"learning_rate"}
    print_predictions = tf.train.LoggingTensorHook(
            tensors_to_log,every_n_iter=1,
            formatter = id2words)
    print_lr = tf.train.LoggingTensorHook(lr,every_n_iter=1000)
    
    '''run model'''
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)    
    tf.reset_default_graph()    #reset graph before importing saved checkpoints
    
    if state=="Training":    
        estimator.train(input_fn=inp_fn,hooks=[print_predictions,print_lr],steps=steps)        
    
    elif state=="Infering":        
        infer_res = list(estimator.predict(input_fn=inp_fn))                
        #id2words(infer_res)        
        
        infer_ids = [i["ids"] for i in infer_res]
        #infer_ids=np.array(infer_ids)
        pos=[i["pos"] for i in infer_res]
        #print(infer_ids[:3])
        c=0
        sl=[]
        for inh in infer_ids:
            sl.append(inh[:,pos[c]])
            c+=1        
        id2words(sl)        
    else:
        eval_results = estimator.evaluate(input_fn=inp_fn,hooks=[print_predictions])
        print(eval_results)
    
    coord.request_stop()
    coord.join(threads)    
    
    #BLEU evaluation    
    hyp = read_text_file(sys)
    bleu = bleuPerSent(fused,hyp)
    print("Bleu score: ", bleu)

#model1 - rbm encoder & gru decoder
def rbmE_gruD(mode,features,labels,params):      
    inp = features["x"]    
    
    if state != "Infering":
        ids = features["ids"]    
        weights = features["weights"]          
        
    batch_size = params["batch_size"]
    
    #Encoder
    enc_cell = rnn.NASCell(num_units=NUM_UNITS)
    enc_out, enc_state = tf.nn.dynamic_rnn(enc_cell,inp,time_major=False,dtype=tf.float32)      
    
    #Decoder    
    cell = rnn.NASCell(num_units=NUM_UNITS)    
    
    _, embeddings = load_processed_embeddings(sess=tf.InteractiveSession())             
    out_lengths = tf.constant(seq_len,shape=[batch_size])                
    if state != "Infering":
        #sampling method for training 
        train_helper=seq2seq.TrainingHelper(labels,out_lengths,time_major=False)
        '''
        train_helper=seq2seq.ScheduledEmbeddingTrainingHelper(inputs=labels,
                                                              sequence_length=out_lengths,
                                                              embedding=embeddings,
                                                              sampling_probability=probs)
        '''
    #sampling method for evaluation     
    start_tokens = tf.zeros([batch_size], dtype=tf.int32)
    infer_helper =seq2seq.GreedyEmbeddingHelper(embedding=embeddings,start_tokens=start_tokens,end_token=END)              
    #infer_helper = seq2seq.SampleEmbeddingHelper(embeddings,start_tokens=start_tokens,end_token=END)           
    #infer_helper=seq2seq.ScheduledEmbeddingTrainingHelper(inputs=inp,sequence_length=out_lengths,embedding=embeddings,sampling_probability=1.0)    
    projection_layer = layers_core.Dense(vocab_size, use_bias=False)
    
    def decode(helper):
        decoder = seq2seq.BasicDecoder(                
                cell=cell,helper=helper,initial_state=enc_state,
                output_layer=projection_layer) 
        #decoder.tracks_own_finished=True
        (dec_outputs,_,_) = seq2seq.dynamic_decode(decoder,maximum_iterations=seq_len)
        #(dec_outputs,_,_) = seq2seq.dynamic_decode(decoder)
        dec_ids = dec_outputs.sample_id          
        logits = dec_outputs.rnn_output                   
        return dec_ids,logits        
    
    #equalize logits, labels and weight lengths incase of early finish in decoder
    def norm_logits_loss(logts,ids,weights):        
        current_ts = tf.to_int32(tf.minimum(tf.shape(ids)[1], tf.shape(logts)[1]))
        logts = tf.slice(logts, begin=[0,0,0], size=[-1, current_ts, -1])       
        ids = tf.slice(ids, begin=[0, 0], size=[-1, current_ts])
        weights = tf.slice(weights, begin=[0, 0], size=[-1, current_ts])
        return logts,ids,weights
    
    #training mode
    if state == "Training":
        dec_ids,logits = decode(train_helper)                
        # some sample_id are overwritten with '-1's
        #dec_ids = tf.argmax(logits, axis=2)
        tf.identity(dec_ids, name="predictions")   
        logits,ids,weights = norm_logits_loss(logits,ids,weights)
        loss = tf.contrib.seq2seq.sequence_loss(logits,ids,weights=weights)
        learning_rate=0.001 #0.0001         

        tf.identity(learning_rate, name="learning_rate")       
    
    #evaluation mode    
    if state == "Evaluating" or state == "Testing":               
        eval_dec_ids,eval_logits = decode(infer_helper)   
        #eval_dec_ids = tf.argmax(eval_logits, axis=2)
        tf.identity(eval_dec_ids, name="predictions")                  
        
        #equalize logits, labels and weight lengths incase of early finish in decoder
        eval_logits,ids,weights = norm_logits_loss(eval_logits,ids,weights)
        '''
        current_ts = tf.to_int32(tf.minimum(tf.shape(ids)[1], tf.shape(eval_logits)[1]))
        ids = tf.slice(ids, begin=[0, 0], size=[-1, current_ts])
        weights = tf.slice(weights, begin=[0, 0], size=[-1, current_ts])
        #mask_ = tf.sequence_mask(lengths=target_sequence_length, maxlen=current_ts, dtype=eval_logits.dtype)
        eval_logits = tf.slice(eval_logits, begin=[0,0,0], size=[-1, current_ts, -1])       
        '''
        eval_loss = tf.contrib.seq2seq.sequence_loss(eval_logits,ids,weights=weights)       
    
    #beamSearch decoder    
    init_state = tf.contrib.seq2seq.tile_batch(enc_state, multiplier=5)            
    beamSearch_decoder = seq2seq.BeamSearchDecoder(cell,embeddings,start_tokens,end_token=END,
                                                   initial_state=init_state,beam_width=5,
                                                   output_layer=projection_layer)
    (infer_outputs,_,_)= seq2seq.dynamic_decode(beamSearch_decoder,maximum_iterations=seq_len)
    infer_ids =infer_outputs.predicted_ids
    infer_probs=infer_outputs.beam_search_decoder_output.scores
    infer_probs = tf.reduce_prod(infer_probs,axis=1)    
    infer_pos= tf.argmax(infer_probs,axis=1)        
    infers = {"ids":infer_ids,"pos":infer_pos}
    
           
    if mode == tf.estimator.ModeKeys.TRAIN:         
        train_op = layers.optimize_loss(
            loss, tf.train.get_global_step(),
            optimizer='Adam',                    
            learning_rate = learning_rate,
            clip_gradients=5.0)
        
        spec = tf.estimator.EstimatorSpec(mode=mode,predictions=dec_ids,
                                          loss = loss,
                                          train_op=train_op)
    #evaluation mode
    elif mode == tf.estimator.ModeKeys.EVAL:
        spec = tf.estimator.EstimatorSpec(mode=mode,loss=eval_loss,predictions=eval_dec_ids)
    else:
        spec = tf.estimator.EstimatorSpec(mode=mode,predictions=infers)
    return spec

#coverts list of ids to words
def id2words(ids):     
    global probs
    if state!="Infering":
        global count    
        ids = ids["pred"]  
        if((count*BATCH_SIZE)%num_examples==0):
            mode="w"            
        else:
            mode="a"
        count+=1
        if state == "Training":
            probs+= inc_prob    
    decoded = []
    for sents in ids:
        sent = ""        
        for word_ids in sents:
            if word_ids != END and word_ids != UNK:
                sent += vocab[word_ids]+" "    
        sent = rmvDups(sent)        
        decoded.append(sent)
        
    if state=="Training":
        file= TRAIN_SYS            
    elif state=="Evaluating":
        file= VAL_SYS
    elif state=="Testing":
        file=TEST_SYS
    else:
        file=INFER_SYS
        mode ="w"
        
    write_to_file(file,decoded,mode)    
    return "Step "+str(count)    

def rmvDups(inp):
    inp = [word[0] for word in groupby(inp.split())]
    return ' '.join(inp)

def ids2words(ids):
    decoded=[]    
    for sents in ids:
        sent = ""
        #sent = []
        for word_ids in sents:
            if word_ids != UNK:
                sent += vocab[word_ids]+" "
            #sent.append(vocab[word_ids])
        decoded.append(sent)
    return decoded

def lookUp_batch_embeddings(dest_dir,sents,extra_pad=False): 
    global vocab_size,embd_dim,seq_len
    
    with tf.Session(graph = tf.Graph()) as sess:        
            vocab, word_embeddings = load_processed_embeddings(sess) 
            vocab_size = len(vocab)            
            ids = get_sents_embedding(sents,word_embeddings,vocab,sess,dest_dir,extra_pad)
            
    with tf.Session(graph = tf.Graph()) as sess:         
            saver = tf.train.import_meta_graph(dest_dir+'temp.ckpt.meta')   
            saver.restore(sess,dest_dir+'temp.ckpt')
            
            sent_embed = sess.run("embeds:0")                    
            n_examples,seq_len,embd_dim = sent_embed.shape                        
    return ids,sent_embed

#read sent1,sent2 and its fused sent from file
def get_data(source1,source2,src_fused):
    #shuffle the positive pairs
    sents1 = read_text_file(source1,num_examples)
    sents2 = read_text_file(source2,num_examples) 
    fused = read_text_file(src_fused,num_examples)   
    #sents1,sents2,fused = shuffle_data(sents1,sents2,fused,num_examples)  
      
    return sents1,sents2,fused

def get_sents_embedding(sents,word_embeddings,vocab,sess,dest_dir,extra_pad,sub_set = None):           
    sents = preprocess(sents)    #temporarily work with the first 1500 sents    
    avg_length = avg_Length(sents,sess) #returns the AVERAGE of all the sentence lenghts 
    
    if extra_pad:    
        avg_length = avg_length+2    
    vocab_dict = create_dict(vocab,avg_length)      
    ids = np.array(list(vocab_dict.transform(sents))) - 1 #transform inputs     
    embed = tf.nn.embedding_lookup(word_embeddings,ids)  
    
    embed = tf.Variable(embed,name="embeds")    
    
    sess.run(embed.initializer) 
    saver = tf.train.Saver([embed])  
    saver.save(sess, dest_dir+'temp.ckpt') 
    return ids

def avg_Length(sent,sessn): 
    words  = [s.split() for s in sent]
    length = tf.constant([len(s) for s in words])
    avg = sessn.run(tf.reduce_mean(length))
    #lenghts = sessn.run([tf.minimum(l,avg) for l in length])
    #print("l",sessn.run(lenghts[:5]))
    return avg


#shuffle sent1,sent2 and its fused sent
def shuffle_data(data1,data2,data3,limit=None):    
    data_tuple =[]
    for d in range(len(data1)):
        pair = (data1[d],data2[d],data3[d])
        data_tuple.append(pair)
    np.random.shuffle(data_tuple)
    
    #randomly picks a group of 'limit(n) tuples
    if not limit is None:
        data_tuple = random.sample(data_tuple,limit)
    
    #reassemble data
    data1 = []
    data2 = []
    data3 = []
    for data in data_tuple:
        d1, d2, d3 = data
        data1.append(d1)
        data2.append(d2)
        data3.append(d3)
    return data1,data2,data3

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
    n_examples = 20000 #40000
    mood= "Training"
    source1 = TRAIN_SOURCE_SENTS1
    source2 = TRAIN_SOURCE_SENTS2
    sourceFused = TRAIN_FUSED_SOURCE
    sys = TRAIN_SYS
    ref= TRAIN_REF
    
    model_wrapper(n_examples,mood,source1,source2,sourceFused,sys,ref)
    
def eval_model():    
    n_examples = 2400
    mood= "Evaluating"
    source1 = SOURCE_SENTS1
    source2 = SOURCE_SENTS2
    sourceFused = FUSED_SOURCE
    sys = VAL_SYS
    ref= VAL_REF
    
    model_wrapper(n_examples,mood,source1,source2,sourceFused,sys,ref)

def test_model():    
    n_examples = 2000
    mood= "Testing"
    source1 = TEST_SOURCE_SENTS1
    source2 = TEST_SOURCE_SENTS2
    sourceFused = TEST_FUSED_SOURCE
    sys = TEST_SYS
    ref= TEST_REF
    
    model_wrapper(n_examples,mood,source1,source2,sourceFused,sys,ref)

def infer_model():    
    n_examples = 2000
    mood= "Infering"
    source1 = TEST_SOURCE_SENTS1
    source2 = TEST_SOURCE_SENTS2
    sourceFused = TEST_FUSED_SOURCE
    sys = INFER_SYS
    ref= INFER_REF
    
    model_wrapper(n_examples,mood,source1,source2,sourceFused,sys,ref)


def preProc(sent): 
    dec_inp = [GO+s for s in sent]    
    return dec_inp

def postProcDecoding(ids): 
    ids = ids.tolist()
    #print(i,type(i))
    #dec_out = [np.append(s[1:],[END]) for s in ids]
    dec_inp = [s+[END] for s in ids]
    dec_out = [s[1:]+[END]+[-1] for s in ids]
    return np.array(dec_inp),np.array(dec_out)
    
def postProc(sent):      
    words  = [s.split() for s in sent]        
    dec_out = [' '.join(s[1:]+[STOP]) for s in words]    
    return dec_out

if __name__ == "__main__":
    main()# -*- coding: utf-8 -*-    