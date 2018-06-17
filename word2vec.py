# -*- coding: utf-8 -*-
'''
    Date Started: Nov 22nd, 2017
    Description: Refer to 
        https://ireneli.eu/2017/01/17/tensorflow-07-word-embeddings-2-loading-pre-trained-vectors/
'''

import numpy as np
import tensorflow as tf
import os
from tensorflow.contrib import learn
from tensorflow.contrib.tensorboard.plugins import projector
from Model._word2vec import isValid
from Model._data_generator import read_text_file
#from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


#IN_FILE = "D_Auxilliary_Code/glove.6B.50d.txt"
IN_FILE = "D_Auxilliary_Code/glove.6B.300d.txt"
#IN_FILE = "D_Auxilliary_Code/glove.840B.300d.txt"
LOG_DIR = "LOG_DIR_300/embeddings/" 

sess = tf.Session()

def main():      
    test = ["the city university lethbridge love elozino loves stte sttt","trying this out"]
    vocab, word_embeddings = load_processed_embeddings(sess)        
    max_length = avg_Length(test,sess)    
    vocab_dict = create_dict(vocab,max_length)      
    ids = np.array(list(vocab_dict.transform(test))) -1    #transform inputs  
    print(ids)
    for i in ids:
        for j in i:
            print(vocab[j])
    a = tf.nn.embedding_lookup(word_embeddings,ids)     
    #print(sess.run(a[0]))        
    visualize_embed(vocab,word_embeddings)
    #print("0\n:", tf.global_variables())
    tf.reset_default_graph()
    sess.close()
    
def load_processed_embeddings(sess):
    try:
        saver = tf.train.import_meta_graph('LOG_DIR_300/embeddings/model.ckpt.meta')                
        saver.restore(sess, 'LOG_DIR_300/embeddings/model.ckpt')
        #graph = tf.get_default_graph()
        word_embeddings = sess.run('embed:0')
        #word_embeddings = graph.get_tensor_by_name('embed:0')
        #word_embeddings = sess.run('embed:0')        
        #print_tensors_in_checkpoint_file(file_name='LOG_DIR/model.ckpt', tensor_name='', all_tensors=False)        
    except Exception as e:        
        print("Error: ", e)
        vocab, word_embeddings = run_glove(sess,"self")   
    else:        
        vocab = read_text_file('LOG_DIR_300/embeddings/metadata.tsv')        
        print("Embeddings loaded")
    return vocab, word_embeddings
    
#loads the words and their pretrained word vectors into an array
def load_glove(file):
    vocab = []
    embd = []
    with open(file, "r", encoding='utf8') as f:
        #for line in f.readlines():
        for line in f:
            row = line.strip().split(' ')
            if isValid(str(row[0])):
                vocab.append(row[0])
                embd.append(row[1:])    
    print('Loaded Glove')
    return vocab, embd

#feeds the word vectors into a tensor
def run_glove(sessn, caller = None):
    #print("-0\n:", tf.global_variables())
    vocab, embd = load_glove(IN_FILE)    
    vocab_size = len(vocab)    
    embedding_dim = len(embd[0])
    embedding = np.asarray(embd)
    
    W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
                trainable=False)
    embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
    embedding_init = W.assign(embedding_placeholder)
    #embedding_init = tf.assign(W,embedding_placeholder)
    feed_dict={embedding_placeholder: embedding}
    #sess.run(feed_dict)
    word_embedding = tf.Variable(sessn.run(embedding_init, feed_dict),name = "embed")
    #word_embedding = sess.run(embedding_init, feed_dict)
    
    sessn.run(tf.global_variables_initializer())
    
    if caller is not None:          
        saver = tf.train.Saver([word_embedding])    
        saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))         
        
    return vocab,word_embedding           
    #return vocab,embedding_placeholder

#creates a dictionary of word ids to their vector representations
def create_dict(vocab,max_length):    
    #init vocab processor
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_length)
    #fit the vocab from glove      
    vocab_dict = vocab_processor.fit(vocab)      
    return vocab_dict

def avg_Length(sent,sessn): 
    words  = [s.split() for s in sent]
    length = tf.constant([len(s) for s in words]) 
    #print(sess.run(length))
    return sessn.run(tf.reduce_mean(length))

def max_Length(sent): 
    words  = [s.split() for s in sent]
    length = tf.constant([len(s) for s in words])     
    return sess.run(tf.reduce_max(length))


def visualize_embed(vocab,word_embeddings):
    
    # Use the same LOG_DIR where you stored your checkpoint.    
    summary_writer = tf.summary.FileWriter(LOG_DIR)
    
    # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
    config = projector.ProjectorConfig()
    
    # You can add multiple embeddings. Here we add only one.
    embedding = config.embeddings.add()    
    embedding.tensor_name = "W"
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = os.path.join(LOG_DIR, "metadata.tsv")

    with open(embedding.metadata_path, 'w') as f:
        for word in vocab:            
            f.write(word + '\n')
    
    # Saves a configuration file that TensorBoard will read during startup.
    projector.visualize_embeddings(summary_writer, config)

if __name__ == "__main__":
    main()
