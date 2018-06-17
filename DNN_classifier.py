# -*- coding: utf-8 -*-
'''
Source: https://www.tensorflow.org/get_started/estimator
'''
#TensorFlow’s high-level machine learning API (tf.estimator) makes it easy to 
#configure, train, and evaluate a variety of machine learning models. In this tutorial,
# you’ll use tf.estimator to construct a neural network classifier and train it on the
#Iris data set to predict flower species based on sepal/petal geometry

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import urllib.request
#import urllib

import numpy as np
import tensorflow as tf
from Model._data_generator import write_to_file

STATUS_LOG = 'LOG_DIR_300/RBM_model/log.txt'

def main():
    print("DNN_classifier")
       
  
def dnn_classifier(desc,features,labels):  
    with tf.Session(graph = tf.Graph()) as sess:        
        num,batchSize,units,dim = features.shape  
        n = num * batchSize
        features = sess.run(tf.reshape(features,[n,units]))  
        run_classifier(desc,features,labels,n)
      
  
def run_classifier(desc,features,labels,n):
    feature_columns = [tf.feature_column.numeric_column("x", shape=[15])]

    '''
    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[10, 20, 10],
                                          n_classes=2,
                                          model_dir="/tmp/fus_classifier")
    '''
    
    classifier = tf.estimator.LinearClassifier(feature_columns=feature_columns,                                          
                                          n_classes=2,
                                          model_dir="/last_tmp_Lin2/fus_classifier")
    # Define the training inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(features)},
        y=np.array(labels),
        num_epochs=1,
        shuffle=True)

    # Train model.
    #classifier.train(input_fn=train_input_fn, steps=6000)

    # Evaluate accuracy.    
    accuracy_score = classifier.evaluate(input_fn=train_input_fn)["accuracy"]      
    status = desc
    status += "\nAccuracy: {0:f}\n".format(accuracy_score)
    #write_to_file(STATUS_LOG,status)  

    print("\nAccuracy: {0:f}\n".format(accuracy_score))
    
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(features)},
      num_epochs=1,
      shuffle=False)

    predictions = list(classifier.predict(input_fn=predict_input_fn))
    predicted_classes = [p["classes"] for p in predictions]    
    pr = [1 if p==b'1' else 0 for p in predicted_classes]
    y=np.array(labels)
    get_metrics(pr,y,desc,n)

    '''
    _,prec= tf.metrics.precision(y,pr)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer()) 
    sess.run(tf.local_variables_initializer())             
    print("Precision",sess.run(prec))
    '''
    
    
def get_metrics(preds,labels,desc,nExamples):
    with tf.Session(graph = tf.Graph()) as sess: 
        correct_prediction = tf.equal(preds, labels)            
        accuracy = sess.run(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))            
        
        _,prec= tf.metrics.precision(labels,preds)
        _,rec= tf.metrics.recall(labels,preds)            
                    
        _,false_pos= tf.metrics.false_positives(labels,preds)
        _,true_pos = tf.metrics.true_positives(labels,preds)
        _,false_neg = tf.metrics.false_negatives(labels,preds)          
              
        sess.run(tf.global_variables_initializer()) 
        sess.run(tf.local_variables_initializer())             
        
        false_pos,true_pos,false_neg = sess.run([false_pos,true_pos,false_neg])
        
        false_pos =false_pos/nExamples 
        true_pos = true_pos/nExamples
        false_neg = false_neg/nExamples
        
        prec,rec = sess.run([prec,rec])
        true_neg = 1 - (false_pos + true_pos + false_neg)
        denom = prec + rec
        if denom > 0:
            f_measure =  2.0 * ((prec*rec) / (prec+rec))
        else:
            f_measure = 0.0
        print_metrics(accuracy,prec,rec,f_measure,false_pos,true_pos,false_neg,true_neg)
        #save_metrics(desc,accuracy,prec,rec,f_measure,false_pos,true_pos,false_neg,true_neg)
    
        
def print_metrics(accuracy,prec,rec,f_measure,false_pos,true_pos,false_neg,true_neg):
                    
        print('\nFalse Positives', false_pos)
        print('True Positives', true_pos)
        print('False Negatives', false_neg)
        print('True Negatives', true_neg)
        
        print ('\nAccuracy: ', accuracy)           
        print('Precision', prec)
        print('Recall', rec)
        print('F_measure', f_measure)
    
def save_metrics(desc,accuracy,prec,rec,f_measure,false_pos,true_pos,false_neg,true_neg):            
        status = '\n' + desc + ' Metrics' 
        write_to_file(STATUS_LOG,status)            
        status = '{0: <20}'.format('Accuracy') 
        #acc = str(accuracy) + ' (' + duration + ')'            
        status += '{0: <27}'.format(': ' + str(accuracy))
        
        status += '\n{0: <20}'.format('Precision')             
        status += '{0: <27}'.format(': ' + str(prec)) 
        
        status += '\n{0: <20}'.format('Recall')             
        status += '{0: <27}'.format(': ' + str(rec))
        
        status += '\n{0: <20}'.format('F_measure')             
        status += '{0: <27}'.format(': ' + str(f_measure))
        
        status += '\n\n{0: <20}'.format('True Positives')             
        status += '{0: <27}'.format(': ' + str(true_pos))
        
        status += '\n{0: <20}'.format('True Negatives')             
        status += '{0: <27}'.format(': ' + str(true_neg))
        
        status += '\n{0: <20}'.format('False Positives')             
        status += '{0: <27}'.format(': ' + str(false_pos))
        
        status += '\n{0: <20}'.format('False Negatives')             
        status += '{0: <27}'.format(': ' + str(false_neg))       
        
        write_to_file(STATUS_LOG,status)                
      
if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-

