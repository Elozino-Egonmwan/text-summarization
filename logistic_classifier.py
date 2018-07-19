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

import numpy as np
import tensorflow as tf
from Helper._data_generator import write_to_file

STATUS_LOG = 'LOG_DIR_300/Extractor/Classifier/log.txt'
BATCH_SIZE=100 #128 steps=6000
epochs =100
n_examples =100 #automatically updated
#"/last_tmp_Lin2/fus_classifier"
def main():
    print("Logistic_classifier")
       
  
def _classifier(desc,features,labels,mood,model_dir): 
    global n_examples
    with tf.Session(graph = tf.Graph()) as sess:        
        num,batchSize,units,dim = features.shape  
        n_examples = num * batchSize
        features = sess.run(tf.reshape(features,[n_examples,units]))  
        run_classifier(desc,features,labels,n_examples,mood,model_dir)      
  
def run_classifier(desc,features,labels,n,mood,model_dir):
    feature_columns = [tf.feature_column.numeric_column("x", shape=[15])]
    num_batches = int(n_examples/BATCH_SIZE)
    steps = num_batches * epochs
    run_config = tf.estimator.RunConfig(save_summary_steps=num_batches)
    classifier = tf.estimator.LinearClassifier(feature_columns=feature_columns,                                          
                                          n_classes=2,
                                          config=run_config,
                                          model_dir=model_dir)
    if(mood=="Train"):
        num_epochs=None
    else:
        num_epochs=1    
    
    if(mood=="Train" or mood=="Eval"):        
        # Define the training inputs
        inp_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": np.array(features)},
            y=np.array(labels),
            num_epochs=num_epochs,   #None for training
            batch_size= BATCH_SIZE,
            shuffle=True)
        
        if mood=="Train":        
            classifier.train(input_fn=inp_fn, steps=steps)             
        else:            
            accuracy_score = classifier.evaluate(input_fn=inp_fn)["accuracy"]      
            status = desc
            status += "\nAccuracy: {0:f}\n".format(accuracy_score)
            #write_to_file(STATUS_LOG,status)      
            print("\nAccuracy: {0:f}\n".format(accuracy_score))
        
    #predict mood    
    else:    
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
          x={"x": np.array(features)},
          num_epochs=1,
          batch_size=BATCH_SIZE,
          shuffle=False)
    
        predictions = list(classifier.predict(input_fn=predict_input_fn))        
        pr = np.reshape([p["class_ids"] for p in predictions],-1)
        #pr = [1 if p==b'1' else 0 for p in predicted_classes]
        y=np.array(labels)
        get_metrics(pr,y,desc,n)
        
        predicted_logits=np.reshape([p["logistic"] for p in predictions],-1)
        #print(predicted_logits)

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
        save_metrics(desc,accuracy,prec,rec,f_measure,false_pos,true_pos,false_neg,true_neg)
        #return false_pos,false_neg
    
        
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
        status = '\n\n' + desc + ' Metrics'
        write_to_file(STATUS_LOG,status)
               
        status = '{0: <20}'.format('Accuracy')         
        status += '{0: <27}'.format(': ' + str(accuracy))
        
        status += '\n{0: <20}'.format('Precision')             
        status += '{0: <27}'.format(': ' + str(prec)) 
        
        status += '\n{0: <20}'.format('Recall')             
        status += '{0: <27}'.format(': ' + str(rec))
        
        status += '\n{0: <20}'.format('F_measure')                     
        status += '{0: <27}'.format(': ' + str(f_measure))        
        
        status += '\n{0: <20}'.format('True Positives')             
        status += '{0: <27}'.format(': ' + str(true_pos))
        
        status += '\n{0: <20}'.format('True Negatives')             
        status += '{0: <27}'.format(': ' + str(true_neg))
        
        
        status = '\n{0: <20}'.format('False Positives')                     
        status += '{0: <27}'.format(': ' + str(false_pos))        
        
        status += '\n{0: <20}'.format('False Negatives')             
        status += '{0: <27}'.format(': ' + str(false_neg))       
        
        write_to_file(STATUS_LOG,status)                
      
if __name__ == "__main__":
    main()