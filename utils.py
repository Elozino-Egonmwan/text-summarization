# -*- coding: utf-8 -*-

from nltk import sent_tokenize

# Function to choose some sentences from the original document accroding to code sequence
# @code_path: filepath, 0-1 sequence
# @document_path: filepath, original document
# @outpput_path: filepath, for output
# return a list, and the concatation of sentences.
def code_to_sentence_transformer(code_path, document_path, output_path):
    
    # output_list = [] # for all summaries
    with open(code_path, 'r') as f:
        with open(document_path, 'r') as g:
            with open(output_path, 'w') as out:    
                while True:
                    code_sequence = f.readline().split() # same length
                    document = sent_tokenize(g.readline()) # same length
                    output = [] # for one summary
                    if code_sequence == []:
                        break # EOF
                
                    # for each sentence
                    for j in range(len(code_sequence)):
                        if code_sequence[j] == '1':
                            output.append(document[j])
                    summary = ''
                    
                    #We should handle stopwords '-lrn-' etc. here.
                    # I have done that manufactly
                    
                    #write file
                    for element in output:
                        summary = summary  + element + ' '
                    summary = summary + '\n'
                    out.write(summary)
            
#document_path = "A_Training/Sum_Corpus/doc.txt"
#code_path = "LOG_DIR_300/Extractor_RNN/tunedExtTrain_matched.txt"
#output_path = "LOG_DIR_300/Extractor_RNN/system/extTrain_generated.txt"
#code_to_sentence_transformer(code_path, document_path, output_path)
                    
#==============================================================================
                    
# Function to re-split 0-1 sequence to match original documents. If the last
# sentences doesn't contain enough numbers, it will be filled with 0.
# @code_path: filepath, 0-1 sequence
# @document_path: filepath, original document
# @outpput_path: filepath, for output
# return: number of lines
def resplit_0_1_sequence_with_doc(code_path, document_path, output_path):
    with open(document_path, 'r') as src:
        with open(code_path, 'r') as code:
            flag = 0 # flag EOF
            count_number = 0 # flag lines
            while True:
                document = sent_tokenize(src.readline())
                code_sequence = []
                for i in range(len(document)):
                    element = code.readline().split()
                    if element == []: # EOF
                        code_sequence = code_sequence + ['0']
                        flag = 1
                    else:
                        code_sequence = code_sequence + element
                count_number = count_number + 1
                
                if flag == 1:
                    break
                
                    # finished one line
                with open(output_path, 'a') as out:
                    summary = ''
                    for i in code_sequence:
                        summary = summary  + i + ' '
                    summary = summary + '\n'
                    out.write(summary)
                
    return count_number

#document_path = "A_Training/Sum_Corpus/doc.txt"
#code_path = "LOG_DIR_300/Extractor_RNN/tunedExtTrain.txt"
#output_path = "LOG_DIR_300/Extractor_RNN/tunedExtTrain_matched.txt"
#print(resplit_0_1_sequence_with_doc(code_path, document_path, output_path))
    
#==============================================================================

def remove_stopwords(input_file, output_file):
    stop_words = ['cnn','-lrb-','-rrb-']
    with open(input_file, 'r') as inp:
        with open(output_file, 'w') as out:
            while True:
                sentence = inp.readline()
                if sentence == '':
                    break
                else:
                    words = sentence.split(' ')
                    condition = lambda t: t not in stop_words
                    filter_list = list(filter(condition, words))
                    summary = ''
                    for i in filter_list:
                        if i == '':
                            continue
                        if i[-1] == '\n':
                            summary = summary + i
                        else:
                            summary = summary + i + ' '
                    out.write(summary)
#input_file = 'LOG_DIR_300/Extractor_RNN/reference/extTest_reference.txt'
#output_file = 'LOG_DIR_300/Extractor_RNN/reference/extTest_reference_clean.txt'
#remove_stopwords(input_file, output_file)
#    
                    
#==============================================================================
                    
def cutting_first_n_lines(input_file, output_file, n):
    with open(input_file, 'r') as inp:
        with open(output_file, 'w') as out:
            for i in range(n):
                sentence = inp.readline()
                out.write(sentence)
                
#input_file = "LOG_DIR_300/Manual_Extractor/system/extVal_generated.txt"
#output_file = 'LOG_DIR_300/Manual_Extractor/system/extVal_generated_1000.txt'
#cutting_first_n_lines(input_file, output_file, 1000)
    
