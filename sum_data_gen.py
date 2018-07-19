from Helper._data_generator import read_text_file, get_abs_art

TRAIN = "CNN_Daily_Mail/train.txt"
TRAIN_DOC = "A_Training/Sum_Corpus/doc.txt"
TRAIN_SUM = "A_Training/Sum_Corpus/sum.txt"

VAL = "CNN_Daily_Mail/val.txt"
VAL_DOC = "B_Validating/Sum_Corpus/doc.txt"
VAL_SUM = "B_Validating/Sum_Corpus/sum.txt"

TESTING = "CNN_Daily_Mail/test.txt"
TEST_DOC = "C_Testing/Sum_Corpus/doc.txt"
TEST_SUM = "C_Testing/Sum_Corpus/sum.txt"

def main():
   lines = read_text_file(TRAIN)   
   print("Processing...")
   docs = ""
   summ = ""
   lines_procd = 0
   for l in range(len(lines)):
       abstract, article = get_abs_art(lines[l])  
       docs+= article+"\n"
       summ += abstract+"\n"
       lines_procd+=1
       if (lines_procd % 500) == 0:
           print(lines_procd, "...")
           
   print("writing to file")    
   write_to_file(TRAIN_DOC,docs,"w")
   write_to_file(TRAIN_SUM,summ,"w")
   print("end")
   
def write_to_file(file,txt,mode):    
    with open(file,mode) as f: 
        #for line in txt:
        f.write(txt)
            
if __name__ == "__main__":
    main()
