# -*- coding: utf-8 -*-
import re

puncts = "–!#$%&\'()*+,''-./:;<=>?@[\\~]^_`{|’}``--..."
puncts+= '"'
regex = re.compile('[%s]' % re.escape(puncts))


def main():    
    print(isValid("happy"))
    
def isPunct(word):
    if not word in puncts:
        return False
    else:
        return True
    
def hasPunct(word):
    newWord = regex.sub('',word)
    return not newWord == word

def normalize(word):    
    return regex.sub('',word)

def isEnglish(word):
    try:
        word.encode(encoding= 'utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def isValid(word):
    return not isPunct(word) and isEnglish(word) and not hasPunct(word)

if __name__ == "__main__":
    main()
