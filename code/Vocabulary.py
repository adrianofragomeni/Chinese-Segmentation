from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk import ngrams
from collections import Counter
import re
import pickle
from unicodedata import normalize
from string import ascii_letters,digits,punctuation

###############################################################################
#### GLOBAL VARIABLES
###############################################################################

punctuation_set=set(["...","—","！","‘","’","／","…","『","』","、","，","。","●","－","～","＇","｀","《","》","：","（","）","【","】","「","」","？","”","“","；","‧","·"]+list(punctuation))



def open_file(name,encode):
###############################################################################
# This function opens the txt file with the labels and utf8 file with the dataset
#
# Input:
#   name: name of the file where the data is saved
#   encode: Boolean variable to encode the utput file as utf8
#    
# Output:
#   data: data with labels or sentences
#   
###############################################################################

    if encode:
        with open('../resources/'+ name,'r', encoding='utf8') as file:
            data=[line.strip() for line in file]
            
    else:
        with open('../resources/'+ name,'r') as file:
            data=[line.strip() for line in file]
            
    return data



def create_vocabulary_unigram(vocab_size):
###############################################################################
# This function creates the unigram vocabulary (unigram to id), considering only the chinese characters
# and taking only the vocab_size-most common unigrams, other kinds of tags are added to consider
# the OOV, the Padding, the Punctuation and the Latin words + Arabic numbers
#
# Input:
#   vocab_size: the maximum length of the vocabulary
#    
# Output:
#   : None
#   
###############################################################################

    dataset_train=open_file('../resources/Train_data.utf8',True)
    
    # consider only the chinese characters
    dataset_chinese="".join(re.findall(r'[\u4e00-\u9fff]+',"".join(dataset_train)))
        
    # create the id_to_unigram, adding new tags
    id_to_unigram=dict(enumerate([char[0] for char in Counter(dataset_chinese).most_common(vocab_size)],start=4))
    id_to_unigram[0]='<PAD>'
    id_to_unigram[1]='<UNK>'
    id_to_unigram[2]='<PUNCT>'
    id_to_unigram[3]='<LATIN>'
   
    # create the id_to_unigram, giving an id to each element of the vocabulary
    unigram_to_id={value:key for key, value in id_to_unigram.items()}
    
    with open("../resources/unigram_dictionary.pkl","bw") as file:
        pickle.dump(unigram_to_id,file)



def create_vocabulary_bigrams(vocab_size):
###############################################################################
# This function creates the bigram vocabulary (bigram to id),and taking 
# only the vocab_size-most common bigrams, other kinds of tags are added to consider
# the OOV and the Padding. 
#
# Input:
#   vocab_size: the maximum length of the vocabulary
#    
# Output:
#   : None
#   
###############################################################################
    
    dataset_train=open_file('../resources/Train_data.utf8',True)
    
    # create the bigrams for the whole dataset
    dataset_chinese=create_bigrams(dataset_train)
            
    
    # create the id_to_bigram, adding new tags
    id_to_bigram=dict(enumerate([bigrm[0] for bigrm in Counter(dataset_chinese).most_common(vocab_size)],start=2))
    id_to_bigram[0]='<PAD>'
    id_to_bigram[1]='<UNK>'


    # create the id_to_bigram, giving an id to each element of the vocabulary
    bigram_to_id={value:key for key, value in id_to_bigram.items()}
    
    with open("../resources/bigram_dictionary.pkl","bw") as file:
        pickle.dump(bigram_to_id,file)




def create_bigrams(data):
###############################################################################
# This function creates the bigrams of a dataset, it splits each sentence in bigrams,
# adding the '</s>' tag at the end sentence and exchange each punctuation or
# latin characters with their unigram token.
#
# Input:
#   data: dataset of chinese sentences
#    
# Output:
#   bigrams: list of bigrams
#   
###############################################################################
    
    bigrams=[]
    
    for sentence in data:
        bigrams+=list(ngrams(exchange_tokens(list(sentence.strip()))+['</s>'],2))

    return bigrams



def exchange_tokens(sentence):
###############################################################################
# This function exchange all the particular characters with their new tag
# e.g.the punctuation with the '<PUNC>' token
#
# Input:
#   sentence: chinese sentence
#    
# Output:
#   characters: list of the new tokens for each character in the sentence
#   
###############################################################################
    characters=[]
    
    for char in sentence:
        
        if normalize("NFKC",char) in list(digits)+list(ascii_letters):
            characters.append('<LATIN>')
            
        elif char in punctuation_set:
            characters.append('<PUNCT>')
            
        else:
            characters.append(char)
            
    return characters
    


def convert_sentences(chinese_dataset,max_length,ngram_to_id,pad=True,unigram_bool=True):
###############################################################################
# This function converts all the sentences in a dataset in bigrams and unigrams
# and giving to them an id from the dictionary, if an unigram or a bigram is not
# in the dictionay the id 1 is given
#
# Input:
#   chinese_dataset: dataset with all the sentences
#   max_length: max length each sentence can have
#   ngram_to_id: unigram dictionary or bigram dictionary
#   pad: Boolean variable, if it is True the padding is applied
#   unigram_bool: Boolean variable, if it is True the unigram dictionary is built
#    
# Output:
#   : list of padded (or not) sequences of unigrams or bigrams for each sentence
#   
###############################################################################
    
    sentences=[]
    
    if unigram_bool:
        
        # Exchange the unigrams in the sentence with their id from the vocabulary
        for sentence in map(list,chinese_dataset):
            nid=[ngram_to_id[unigram] if unigram in ngram_to_id else 1 for unigram in exchange_tokens(sentence)]
            
            sentences.append(nid)
    
    else:
        
        # Exchange the unigrams in the sentence with their id from the vocabulary      
        for sentence in chinese_dataset:
            nid=[ngram_to_id[bigram] if bigram in ngram_to_id else 1 
                 for bigram in list(ngrams(exchange_tokens(list("".join(sentence.split())))+['</s>'],2))]
            
            sentences.append(nid)
    # pad the sequence 
    return  pad_sequences(sentences, truncating='pre', padding='post', maxlen=max_length) if pad else sentences


    
def convert_labels(labels_dataset,max_length):
###############################################################################
# This function converts all the labels from BIES format into integer id,
# padding them if it is necessary.
#
# Input:
#   labels_dataset: dataset with all the labels
#   max_length: max length each sentence can have
#    
# Output:
#   labels: list of padded labels
#   
###############################################################################   
    labels_to_id={'B':0,'I':1,'S':2,'E':3}

    labels=[]
    
    for sentence_label in map(list,labels_dataset):
        lid=[labels_to_id[label] for label in sentence_label]
        labels.append(lid)
    
    return pad_sequences(labels, truncating='pre', padding='post', maxlen=max_length)




    
    
    
    
    
    
    
    
    

