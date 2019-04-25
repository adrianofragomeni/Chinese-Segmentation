import re
from hanziconv import HanziConv

###############################################################################
#### GLOBAL VARIABLES
###############################################################################

# Define paths for each dataset
paths_MSR=[['../dataset/training/msr_training.utf8',True],['../dataset/gold/msr_test_gold.utf8',True]]
paths_PKU=[['../dataset/training/pku_training.utf8',False],['../dataset/gold/pku_test_gold.utf8',False]]
paths_AS=[['../dataset/training/as_training.utf8',False],['../dataset/gold/as_testing_gold.utf8',False]]
paths_cityu=[['../dataset/training/cityu_training.utf8',True],['../dataset/gold/cityu_test_gold.utf8',True]]


def open_dataset(path,simplified, limit_length):
###############################################################################
# This function opens txt file( for labels) and utf8 file (for the dataset),
# removing all the sentences less than a specifi value characters
# (just for the training set), splitting the words also when there is a punctuation
# and converting the dataset from Traditional chinese to simplified chinese, if it is needed.
#
# Input:
#   path: path of the file 
#   simplified: Boolean variable for the convertion of the dataset (if it is True)
#   limit_length: value to choose sentences larger than it
#    
# Output:
#   chinese_sentences: list of sentences
#
###############################################################################
    
    # condition to open file ( if the dataset is for training or for Dev set)
    with open(path, 'r', encoding='utf8') as file:
    
        if simplified:
            chinese_sentences=[split_punctuation(line.strip().split()) for line in file if len(line.strip().split())>limit_length]
            
        else:
            chinese_sentences=[HanziConv.toSimplified(split_punctuation(line.strip().split())) 
                                for line in file if len(line.strip().split())>limit_length]        
    

    
    return chinese_sentences 



def create_labels(sentence):
###############################################################################
# This function creates the BIES label for one sentence (with blank space between words)
# based on the length of each words, if the word's length is 1, the label is S,
# if the length is 2, the label is BE, otherwise many I are added 
# based on how many characters there are between the first letter and the last one
#
# Input:
#   sentence: chinese sentence with blank space
#    
# Output:
#   :BIES label associated to a sentence
#   
###############################################################################

    sentence_bies=[]
    
    for word in sentence.split():
        
        if len(word)==1:
            sentence_bies.append('S')
            
        elif len(word)==2:
            sentence_bies.append('BE')
            
        else:
            bies='B'+'I'*(len(word)-2)+'E'
            
            sentence_bies.append(bies)
            
    return "".join(sentence_bies)
    


def split_punctuation(sentence):
###############################################################################
# This function splits the sentence with respect to the punctuation, because some words are merged
# with the punctuation, in addition all the None element are removed
#
# Input:
#   sentence: chinese sentence with blank space
#
# Output:
#   : chinese sentence
#   
###############################################################################

    context=[]
    
    for string in sentence:
      
        # split sentence using the punctuation
       context +=re.split(r"([！—‘’／…『』、，。●－～＇·｀《》：（）\(\)【】「」？”“；‧])",string)
    
    # remove the None elemente and convert the list in string with blank space   
    return " ".join(filter(lambda element: not re.match(r'^\s*$', element), context))
 
 
    
def save_file(name_file,data,encode):
###############################################################################
# This function saves the sentences, without blank space between words, as utf8 file
# and the labels as txt file
#
# Input:
#   name_file: name of the file where the data is saved
#   data: data which are saved
#   encode: Boolean variable to encode the utput file as utf8
#
# Output:
#   :None
#
###############################################################################
    
    if encode:
        
        with open('../resources/'+name_file,'w',encoding='utf8') as file:
            file.write("\n".join(element for element in data))
            
    else:
        
        with open('../resources/'+name_file,'w') as file:
            file.write("\n".join(element for element in data))
    


def create_input(sentence):
###############################################################################
# This function joins a sentence, removing the blank space
#
# Input:
#   sentence: chinese sentence with blank space
#
# Output:
#   : chinese sentence without blank space
#   
###############################################################################

    return "".join(sentence.split())



def remove_wrong_sentence(dataset):
###############################################################################
# This function removes all the duplicate sentences in the dataset and the blank lines 
#
# Input:
#   dataset: the whole dataset with all the sentences
#
# Output:
#   : cleaned dataset
#   
###############################################################################

    seen=set()
    seen_add=seen.add
    
    # remove duplicates using a set
    sentences=[sentence for sentence in dataset if not (sentence in seen or seen_add(sentence))]
    
    #removing the blank lines
    return list(filter(None,sentences))



def save_dataset(dev_test,limit_length,dataset=None):
###############################################################################
# This function creates and saves the dataset with its labels, giving the choice
# whether using all the datasets or just one of them 
#
# Input:
#   dev_test: Boolean variable to save the Dev dataset
#   limit_length: minimum length of the sentence to be used for the training
#   dataset: Name of the dataset, which will be used, if it None, all the datasets are used
#
# Output:
#   :None   
###############################################################################
    
    # Name of the file
    if dev_test:
        name_data='Dev_data.utf8'
        name_label='Dev_label.txt'

    else:
        name_data='Train_data.utf8'
        name_label='Train_label.txt'
    
    # if dataset is None, all 4 datasets are used
    if dataset==None:
        dataset_train=[]
        
        for path in [paths_AS[int(dev_test)],paths_cityu[int(dev_test)],paths_MSR[int(dev_test)],paths_PKU[int(dev_test)]]:
            dataset_train+=open_dataset(*path,limit_length)
       
        dataset_train=remove_wrong_sentence(dataset_train)
        # Create dataset which will be used to train the model
        data=list(map(create_input,dataset_train))
        label=list(map(create_labels,dataset_train))
        
        save_file(name_data,data,encode=True)
        save_file(name_label,label,encode=False)
        
    # only one dataset is used
    else:
        dataset_train=open_dataset(*dataset[int(dev_test)],limit_length)
        dataset_train=remove_wrong_sentence(dataset_train)

        # Create dataset which will be used to train the model
        data=list(map(create_input,dataset_train))
        label=list(map(create_labels,dataset_train))
        
        save_file(name_data,data,encode=True)
        save_file(name_label,label,encode=False)

    
    
    
    
    
    