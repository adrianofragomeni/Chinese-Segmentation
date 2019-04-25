import Embedding
import random
import preprocessing as prep
import Vocabulary
import model
import tensorflow as tf
import pickle
from keras.utils import to_categorical
import numpy as np

###############################################################################
#### GLOBAL VARIABLES
###############################################################################
random.seed('abc')
EPOCHS = 30
STEP_EPOCHS=800
MAX_LENGTH=200
VOCAB_UNIGRAM_SIZE=5000
VOCAB_BIGRAM_SIZE=40000

# Define paths for each dataset
dictionary_paths={
'paths_MSR':[['../dataset/training/msr_training.utf8',True],['../dataset/gold/msr_test_gold.utf8',True]],
'paths_PKU':[['../dataset/training/pku_training.utf8',False],['../dataset/gold/pku_test_gold.utf8',False]],
'paths_AS':[['../dataset/training/as_training.utf8',False],['../dataset/gold/as_testing_gold.utf8',False]],
'paths_cityu':[['../dataset/training/cityu_training.utf8',True],['../dataset/gold/cityu_test_gold.utf8',True]]}


def prepare_dataset(sentences,labels,max_length):
###############################################################################
# This function converts the inputs of the model in integer: chinese unigrams 
# and bigrams are converted into integer id using the vocabularies, whereas
# the BIES labels are converted in one hot encoding representation
#
# Input:
#   sentences: senteces of the dataset
#   labels: ground truth of the dataset
#   max_length= the maximum length of a sentence
#
# Output:
#   :A list with converted unigrams, bigrams and labels 
###############################################################################
    
    # load unigram and bigram dictionaries
    vocab_unigram,vocab_bigram=load_dictionaries()
    
    # convert unigrams and bigrams into id
    sentences_unigrams=Vocabulary.convert_sentences(sentences,max_length,vocab_unigram)
    sentences_bigrams=Vocabulary.convert_sentences(sentences,max_length,vocab_bigram,unigram_bool=False)
    
    # convert BIES labels in integer labels for keras
    labels=np.array([to_categorical(label,4) for label in Vocabulary.convert_labels(labels,max_length)])
    
    return [sentences_unigrams,sentences_bigrams, labels]


def datasets_features(name_dataset_train=None,name_dataset_dev=None,creation_features=True):
###############################################################################
# This function creates or loads all the main features to train the model 
# (train data, dev_data, vocabulary)
#
# Input:
#   name_dataset_train: name of the train dataset
#   name_dataset_dev: name of the dev dataset
#   creation_features: Boolean variable for the creation of the features
#
# Output:
#   train_data: Training data to train the model
#   dev_data: Dev data to test the model
###############################################################################
    
    try:
        path_train=dictionary_paths[name_dataset_train]
    except:
        path_train=None
        
    try:
        path_dev=dictionary_paths[name_dataset_dev]
    except:
        path_dev=None
        
    if creation_features:
        
        # Creation features
        prep.save_dataset(dev_test=False,dataset=path_train,limit_length=2)
        prep.save_dataset(dev_test=True,dataset=path_dev,limit_length=0) 
        Vocabulary.create_vocabulary_unigram(VOCAB_UNIGRAM_SIZE)
        Vocabulary.create_vocabulary_bigrams(VOCAB_BIGRAM_SIZE)  
        Embedding.embedding_matrix('vocab_unigram.npy')
        Embedding.embedding_matrix('vocab_bigram.npy',False)
    
    else:
        
        # load Train set
        train_x=Vocabulary.open_file('Train_data.utf8',True)
        train_y=Vocabulary.open_file('Train_label.txt',False)
        
        # load Dev set
        dev_x=Vocabulary.open_file('Dev_data.utf8',True)
        dev_y=Vocabulary.open_file('Dev_label.txt',False)
        
        # Creation of the training and dev dataset
        train_data=prepare_dataset(train_x,train_y,MAX_LENGTH)
        dev_data=prepare_dataset(dev_x,dev_y,MAX_LENGTH)
    
        return train_data, dev_data
    



def trainer(name_dataset_train=None,name_dataset_dev=None):
###############################################################################
# This function defines and trains the model for the chinese segmentation, 
# all the features are loaded and passed to the model for the training process 
#
# Input:
#   name_dataset_train: name of the train dataset
#   name_dataset_dev: name of the dev dataset
#
# Output:
#   history: history of the model
###############################################################################  
 
    try:
        path_train=dictionary_paths[name_dataset_train]
    except:
        path_train=None
        
    try:
        path_dev=dictionary_paths[name_dataset_dev]
    except:
        path_dev=None
        
    # load dictionaries
    vocab_unigram,vocab_bigram=load_dictionaries()

    vocab_size_unigram=len(vocab_unigram)
    vocab_size_bigram= len(vocab_bigram)
    
    # Creation of the model
    training_model = model.lstm_model(vocab_size_unigram,vocab_size_bigram)
    training_model.summary()
    
    
    cbk = tf.keras.callbacks.TensorBoard("logging/keras_model")
    
    # Trainin the model
    train_data,dev_data=datasets_features(path_train,path_dev,False)    
    data_gen= model.batch_creation(*train_data)
    history=training_model.fit_generator(data_gen,STEP_EPOCHS, EPOCHS,validation_data=([*dev_data[:2]],dev_data[2]), callbacks=[cbk])
    
    save_model(training_model)
    
    return history


def save_model(model):
###############################################################################
# This function saves the weights and the model
#
# Input:
#   model: This is the model that it will be saved

# Output:
#   None
###############################################################################
    
    model_json = model.to_json()
    
    with open("../resources/model.json","w") as json_file:
        json_file.write(model_json)
    
    model.save_weights("../resources/weights.h5")



def load_dictionaries():
###############################################################################
# This function loads the unigram and bigram vocabularies
#
# Input:
#   None

# Output:
#   vocab_unigram: vocabulary with the considered unigrams
#   vocab_bigram: vocabulary with the considered bigrams
###############################################################################
       

    with open('../resources/unigram_dictionary.pkl', 'rb') as file:
        vocab_unigram = pickle.load(file)
        
    with open('../resources/bigram_dictionary.pkl', 'rb') as file:
        vocab_bigram = pickle.load(file)
        
    return vocab_unigram,vocab_bigram
