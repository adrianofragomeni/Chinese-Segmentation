from gensim.models import KeyedVectors
import numpy as np
import random
import pickle


###############################################################################
#### GLOBAL VARIABLES
###############################################################################

EMBEDDING = 300
random.seed('abc')
model_w2v = KeyedVectors.load_word2vec_format('../resources/fasttext.vec', binary=False)


def embedding_matrix(name_file,unigram=True):
###############################################################################
# This function calculates the initial embedding matrix 
# for the segmentation model, it initializes some uniform random values
# for each word in the vocabulary and then exchange this value,
# if the word is inside the Fasttext vocabulary
#
# Input:
#   name_file: name of the file where the embedding matrix will be saved
#   unigram: True, if the vocabulary's elements are unigrams, otherwise False

# Output:
#   None
###############################################################################     

    # load vocabulary
    if unigram:
        with open("../resources/unigram_dictionary.pkl","br") as file:
            vocabulary = pickle.load(file)
            
    else:
        with open("../resources/bigram_dictionary.pkl","br") as file:
            vocabulary = pickle.load(file)

    # initialize random values for the embedding matrix    
    embedding_weights=np.random.uniform(low=-1, high=1, size=(len(vocabulary), EMBEDDING) ) 

    for character, id_ in vocabulary.items():
        
        if unigram:
            character="".join(character)

        # check if the character has an embedding vector in the pretrained model            
        try:
            embedding_weights[id_, :] = model_w2v[character][0] 
        except KeyError:
            pass
                    
    np.save('../resources/'+ name_file, embedding_weights)    
       



