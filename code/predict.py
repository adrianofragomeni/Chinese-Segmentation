from argparse import ArgumentParser
import pickle
import Vocabulary
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import model_from_json
import numpy as np
from itertools import compress
from operator import itemgetter


###############################################################################
#### GLOBAL VARIABLES
###############################################################################

MAX_LENGTH=200

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_path", help="The path of the input file")
    parser.add_argument("output_path", help="The path of the output file")
    parser.add_argument("resources_path", help="The path of the resources needed to load your model")

    return parser.parse_args()



def load_trained_model(path):
###############################################################################
# This function loads the model and its weights
#
# Input:
#   path: path where the model and its weights are stored
#    
# Output:
#   model: the trained model
#   
###############################################################################
    
    # load model
    with open(path+"model.json","r") as file:
        loaded_model_json = file.read()

    #Import weights
    model=model_from_json(loaded_model_json)
    model.load_weights(path+"weights.h5")
    
    return model



def load_features(resources_path,input_path):
###############################################################################
# This function loads all the main features to test the model: the unigram dictionary,
# the bigram dictionary and the dataset
#
# Input:
#   input_path: path where the dataset is stored
#   resources_path: path where the dictionaries are stored
#    
# Output:
#   unigram_vocab: unigram vocabulary
#   bigram_vocab: bigram vocabulary
#   dataset: Test dataset
#   
###############################################################################
    with open(resources_path+"unigram_dictionary.pkl","br") as file:
        unigram_vocab = pickle.load(file)
    
    with open(resources_path+"bigram_dictionary.pkl","br") as file:
        bigram_vocab = pickle.load(file)
    
    with open(input_path,'r', encoding='utf8') as file:
        dataset=[line.strip() for line in file]
        
    return unigram_vocab, bigram_vocab, dataset



def features_sentences(unigram,bigram,indices,longest=True):
###############################################################################
# This function creates the needed features for the model (unigrams and bigrams),
# if the sentence is longer than the fixed maximum, it is splitted in n sentences
#
# Input:
#   unigram: list of unigrams for each sentence
#   bigram: list of bigrams for each sentence
#   indices: list of positions of the sentences in the list 
#   longest: Boolean variable to identify when the sentence is longer than the maximum
#    
# Output:
#   dataset_unigram: unigram features of the dataset
#   dataset_bigram: bigram features of the dataset
#   length_list: list of subsentences for each long sentence (just for long sentence)
#   
###############################################################################
    
    if longest:
        
        # split the long sentence in subsentences and for each of them, convert the sentence in unigram and bigram padding them
        unigrams=list(map(lambda unigram_line:pad_sequences(chunks(unigram_line),padding='post', maxlen=MAX_LENGTH),
                                 list(itemgetter(*indices)(unigram))))            
        bigrams=list(map(lambda bigram_line:pad_sequences(chunks(bigram_line),padding='post', maxlen=MAX_LENGTH),
                                list(itemgetter(*indices)(bigram))))
    
        # save the number of subsentences for each long sentence
        length_list=[0]+np.cumsum(list(map(len,unigrams))).tolist()    
        
    else:
        
        # lists of unigrams and bigram for the sentences shorter then the fixed maximum length
        unigrams=pad_sequences(list(itemgetter(*indices)(unigram)),padding='post', maxlen=MAX_LENGTH)              
        bigrams=pad_sequences(list(itemgetter(*indices)(bigram)),padding='post', maxlen=MAX_LENGTH) 
    
    # stack the unigrams and bigrams to pass them to the model
    dataset_unigram=np.vstack(unigrams).tolist()
    dataset_bigram=np.vstack(bigrams).tolist()

    if longest:
        return dataset_unigram, dataset_bigram,length_list
    
    else:    
        return dataset_unigram, dataset_bigram



def predict(input_path, output_path, resources_path):
    """
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the BIES format.
    
    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.
    
    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.

    :param input_path: the path of the input file to predict.
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    model=load_trained_model(resources_path)   
    unigram_vocab, bigram_vocab, dataset= load_features(resources_path,input_path)

    # unigrams and bigrams of the given dataset
    unigrams=Vocabulary.convert_sentences(dataset,MAX_LENGTH,unigram_vocab,pad=False)
    bigrams= Vocabulary.convert_sentences(dataset,MAX_LENGTH,bigram_vocab,pad=False,unigram_bool=False)
    
    # Calculate the position of each sentence (splitting the position of the long sentences and the short sentences in two different list)
    S_indices=list(compress(range(len(dataset)), map(lambda sentence: len(sentence)<=MAX_LENGTH,dataset)))
    L_indices=list(compress(range(len(dataset)), map(lambda sentence: len(sentence)>MAX_LENGTH,dataset)))
    
    # create the list of bigrams and unigrams, for short and long sentences, and then merge them to have the final lists of 
    # bigrams and unigrams
    if L_indices!=[]: 
        short_unigram=[]
        short_bigram=[]
        
        L_unigrams, L_bigrams, cumulative_lengths=features_sentences(unigrams,bigrams,L_indices)
        
        if S_indices!=[]:           
            S_unigrams, S_bigrams=features_sentences(unigrams,bigrams,S_indices,longest=False)
            short_unigram+=S_unigrams
            short_bigram+=S_bigrams
            
        # the dataset is created putting at the beginning the short sentences and at the end the long sentences
        dataset_unigram=short_unigram+L_unigrams
        dataset_bigram=short_bigram+L_bigrams
        
    else:        
        S_unigrams, S_bigrams=features_sentences(unigrams,bigrams,S_indices,longest=False)
        dataset_unigram,dataset_bigram=S_unigrams,S_bigrams


    prediction=model.predict([dataset_unigram,dataset_bigram])
    
    # reconstruct the prediction of the split sentences into one sentence 
    # and order the predictions with the same order of the input file
    prediction=correct_prediction(prediction,cumulative_lengths,S_indices,L_indices)

    with open(output_path,'w') as file:
        
        # save the predictions removing the predictions for the padding
        file.write("\n".join("".join(prediction[pos])[:len(dataset[pos])] for pos in range(len(prediction))))
        
    pass



def correct_prediction(prediction,cum_lengths,S_id,L_id):
###############################################################################
# This function reconstructs the predictions, merging the prediction of the subsentences into one prediction,
# and order all the sentences with the same order of the input file
#
# Input:
#   prediction: list of predictions (where the prediction of the long sentence is split in more than one )
#   cum_lengths: list of the numbers of subsentences per long sentence
#   S_id: positions of the short sentences
#   L_id: positions of the long sentence
#    
# Output:
#   prediction: prediction for each sentence in the dataset
#   
###############################################################################
    
    # convert the predictions from one hot encoding to BIES
    prediction_sentences=[convert_from_labels(np.argmax(prediction[pos],1)) for pos in range(len(prediction))]
    
    if L_id!=[]:
        
        # Reconstruct the long sentences, using the saved lengths (number of subsentences for each long sentence)
        long_sentences=[sum(prediction_sentences[len(S_id):][cum_lengths[pos-1]:cum_lengths[pos]],[]) 
                    for pos in range(1,len(cum_lengths))]
        
        # create the ordered predictions using the saved indices (positions) of all the sentences of the dataset
        prediction=[pred_BIES for index,pred_BIES in sorted(zip(S_id+L_id, prediction_sentences[:len(S_id)]+long_sentences))] 
    else:
        prediction=prediction_sentences
    
    return prediction

    
    
def chunks(list_):
###############################################################################
# This function returns all the sublists (w.r.t. a given number) of a given list
#
# Input:
#   list_: list of elements
#    
# Output:
#   : list with all the sublists of the input list
#   
###############################################################################

    return [list_[pos:pos + MAX_LENGTH] for pos in range(0, len(list_), MAX_LENGTH)]



def convert_from_labels(predicted_sentence):
###############################################################################
# This function converts the prediction into the BIES format
#
# Input:
#   predicted_sentence: prediction sentence
#    
# Output:
#   : prediction in the BIES format
#   
###############################################################################   
    
    id_to_labels={0:"B",1:"I",2:"S",3:"E"}
    return [id_to_labels[character] for character in predicted_sentence]

 

if __name__ == '__main__':
    args = parse_args()
    predict(args.input_path, args.output_path, args.resources_path)

