import numpy as np
import tensorflow as tf
import random


###############################################################################
#### GLOBAL VARIABLES
###############################################################################
BATCH_SIZE = 64
HIDDEN_SIZE=256
EMBEDDING_DIM_UNIG =300
EMBEDDING_DIM_BIGR = 300
random.seed('abc')


def batch_creation(X_unigram,X_bigram,Y):
###############################################################################
# This function is a batch generator, it creates infinite batches which are passed
# to model to train it
#
# Input:
#   X_unigram: lists of unigrams of the dataset
#   X_bigram: lists of bigrams of the dataset 
#   Y: labels of the dataset
#    
# Output:
#   :batch of unigrams and bigrams
#   :batch of the labels 
###############################################################################
    
    while True:
        # random permutation of the elements
        perm = np.random.permutation(len(Y))
        for start in range(0, len(X_unigram), BATCH_SIZE):
            end = start + BATCH_SIZE
            yield [X_unigram[perm[start:end]],X_bigram[perm[start:end]]], Y[perm[start:end]]



def lstm_model(vocab_size_unigram,vocab_size_bigram):
###############################################################################
# This function is the implemented model; using unigrams and bigrams, it creates
# and concatenates two embedding  matrices, one for bigrams and one for unigrams. 
# Then a Bidirectional RNN (using LSTM as cells) is used, concatenating
# the outputs of the forward and backward LSTM
# 
# Input:
#   vocab_size_unigram: size of the unigram dictionary
#   vocab_size_bigram: size of the bigram dictionary
#    
# Output:
#   model: model object
###############################################################################
        
    unigram = tf.keras.Input(name='unigram', shape=(None,), dtype=tf.int32)
    bigram=tf.keras.Input(name='bigram', shape=(None,), dtype=tf.int32)

    embedding_unigram = tf.keras.layers.Embedding(name='embedding_unigram',input_dim=vocab_size_unigram, output_dim=EMBEDDING_DIM_UNIG,mask_zero=True,
                                                  weights=[np.load('../resources/vocab_unigram.npy')])(unigram)
    
    embedding_bigram = tf.keras.layers.Embedding(name='embedding_bigram',input_dim=vocab_size_bigram, output_dim=EMBEDDING_DIM_BIGR,mask_zero=True,
                                                 weights=[np.load('../resources/vocab_bigram.npy')])(bigram)
    
    concat_embedding=tf.keras.layers.concatenate(name='concatenate_embedding',inputs=[embedding_unigram,embedding_bigram])


    bi_directional_unigram = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(HIDDEN_SIZE,name='Bidirectional_LSTM',
                                                                                dropout=0.3, recurrent_dropout=0.25, return_sequences=True),name='Bidirectional_RNN',merge_mode='concat')(concat_embedding)
    
    predicted_char = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(4, activation='softmax'),name='Output')(bi_directional_unigram)
    
    model = tf.keras.Model(inputs=[unigram,bigram], outputs=[predicted_char])
    model.compile(optimizer=tf.keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004),loss='categorical_crossentropy',metrics=['acc'])
    
    return model

