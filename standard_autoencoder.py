import numpy as np
from multiphenotype_utils import (get_continuous_features_as_matrix, add_id, remove_id_and_get_mat, 
    partition_dataframe_into_binary_and_continuous, divide_idxs_into_batches)
import pandas as pd
import tensorflow as tf
from dimreducer import DimReducer
from copy import deepcopy

from general_autoencoder import GeneralAutoencoder

class StandardAutoencoder(GeneralAutoencoder):
    """
    Implements a standard, deterministic feed-forward autoencoder.
    """
    def __init__(self, 
                 encoder_layer_sizes,
                 decoder_layer_sizes,
                 **kwargs):

        super(StandardAutoencoder, self).__init__(**kwargs)   
        # Does not include input_dim, but includes last hidden layer
        self.encoder_layer_sizes = deepcopy(encoder_layer_sizes) # make a deepcopy so we don't modify the original data accidentally. 
        self.k = self.encoder_layer_sizes[-1]

        self.decoder_layer_sizes = deepcopy(decoder_layer_sizes)
        
        self.non_linearity = tf.nn.relu
        self.initialization_function = self.glorot_init


    def init_network(self):
        self.weights = {}
        self.biases = {}        
        
        # Encoder layers.         
        for encoder_layer_idx, encoder_layer_size in enumerate(self.encoder_layer_sizes):
            if encoder_layer_idx == 0:
                input_dim = len(self.feature_names)
            else:
                input_dim = self.encoder_layer_sizes[encoder_layer_idx - 1]
            output_dim = self.encoder_layer_sizes[encoder_layer_idx]
            print("Added encoder layer with input dimension %i and output dimension %i" % (input_dim, output_dim))
            self.weights['encoder_h%i' % encoder_layer_idx] = tf.Variable(
                self.initialization_function([input_dim, output_dim]))
            self.biases['encoder_b%i' % encoder_layer_idx] = tf.Variable(
                self.initialization_function([output_dim]))

        # Decoder layers. 
        self.decoder_layer_sizes.append(len(self.feature_names))
        for decoder_layer_idx, decoder_layer_size in enumerate(self.decoder_layer_sizes):
            if decoder_layer_idx == 0:
                input_dim = self.k
            else:
                input_dim = self.decoder_layer_sizes[decoder_layer_idx - 1]
            output_dim = self.decoder_layer_sizes[decoder_layer_idx]
            print("Added decoder layer with input dimension %i and output dimension %i" % (input_dim, output_dim))
            self.weights['decoder_h%i' % decoder_layer_idx] = tf.Variable(
                self.initialization_function([input_dim, output_dim]))
            self.biases['decoder_b%i' % decoder_layer_idx] = tf.Variable(
                self.initialization_function([output_dim]))
        

    def encode(self, X):  
        num_layers = len(self.encoder_layer_sizes)
        Z = X
        for idx in range(num_layers):
            Z = tf.matmul(Z, self.weights['encoder_h%i' % (idx)]) \
                + self.biases['encoder_b%i' % (idx)]
            # No non-linearity on the last layer
            # if idx != num_layers - 1:
                # Z = self.non_linearity(Z) 
        return Z


    def decode(self, Z):
        num_layers = len(self.decoder_layer_sizes)
        X_with_logits = Z
        for idx in range(num_layers):
            X_with_logits = tf.matmul(X_with_logits, self.weights['decoder_h%i' % idx]) \
                + self.biases['decoder_b%i' % idx]
            # No non-linearity on the last layer
            if idx != num_layers - 1:
                X_with_logits = self.non_linearity(X_with_logits) 
        
        return X_with_logits


    def get_loss(self):     
        """
        Uses self.X and self.Xr
        """
        X_binary, X_continuous = self.split_into_binary_and_continuous(self.X)
        Xr_logits, Xr_continuous = self.split_into_binary_and_continuous(self.Xr)
        
        if len(self.binary_feature_idxs) == 0:
            binary_loss = tf.zeros(1)
        else:
            binary_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=Xr_logits, 
                        labels=X_binary),
                    axis=1),
                axis=0)

        if len(self.continuous_feature_idxs) == 0:
            continuous_loss = tf.zeros(1)
        else:
            continuous_loss = .5 * (
                tf.reduce_mean(
                    tf.reduce_sum(
                        tf.square(X_continuous - Xr_continuous), 
                        axis=1),
                    axis=0))

        reg_loss = tf.zeros(1)
        combined_loss = binary_loss + continuous_loss + reg_loss        

        return combined_loss, binary_loss, continuous_loss, reg_loss
