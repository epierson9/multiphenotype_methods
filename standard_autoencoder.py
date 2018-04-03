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
                 learn_continuous_variance=False,
                 **kwargs):

        super(StandardAutoencoder, self).__init__(**kwargs)   
        # Does not include input_dim, but includes last hidden layer
        self.encoder_layer_sizes = deepcopy(encoder_layer_sizes) # make a deepcopy so we don't modify the original data accidentally. 
        self.k = self.encoder_layer_sizes[-1]

        self.decoder_layer_sizes = deepcopy(decoder_layer_sizes)
        self.learn_continuous_variance = learn_continuous_variance

    def init_network(self):
        self.weights = {}
        self.biases = {}        
        
        # Encoder layers.         
        for encoder_layer_idx, encoder_layer_size in enumerate(self.encoder_layer_sizes):
            if encoder_layer_idx == 0:
                input_dim = len(self.feature_names) + self.include_age_in_encoder_input # if we include age in input, need one extra feature
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

    def get_setter_ops(self):
        self.weights_placeholders = {}
        self.weights_setters = {}
        for key in self.weights:
            self.weights_placeholders[key] = tf.placeholder(
                tf.float32,
                shape=self.weights[key].shape)
            self.weights_setters[key] = tf.assign(
                self.weights[key], 
                self.weights_placeholders[key],
                validate_shape=True)

        self.biases_placeholders = {}
        self.biases_setters = {}
        for key in self.biases:
            self.biases_placeholders[key] = tf.placeholder(
                tf.float32,
                shape=self.biases[key].shape)
            self.biases_setters[key] = tf.assign(
                self.biases[key], 
                self.biases_placeholders[key],
                validate_shape=True)      

    def assign_weights_and_biases(self, weights, biases):
        """
        weights and biases are dicts that can be partially defined.
        """
        for key in weights:
            assert key in self.weights
            self.sess.run(
                self.weights_setters[key], 
                feed_dict={self.weights_placeholders[key]:weights[key]})

        for key in biases:
            assert key in self.biases
            self.sess.run(
                self.biases_setters[key], 
                feed_dict={self.biases_placeholders[key]:biases[key]})            

    def encode(self, X):  
        num_layers = len(self.encoder_layer_sizes)
        Z = X
        for idx in range(num_layers):
            Z = tf.matmul(Z, self.weights['encoder_h%i' % (idx)]) \
                + self.biases['encoder_b%i' % (idx)]
            # No non-linearity on the last layer
            if idx != num_layers - 1:
                Z = self.non_linearity(Z) 
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

    def get_continuous_loss(self, X_continuous, Xr_continuous):
        # Given the true values X_continuous and the reconstructed values Xr_continuous
        # returns the continuous loss. 
        if len(self.continuous_feature_idxs) == 0:
            continuous_loss = tf.zeros(1)
        else:
            if self.learn_continuous_variance:
                # if we do not assume the variance is one, the continuous loss is the 
                # negative Gaussian log likelihood with all constant terms.
                continuous_variance = tf.exp(self.log_continuous_variance)
                continuous_loss = (
                    .5 * (
                        tf.reduce_mean(
                            tf.reduce_sum(
                                tf.square(X_continuous - Xr_continuous) / continuous_variance, 
                                axis=1),
                            axis=0)) 
                    + .5 * (
                        self.log_continuous_variance + 
                        tf.log(2 * np.pi)) 
                    * len(self.continuous_feature_idxs))
            else:
                # otherwise, it is just a squared-error loss.
                continuous_loss = .5 * (
                    tf.reduce_mean(
                        tf.reduce_sum(
                            tf.square(X_continuous - Xr_continuous), 
                            axis=1),
                        axis=0))
        return continuous_loss
        
    def get_binary_loss(self, X_binary, Xr_logits):
        # Given the true values X_binary and the reconstructed values Xr_logits
        # returns the binary loss.
        
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
            # upweight binary loss by the binary loss weighting. 
            binary_loss = self.binary_loss_weighting * binary_loss
        return binary_loss
        
    def get_loss(self, X, Xr):     
        """
        Uses self.X and self.Xr. 
        """
        X_binary, X_continuous = self.split_into_binary_and_continuous(X)
        Xr_logits, Xr_continuous = self.split_into_binary_and_continuous(Xr)
        
        binary_loss = self.get_binary_loss(X_binary, Xr_logits)
        continuous_loss = self.get_continuous_loss(X_continuous, Xr_continuous)

        reg_loss = tf.zeros(1)
        combined_loss = self.combine_loss_components(binary_loss, continuous_loss, reg_loss)    

        return combined_loss, binary_loss, continuous_loss, reg_loss
