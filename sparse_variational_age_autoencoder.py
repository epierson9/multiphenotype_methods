import numpy as np
from multiphenotype_utils import (get_continuous_features_as_matrix, add_id, remove_id_and_get_mat, 
    partition_dataframe_into_binary_and_continuous, divide_idxs_into_batches)
import pandas as pd
import tensorflow as tf
from dimreducer import DimReducer

from general_autoencoder import GeneralAutoencoder
from standard_autoencoder import StandardAutoencoder
from variational_autoencoder import VariationalAutoencoder
from variational_age_autoencoder import VariationalAgeAutoencoder


class SparseVariationalAgeAutoencoder(VariationalAgeAutoencoder):
    """
    Implements a variational autoencoder with an age prior and sparsity.
    """    
    def __init__(self,
                 k_age,
                 Z_age_coef,
                 sparsity_weighting = .1,
                 **kwargs):

        super(SparseVariationalAgeAutoencoder, self).__init__(k_age = k_age, 
                                                        Z_age_coef = Z_age_coef, 
                                                        **kwargs)   
        # Does not include input_dim, but includes last hidden layer
        # self.encoder_layer_sizes = encoder_layer_sizes

        self.k_age = k_age
        self.Z_age_coef = Z_age_coef
        assert self.k >= self.k_age
        self.need_ages = True

        self.sparsity_weighting = sparsity_weighting
        
        # assert we only have a single decoder layer (otherwise the sparsity loss doesn't make sense). 
        #assert(len([layer_name for layer_name in self.weights if 'decoder' in layer_name]) == 1)

    def get_loss(self):
        """
        Uses self.X, self.Xr, self.Z_sigma, self.Z_mu, self.kl_weighting
        """
        _, binary_loss, continuous_loss, kl_div_loss = super(SparseVariationalAgeAutoencoder, self).get_loss()   
        
        decoder_layer_number = 0
        layer_name = 'decoder_h%i' % decoder_layer_number
        while layer_name in self.weights:
            print('adding sparsity loss to decoder layer ' + layer_name)
            if decoder_layer_number == 0:
                combined_weight_matrix = tf.abs(self.weights[layer_name])
            else:
                combined_weight_matrix = tf.matmul(combined_weight_matrix, tf.abs(self.weights[layer_name]))
            decoder_layer_number += 1
            layer_name = 'decoder_h%i' % decoder_layer_number
            
        sparsity_loss = tf.reduce_sum(tf.abs(combined_weight_matrix))
        regularization_loss = kl_div_loss + sparsity_loss * self.sparsity_weighting
        combined_loss = self.combine_loss_components(binary_loss, continuous_loss, regularization_loss)

        return combined_loss, binary_loss, continuous_loss, regularization_loss  
