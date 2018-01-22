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
    The loss function for this only really makes sense when we have a single decoder layer, so we assert that. 
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
        # self.k = self.encoder_layer_sizes[-1]        

        # self.decoder_layer_sizes = decoder_layer_sizes
        self.k_age = k_age
        self.Z_age_coef = Z_age_coef
        assert self.k >= self.k_age
        self.need_ages = True

        self.sigma_scaling = .1
        self.sparsity_weighting = sparsity_weighting
        
        # assert we only have a single decoder layer (otherwise the sparsity loss doesn't make sense). 
        #assert(len([layer_name for layer_name in self.weights if 'decoder' in layer_name]) == 1)

    def get_loss(self):
        """
        Uses self.X, self.Xr, self.Z_sigma, self.Z_mu, self.kl_weighting
        """
        _, binary_loss, continuous_loss, kl_div_loss = super(SparseVariationalAgeAutoencoder, self).get_loss()   
        
        sparsity_loss = 0
        layers_examined = 0
        for layer_name in self.weights:
            if 'decoder' in layer_name:
                if layers_examined == 0:
                    combined_weight_matrix = tf.abs(self.weights[layer_name])
                else:
                    combined_weight_matrix = tf.matmul(combined_weight_matrix, tf.abs(self.weights[layer_name]))
                layers_examined += 1
        sparsity_loss += tf.reduce_sum(tf.abs(combined_weight_matrix))
        sparsity_loss = sparsity_loss * self.sparsity_weighting

        combined_loss = self.combine_loss_components(binary_loss, continuous_loss, kl_div_loss + sparsity_loss)

        return combined_loss, binary_loss, continuous_loss, kl_div_loss  
