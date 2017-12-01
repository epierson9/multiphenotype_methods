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

        self.initialization_function = self.glorot_init
        self.kl_weighting = 1
        self.non_linearity = tf.nn.sigmoid
        self.sigma_scaling = .1
        self.sparsity_weighting = sparsity_weighting
        
        # assert we only have a single decoder layer (otherwise the sparsity loss doesn't make sense). 
        assert(len([layer_name for layer_name in self.weights if 'decoder' in layer_name]) == 1)

    def get_loss(self):
        """
        Uses self.X, self.Xr, self.Z_sigma, self.Z_mu, self.kl_weighting
        """
        _, binary_loss, continuous_loss, _ = super(VariationalAutoencoder, self).get_loss()   

        # Subtract off self.Z_age_coef * self.ages from the components of self.Z_mu 
        # that are supposed to correlate with age
        # This assumes that the age-related components are drawn from N(Z_age_coef * age, 1)
        Z_mu_age = self.Z_mu[:, :self.k_age] - self.Z_age_coef * tf.reshape(self.ages, (-1, 1)) # Relies on broadcasting

        # Leave the other components untouched
        Z_mu_others = self.Z_mu[:, self.k_age:]

        # Z_mu_diffs is the difference between Z_mu and the priors
        Z_mu_diffs = tf.concat((Z_mu_age, Z_mu_others), axis=1)

        kl_div_loss = -.5 * (
            1 + 
            2 * tf.log(self.Z_sigma) - tf.square(Z_mu_diffs) - tf.square(self.Z_sigma))
        kl_div_loss = tf.reduce_mean(
            tf.reduce_sum(
                kl_div_loss,
                axis=1),
            axis=0) * self.kl_weighting
        
        sparsity_loss = 0
        for layer_name in self.weights:
            if 'decoder' in layer_name:
                sparsity_loss += tf.reduce_sum(tf.abs(self.weights[layer_name]))
        sparsity_loss = sparsity_loss * self.sparsity_weighting

        combined_loss = binary_loss + continuous_loss + kl_div_loss + sparsity_loss

        return combined_loss, binary_loss, continuous_loss, kl_div_loss  
