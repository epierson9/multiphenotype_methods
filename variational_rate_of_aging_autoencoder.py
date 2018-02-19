import numpy as np
from multiphenotype_utils import (get_continuous_features_as_matrix, add_id, remove_id_and_get_mat, 
    partition_dataframe_into_binary_and_continuous, divide_idxs_into_batches)
import pandas as pd
import tensorflow as tf
from dimreducer import DimReducer

from general_autoencoder import GeneralAutoencoder
from standard_autoencoder import StandardAutoencoder
from variational_autoencoder import VariationalAutoencoder

class VariationalRateOfAgingAutoencoder(VariationalAutoencoder):
    """
    Implements a variational rate-of-aging autoencoder.
    """    
    
    def __init__(self,
                 k_age,
                 **kwargs):
        
        # TODO: 
        # not confident we're training correctly. Things seem to be exploding. Why? 
        # I think multiplying age by the exponentiated log_aging_rate is causing problems. 

        super(VariationalRateOfAgingAutoencoder, self).__init__(**kwargs)   
        self.k_age = k_age
        assert self.k >= self.k_age
        self.need_ages = True
        self.age_preprocessing_method = 'scale_so_max_is_one'
        self.include_age_in_encoder_input = True
        self.aging_rate_scaling = .1 # THERE IS A BETTER WAY TO DO THIS. DO NOT WANT HUGE COMPONENTS OF Z. THINK MORE. 
        
    def sample_Z(self, age, n):
        """
        TODO: implement. 
        """
        
        return None
    
    def encode(self, X, ages):  
        # first k_age dimensions are log(rate_of_aging). 
        # dimensions after that are the residual. 
        
        # note that we use the same notation -- Z_mu, Z_sigma -- here as before
        # but Z_mu and Z_sigma have different interpretations for the age components. 
        # they are the mean and std of the LOG RATE OF AGING, not of Z. 
        
        X_with_age = tf.concat([X, tf.reshape(ages, [-1, 1])], axis=1) # make age 2d. 
        
        num_layers = len(self.encoder_layer_sizes)
        # Get mu 
        mu = X_with_age
        for idx in range(num_layers):
            mu = tf.matmul(mu, self.weights['encoder_h%i' % (idx)]) \
                + self.biases['encoder_b%i' % (idx)]
            # No non-linearity on the last layer
            if idx != num_layers - 1:
                mu = self.non_linearity(mu)
        self.Z_mu = mu

        # Get sigma
        sigma = X_with_age
        for idx in range(num_layers):
            sigma = tf.matmul(sigma, self.weights['encoder_h%i_sigma' % (idx)]) \
                + self.biases['encoder_b%i_sigma' % (idx)]
            # No non-linearity on the last layer
            if idx != num_layers - 1:
                sigma = self.non_linearity(sigma)
        sigma = sigma * self.sigma_scaling # scale so sigma doesn't explode when we exponentiate it. 
        sigma = tf.exp(sigma)
        self.Z_sigma = sigma

        # Sample from N(mu, sigma). The first k_age components are the LOG AGING RATE
        # the components after that are the residual (Z_non_age)
        self.eps = tf.random_normal(tf.shape(self.Z_mu), dtype=tf.float32, mean=0., stddev=1.0, seed=self.random_seed)
        log_aging_rate_plus_residual = self.Z_mu + self.Z_sigma * self.eps 
        
        # generate age components. 
        # Exponentiate log aging rate and then multiply by age
        # (and a constant factor to keep Z_age and Z_non_age on the same scale) 
        log_aging_rate = self.aging_rate_scaling * log_aging_rate_plus_residual[:, :self.k_age]
        aging_rate = tf.clip_by_value(tf.exp(log_aging_rate), 0, 50)
        Z_age = aging_rate * tf.reshape(ages, [-1, 1]) # relies on broadcasting to reshape age vector. 
        #print tf.shape(tf.tile(ages, [1, self.k_age]))
        
        # generate non-age components. This just means taking the last components of log_aging_rate_plus_residual. 
        Z_non_age = log_aging_rate_plus_residual[:, self.k_age:]
        Z = tf.concat([Z_age, Z_non_age], axis=1)
        return Z
        
