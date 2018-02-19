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
    Generative model: 
    rate_of_aging ~ some_prior (currently, rate_of_aging = exp(scaling_factor * N(0, I)), so this is a log-normal prior centered around 1). 
    Z_age = age * rate_of_aging
    Z_non_age ~ N(0, I)
    """    
    
    def __init__(self,
                 k_age,
                 aging_rate_scaling_factor=.1,
                 **kwargs):
        super(VariationalRateOfAgingAutoencoder, self).__init__(**kwargs)   
        self.k_age = k_age
        assert self.k >= self.k_age
        self.need_ages = True
        self.age_preprocessing_method = 'divide_by_a_constant' # important not to zero-mean age here. 
        # otherwise we end up with people with negative ages, which will mess up the interpretation of the aging rate. 
        # we divide by a constant to put age on roughly the same scale as the other features. 

        self.include_age_in_encoder_input = True
        self.aging_rate_scaling_factor = aging_rate_scaling_factor 
        # log_unscaled_aging_rate ~ N(0, 1)
        # Z_age = age * exp(self.aging_rate_scaling_factor * log_unscaled_aging_rate) 
        # so aging_rate_scaling_factor controls how much spread we have on the aging rate. 
        
    def sample_Z(self, age, n):
        # sample age components.
        log_unscaled_aging_rate = np.random.multivariate_normal(mean = np.zeros([self.k_age,]), cov = np.eye(self.k_age), size = n)
        aging_rate = np.exp(self.aging_rate_scaling_factor * log_unscaled_aging_rate)
        Z_age = age * aging_rate
        
        # sample non-age components.
        k_non_age = self.k - self.k_age
        Z_non_age = np.random.multivariate_normal(mean = np.zeros([k_non_age,]), cov = np.eye(k_non_age), size = n)

        return np.concatenate([Z_age, Z_non_age], axis=1)
    
    def encode(self, X, ages):  
        # note that we use the same notation -- Z_mu, Z_sigma -- here as with the standard autoencoder 
        # and essentially the same procedure for generating both (with the same loss function)
        # but Z_mu and Z_sigma have different interpretations for the age components. 
        # they are the mean and std of the log_unscaled_aging_rate, not of Z. 
        # so we have Z_age = age * exp(self.aging_rate_scaling_factor * log_unscaled_aging_rate) 
        # where log_unscaled_aging_rate ~ N(0, 1). 
        
        # concatenate age onto X, since we use both to generate the posterior over Z.
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

        # Sample from N(mu, sigma). The first k_age components are the log unscaled aging rate
        # the components after that are the residual (Z_non_age, just as before)
        self.eps = tf.random_normal(tf.shape(self.Z_mu), dtype=tf.float32, mean=0., stddev=1.0, seed=self.random_seed)
        log_unscaled_aging_rate_plus_residual = self.Z_mu + self.Z_sigma * self.eps 
        
        # generate age components. 
        # Exponentiate log aging rate and then multiply by age
        log_unscaled_aging_rate = log_unscaled_aging_rate_plus_residual[:, :self.k_age]
        log_aging_rate = self.aging_rate_scaling_factor * log_unscaled_aging_rate
        aging_rate = tf.clip_by_value(tf.exp(log_aging_rate), 
                clip_value_min=0, 
                clip_value_max=50) # keep from exploding. We should never be near either of these bounds anyway.  
        Z_age = aging_rate * tf.reshape(ages, [-1, 1]) # relies on broadcasting to reshape age vector. 
        
        # generate non-age components. This just means taking the last components of log_unscaled_aging_rate_plus_residual. 
        Z_non_age = log_unscaled_aging_rate_plus_residual[:, self.k_age:]
        Z = tf.concat([Z_age, Z_non_age], axis=1)
        return Z
        
