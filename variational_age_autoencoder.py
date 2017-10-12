import numpy as np
from multiphenotype_utils import (get_continuous_features_as_matrix, add_id, remove_id_and_get_mat, 
    partition_dataframe_into_binary_and_continuous, divide_idxs_into_batches)
import pandas as pd
import tensorflow as tf
from dimreducer import DimReducer

from general_autoencoder import GeneralAutoencoder
from standard_autoencoder import StandardAutoencoder
from variational_autoencoder import VariationalAutoencoder

class VariationalAgeAutoencoder(VariationalAutoencoder):
    """
    Implements a standard variational autoencoder (diagonal Gaussians everywhere).
    """    
    def __init__(self,
                 k_age,
                 Z_age_coef,
                 **kwargs):

        super(VariationalAgeAutoencoder, self).__init__(**kwargs)   
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

    def get_loss(self):
        """
        Uses self.X, self.Xr, self.Z_sigma, self.Z_mu, self.kl_weighting
        """
        _, binary_loss, continuous_loss, _ = super(VariationalAutoencoder, self).get_loss()   

        # This creates a matrix where each row looks like 
        # [self.Z_age_coef * age] (repeated self.k_age times) + [0] (repeated self.k - self.k_age times)
        # np.array(self.Z_age_coef * self.ages)
        # age_prior_mat = np.array([self.Z_age_coef * self.ages] * self.k_age + [0] * (self.k - self.k_age))

        Z_mu_age = self.Z_mu[:, :self.k_age] - self.Z_age_coef * tf.reshape(self.ages, (-1, 1)) # Relies on broadcasting
        Z_mu_others = self.Z_mu[:, self.k_age:]
        Z_mu_diffs = tf.concat((Z_mu_age, Z_mu_others), axis=1)

        # Z_mu_age, Z_mu_others = tf.gather(self.Z_mu, indices=self.binary_feature_idxs, axis=1)
        # Z_mu_diffs = self.Z_mu
        # Z_mu_diffs[:, :self.k_age] = self.Z_age_coef * self.ages
        # Z_mu_diffs[:, 1] = self.Z_age_coef * self.ages

        kl_div_loss = -.5 * (
            1 + 
            2 * tf.log(self.Z_sigma) - tf.square(Z_mu_diffs) - tf.square(self.Z_sigma))
        kl_div_loss = tf.reduce_mean(
            tf.reduce_sum(
                kl_div_loss,
                axis=1),
            axis=0) * self.kl_weighting

        combined_loss = binary_loss + continuous_loss + kl_div_loss

        return combined_loss, binary_loss, continuous_loss, kl_div_loss  
