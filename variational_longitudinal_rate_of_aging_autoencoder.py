from copy import deepcopy
import numpy as np
from multiphenotype_utils import (get_continuous_features_as_matrix, add_id, remove_id_and_get_mat, 
    partition_dataframe_into_binary_and_continuous, divide_idxs_into_batches)
import pandas as pd
import tensorflow as tf
from dimreducer import DimReducer

from general_autoencoder import GeneralAutoencoder
from standard_autoencoder import StandardAutoencoder
from variational_rate_of_aging_autoencoder import VariationalRateOfAgingAutoencoder

class VariationalLongitudinalRateOfAgingAutoencoder(VariationalRateOfAgingAutoencoder):
    """
    Implements a variational rate-of-aging autoencoder.
    Generative model: 
    rate_of_aging ~ some_prior (currently, rate_of_aging = exp(scaling_factor * N(0, I)), so this is a log-normal prior. 
    Note that the log-normal prior will have a peak at one, but its mean will be greater than one
    (because the log-normal mean is right-shifted: https://en.wikipedia.org/wiki/Log-normal_distribution)
    Z_age = age * rate_of_aging
    Z_non_age ~ N(0, I)
    """    
    
    def __init__(self,
                 k_age,
                 sparsity_weighting=0,
                 aging_rate_scaling_factor=.1,
                 **kwargs):
        super(VariationalLongitudinalRateOfAgingAutoencoder, self).__init__(**kwargs)     
        self.uses_longitudinal_data = True
        
    def get_loss(self):
        # uses X, ages, followup_X, followup_ages. 
        if self.followup_X is None and self.followup_ages is None:
            # in this case, we are just doing the standard cross-sectional loss (no followup data)
            return super(VariationalLongitudinalRateOfAgingAutoencoder, self).get_loss()
        else:
            # i
            Z0 = self.encode(self.X, self.ages)
            
            # now project Z0 forward to get Z1. 
            Z1 = deepcopy(Z0)
            for k in range(self.k_age):
                Z1['z%i' % k] = Z1['z%i' % k] * (1.0*self.followup_ages / self.ages)
               
            # reconstruct X from Z0 and Z1. 
            Xr0 = self.decode(Z0)
            Xr1 = self.decode(Z1)
            
            # split the reconstructions up into binary and continuous components
            Xr0_logits, Xr0_continuous = self.split_into_binary_and_continuous(Xr0) 
            Xr1_logits, Xr1_continuous = self.split_into_binary_and_continuous(Xr1)
            
            # do the same for the true data
            X0_binary, X0_continuous = self.split_into_binary_and_continuous(self.X)
            X1_binary, X1_continuous = self.split_into_binary_and_continuous(self.followup_X)
            
            # compute losses
            binary_loss_0 = self.get_binary_loss(X0_binary, Xr0_logits)
            continuous_loss_0 = self.get_continuous_loss(X0_continuous, Xr0_continuous)
            binary_loss_1 = self.get_binary_loss(X1_binary, Xr1_logits)
            continuous_loss_1 = self.get_continuous_loss(X1_continuous, Xr1_continuous)
            
            # combine them 
            binary_loss = binary_loss_0 + binary_loss_1
            continuous_loss = continuous_loss_0 + continuous_loss_1
            
            # KL div loss is the same for both because encoder_mu and encoder_sigma are the same for both. 
            # TODO: is this right? 
            kl_div_loss = -.5 * (
                1 + 
                2 * tf.log(self.encoder_sigma) - tf.square(self.encoder_mu) - tf.square(self.encoder_sigma))
            kl_div_loss = tf.reduce_mean(
                tf.reduce_sum(
                    kl_div_loss,
                    axis=1),
                axis=0)
        
            # add in a sparsity loss: L1 penalty on the age state decoder. 
            if self.sparsity_weighting > 0:
                sparsity_loss = tf.reduce_sum(tf.abs(self.weights['decoder_Z_age_h0']))
                regularization_loss = kl_div_loss + sparsity_loss * self.sparsity_weighting
            else:
                regularization_loss = kl_div_loss

            # multiply regularization loss by two because we are actually evaluating at two timepoints 
            regularization_loss = regularization_loss * 2

            combined_loss = self.combine_loss_components(binary_loss, continuous_loss, regularization_loss)

            return combined_loss, binary_loss, continuous_loss, regularization_loss
    