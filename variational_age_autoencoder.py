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
    Implements a variational autoencoder with an age prior.
    """    
    def __init__(self,
                 k_age,
                 Z_age_coef,
                 **kwargs):

        super(VariationalAgeAutoencoder, self).__init__(**kwargs)   
        self.k_age = k_age
        self.Z_age_coef = Z_age_coef
        assert self.k >= self.k_age
        self.need_ages = True
        
    def sample_Z(self, age, n):
        """
        samples Z from the age autoencoder prior. 
        Important note: this function automatically applies the age preprocessing function
        to the passed in age, so there is no need to transform age ahead of time. 
        """
        preprocessed_age = self.age_preprocessing_function(age)
        Z = np.zeros([n, self.k])
        # For age components, need to add age shift. Other components are zero-centered. 
        Z[:, :self.k_age] = preprocessed_age * self.Z_age_coef
        
        # add noise to all components. 
        Z = Z + np.random.multivariate_normal(mean = np.zeros([self.k,]),
                                              cov = np.eye(self.k),
                                              size = n)
        return Z
    
    def set_up_regularization_loss_structure(self):
        """
        This function sets up the basic loss structure. Should define self.reg_loss. 
        """
        self.reg_loss = self.get_regularization_loss(self.ages, self.Z_mu, self.Z_sigma)   
        
    def get_regularization_loss(self, ages, Z_mu, Z_sigma):
        # Subtract off self.Z_age_coef * self.ages from the components of self.Z_mu 
        # that are supposed to correlate with age
        # This assumes that the age-related components are drawn from N(Z_age_coef * age, 1)
        Z_mu_age = Z_mu[:, :self.k_age] - self.Z_age_coef * tf.reshape(ages, (-1, 1)) # Relies on broadcasting

        # Leave the other components untouched
        Z_mu_others = Z_mu[:, self.k_age:]

        # Z_mu_diffs is the difference between Z_mu and the priors
        Z_mu_diffs = tf.concat((Z_mu_age, Z_mu_others), axis=1)

        kl_div_loss = -.5 * (
            1 + 
            2 * tf.log(Z_sigma) - tf.square(Z_mu_diffs) - tf.square(Z_sigma))
        kl_div_loss = tf.reduce_mean(
            tf.reduce_sum(
                kl_div_loss,
                axis=1),
            axis=0)
        
        return kl_div_loss

    def fast_forward_Z(self, Z0, train_df, years_to_move_forward):
        Z0_projected_forward = copy.deepcopy(Z0)
        # move age components forward. 
        for k in range(self.k_age):
            Z0_projected_forward['z%i' % k] = Z0_projected_forward['z%i' % k] + \
            self.model_learned_age_coefs[k] * np.array(years_to_move_forward)
            
        return Z0_projected_forward
