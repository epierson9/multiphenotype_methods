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
        
    def get_loss(self):
        """
        Uses self.X, self.Xr, self.Z_sigma, self.Z_mu, self.kl_weighting
        """
        _, binary_loss, continuous_loss, _ = super(VariationalAgeAutoencoder, self).get_loss()   

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
            axis=0)

        combined_loss = self.combine_loss_components(binary_loss, continuous_loss, kl_div_loss)

        return combined_loss, binary_loss, continuous_loss, kl_div_loss  
    
    def project_forward(self, train_df, years_to_move_forward, project_onto_mean=True):
        """
        given a df and an autoencoder model, projects the train_df down into Z-space, moves it 
        years_to_move_forward in Z-space, then projects it back up. 
        Note that Z here will be sampled stochastically if project_onto_mean = False
        and X will be sampled stochastically given Z. 
        Sampling X will introduce a lot of noise, so it is good for comparing distributions but maybe not for individual 
        aging trajectories (eg, in the longitudinal data). 
        """
        Z0 = self.get_projections(train_df, project_onto_mean=project_onto_mean)
        Z0_projected_forward = copy.deepcopy(Z0)
        for k in range(self.k_age):
            Z0_projected_forward['z%i' % k] = Z0_projected_forward['z%i' % k] + \
            self.model_learned_age_coefs[k] * years_to_move_forward
        projected_trajectory = self.sample_X_given_Z(remove_id_and_get_mat(Z0_projected_forward))
        assert projected_trajectory.shape[1] == len(self.feature_names)
        return projected_trajectory
