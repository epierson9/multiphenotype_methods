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


class MortalityWeightedVariationalAgeAutoencoder(VariationalAgeAutoencoder):
    """
    Implements a variational autoencoder with an age prior 
    that weights features by how much they predict mortality. 
    """    
    def __init__(self,
                 k_age,
                 Z_age_coef,
                 mortality_weighting_dict,
                 **kwargs):

        super(MortalityWeightedVariationalAgeAutoencoder, self).__init__(k_age = k_age, 
                                                        Z_age_coef = Z_age_coef, 
                                                        **kwargs)   
        # mortality_weighting_dict should be a dictionary whose keys are features and values are weights. 
        self.mortality_weighting_dict = mortality_weighting_dict

    def get_loss(self, X, Xr):
        """
        For each feature x_i, the loss is abs(beta_i * (Xr_i - X_i)). For binary features, Xr is \in [0, 1] 
        (ie, it is the predicted probability of the feature being on). The betas are given by the mortality weighting dict
        and describe each feature's contribution to mortality, but one could imagine some other way of weighting. 
        Unfortunately, this doesn't seem to work very well for binary features: very hard to train because the logistics saturate. 
        """
        # TODO: should probably move these two lines into the data preprocessing function somehow. 
        continuous_feature_mortality_weights = np.atleast_2d( \
            [np.abs(self.mortality_weighting_dict[self.feature_names[i]]) for i in self.continuous_feature_idxs]).transpose().astype(np.float32)
        binary_feature_mortality_weights = np.atleast_2d( \
            [np.abs(self.mortality_weighting_dict[self.feature_names[i]]) for i in self.binary_feature_idxs]).transpose().astype(np.float32)
        
        # partition reconstruction into binary and continuous features as usual. 
        X_binary, X_continuous = self.split_into_binary_and_continuous(X)
        Xr_logits, Xr_continuous = self.split_into_binary_and_continuous(Xr)
        
        # feed them logits through a logistic transform to put them on the probability scale. 
        Xr_binary_probabilities = tf.nn.sigmoid(Xr_logits)

        # now compute losses: we weight the errors by how much they affect the mortality prediction. 
        continuous_loss = tf.reduce_mean(tf.matmul(tf.abs(X_continuous - Xr_continuous), continuous_feature_mortality_weights))
        binary_loss = tf.reduce_mean(tf.matmul(tf.abs(X_binary - Xr_binary_probabilities), binary_feature_mortality_weights))

        binary_loss = binary_loss * self.binary_loss_weighting
   
        # KL div loss comes from the variational age autoencoder. 
        _, _, _, kl_div_loss = super(MortalityWeightedVariationalAgeAutoencoder, self).get_loss()
        combined_loss = self.combine_loss_components(binary_loss, continuous_loss, kl_div_loss)

        return combined_loss, binary_loss, continuous_loss, kl_div_loss  
