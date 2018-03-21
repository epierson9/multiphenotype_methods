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


class SparseCorrelationVariationalAgeAutoencoder(VariationalAgeAutoencoder):
    """
    Implements a variational autoencoder with an age prior and sparsity on the X-Z correlation matrix.
    """    
    def __init__(self,
                 k_age,
                 Z_age_coef,
                 sparsity_weighting = .1,
                 batch_size=512,
                 min_corr_value = .05,
                 use_age_adjusted_X=True,
                 **kwargs):

        super(SparseCorrelationVariationalAgeAutoencoder, self).__init__(k_age = k_age, 
                                                        Z_age_coef = Z_age_coef, 
                                                        batch_size=batch_size,
                                                        **kwargs)   
        self.sparsity_weighting = sparsity_weighting # weighting on the L1 X-Z correlation matrix loss. 
        self.use_age_adjusted_X = use_age_adjusted_X # if True, computes correlations with the age state using X that has been decorrelated with age. 
        self.min_corr_value = min_corr_value # correlations below this value are treated as equivalent. 
        
    def compute_pearson_correlation(self, v1, v2):
        """
        slow (non-vectorized) way of computing the pearson correlation. 
        pearson correlation:
        https://en.wikipedia.org/wiki/Correlation_and_dependence
        Not being used at present .
        Verified that this matches up with pearsonr(x, y) for random vectors. 
        """
        # The mean and variance are calculated by aggregating the contents of x across axes. 
        # If x is 1-D and axes = [0] this is just the mean and variance of a vector.
        mu_1, variance_1 = tf.nn.moments(v1, axes=[0])
        mu_2, variance_2 = tf.nn.moments(v2, axes=[0])
        
        sigma_1 = tf.sqrt(variance_1)
        sigma_2 = tf.sqrt(variance_2)
        pearsonr = tf.reduce_mean((v1 - mu_1) * (v2 - mu_2)) / (sigma_1 * sigma_2)
        return pearsonr
    
    def compute_correlation_sparsity_loss(self, Z, X):
        """
        this is a vectorized version of the above function which is faster but equivalent. 
        Verified that it agrees with pearsonr on random matrices. 
        """
        mu_X, variance_X = tf.nn.moments(X, axes=[0])
        mu_Z, variance_Z = tf.nn.moments(Z, axes=[0])
        std_X = tf.reshape(tf.sqrt(variance_X), shape=[len(self.feature_names), 1])
        std_Z = tf.reshape(tf.sqrt(variance_Z), shape=[1, tf.shape(variance_Z)[0]])
        
        zero_mean_X = X - mu_X # this subtracts off the mean of each column of X. 
        zero_mean_Z = Z - mu_Z # similarly for Z. 
        n_samples = tf.cast(tf.shape(X)[0], tf.float32)
        expected_product = tf.matmul(tf.transpose(zero_mean_X), zero_mean_Z) / n_samples
        product_of_stds = tf.matmul(std_X, std_Z)
        pearsonr_matrix = expected_product / product_of_stds
        clipped_pearsonr_matrix = tf.clip_by_value(tf.abs(pearsonr_matrix), self.min_corr_value, np.inf)
        sparsity_loss = tf.reduce_sum(clipped_pearsonr_matrix)
        return sparsity_loss
    
    def get_loss(self):
        """
        Adds a correlation sparsity loss to the regularization term. 
        """
        _, binary_loss, continuous_loss, kl_div_loss = super(SparseCorrelationVariationalAgeAutoencoder, self).get_loss()   
        
        if self.use_age_adjusted_X:
            # for non-age states, use correlation with X to compute sparsity loss. 
            # for age states, use correlation with age_adjusted_X. 
            sparsity_loss = self.compute_correlation_sparsity_loss(self.Z[:, self.k_age:], self.X)
            if self.k_age > 0:
                sparsity_loss += self.compute_correlation_sparsity_loss(self.Z[:, :self.k_age], self.age_adjusted_X)
        else:
            # if we're not using age adjusted X for the age states, just compute the sparse correlation matrix with X. 
            sparsity_loss = self.compute_correlation_sparsity_loss(self.Z, self.X)
        
        regularization_loss = kl_div_loss + sparsity_loss * self.sparsity_weighting
        combined_loss = self.combine_loss_components(binary_loss, continuous_loss, regularization_loss)

        return combined_loss, binary_loss, continuous_loss, regularization_loss  
