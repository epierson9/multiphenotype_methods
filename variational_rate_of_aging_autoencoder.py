from copy import deepcopy
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
        super(VariationalRateOfAgingAutoencoder, self).__init__(**kwargs)   
        self.k_age = k_age
        assert self.k >= self.k_age
        self.need_ages = True
        self.sparsity_weighting = sparsity_weighting
        self.can_calculate_Z_mu = False
        self.age_preprocessing_method = 'divide_by_a_constant' # important not to zero-mean age here. 
        # otherwise we end up with people with negative ages, which will mess up the interpretation of the aging rate. 
        # we divide by a constant to put age on roughly the same scale as the other features. 

        self.include_age_in_encoder_input = True
        self.aging_rate_scaling_factor = aging_rate_scaling_factor 
        # log_unscaled_aging_rate ~ N(0, 1)
        # Z_age = age * exp(self.aging_rate_scaling_factor * log_unscaled_aging_rate) 
        # so aging_rate_scaling_factor controls how much spread we have on the aging rate.
        
        # assert we only have a single decoder layer (otherwise the sparsity loss doesn't make sense). 
        if self.sparsity_weighting > 0:
            assert(len(self.decoder_layer_sizes) == 0)
        
    def sample_Z(self, age, n):
        # this function automatically applies the age preprocessing function to the passed in age
        # so there is no need to transform age ahead of time. 
        
        # sample age components.
        preprocessed_age = self.age_preprocessing_function(age)
        log_unscaled_aging_rate = np.random.multivariate_normal(mean = np.zeros([self.k_age,]), cov = np.eye(self.k_age), size = n)
        aging_rate = np.exp(self.aging_rate_scaling_factor * log_unscaled_aging_rate)
        Z_age = preprocessed_age * aging_rate
        
        # sample non-age components.
        k_non_age = self.k - self.k_age
        Z_non_age = np.random.multivariate_normal(mean = np.zeros([k_non_age,]), cov = np.eye(k_non_age), size = n)

        return np.concatenate([Z_age, Z_non_age], axis=1)
    
    def encode(self, X, ages):  
        # note that we use similar notation -- mu, sigma -- here as with the standard autoencoder 
        # and essentially the same procedure for generating both (with the same loss function)
        # but mu and sigma have different interpretations for the age components. 
        # they are the mean and std of the log_unscaled_aging_rate, not of Z. 
        # so we have Z_age = age * exp(self.aging_rate_scaling_factor * log_unscaled_aging_rate) 
        # where log_unscaled_aging_rate ~ N(0, 1). 
        
        
        # concatenate age onto X, since we use both to generate the posterior over Z.
        X_with_age = tf.concat([X, tf.reshape(ages, [-1, 1])], axis=1) # make age 2d. 
        
        num_layers = len(self.encoder_layer_sizes)
        # Get mu.  
        mu = X_with_age
        for idx in range(num_layers):
            mu = tf.matmul(mu, self.weights['encoder_h%i' % (idx)]) \
                + self.biases['encoder_b%i' % (idx)]
            # No non-linearity on the last layer
            if idx != num_layers - 1:
                mu = self.non_linearity(mu)
        self.encoder_mu = mu

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
        self.encoder_sigma = sigma

        # Sample from N(mu, sigma). The first k_age components are the log unscaled aging rate
        # the components after that are the residual (Z_non_age, just as before)
        self.eps = tf.random_normal(tf.shape(self.encoder_mu), 
                                    dtype=tf.float32, 
                                    mean=0., 
                                    stddev=1.0, 
                                    seed=self.random_seed)
        log_unscaled_aging_rate_plus_residual = self.encoder_mu + self.encoder_sigma * self.eps
        
        # generate age components. 
        # Exponentiate log aging rate and then multiply by age
        # define small helper method to do this. 
        def get_Z_age_from_aging_rate(log_unscaled_aging_rate):
            # small helper method: feeds a given unscaled aging rate through the exponential. 
            log_aging_rate = self.aging_rate_scaling_factor * log_unscaled_aging_rate
            aging_rate = tf.clip_by_value(tf.exp(log_aging_rate), 
                    clip_value_min=0, 
                    clip_value_max=50) # keep from exploding. We should never be near either of these bounds anyway.  
            Z_age = aging_rate * tf.reshape(ages, [-1, 1]) # relies on broadcasting to reshape age vector. 
            return Z_age
        
        log_unscaled_aging_rate = log_unscaled_aging_rate_plus_residual[:, :self.k_age]
        Z_age = get_Z_age_from_aging_rate(log_unscaled_aging_rate)
        
        # generate non-age components. This just means taking the last components of log_unscaled_aging_rate_plus_residual. 
        Z_non_age = log_unscaled_aging_rate_plus_residual[:, self.k_age:]
        Z = tf.concat([Z_age, Z_non_age], axis=1)
        
        # We may want to generate Z_mu and Z_sigma because other classes make use of them.  
        # both of these are calculable but a little convoluted; set to np.nan for now. 
        # https://en.wikipedia.org/wiki/Log-normal_distribution
        # nans should make it obvious if we are erroneously calling these values. 
        self.Z_mu = tf.zeros(tf.shape(Z)) * np.nan
        self.Z_sigma = tf.zeros(tf.shape(Z)) * np.nan
        
        return Z
    
    def init_network(self):
        """
        the only difference here is with the decoder, since we need to split out the age state and the residual. 
        """
        self.weights = {}
        self.biases = {} 
        
        if self.learn_continuous_variance:
            # we exponentiate this because it has to be non-negative. 
            self.log_continuous_variance = tf.Variable(self.initialization_function([1]))
        
        # Encoder layers -- the same.       
        for encoder_layer_idx, encoder_layer_size in enumerate(self.encoder_layer_sizes):
            if encoder_layer_idx == 0:
                input_dim = len(self.feature_names) + self.include_age_in_encoder_input # if we include age in input, need one extra feature. 
            else:
                input_dim = self.encoder_layer_sizes[encoder_layer_idx - 1]
            output_dim = self.encoder_layer_sizes[encoder_layer_idx]
            print("Added encoder layer with input dimension %i and output dimension %i" % (input_dim, output_dim))
            self.weights['encoder_h%i' % encoder_layer_idx] = tf.Variable(
                self.initialization_function([input_dim, output_dim]))
            self.biases['encoder_b%i' % encoder_layer_idx] = tf.Variable(
                self.initialization_function([output_dim]))
            self.weights['encoder_h%i_sigma' % encoder_layer_idx] = tf.Variable(
                self.initialization_function([input_dim, output_dim]))
            self.biases['encoder_b%i_sigma' % encoder_layer_idx] = tf.Variable(
                self.initialization_function([output_dim])) 

        # Decoder layers -- here, we need to split out the age state and the residual. 
        # so the decoder produces f(Z_age) + g(residual)
        
        self.decoder_layer_sizes.append(len(self.feature_names))
        for decoder_name in ['Z_age', 'residual']:
            for decoder_layer_idx, decoder_layer_size in enumerate(self.decoder_layer_sizes):
                if decoder_layer_idx == 0:
                    if decoder_name == 'Z_age':
                        input_dim = self.k_age
                    else:
                        input_dim = self.k - self.k_age
                else:
                    input_dim = self.decoder_layer_sizes[decoder_layer_idx - 1]
                output_dim = self.decoder_layer_sizes[decoder_layer_idx]
                print("Added decoder layer for %s with input dimension %i and output dimension %i" % (decoder_name, 
                                                                                                      input_dim, 
                                                                                                      output_dim))
                self.weights['decoder_%s_h%i' % (decoder_name, decoder_layer_idx)] = tf.Variable(
                    self.initialization_function([input_dim, output_dim]))
                self.biases['decoder_%s_b%i' % (decoder_name, decoder_layer_idx)] = tf.Variable(
                    self.initialization_function([output_dim]))
    
    def decode(self, Z):
        """
        we have to separately decode Z_age and the residual, then add them back together. 
        Decoding procedure is identical for both Z_age and residual although they use different weights. 
        """
        num_layers = len(self.decoder_layer_sizes)
        
        Z_age = Z[:, :self.k_age]
        residual = Z[:, self.k_age:]
        
        for idx in range(num_layers):
            Z_age = tf.matmul(Z_age, self.weights['decoder_Z_age_h%i' % idx]) \
                + self.biases['decoder_Z_age_b%i' % idx]
            residual = tf.matmul(residual, self.weights['decoder_residual_h%i' % idx]) \
                + self.biases['decoder_residual_b%i' % idx]
            # No non-linearity on the last layer
            if idx != num_layers - 1:
                Z_age = self.non_linearity(Z_age) 
                residual = self.non_linearity(residual)
        
        X_with_logits = Z_age + residual
        
        return X_with_logits
    
    def get_loss(self):
        # The KL loss is just the KL loss for N(0, I) computed on encoder_mu and encoder_sigma. 
        _, binary_loss, continuous_loss, _ = super(VariationalRateOfAgingAutoencoder, self).get_loss()   

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
            
        combined_loss = self.combine_loss_components(binary_loss, continuous_loss, regularization_loss)

        return combined_loss, binary_loss, continuous_loss, regularization_loss
    
    def get_rate_of_aging_plus_residual(self, Z_df, train_df):
        """
        helper method: given Z, divides by age to return the rate of aging plus the residual. 
        """
        rate_of_aging_plus_residual = deepcopy(Z_df)
        preprocessed_ages = self.get_ages(train_df)
        for k in range(self.k_age):
            rate_of_aging_plus_residual['z%i' % k] = rate_of_aging_plus_residual['z%i' % k] / preprocessed_ages
        return rate_of_aging_plus_residual

    def fast_forward_Z(self, Z0, train_df, years_to_move_forward):
        # get rate of aging. 
        rate_of_aging_plus_residual = self.get_rate_of_aging_plus_residual(Z0, train_df)
        
        # compute the fast-forwarded ages. 
        fastforwarded_ages = self.age_preprocessing_function(train_df['age_sex___age'] + years_to_move_forward)
        
        # project Z forward. 
        Z0_projected_forward = deepcopy(Z0)
        for k in range(self.k_age):
            Z0_projected_forward['z%i' % k] = rate_of_aging_plus_residual['z%i' % k] * fastforwarded_ages
            
        return Z0_projected_forward
