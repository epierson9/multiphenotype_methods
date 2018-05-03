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
                 preset_aging_rate_scaling_factor=.1,
                 learn_aging_rate_scaling_factor_from_data=False,
                 age_preprocessing_method='divide_by_a_constant',
                 weight_constraint_implementation=None,
                 # for rate of aging autoencoders we default to starting age at (approximately) 0 because
                 # it seems safer to only assume linear movement through Z-space over the range where we have data. 
                 **kwargs):
        super(VariationalRateOfAgingAutoencoder, self).__init__(is_rate_of_aging_model=True,
                                                                age_preprocessing_method=age_preprocessing_method,
                                                                **kwargs)   
        self.k_age = k_age
        assert self.k >= self.k_age
        assert weight_constraint_implementation in [None, 'take_absolute_value']
        self.need_ages = True
        self.sparsity_weighting = sparsity_weighting
        self.can_calculate_Z_mu = True
        self.weight_constraint_implementation = weight_constraint_implementation
        self.include_age_in_encoder_input = True
        
        if self.weight_constraint_implementation is None:
            self.weight_preprocessing_fxn = tf.identity
        elif self.weight_constraint_implementation == 'take_absolute_value':
            self.weight_preprocessing_fxn = tf.abs
        print("Weight constraint method is %s" % self.weight_constraint_implementation)
        
        # we can either preset the aging_rate_scaling_factor or learn it from the data; ensure we're only doing one of these. 
        if learn_aging_rate_scaling_factor_from_data:
            assert preset_aging_rate_scaling_factor is None
        else:
            assert preset_aging_rate_scaling_factor is not None
        self.learn_aging_rate_scaling_factor_from_data = learn_aging_rate_scaling_factor_from_data
        self.preset_aging_rate_scaling_factor = preset_aging_rate_scaling_factor 
        
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
        if self.learn_aging_rate_scaling_factor_from_data:
            aging_rate_scaling_factor = self.sess.run(self.aging_rate_scaling_factor)[0]
        else:
            aging_rate_scaling_factor = self.aging_rate_scaling_factor
            
        aging_rate = np.exp(aging_rate_scaling_factor * log_unscaled_aging_rate)
        Z_age = preprocessed_age * aging_rate
        
        # sample non-age components.
        k_non_age = self.k - self.k_age
        Z_non_age = np.random.multivariate_normal(mean = np.zeros([k_non_age,]), cov = np.eye(k_non_age), size = n)

        return np.concatenate([Z_age, Z_non_age], axis=1)
    
    def set_up_encoder_structure(self):
        self.Z, self.Z_mu, self.Z_sigma, self.encoder_mu, self.encoder_sigma = self.encode(self.X, self.ages)
        
    def set_up_regularization_loss_structure(self):
        self.reg_loss = self.get_regularization_loss(self.encoder_mu, self.encoder_sigma) 
        
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
        encoder_mus = {} # stores one mu for age state, one for residual.
        for encoder_name in ['Z_age', 'residual']:
            mu = X_with_age
            for idx in range(num_layers):
                W = self.weights['encoder_%s_h%i' % (encoder_name, idx)]
                if encoder_name == 'Z_age':
                    W = self.weight_preprocessing_fxn(W) # constrain weights on encoder means. 
                
                mu = tf.matmul(mu, W) + self.biases['encoder_%s_b%i' % (encoder_name, idx)]
                # No non-linearity on the last layer
                if idx != num_layers - 1:
                    mu = self.non_linearity(mu)
            encoder_mus[encoder_name] = mu
        encoder_mu = tf.concat([encoder_mus['Z_age'], encoder_mus['residual']], axis=1)

        # Get sigma
        encoder_sigmas = {}
        for encoder_name in ['Z_age', 'residual']:
            sigma = X_with_age
            for idx in range(num_layers):
                sigma = tf.matmul(sigma, 
                                  self.weights['encoder_%s_h%i_sigma' % (encoder_name, idx)]) \
                    + self.biases['encoder_%s_b%i_sigma' % (encoder_name, idx)]
                # No non-linearity on the last layer
                if idx != num_layers - 1:
                    sigma = self.non_linearity(sigma)
            sigma = sigma * self.sigma_scaling # scale so sigma doesn't explode when we exponentiate it. 
            sigma = tf.exp(sigma)
            encoder_sigmas[encoder_name] = sigma
        encoder_sigma = tf.concat([encoder_sigmas['Z_age'], encoder_sigmas['residual']], axis=1)

        # Sample from N(mu, sigma). The first k_age components are the log unscaled aging rate
        # the components after that are the residual (Z_non_age, just as before)
        eps = tf.random_normal(tf.shape(encoder_mu), 
                                    dtype=tf.float32, 
                                    mean=0., 
                                    stddev=1.0, 
                                    seed=self.random_seed)
        log_unscaled_aging_rate_plus_residual = encoder_mu + encoder_sigma * eps
        
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
        
        # We generate Z_mu and Z_sigma because other classes make use of them.  
        # both of these are calculable but a little convoluted.
        
        # for age components, we use the expressions here: https://en.wikipedia.org/wiki/Log-normal_distribution
        # first we compute the parameters of the log-normal distribution, which requires multiplying by the scaling factor and by age. 
        log_normal_mu_parameter = encoder_mu[:, :self.k_age] * self.aging_rate_scaling_factor
        log_normal_sigma_parameter = encoder_sigma[:, :self.k_age] * self.aging_rate_scaling_factor
        
        # then we plug those in to calculate the mean and sigma of the rate of aging. 
        rate_of_aging_mu = tf.exp(log_normal_mu_parameter + log_normal_sigma_parameter**2/2.0)
        rate_of_aging_sigma = tf.sqrt(
            (tf.exp(log_normal_sigma_parameter**2) - 1) * tf.exp(2*log_normal_mu_parameter + log_normal_sigma_parameter**2))
        
        # finally, multiply by age to get age_Z_mu and age_Z_sigma (since age_Z = rate_of_aging * age)
        age_Z_mu = rate_of_aging_mu * tf.reshape(ages, [-1, 1])
        age_Z_sigma = rate_of_aging_sigma * tf.reshape(ages, [-1, 1])
        
        # for non-age components, Z_mu and Z_sigma are just encoder_mu and encoder_sigma, as before. 
        # Concatenate to produce the final tensors. 
        Z_mu = tf.concat([age_Z_mu, encoder_mu[:, self.k_age:]], axis=1)
        Z_sigma = tf.concat([age_Z_sigma, encoder_sigma[:, self.k_age:]], axis=1)
        
        return Z, Z_mu, Z_sigma, encoder_mu, encoder_sigma
    
    def init_network(self):
        """
        the only difference here is with the decoder, since we need to split out the age state and the residual. 
        """
        self.weights = {}
        self.biases = {} 
        
        if self.learn_aging_rate_scaling_factor_from_data:
            # we exponentiate this because it has to be non-negative. 
            print("Learning aging rate scaling factor from data.")
            self.log_aging_rate_scaling_factor = tf.Variable(tf.random_normal(shape=[1], 
                                                                              mean=-2,
                                                                              stddev=.1,
                                                                              seed=self.random_seed))
            self.aging_rate_scaling_factor = tf.exp(self.log_aging_rate_scaling_factor)
        else:
            print("Setting aging rate scaling factor to %2.3f" % self.preset_aging_rate_scaling_factor)
            self.aging_rate_scaling_factor = self.preset_aging_rate_scaling_factor
        
        if self.learn_continuous_variance:
            # we exponentiate this because it has to be non-negative. 
            self.log_continuous_variance = tf.Variable(self.initialization_function([1]))
        
        # Encoder layers -- the same. 
        for encoder_name in ['Z_age', 'residual']:
            for encoder_layer_idx, encoder_layer_size in enumerate(self.encoder_layer_sizes):
                # require a special case for the first layer input size
                if encoder_layer_idx == 0:
                    input_dim = len(self.feature_names) + self.include_age_in_encoder_input # if we include age in input, need one extra feature. 
                else:
                    input_dim = self.encoder_layer_sizes[encoder_layer_idx - 1]
                
                # also require a special case for the last encoder layer
                # depending on whether it is the age encoder or the residual encoder. 
                if encoder_layer_idx == len(self.encoder_layer_sizes) - 1: 
                    if encoder_name == 'Z_age':
                        output_dim = self.k_age
                    else:
                        output_dim = self.k - self.k_age
                else:
                    output_dim = self.encoder_layer_sizes[encoder_layer_idx]
                print("Added encoder layer for %s with input dimension %i and output dimension %i" % (encoder_name, 
                                                                                                      input_dim, 
                                                                                                      output_dim))
                self.weights['encoder_%s_h%i' % (encoder_name, encoder_layer_idx)] = tf.Variable(
                    self.initialization_function([input_dim, output_dim]))
                self.biases['encoder_%s_b%i' % (encoder_name, encoder_layer_idx)] = tf.Variable(
                    self.initialization_function([output_dim]))
                self.weights['encoder_%s_h%i_sigma' % (encoder_name, encoder_layer_idx)] = tf.Variable(
                    self.initialization_function([input_dim, output_dim]))
                self.biases['encoder_%s_b%i_sigma' % (encoder_name, encoder_layer_idx)] = tf.Variable(
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
            Z_age = tf.matmul(Z_age, 
                              self.weight_preprocessing_fxn(self.weights['decoder_Z_age_h%i' % idx])) \
                + self.biases['decoder_Z_age_b%i' % idx]
                
            # no weight constraints on residual decoder. 
            residual = tf.matmul(residual, self.weights['decoder_residual_h%i' % idx]) \
                + self.biases['decoder_residual_b%i' % idx]
            # No non-linearity on the last layer
            if idx != num_layers - 1:
                Z_age = self.non_linearity(Z_age) 
                residual = self.non_linearity(residual)
        
        X_with_logits = Z_age + residual
        
        return X_with_logits
    
    def get_regularization_loss(self, encoder_mu, encoder_sigma):
        # Pull this out into a method because subclasses use it. 
        kl_div_loss = -.5 * (
            1 + 
            2 * tf.log(encoder_sigma) - tf.square(encoder_mu) - tf.square(encoder_sigma))
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
        return regularization_loss
    
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
        fastforwarded_ages = self.age_preprocessing_function(train_df['age_sex___age'] + np.array(years_to_move_forward))
        
        # project Z forward. 
        Z0_projected_forward = deepcopy(Z0)
        for k in range(self.k_age):
            Z0_projected_forward['z%i' % k] = np.array(rate_of_aging_plus_residual['z%i' % k]) * fastforwarded_ages
            
        return Z0_projected_forward
