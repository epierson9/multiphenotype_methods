import numpy as np
from multiphenotype_utils import (get_continuous_features_as_matrix, add_id, remove_id_and_get_mat, 
    partition_dataframe_into_binary_and_continuous, divide_idxs_into_batches)
import pandas as pd
import tensorflow as tf
from dimreducer import DimReducer
from scipy.special import expit

from general_autoencoder import GeneralAutoencoder
from standard_autoencoder import StandardAutoencoder

class VariationalAutoencoder(StandardAutoencoder):
    """
    Implements a standard variational autoencoder (diagonal Gaussians everywhere).
    """    
    def __init__(self, 
                 **kwargs):
        super(VariationalAutoencoder, self).__init__(**kwargs)   
        self.sigma_scaling = .1 # keeps the sigmas (ie, the std of the normal from which Z is drawn) from getting too large. 
        
        
    def init_network(self):
        self.weights = {}
        self.biases = {} 
        
        if self.learn_continuous_variance:
            # we exponentiate this because it has to be non-negative. 
            self.log_continuous_variance = tf.Variable(self.initialization_function([1]))
        
        # Encoder layers.         
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

        # Decoder layers. 
        self.decoder_layer_sizes.append(len(self.feature_names))
        for decoder_layer_idx, decoder_layer_size in enumerate(self.decoder_layer_sizes):
            if decoder_layer_idx == 0:
                input_dim = self.k
            else:
                input_dim = self.decoder_layer_sizes[decoder_layer_idx - 1]
            output_dim = self.decoder_layer_sizes[decoder_layer_idx]
            print("Added decoder layer with input dimension %i and output dimension %i" % (input_dim, output_dim))
            self.weights['decoder_h%i' % decoder_layer_idx] = tf.Variable(
                self.initialization_function([input_dim, output_dim]))
            self.biases['decoder_b%i' % decoder_layer_idx] = tf.Variable(
                self.initialization_function([output_dim]))
                

    def encode(self, X):          
        num_layers = len(self.encoder_layer_sizes)
        # Get mu 
        mu = X
        for idx in range(num_layers):
            mu = tf.matmul(mu, self.weights['encoder_h%i' % (idx)]) \
                + self.biases['encoder_b%i' % (idx)]
            # No non-linearity on the last layer
            if idx != num_layers - 1:
                mu = self.non_linearity(mu)
        self.Z_mu = mu

        # Get sigma
        sigma = X
        for idx in range(num_layers):
            sigma = tf.matmul(sigma, self.weights['encoder_h%i_sigma' % (idx)]) \
                + self.biases['encoder_b%i_sigma' % (idx)]
            # No non-linearity on the last layer
            if idx != num_layers - 1:
                sigma = self.non_linearity(sigma)
        sigma = sigma * self.sigma_scaling # scale so sigma doesn't explode when we exponentiate it. 
        sigma = tf.exp(sigma)
        self.Z_sigma = sigma

        # Sample from N(mu, sigma)
        self.eps = tf.random_normal(tf.shape(self.Z_mu), dtype=tf.float32, mean=0., stddev=1.0, seed=self.random_seed)
        Z = self.Z_mu + self.Z_sigma * self.eps        
        return Z
    
    def sample_X_given_Z(self, Z):
        """
        given a Z, samples X. Adds noise for both binary and continuous features following the autoencoder model. 
        """
        Xr = self.sess.run(self.Xr, feed_dict = {self.Z:Z})
        # for binary features, need to convert logits to 1s and 0s by sampling. 
        Xr[:, self.binary_feature_idxs] = np.random.random(Xr[:, self.binary_feature_idxs].shape) < \
                                                           expit(Xr[:, self.binary_feature_idxs])
        # for continuous features, need to add noise. 
        if self.learn_continuous_variance:
            continuous_variance = np.exp(self.sess.run(self.log_continuous_variance)[0])
            std = np.sqrt(continuous_variance)
        else:
            std = 1

        Xr[:, self.continuous_feature_idxs] = Xr[:, self.continuous_feature_idxs] + \
        np.random.normal(loc=0, scale=std, size=Xr[:, self.continuous_feature_idxs].shape)
        return Xr
    
    def sample_X(self, age, n):
        """
        samples X by first sampling Z from the autoencoder prior, then feeding it through the model. 
        Draws n samples for people of a given age. 
        Important note: in our age autoencoder formulation, age is zero-centered 
        (ie, we train the model with ages whose mean has been subtracted off). 
        So you probably want the age you pass in to account for that. 
        """
        Z = self.sample_Z(age, n)
        return self.sample_X_given_Z(Z)
    
    def sample_Z(self, age, n):
        return np.random.multivariate_normal(mean = np.zeros([self.k,]), cov = np.eye(self.k), size = n)
        
    def get_loss(self):
        """
        Uses self.X, self.Xr, self.Z_sigma, self.Z_mu, self.kl_weighting
        """
        _, binary_loss, continuous_loss, _ = super(VariationalAutoencoder, self).get_loss()   

        kl_div_loss = -.5 * (
            1 + 
            2 * tf.log(self.Z_sigma) - tf.square(self.Z_mu) - tf.square(self.Z_sigma))
        kl_div_loss = tf.reduce_mean(
            tf.reduce_sum(
                kl_div_loss,
                axis=1),
            axis=0)

        combined_loss = self.combine_loss_components(binary_loss, continuous_loss, kl_div_loss)

        return combined_loss, binary_loss, continuous_loss, kl_div_loss  


    def compute_elbo(self, df, continuous_variance=1):
        if self.learn_continuous_variance:
            continuous_variance = np.exp(self.sess.run(self.log_continuous_variance)[0])
            print("Warning: ignoring continuous variance input because we have already learned continuous variance: %2.3f" % continuous_variance)
            
        data, binary_feature_idxs, continuous_feature_idxs, feature_names = \
            partition_dataframe_into_binary_and_continuous(df)
        ages = None
        age_adjusted_data = None
        if self.need_ages:
            # in general, self.need_ages is True for the variational age autoencoders 
            # (most of our interesting models). It is False for models that have nothing to do with age: 
            # eg, the simple variational autoencoder. 
            ages = self.get_ages(df)
            # some models additionally require us to compute the age_adjusted_data, so we compute that just in case. 
            # eg, if we want to enforce sparse correlations between X (adjusted for age) and Z. 
            # age_adjusted_data is not actually used for most models. 
            age_adjusted_data = self.decorrelate_data_with_age(data, ages)            
        
        assert np.all(binary_feature_idxs == self.binary_feature_idxs)
        assert np.all(continuous_feature_idxs == self.continuous_feature_idxs)
        assert np.all(feature_names == self.feature_names)

        print(("Computing ELBO with %i continuous features, %i binary features, "
            "%i examples, continuous variance = %2.3f") %
              (len(continuous_feature_idxs), 
               len(binary_feature_idxs), 
               len(data), 
               continuous_variance))
 
        num_iter = 1 # set to more than one if you want to average runs together. 
        mean_binary_loss = 0
        mean_continuous_loss = 0
        mean_reg_loss = 0

        for i in range(num_iter):              
            _, binary_loss, continuous_loss, reg_loss = self.minibatch_mean_eval(data, 
                                                                                 ages,
                                                                                 age_adjusted_data=age_adjusted_data,
                                                                                 regularization_weighting=1)
                                                                                 
            mean_binary_loss += binary_loss / num_iter
            mean_continuous_loss += continuous_loss / num_iter
            mean_reg_loss += reg_loss / num_iter
                    
        # https://www.statlect.com/fundamentals-of-statistics/normal-distribution-maximum-likelihood
        if not self.learn_continuous_variance:
            # in this case, we have to add in the variance term since the mean_continuous_loss is just a squared-error term.
            # if we learn the variance, the continuous loss is actually the full negative Gaussian log likelihood
            # and there is no correction needed. 
            constant_offset_per_sample_and_feature = .5 * np.log(2 * np.pi) + .5 * np.log(continuous_variance)
            mean_continuous_loss = constant_offset_per_sample_and_feature * len(continuous_feature_idxs) + mean_continuous_loss / continuous_variance
            
        # note that we compute the elbo using a weight of 1 for the regularization loss regardless of the regularization weighting. 
        mean_combined_loss = mean_binary_loss + mean_continuous_loss + mean_reg_loss
        elbo = -mean_combined_loss
        print("Average ELBO per sample = %2.3f" % elbo)
        return elbo
