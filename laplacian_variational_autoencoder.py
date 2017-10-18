import numpy as np
from multiphenotype_utils import (get_continuous_features_as_matrix, add_id, remove_id_and_get_mat, 
    partition_dataframe_into_binary_and_continuous, divide_idxs_into_batches)
import pandas as pd
import tensorflow as tf
from dimreducer import DimReducer

from general_autoencoder import GeneralAutoencoder
from standard_autoencoder import StandardAutoencoder

class VariationalLaplacianAutoencoder(StandardAutoencoder):
    """
    Implements a variational autoencoder with independent Laplacian priors. 
    This code is identical to the Gaussian variational except where explicitly noted in comments. 
    """    
    def __init__(self, 
                 kl_weighting = 1,
                 **kwargs):

        super(VariationalLaplacianAutoencoder, self).__init__(**kwargs)   
        # Does not include input_dim, but includes last hidden layer
        # self.encoder_layer_sizes = encoder_layer_sizes
        # self.k = self.encoder_layer_sizes[-1]        

        # self.decoder_layer_sizes = decoder_layer_sizes

        self.initialization_function = self.glorot_init
        self.kl_weighting = kl_weighting
        self.non_linearity = tf.nn.sigmoid
        self.sigma_scaling = .1

    def init_network(self):
        self.weights = {}
        self.biases = {}        
        
        # Encoder layers.         
        for encoder_layer_idx, encoder_layer_size in enumerate(self.encoder_layer_sizes):
            if encoder_layer_idx == 0:
                input_dim = len(self.feature_names)
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

        # Get sigma (often called b, but we call it sigma to avoid renaming everything). 
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
        # Important: this deviates from the standard Gaussian autoencoder
        # Sample from Laplacian(mu, sigma). 
        # See https://en.wikipedia.org/wiki/Laplace_distribution#Generating_random_variables_according_to_the_Laplace_distribution
        # Z = mu - b * sgn(eps) * ln(1 - 2|eps|) where eps ~ U(-.5, .5)
        # add in small constant (1e-8) for numerical stability in sampling; otherwise log can explode. 

        self.eps = tf.random_uniform(tf.shape(self.Z_mu), 
                                     dtype=tf.float32, 
                                     minval=-.5, 
                                     maxval=.5,
                                     seed=self.random_seed)
        Z = self.Z_mu - self.Z_sigma * tf.sign(self.eps) * tf.log(1 - 2 * tf.abs(self.eps) + 1e-8) 
        return Z


    def get_loss(self):
        """
        Uses self.X, self.Xr, self.Z_sigma, self.Z_mu, self.kl_weighting
        """
        _, binary_loss, continuous_loss, _ = super(VariationalLaplacianAutoencoder, self).get_loss()   
        
        # Important: this deviates from the standard Gaussian autoencoder
        # We assume that the prior is a Laplacian with sigma = 1, mu = 0.
        # to compute q log p: https://www.wolframalpha.com/input/?i=integrate++1+%2F+(2+*+pi)+*+exp(-abs(x+-+7)+%2F+pi)+*+(-abs(x)+%2F+1)+from+-infinity+to+infinity
        # to compute q log q: https://www.wolframalpha.com/input/?i=integrate++1+%2F+(2+*+pi)+*+exp(-abs(x+-+7)+%2F+pi)+*+(-abs(x+-+7)+%2F+pi)+from+-infinity+to+infinity
        # We want to compute KL(Q, P) and we have 
        # mu_p = 0, sigma_p = 1. Then: 
        # KL(Q, P) = -log(sigma) - 1 - (- abs(mu) - sigma * exp(-abs(mu) / sigma))
        # KL(Q, P) = -log(sigma) - 1 + abs(mu) + sigma * exp(-abs(mu) / sigma)
        # which vanishes, as it should, when sigma = 1, mu = 0. 
        
        kl_div_loss = -tf.log(self.Z_sigma) - 1 + tf.abs(self.Z_mu) + self.Z_sigma * tf.exp(tf.abs(self.Z_mu) / self.Z_sigma)
        kl_div_loss = tf.reduce_mean(
            tf.reduce_sum(
                kl_div_loss,
                axis=1),
            axis=0) * self.kl_weighting
        kl_div_loss = kl_div_loss
        combined_loss = binary_loss + continuous_loss + kl_div_loss

        return combined_loss, binary_loss, continuous_loss, kl_div_loss  


    def compute_elbo(self, df, continuous_variance=1):
        data, binary_feature_idxs, continuous_feature_idxs, feature_names = \
            partition_dataframe_into_binary_and_continuous(df)
        assert np.all(binary_feature_idxs == self.binary_feature_idxs)
        assert np.all(continuous_feature_idxs == self.continuous_feature_idxs)
        assert np.all(feature_names == self.feature_names)

        print(("Computing ELBO with %i continuous features, %i binary features, "
            "%i examples, continuous variance = %2.3f") %
              (len(continuous_feature_idxs), 
               len(binary_feature_idxs), 
               len(data), 
               continuous_variance))

        # https://www.statlect.com/fundamentals-of-statistics/normal-distribution-maximum-likelihood
        num_iter = 1
        mean_combined_loss = 0
        mean_binary_loss = 0
        mean_continuous_loss = 0
        mean_reg_loss = 0
        for i in range(num_iter):              
            combined_loss, binary_loss, continuous_loss, reg_loss = self.minibatch_mean_eval(data)
            mean_combined_loss += combined_loss / num_iter
            mean_binary_loss += binary_loss / num_iter
            mean_continuous_loss += continuous_loss / num_iter
            mean_reg_loss += reg_loss / num_iter
                    
        
        constant_offset_per_sample_and_feature = .5 * np.log(2 * np.pi) + .5 * np.log(continuous_variance)
        mean_continuous_loss = constant_offset_per_sample_and_feature * len(continuous_feature_idxs) + mean_continuous_loss / continuous_variance
        mean_combined_loss = mean_binary_loss + mean_continuous_loss + mean_reg_loss
        elbo = -mean_combined_loss
        print("Average ELBO per sample = %2.3f" % elbo)
        return elbo
