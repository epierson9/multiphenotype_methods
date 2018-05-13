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
                
    def set_up_encoder_structure(self):
        """
        This function sets up the basic encoder structure and return arguments. 
        We need to return Z, Z_mu, and Z_sigma. 
        """
        self.Z, self.Z_mu, self.Z_sigma = self.encode(self.X) 
        
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
        Z_mu = mu

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
        Z_sigma = sigma

        # Sample from N(mu, sigma)
        eps = tf.random_normal(tf.shape(Z_mu), dtype=tf.float32, mean=0., stddev=1.0, seed=self.random_seed)
        Z = Z_mu + Z_sigma * eps        
        return Z, Z_mu, Z_sigma
    
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
        Important note: in general, this function relies on sample_Z, which automatically applies the age preprocessing function
        to the passed in age, so there is no need to transform age ahead of time (for either sample_X or sample_Z). 
        """
        Z = self.sample_Z(age, n)
        return self.sample_X_given_Z(Z)
    
    def sample_Z(self, age, n):
        return np.random.multivariate_normal(mean = np.zeros([self.k,]), cov = np.eye(self.k), size = n)
       
    def set_up_regularization_loss_structure(self):
        self.reg_loss = self.get_regularization_loss(self.Z_mu, self.Z_sigma)
        
    def get_regularization_loss(self, Z_mu, Z_sigma):
        kl_div_loss = -.5 * (
            1 + 
            2 * tf.log(Z_sigma) - tf.square(Z_mu) - tf.square(Z_sigma))
        kl_div_loss = tf.reduce_mean(
            tf.reduce_sum(
                kl_div_loss,
                axis=1),
            axis=0)
        return kl_div_loss
    
    def project_forward(self, train_df, years_to_move_forward, add_noise_to_Z, add_noise_to_X):
        """
        given a df and an autoencoder model, projects the train_df down into Z-space, moves it 
        years_to_move_forward in Z-space, then projects it back up. years_to_move_forward can be an array or a scalar. 
        
        This will not make sense unless the model has some notion of an age state and how it evolves, 
        so to implement this method, you need to implement fast_forward_Z. 
        
        if add_noise_to_Z is False, Z is projected onto the mean; otherwise, it's sampled. 
        if add_noise_to_X is False, X is decoded directly from Z (ie, it is Xr); otherwise, it's sampled. 
        """
        
        # cast years_to_move_forward to an array (I think this should be fine even if it is a scalar?)
        years_to_move_forward = np.array(years_to_move_forward)
        
        # project down to latent state. 
        if add_noise_to_Z:
            Z0 = self.get_projections(train_df, project_onto_mean=False)
        else:
            Z0 = self.get_projections(train_df, project_onto_mean=True)
        
        if (years_to_move_forward == 0).all():
            # if we're not moving forward at all, Z0 is just Z. This is equivalent to reconstruction. 
            # we shouldn't actually need this if-branch, but it makes what is happening a little more explicit. 
            Z0_projected_forward = remove_id_and_get_mat(Z0)
        else:
            # move age components forward following the model's evolution rule. 
            Z0_projected_forward = self.fast_forward_Z(Z0, train_df, years_to_move_forward)
            Z0_projected_forward = remove_id_and_get_mat(Z0_projected_forward)
            
        # sample X again. 
        if add_noise_to_X:
            projected_trajectory = self.sample_X_given_Z(Z0_projected_forward)
        else:
            projected_trajectory = self.sess.run(self.Xr, feed_dict = {self.Z:Z0_projected_forward})
            
        assert projected_trajectory.shape[1] == len(self.feature_names)
        return projected_trajectory
    
    
    def project_forward_by_sampling_Z_and_then_sampling_X(self, train_df, years_to_move_forward):
        """
        alternate way of projecting forward: sample Z from p(Z | X), then sample X. Not using this at present. 
        """
        # cast years_to_move_forward to an array (I think this should be fine even if it is a scalar?)
        years_to_move_forward = np.array(years_to_move_forward)
        
        n_iterates = 100
        
        for i in range(n_iterates):
            Z0 = self.get_projections(train_df, project_onto_mean=False)
        
            if (years_to_move_forward == 0).all():
                # if we're not moving forward at all, Z0 is just Z. This is equivalent to reconstruction. 
                # we shouldn't actually need this if-branch, but it makes what is happening a little more explicit. 
                Z0_projected_forward = remove_id_and_get_mat(Z0)
            else:
                # move age components forward following the model's evolution rule. 
                Z0_projected_forward = self.fast_forward_Z(Z0, train_df, years_to_move_forward)
                Z0_projected_forward = remove_id_and_get_mat(Z0_projected_forward)
            
            sampled_X = self.sample_X_given_Z(Z0_projected_forward)
            if i == 0:
                projected_trajectory = sampled_X
            else:
                projected_trajectory = projected_trajectory + sampled_X

          
            assert projected_trajectory.shape[1] == len(self.feature_names)
        
        return projected_trajectory / n_iterates
    
    def fast_forward_Z(self, Z0, train_df, years_to_move_forward):
        """
        given a Z, evolve it following the model's aging rule for years_to_move_forward years. 
        """
        raise NotImplementedException
        
        
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
