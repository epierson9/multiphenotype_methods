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
from variational_rate_of_aging_autoencoder import VariationalRateOfAgingAutoencoder

class VariationalRateOfAgingMonotonicAutoencoder(VariationalRateOfAgingAutoencoder):
    """
    """    
    
    def __init__(self, 
                 polynomial_powers_to_fit, 
                 weight_constraint_implementation='take_absolute_value', 
                 constrain_encoder=False,
                 non_monotonic_features=None,
                 **kwargs):
        super(VariationalRateOfAgingMonotonicAutoencoder, self).__init__(
            weight_constraint_implementation=weight_constraint_implementation,
            constrain_encoder=constrain_encoder,
            **kwargs)   
        self.polynomial_powers_to_fit = polynomial_powers_to_fit
        if non_monotonic_features is None:
            non_monotonic_features = set([])
        self.non_monotonic_features = set(non_monotonic_features)
        assert len(self.polynomial_powers_to_fit) > 0
        assert (np.array(self.polynomial_powers_to_fit) > 0).all()
        assert 1 in list(self.polynomial_powers_to_fit) 
    
    def init_network(self):
        """
        substantial differences with superclass here. 
        
        """
        super(VariationalRateOfAgingMonotonicAutoencoder, self).init_network()
        
        # get the indices of the monotonic and non-monotonic feature indices. 
        self.non_monotonic_idxs = np.array(sorted([self.feature_names.index(feature) for feature in self.non_monotonic_features])).astype('int32')
        self.monotonic_idxs = np.array([a for a in range(len(self.feature_names)) if a not in self.non_monotonic_idxs]).astype('int32')
        self.monotonic_feature_names = np.array(self.feature_names)[self.monotonic_idxs]
        assert len(self.non_monotonic_idxs) + len(self.monotonic_idxs) == len(self.feature_names)
        
        # remove the decoder for the age state because we need a new one. 
        # This is a little messy but at least it's transparent. 
        all_weight_names = list(self.weights.keys())
        all_bias_names = list(self.biases.keys())
        for k in all_weight_names:
            if 'decoder_Z_age' in k:
                del self.weights[k]
        for k in all_bias_names:
            if 'decoder_Z_age' in k:
                del self.biases[k]

        # make new decoder for age state. 
        # first, add non-monotonic bit if necessary. 
        if len(self.non_monotonic_idxs) > 0:
            for decoder_layer_idx, decoder_layer_size in enumerate(self.decoder_layer_sizes):
                # input layer size is the same as before
                if decoder_layer_idx == 0:
                    input_dim = self.k_age
                else:
                    input_dim = self.decoder_layer_sizes[decoder_layer_idx - 1]
                # output layer size is the length of the non-monotonic idxs if it's the last layer. 
                if decoder_layer_idx == len(self.decoder_layer_sizes) - 1:
                    output_dim = len(self.non_monotonic_idxs)
                else:
                    output_dim = self.decoder_layer_sizes[decoder_layer_idx]
                print("Added non-monotonic age decoder layer with input dimension %i and output dimension %i" % (input_dim, 
                                                                                                      output_dim))
                self.weights['decoder_Z_age_nonmonotonic_h%i' % decoder_layer_idx] = tf.Variable(
                    self.initialization_function([input_dim, output_dim]))
                self.biases['decoder_Z_age_nonmonotonic_b%i' % decoder_layer_idx] = tf.Variable(
                    self.initialization_function([output_dim]))
        
        # now, add monotonic bit to generate monotonic features. 
        # first a linear transformation. 
        self.age_decoder_linear_weights = tf.Variable(self.initialization_function([self.k_age, len(self.monotonic_idxs)]))
                
        # initialize nonlinearity matrix carefully -- approximately linear. 
        n_polynomial_terms = len(self.polynomial_powers_to_fit)
        nonlinearity_matrix = .01 * np.random.random([n_polynomial_terms, len(self.monotonic_idxs)]).astype('float32')
        nonlinearity_matrix[list(self.polynomial_powers_to_fit).index(1), :] = 1.
        self.nonlinearity_weights = tf.Variable(initial_value=nonlinearity_matrix)
        
    def decode(self, Z):
        """
        we have to separately decode Z_age and the residual, then add them back together. 
        """
        num_layers = len(self.decoder_layer_sizes)
                
        # first construct residual term by feeding through network as usual. 
        residual = Z[:, self.k_age:]
        for idx in range(num_layers):
            residual = tf.matmul(residual, self.weights['decoder_residual_h%i' % idx]) \
                + self.biases['decoder_residual_b%i' % idx]
            # No non-linearity on the last layer
            if idx != num_layers - 1:
                residual = self.non_linearity(residual)
        
        # Now construct the age term for monotonic features. 
        # 1. Positive linear transformation. Enforce positivity with self.weight_preprocessing_fxn. 
        monotonic_age_term = Z[:, :self.k_age]
        monotonic_age_term = tf.matmul(monotonic_age_term, self.weight_preprocessing_fxn(self.age_decoder_linear_weights))
        # 2. Polynomial transformation (positive coefficients)
        for idx in range(len(self.polynomial_powers_to_fit)):
            weights_for_polynomial_term = self.weight_preprocessing_fxn(self.nonlinearity_weights[idx, :])
            polynomial_term = tf.pow(monotonic_age_term, 
                                     self.polynomial_powers_to_fit[idx]) * weights_for_polynomial_term # broadcasting
            if idx == 0:
                summed_polynomial_age_terms = polynomial_term
            else:
                summed_polynomial_age_terms = summed_polynomial_age_terms + polynomial_term
        # No need for bias because we already have that in the residual term. 
        monotonic_age_term = summed_polynomial_age_terms
        
        # If there are no non-monotonic features, age_term is just the monotonic age term. 
        if len(self.non_monotonic_features) == 0:
            age_term = monotonic_age_term
        else:
            # Otherwise, we have to construct the non-monotonic age term
            # and combine it with the monotonic age term. 
            non_monotonic_age_term = Z[:, :self.k_age]
            for idx in range(num_layers):
                non_monotonic_age_term = tf.matmul(non_monotonic_age_term, 
                                               self.weights['decoder_Z_age_nonmonotonic_h%i' % idx]) \
                    + self.biases['decoder_Z_age_nonmonotonic_b%i' % idx]
                # No non-linearity on the last layer
                if idx != num_layers - 1:
                    non_monotonic_age_term = self.non_linearity(non_monotonic_age_term)
            
            if len(self.non_monotonic_features) == len(self.feature_names):
                age_term = non_monotonic_age_term
            else:
                # we have to insert the monotonic and non-monotonic terms into the appropriate columns. 
                # this is a bit messy; we do it using tf.scatter_nd
                # but I can only get this to work properly by transposing the matrix
                # (eg, inserting into the rows of the transpose)
                # This should work; for a MWE, try doing
                """
                n_features = 50
                n_points = 10
                non_monotonic_indices = [[1], [2], [10], [45]]
                indices = non_monotonic_indices
                Z_age_monotonic = np.random.random([n_points, len(non_monotonic_indices)])
                updates = tf.constant(Z_age_monotonic.transpose())
                shape = tf.constant([n_features, n_points])
                scatter = tf.transpose(tf.scatter_nd(indices, updates, shape))
                with tf.Session() as sess:
                    M = sess.run(scatter)
                """
                   
                shape = tf.shape(tf.transpose(residual))
                monotonic_age_term = tf.transpose(monotonic_age_term)
                monotonic_age_term = tf.transpose(
                    tf.scatter_nd(
                        tf.reshape(self.monotonic_idxs, [-1, 1]), 
                        monotonic_age_term, 
                        shape)
                )

                non_monotonic_age_term = tf.transpose(non_monotonic_age_term)
                non_monotonic_age_term = tf.transpose(
                    tf.scatter_nd(
                        tf.reshape(self.non_monotonic_idxs, [-1, 1]), 
                        non_monotonic_age_term, 
                        shape)
                )

                age_term = non_monotonic_age_term + monotonic_age_term
        
        X_with_logits = residual + age_term
        
        return X_with_logits

