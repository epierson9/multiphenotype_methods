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
    Decoder layer sizes here refers only to the depth of the RESIDUAL decoder. Age state decoder has depth 1. 
    """    
    
    def __init__(self, 
                 polynomial_powers_to_fit, 
                 weight_constraint_implementation='take_absolute_value', 
                 constrain_encoder=False,
                 **kwargs):
        super(VariationalRateOfAgingMonotonicAutoencoder, self).__init__(
            weight_constraint_implementation=weight_constraint_implementation,
            constrain_encoder=constrain_encoder,
            **kwargs)   
        self.polynomial_powers_to_fit = polynomial_powers_to_fit
        assert len(self.polynomial_powers_to_fit) > 0
        assert (np.array(self.polynomial_powers_to_fit) > 0).all()
        
    
    def init_network(self):
        """
        the only difference here is with the decoder, since we need to split out the age state and the residual. 
        """
        super(VariationalRateOfAgingMonotonicAutoencoder, self).init_network()
    
        n_polynomial_terms = len(self.polynomial_powers_to_fit)
        self.age_decoder_linear_weights = tf.Variable(self.initialization_function([self.k_age, len(self.feature_names)]))
        self.nonlinearity_weights = tf.Variable(self.initialization_function([n_polynomial_terms, len(self.feature_names)]))
    
    def decode(self, Z):
        """
        we have to separately decode Z_age and the residual, then add them back together. 
        """
        num_layers = len(self.decoder_layer_sizes)
        
        Z_age = Z[:, :self.k_age]
        residual = Z[:, self.k_age:]
        
        # first construct residual term by feeding through network as usual. 
        for idx in range(num_layers):
            # no weight constraints on residual decoder. 
            residual = tf.matmul(residual, self.weights['decoder_residual_h%i' % idx]) \
                + self.biases['decoder_residual_b%i' % idx]
            # No non-linearity on the last layer
            if idx != num_layers - 1:
                residual = self.non_linearity(residual)
        
        # Now construct the age term.
        # 1. Positive linear transformation
        Z_age = tf.matmul(Z_age, self.weight_preprocessing_fxn(self.age_decoder_linear_weights))
        # 2. Polynomial transformation (positive coefficients)
        for idx in range(len(self.polynomial_powers_to_fit)):
            weights_for_polynomial_term = self.weight_preprocessing_fxn(self.nonlinearity_weights[idx, :])
            polynomial_term = tf.pow(Z_age, self.polynomial_powers_to_fit[idx]) * weights_for_polynomial_term # broadcasting
            if idx == 0:
                summed_polynomial_age_terms = polynomial_term
            else:
                summed_polynomial_age_terms = summed_polynomial_age_terms + polynomial_term
        # No need for bias because we already have that in the residual term. 
        
        X_with_logits = summed_polynomial_age_terms + residual
        return X_with_logits

