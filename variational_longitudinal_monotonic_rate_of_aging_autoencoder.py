from copy import deepcopy
import numpy as np
from multiphenotype_utils import (get_continuous_features_as_matrix, add_id, remove_id_and_get_mat, 
    partition_dataframe_into_binary_and_continuous, divide_idxs_into_batches)
import pandas as pd
import tensorflow as tf
from dimreducer import DimReducer

from general_autoencoder import GeneralAutoencoder
from standard_autoencoder import StandardAutoencoder
from variational_rate_of_aging_monotonic_autoencoder import VariationalRateOfAgingMonotonicAutoencoder

class VariationalLongitudinalMonotonicRateOfAgingAutoencoder(VariationalRateOfAgingMonotonicAutoencoder):
    """
    Implements a variational rate-of-aging autoencoder with longitudinal data. 
    Does this by defining a combined_cross_sectional_plus_lon_loss and minimizing that instead. 
    combined_cross_sectional_plus_lon_loss is the standard cross-sectional loss plus an additional longitudinal loss. 
    """    
    
    def __init__(self,
                 lon_loss_weighting_factor=1,
                 lon_batch_size=128,
                 **kwargs):
        super(VariationalLongitudinalMonotonicRateOfAgingAutoencoder, self).__init__(uses_longitudinal_data=True, 
                                                                            **kwargs)  
        
        self.lon_loss_weighting_factor = lon_loss_weighting_factor
        self.lon_batch_size = lon_batch_size
         
    def init_network(self):
        # define four additional placeholders to store the followup longitudinal ages and values
        self.lon_ages0 = tf.placeholder("float32", None, name='lon_ages0')
        self.lon_X0 = tf.placeholder("float32", None, name='lon_X0')
        self.lon_ages1 = tf.placeholder("float32", None, name='lon_ages1')
        self.lon_X1 = tf.placeholder("float32", None, name='lon_X1')
        super(VariationalLongitudinalMonotonicRateOfAgingAutoencoder, self).init_network()
    
    def set_up_encoder_structure(self):
        # set up both longitudinal encoder and cross-sectional encoder. 
        # cross-sectional encoder is straightforward.
        self.Z, self.Z_mu, self.Z_sigma, self.encoder_mu, self.encoder_sigma = self.encode(self.X, self.ages)
        
        # longitudinal initial position. 
        (self.lon_Z0,
         self.lon_Z_mu, 
         self.lon_Z_sigma,
         self.lon_encoder_mu, 
         self.lon_encoder_sigma) = self.encode(self.lon_X0, self.lon_ages0)
        
        # fast forward Z0 to get Z1. 
        self.lon_Z1 = tf.concat([self.lon_Z0[:, :self.k_age] * tf.reshape(1.0*self.lon_ages1 / self.lon_ages0, [-1, 1]),
                                 self.lon_Z0[:, self.k_age:]], axis=1)
    
    def set_up_regularization_loss_structure(self):
        self.reg_loss = self.get_regularization_loss(self.encoder_mu, self.encoder_sigma)
        self.reg_lon_loss = self.get_regularization_loss(self.lon_encoder_mu, self.lon_encoder_sigma)
    
    def set_up_longitudinal_loss_and_optimization_structure(self):
        # define the longitudinal loss, and change the optimizer so it minimizes the longitudinal loss + the cross sectional loss. 
        self.lon_Xr0 = self.decode(self.lon_Z0)
        self.lon_Xr1 = self.decode(self.lon_Z1)
        self.lon_binary_loss0, self.lon_continuous_loss0 = self.get_binary_and_continuous_loss(self.lon_X0, self.lon_Xr0)
        self.lon_binary_loss1, self.lon_continuous_loss1 = self.get_binary_and_continuous_loss(self.lon_X1, self.lon_Xr1)

        # multiply all loss components by longitudinal loss weighting factor
        binary_lon_loss = (self.lon_binary_loss0 + self.lon_binary_loss1) * self.lon_loss_weighting_factor
        continuous_lon_loss = (self.lon_continuous_loss0 + self.lon_continuous_loss1) * self.lon_loss_weighting_factor
        reg_lon_loss = self.reg_lon_loss * self.lon_loss_weighting_factor

        self.combined_lon_loss = binary_lon_loss + continuous_lon_loss + self.regularization_weighting * reg_lon_loss

        self.combined_cross_sectional_plus_lon_loss = (self.combined_lon_loss + self.combined_loss)
        self.optimizer = self.optimization_method(learning_rate=self.learning_rate).minimize(
            self.combined_cross_sectional_plus_lon_loss)
    
    def _train_epoch(self, regularization_weighting):
        # store the data we need in local variables.
        # cross-sectional data.
        data = self.train_data
        ages = self.train_ages
        age_adjusted_data = self.age_adjusted_train_data
        
        # longitudinal data. 
        train_lon_X0 = self.train_lon_X0 
        train_lon_X1 = self.train_lon_X1
        train_lon_ages0 = self.train_lon_ages0
        train_lon_ages1 = self.train_lon_ages1
        
        # permute cross-sectional data
        perm = np.arange(data.shape[0])
        np.random.shuffle(perm)
        data = data[perm, :]
        ages = ages[perm]
        train_batches = divide_idxs_into_batches(
            np.arange(data.shape[0]), 
            self.batch_size)
        n_cross_sectional_points = data.shape[0]
        n_cross_sectional_batches = len(train_batches)        
        n_lon_points = train_lon_X0.shape[0]
        # create longitudinal batches (one for each cross-sectional branch). Sample with replacement. 
        # each row is one set of idxs. 
        lon_train_batches = np.random.choice(n_lon_points, 
                                             size=[n_cross_sectional_batches, self.lon_batch_size],
                                             replace=True)
                                                                               
        # train. For each cross-sectional batch, we sample a random longitudinal batch of size self.lon_batch_size
        total_lon_loss = 0
        total_cross_sectional_loss = 0
        total_lon_points = 0
        total_cross_sectional_points = 0
        
        for i in range(n_cross_sectional_batches):
            # first fill standard cross-sectional feed dict. 
            cross_sectional_idxs = train_batches[i]
            combined_feed_dict = self.fill_feed_dict(
                data=data,
                regularization_weighting=regularization_weighting,
                ages=ages,
                idxs=cross_sectional_idxs, 
                age_adjusted_data=age_adjusted_data)
            
            # now add the stuff we need for longitudinal computations. 
            lon_idxs = lon_train_batches[i, :]
            combined_feed_dict[self.lon_X0] = train_lon_X0[lon_idxs, :]
            combined_feed_dict[self.lon_ages0] = train_lon_ages0[lon_idxs]
            combined_feed_dict[self.lon_X1] = train_lon_X1[lon_idxs, :]
            combined_feed_dict[self.lon_ages1] = train_lon_ages1[lon_idxs]
            
            # optimize and store losses. 
            _, batch_lon_loss, batch_cross_sectional_loss = self.sess.run([self.optimizer,
                                                   self.combined_lon_loss, 
                                                   self.combined_loss],
                                                   feed_dict=combined_feed_dict) 

                              
            total_cross_sectional_loss = (total_cross_sectional_loss + batch_cross_sectional_loss * len(cross_sectional_idxs))
            total_lon_loss = (total_lon_loss + batch_lon_loss * len(lon_idxs))
            
            total_cross_sectional_points = total_cross_sectional_points + len(cross_sectional_idxs)
            total_lon_points = total_lon_points + len(lon_idxs)
            
                 
        total_cross_sectional_loss = total_cross_sectional_loss / total_cross_sectional_points
        total_lon_loss = total_lon_loss / total_lon_points
        total_lon_loss = total_lon_loss / self.lon_loss_weighting_factor
        print(("Cross-sectional loss: %2.3f; " + 
               "longitudinal loss: %2.3f; " + 
               "longitudinal weighting factor: %2.3f\n" + 
               "(losses are per-example, PRIOR to multiplying by the longitudinal weighting factor)") % (
            total_cross_sectional_loss,
            total_lon_loss, 
            self.lon_loss_weighting_factor))


            
            
    
