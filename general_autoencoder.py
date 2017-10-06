import numpy as np
from multiphenotype_utils import (get_continuous_features_as_matrix, add_id, remove_id_and_get_mat, 
    partition_dataframe_into_binary_and_continuous, divide_idxs_into_batches)
import pandas as pd
import tensorflow as tf
from dimreducer import DimReducer


class GeneralAutoencoder(DimReducer):
    """
    Base autoencoder class that other classes derive from. 
    Not intended to be run on its own.
    Has code that's common to autoencoders in general, 
    e.g., default parameter settings, preprocessing functions, training procedure.
    """
    def __init__(self, 
        learning_rate=0.01,
        max_epochs=300, 
        random_seed=0):

        self.need_ages = False

        # How many epochs should pass before we evaluate and print out
        # the loss on the training/validation datasets?
        self.num_epochs_before_eval = 1

        # How many rounds of evaluation without validation improvement
        # should pass before we quit training?        
        # Roughly, 
        # max_epochs_without_improving = num_epochs_before_eval * max_evals_without_improving
        self.max_evals_without_improving = 100

        self.max_epochs = max_epochs

        # Set random seed
        self.random_seed = random_seed

        self.valid_frac = .2

        self.batch_size = 100
        self.learning_rate = learning_rate
        self.optimization_method = tf.train.AdamOptimizer
        self.initialization_function = tf.random_normal        

    def data_preprocessing_function(self, df):
        X, self.binary_feature_idxs, self.continuous_feature_idxs, self.feature_names = \
            partition_dataframe_into_binary_and_continuous(df)
        print("Number of continuous features: %i; binary features %i" % (
            len(self.continuous_feature_idxs), 
            len(self.binary_feature_idxs)))
        return X

    def split_into_binary_and_continuous(self, X):
        if len(self.binary_feature_idxs) > 0:        
            binary_features = tf.gather(X, indices=self.binary_feature_idxs, axis=1)
        else:
            binary_features = tf.zeros([tf.shape(X)[0], 0])

        if len(self.continuous_feature_idxs) > 0:
            continuous_features = tf.gather(X, indices=self.continuous_feature_idxs, axis=1)
        else:
            continuous_features = tf.zeros([tf.shape(X)[0], 0])

        return binary_features, continuous_features
        
    def glorot_init(self, shape):
        return tf.random_normal(shape=shape, stddev=tf.sqrt(2. / shape[0]))
 
    def init_network(self):
        raise NotImplementedError

    def encode(self, X):
        raise NotImplementedError

    def decode(self, Z):
        raise NotImplementedError

    def get_loss():
        raise NotImplementedError

    def fit(self, train_df, valid_df):
        print("Fitting model using method %s." % self.__class__.__name__)
        
        assert train_df.shape[1] == valid_df.shape[1]
        assert np.all(train_df.columns == valid_df.columns)
        train_data = self.data_preprocessing_function(train_df)
        valid_data = self.data_preprocessing_function(valid_df)

        self._fit_from_processed_data(train_data, valid_data)

    def _fit_from_processed_data(self, train_data, valid_data):
        """
        train_data and valid_data are data matrices
        """
        self.train_data = train_data
        self.valid_data = valid_data
        
        print("Train size %i; valid size %i" % (
            self.train_data.shape[0], self.valid_data.shape[0]))
                
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            np.random.seed(self.random_seed)

            self.X = tf.placeholder("float32", [None, len(self.feature_names)])

            self.init_network()
            self.Z = self.encode(self.X)
            self.Xr = self.decode(self.Z)
            self.combined_loss, self.binary_loss, self.continuous_loss, self.reg_loss = self.get_loss()

            self.optimizer = self.optimization_method(learning_rate=self.learning_rate).minimize(self.combined_loss)
            init = tf.global_variables_initializer()
            
            # with tf.Session() as self.sess:
            self.sess = tf.Session()  
            self.sess.run(init)
            min_valid_loss = np.nan
            n_epochs_without_improvement = 0

            params = self.sess.run(self.weights)
            print('Norm of params: %s' % np.linalg.norm(params['encoder_h0']))
            for epoch in range(self.max_epochs):
                # print('eps', self.sess.run(self.eps, feed_dict={self.X:self.train_data}))

                self._train_epoch(self.train_data)

                if (epoch % self.num_epochs_before_eval == 0) or (epoch == self.max_epochs - 1):
                    train_mean_combined_loss, train_mean_binary_loss, \
                        train_mean_continuous_loss, train_mean_reg_loss = \
                        self.minibatch_mean_eval(self.train_data)
                    valid_mean_combined_loss, valid_mean_binary_loss, \
                        valid_mean_continuous_loss, valid_mean_reg_loss = \
                        self.minibatch_mean_eval(self.valid_data)                    
                    print('Epoch %i:\nTrain: mean loss %2.3f (%2.3f + %2.3f + %2.3f).  '
                        'Valid: mean loss %2.3f (%2.3f + %2.3f + %2.3f)' % (
                        epoch, 
                        train_mean_combined_loss, 
                        train_mean_binary_loss,
                        train_mean_continuous_loss,
                        train_mean_reg_loss,
                        valid_mean_combined_loss,
                        valid_mean_binary_loss,
                        valid_mean_continuous_loss,
                        valid_mean_reg_loss
                        ))
                    if 'encoder_h0_sigma' in self.weights:
                        # make sure latent state for VAE looks ok by printing out diagnostics
                        sampled_Z, mu, sigma = self.sess.run([self.Z, self.Z_mu, self.Z_sigma], feed_dict = {self.X:self.train_data})
                        sampled_cov_matrix = np.cov(sampled_Z.transpose())
                        print('mean value of each Z component')
                        print(sampled_Z.mean(axis = 0))
                        print("diagonal elements of Z covariance matrix")
                        print(np.diag(sampled_cov_matrix))
                        upper_triangle = np.triu_indices(n = sampled_cov_matrix.shape[0], k = 1)
                        print("mean absolute value of off-diagonal covariance elements: %2.3f" % 
                              (np.abs(sampled_cov_matrix[upper_triangle]).mean()))
                        
                        print('mean value of Z_mu')
                        print(mu.mean(axis = 0))
                        print('mean value of Z_sigma')
                        print(sigma.mean(axis = 0))
                        
                        

                    

                    # fmin ignores nan's, so this handles the case when epoch=0
                    min_valid_loss = np.fmin(min_valid_loss, valid_mean_combined_loss)
                    if min_valid_loss < valid_mean_combined_loss:
                        print('Warning! valid loss not decreasing this epoch')
                        n_epochs_without_improvement += 1
                        if n_epochs_without_improvement > self.max_evals_without_improving:
                            print("No improvement for too long; quitting")
                            break        
                    else:
                        n_epochs_without_improvement = 0


    def minibatch_mean_eval(self, data):
        """
        Takes in a data matrix and computes the average per-example loss on it.
        Note: 'data' in this class is always a matrix.
        """
        batches = divide_idxs_into_batches(
            np.arange(data.shape[0]), 
            self.batch_size)

        mean_combined_loss = 0
        mean_binary_loss = 0
        mean_continuous_loss = 0
        mean_reg_loss = 0

        for idxs in batches:
            feed_dict = {self.X:data[idxs, :]}
            combined_loss, binary_loss, continuous_loss, reg_loss = self.sess.run(
                [self.combined_loss, self.binary_loss, self.continuous_loss, self.reg_loss], 
                feed_dict=feed_dict)
            mean_combined_loss += combined_loss * len(idxs) / data.shape[0]
            mean_binary_loss += binary_loss * len(idxs) / data.shape[0]
            mean_continuous_loss += continuous_loss * len(idxs) / data.shape[0]
            mean_reg_loss += reg_loss * len(idxs) / data.shape[0]

        return mean_combined_loss, mean_binary_loss, mean_continuous_loss, mean_reg_loss


    def _train_epoch(self, data):

        perm = np.arange(data.shape[0])
        np.random.shuffle(perm)
        data = data[perm, :]

        train_batches = divide_idxs_into_batches(
            np.arange(data.shape[0]), 
            self.batch_size)
        
        for idxs in train_batches:
            feed_dict = {self.X:data[idxs, :]}          
            self.sess.run([self.optimizer], feed_dict=feed_dict)


    def reconstruct_data(self, Z_df):
        """
        Input: n x (k+1) data frame with ID column and k latent components
        Output: n x (d+1) data frame with ID column and data projected into the original (post-processed) space
        """
        Z = remove_id_and_get_mat(Z_df) 
        X = self.sess.run(self.Xr, feed_dict={self.Z:Z})
        df = add_id(Z=X, df_with_id=Z_df)
        df.columns = ['individual_id'] + self.feature_names
        return df


    def _get_projections_from_processed_data(self, data):
        chunk_size = 10000 # break into chunks so GPU doesn't run out of memory BOOO. 
        start = 0
        Zs = []
        while start < len(data):
            data_i = data[start:(start + chunk_size),]
            start += chunk_size
            Zs.append(self.sess.run(self.Z, feed_dict = {self.X:data_i}))
        Z = np.vstack(Zs)
        print("Shape of autoencoder projections is", Z.shape)
        return Z





