import numpy as np
from multiphenotype_utils import (get_continuous_features_as_matrix, add_id, remove_id_and_get_mat, 
    partition_dataframe_into_binary_and_continuous, divide_idxs_into_batches)
import pandas as pd
import tensorflow as tf
from dimreducer import DimReducer
import time
from scipy.stats import pearsonr

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
        random_seed=0, 
        non_linearity='relu'):

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
        if non_linearity == 'sigmoid':
            self.non_linearity = tf.nn.sigmoid
        elif non_linearity == 'relu':
            self.non_linearity = tf.nn.relu
        else:
            raise Exception("not a valid nonlinear activation")
            
        self.learning_rate = learning_rate
        self.optimization_method = tf.train.AdamOptimizer
        self.initialization_function = tf.random_normal
        self.all_losses_by_epoch = []
                    
    def data_preprocessing_function(self, df):
        X, self.binary_feature_idxs, self.continuous_feature_idxs, self.feature_names = \
            partition_dataframe_into_binary_and_continuous(df)
        print("Number of continuous features: %i; binary features %i" % (
            len(self.continuous_feature_idxs), 
            len(self.binary_feature_idxs)))

        return X
        
    def get_projections(self, df, **projection_kwargs):
        """
        use the fitted model to get projections for df. 
        """
        print("Getting projections using method %s." % self.__class__.__name__)
        X = self.data_preprocessing_function(df)
        Z = self._get_projections_from_processed_data(X, **projection_kwargs)
        Z_df = add_id(Z, df)
        Z_df.columns = ['individual_id'] + ['z%s' % i for i in range(Z.shape[1])]

        return Z_df            

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

    def get_ages(self, df):
        ages = np.array(df['age_sex___age'].values, dtype=np.float32)
        ages -= np.mean(ages)
        return ages

    def fit(self, train_df, valid_df):
        print("Fitting model using method %s." % self.__class__.__name__)
        
        assert train_df.shape[1] == valid_df.shape[1]
        assert np.all(train_df.columns == valid_df.columns)
        
        train_data = self.data_preprocessing_function(train_df)
        valid_data = self.data_preprocessing_function(valid_df)
        
        train_ages = None
        valid_ages = None                
        if self.need_ages:
            train_ages = self.get_ages(train_df)
            valid_ages = self.get_ages(valid_df)
            
        self._fit_from_processed_data(train_data, valid_data, train_ages, valid_ages)
    
    def _fit_from_processed_data(self, train_data, valid_data, train_ages=None, valid_ages=None):
        """
        train_data and valid_data are data matrices
        """
        if self.need_ages:
            assert train_ages is not None
            assert valid_ages is not None
        
        self.train_data = train_data
        self.valid_data = valid_data
        self.train_ages = train_ages
        self.valid_ages = valid_ages

        print("Train size %i; valid size %i" % (
            self.train_data.shape[0], self.valid_data.shape[0]))
                
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            np.random.seed(self.random_seed)

            self.X = tf.placeholder("float32", [None, len(self.feature_names)])
            self.ages = tf.placeholder("float32", None)
            self.init_network()
            self.Z = self.encode(self.X)
            self.Xr = self.decode(self.Z)
            self.combined_loss, self.binary_loss, self.continuous_loss, self.reg_loss = self.get_loss()

            self.optimizer = self.optimization_method(learning_rate=self.learning_rate).minimize(self.combined_loss)
            init = tf.global_variables_initializer()
            
            # with tf.Session() as self.sess:
            #config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.4))
            #self.sess = tf.Session(config=config)  
            
            # create a saver object so we can save the model if we want. 
            self.saver = tf.train.Saver()
            self.sess = tf.Session()  
            self.sess.run(init)
            min_valid_loss = np.nan
            n_epochs_without_improvement = 0

            params = self.sess.run(self.weights)
            print('Norm of params: %s' % np.linalg.norm(params['encoder_h0']))
            for epoch in range(self.max_epochs):
                # print('eps', self.sess.run(self.eps, feed_dict={self.X:self.train_data}))
                t0 = time.time()
                self._train_epoch(self.train_data, self.train_ages)
                
                if (epoch % self.num_epochs_before_eval == 0) or (epoch == self.max_epochs - 1):
                    
                    train_mean_combined_loss, train_mean_binary_loss, \
                        train_mean_continuous_loss, train_mean_reg_loss = \
                        self.minibatch_mean_eval(self.train_data, self.train_ages)
                    valid_mean_combined_loss, valid_mean_binary_loss, \
                        valid_mean_continuous_loss, valid_mean_reg_loss = \
                        self.minibatch_mean_eval(self.valid_data, self.valid_ages)    

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
                    # log losses so that we can see if the model's training well. 
                    self.all_losses_by_epoch.append({'epoch':epoch, 
                                                    'train_mean_combined_loss':train_mean_combined_loss, 
                                                    'train_mean_binary_loss':train_mean_binary_loss,
                                                    'train_mean_continuous_loss':train_mean_continuous_loss,
                                                    'train_mean_reg_loss':train_mean_reg_loss,
                                                    'valid_mean_combined_loss':valid_mean_combined_loss,
                                                    'valid_mean_binary_loss':valid_mean_binary_loss,
                                                    'valid_mean_continuous_loss':valid_mean_continuous_loss,
                                                    'valid_mean_reg_loss':valid_mean_reg_loss})
                                                     

                    if 'encoder_h0_sigma' in self.weights:
                        # make sure latent state for VAE looks ok by printing out diagnostics
                        sampled_Z, mu, sigma = self.sess.run([self.Z, self.Z_mu, self.Z_sigma], feed_dict = {self.X:self.train_data})
                        sampled_cov_matrix = np.cov(sampled_Z.transpose())
                        print('mean value of each Z component:')
                        print(sampled_Z.mean(axis = 0))
                        if self.need_ages:
                            print('correlation of each Z component with age:')
                            for i in range(sampled_Z.shape[1]):
                                print('%.2f' % pearsonr(sampled_Z[:, i], self.train_ages)[0], end=' ')
                            print('')
                        print("diagonal elements of Z covariance matrix:")
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
                print("Total time to run epoch: %2.3f seconds" % (time.time() - t0))
    
    def save_model(self, path_to_save_model):
        print("Done training model; saving at path %s." % path_to_save_model)
        self.saver.save(self.sess, save_path=path_to_save_model)
                    
    def fill_feed_dict(self, data, ages=None, idxs=None):
        """
        Returns a dictionary that has two keys:
            self.ages: ages[idxs]
            self.X: data[idxs, :]
        and handles various parameters being set to None.
        """
        if idxs is not None:
            if ages is not None:
                indexed_ages = ages[idxs]
            else:
                indexed_ages = ages
            indexed_data = data[idxs, :]
        else:
            indexed_ages = ages
            indexed_data = data

       
        if self.need_ages:
            feed_dict = {
                self.ages:indexed_ages, 
                self.X:indexed_data}
        else:
            feed_dict = {self.X:indexed_data}
        return feed_dict

    def minibatch_mean_eval(self, data, ages):
        """
        Takes in a data matrix and computes the average per-example loss on it.
        Note: 'data' in this class is always a matrix.
        """
        if self.need_ages:
            assert ages is not None

        batches = divide_idxs_into_batches(
            np.arange(data.shape[0]), 
            self.batch_size)

        mean_combined_loss = 0
        mean_binary_loss = 0
        mean_continuous_loss = 0
        mean_reg_loss = 0

        for idxs in batches:
            feed_dict = self.fill_feed_dict(data, ages, idxs)
            
            combined_loss, binary_loss, continuous_loss, reg_loss = self.sess.run(
                [self.combined_loss, self.binary_loss, self.continuous_loss, self.reg_loss], 
                feed_dict=feed_dict)
            mean_combined_loss += combined_loss * len(idxs) / data.shape[0]
            mean_binary_loss += binary_loss * len(idxs) / data.shape[0]
            mean_continuous_loss += continuous_loss * len(idxs) / data.shape[0]
            mean_reg_loss += reg_loss * len(idxs) / data.shape[0]

        return mean_combined_loss, mean_binary_loss, mean_continuous_loss, mean_reg_loss


    def _train_epoch(self, data, ages):
        if self.need_ages:
            assert ages is not None

        perm = np.arange(data.shape[0])
        np.random.shuffle(perm)
        data = data[perm, :]
        if ages is not None:
            ages = ages[perm]

        train_batches = divide_idxs_into_batches(
            np.arange(data.shape[0]), 
            self.batch_size)
        
        for idxs in train_batches:            
            feed_dict = self.fill_feed_dict(data, ages, idxs)      
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


    def _get_projections_from_processed_data(self, data, project_onto_mean=True):
        """
        if project_onto_mean=True, projects onto the mean value of Z (Z_mu). Otherwise, samples Z.  
        """
        chunk_size = 10000 # break into chunks so GPU doesn't run out of memory BOOO. 
        start = 0
        Zs = []
        while start < len(data):
            data_i = data[start:(start + chunk_size),]
            start += chunk_size
            if project_onto_mean:
                Zs.append(self.sess.run(self.Z_mu, feed_dict = {self.X:data_i}))
            else:
                Zs.append(self.sess.run(self.Z, feed_dict = {self.X:data_i}))
        Z = np.vstack(Zs)
        print("Shape of autoencoder projections is", Z.shape)
        return Z





