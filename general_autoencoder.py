import numpy as np
from multiphenotype_utils import (get_continuous_features_as_matrix, add_id, remove_id_and_get_mat, 
    partition_dataframe_into_binary_and_continuous, divide_idxs_into_batches)
import pandas as pd
import tensorflow as tf
from dimreducer import DimReducer
import time
from scipy.stats import pearsonr, linregress
from scipy.special import expit
import copy

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
        binary_loss_weighting=1.0,
        non_linearity='relu', 
        batch_size=128,
        age_preprocessing_method='subtract_a_constant',
        include_age_in_encoder_input=False,  
        uses_longitudinal_data=False,
        regularization_weighting_schedule={'schedule_type':'constant', 'constant':1}):

        self.need_ages = False # whether ages are needed to compute loss or other quantities. 
        assert age_preprocessing_method in ['subtract_a_constant', 'divide_by_a_constant']
        self.age_preprocessing_method = age_preprocessing_method
        self.include_age_in_encoder_input = include_age_in_encoder_input   
        # include_age_in_encoder_input is whether age is used to approximate the posterior over Z. 
        # Eg, we need this for rate-of-aging autoencoder. 
        
        self.can_calculate_Z_mu = True # does the variable Z_mu make any sense for the model. 
        
        self.uses_longitudinal_data = False # does the model accomodate longitudinal data as well. 
        
        # How many epochs should pass before we evaluate and print out
        # the loss on the training/validation datasets?
        self.num_epochs_before_eval = 10

        # How many rounds of evaluation without validation improvement
        # should pass before we quit training?        
        # Roughly, 
        # max_epochs_without_improving = num_epochs_before_eval * max_evals_without_improving
        self.max_evals_without_improving = 500

        self.max_epochs = max_epochs

        # Set random seed
        self.random_seed = random_seed
        
        # binary loss weighting. This is used to make sure that the model doesn't just ignore binary features. 
        self.binary_loss_weighting = binary_loss_weighting
        
        # save the regularization_weighting_schedule. This controls how heavily we weight the regularization loss
        # as a function of epoch. 
        self.regularization_weighting_schedule = regularization_weighting_schedule
        assert regularization_weighting_schedule['schedule_type'] in ['constant', 'logistic']

        self.batch_size = batch_size
        if non_linearity == 'sigmoid':
            self.non_linearity = tf.nn.sigmoid
        elif non_linearity == 'relu':
            self.non_linearity = tf.nn.relu
        else:
            raise Exception("not a valid nonlinear activation")
            
        self.learning_rate = learning_rate
        self.optimization_method = tf.train.AdamOptimizer
        self.initialization_function = self.glorot_init
        self.all_losses_by_epoch = []
        self.binary_feature_idxs = None
        self.continuous_feature_idxs = None
        self.feature_names = None
                    
    def data_preprocessing_function(self, df):
        # this function is used to process multiple dataframes so make sure that they are in the same format
        old_binary_feature_idxs = copy.deepcopy(self.binary_feature_idxs)
        old_continuous_feature_idxs = copy.deepcopy(self.continuous_feature_idxs)
        old_feature_names = copy.deepcopy(self.feature_names)
        
        X, self.binary_feature_idxs, self.continuous_feature_idxs, self.feature_names = \
            partition_dataframe_into_binary_and_continuous(df)
        print("Number of continuous features: %i; binary features %i" % (
            len(self.continuous_feature_idxs), 
            len(self.binary_feature_idxs)))
        if old_binary_feature_idxs is not None:
            assert list(self.binary_feature_idxs) == list(old_binary_feature_idxs)
        if old_continuous_feature_idxs is not None:
            assert list(self.continuous_feature_idxs) == list(old_continuous_feature_idxs)
        if old_feature_names is not None:
            assert list(self.feature_names) == list(old_feature_names)
        
        return X
        
    def get_projections(self, df, project_onto_mean, **projection_kwargs):
        """
        use the fitted model to get projections for df. 
        if project_onto_mean=True, projects onto the mean value of Z (Z_mu). Otherwise, samples Z.
        """
        print("Getting projections using method %s." % self.__class__.__name__)
        X = self.data_preprocessing_function(df)
        ages = self.get_ages(df)
        Z = self._get_projections_from_processed_data(X, ages, project_onto_mean, **projection_kwargs)
        Z_df = add_id(Z, df) # Z_df and df will have the same id and individual_id. 
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

    def age_preprocessing_function(self, ages):
        # two possibilities: either subtract a constant (to roughly zero-mean ages) 
        # or divide by a constant (to keep age roughly on the same-scale as the other features)
        # in both cases, we hard-code the constant in rather than deriving from data 
        # to avoid weird bugs if we train on people with young ages or something and then test on another group. 
        # the constant is chosen for UKBB data, which has most respondents 40 - 70. 
        
        if self.age_preprocessing_method == 'subtract_a_constant':
            ages = ages - 55. 
        elif self.age_preprocessing_method == 'divide_by_a_constant':
            ages = ages / 70. 
        else:
            raise Exception("Invalid age processing method")
        return np.array(ages)
    
    def get_ages(self, df):
        ages = np.array(df['age_sex___age'].values, dtype=np.float32)
        return self.age_preprocessing_function(ages)
            
    def combine_loss_components(self, binary_loss, continuous_loss, regularization_loss):
        return binary_loss + continuous_loss + self.regularization_weighting * regularization_loss

    def fit(self, 
            train_df, 
            valid_df, 
            train_longitudinal_df0=None, # longitudinal data at the first and second timepoint, respectively. 
            train_longitudinal_df1=None):
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
            
        # preprocess longitudinal data
        train_longitudinal_X0 = None
        train_longitudinal_X1 = None
        train_longitudinal_ages0 = None
        train_longitudinal_ages1 = None
        if self.uses_longitudinal_data:
            assert train_longitudinal_df0 is not None
            assert train_longitudinal_df1 is not None
            assert len(train_longitudinal_df0) == len(train_longitudinal_df1)
            train_longitudinal_X0 = self.data_preprocessing_function(train_longitudinal_df0)
            train_longitudinal_X1 = self.data_preprocessing_function(train_longitudinal_df1)
            train_longitudinal_ages0 = self.get_ages(train_longitudinal_df0)
            train_longitudinal_ages1 = self.get_ages(train_longitudinal_df1)
        else:
            assert train_longitudinal_df0 is None
            assert train_longitudinal_df1 is None
            
        self._fit_from_processed_data(train_data=train_data, 
                                      valid_data=valid_data, 
                                      train_ages=train_ages, 
                                      valid_ages=valid_ages, 
                                      train_longitudinal_X0=train_longitudinal_X0, 
                                      train_longitudinal_X1=train_longitudinal_X1, 
                                      train_longitudinal_ages0=train_longitudinal_ages0, 
                                      train_longitudinal_ages1=train_longitudinal_ages1)
                                      
    
    def get_regularization_weighting_for_epoch(self, epoch):
        if self.regularization_weighting_schedule['schedule_type'] == 'constant':
            weighting = self.regularization_weighting_schedule['constant']
        elif self.regularization_weighting_schedule['schedule_type'] == 'logistic':
            # scales the weighting up following a sigmoid
            fraction_of_way_through_training = 1.0 * epoch / self.max_epochs
            max_weight = self.regularization_weighting_schedule['max_weight']
            slope = self.regularization_weighting_schedule['slope']
            intercept = self.regularization_weighting_schedule['intercept']
            weighting = max_weight * expit(fraction_of_way_through_training * slope + intercept)
        else:
            raise Exception("Invalid schedule type.")
        assert (weighting <= 1) and (weighting >= 0)
        print("Regularization weighting at epoch %i is %2.3e" % (epoch, weighting))
        return weighting
    
    def model_features_as_function_of_age(self, data, ages):
        """
        given a processed data matrix, computes the age slope and intercept for each feature. 
        Returns a dictionary where feature names match to age slopes and intercepts. 
        """
        if len(ages) < 10000:
            raise Exception("You are trying to compute age trends on data using very few datapoints. This seems bad.")
        assert len(data) == len(ages)
        features_to_age_slope_and_intercept = {}
        for i, feature in enumerate(self.feature_names):
            slope, intercept, _, _, _ = linregress(ages, data[:, i])
            features_to_age_slope_and_intercept[feature] = {'slope':slope, 'intercept':intercept}
            assert np.abs(pearsonr(data[:, i] - slope * ages - intercept, ages)[0]) < 1e-6
        return features_to_age_slope_and_intercept
    
    def decorrelate_data_with_age(self, data, ages):
        """
        given a processed data matrix, uses the previously fitted age model to remove age trends from each feature.
        Age trends are modeled linearly. 
        This relies on having previously fitted age_adjusted_models (ie, self.age_adjusted_models should not be None).
        """
        decorrelated_data = copy.deepcopy(data)
        for i, feature in enumerate(self.feature_names):
            slope = self.age_adjusted_models[feature]['slope']
            intercept = self.age_adjusted_models[feature]['intercept']
            decorrelated_data[:, i] = decorrelated_data[:, i] - slope * ages - intercept
        return decorrelated_data
    
    def _fit_from_processed_data(self, 
                                 train_data, 
                                 valid_data, 
                                 train_ages=None, 
                                 valid_ages=None, 
                                 train_longitudinal_X0=None, 
                                 train_longitudinal_X1=None, 
                                 train_longitudinal_ages0=None, 
                                 train_longitudinal_ages1=None):
        """
        train_data and valid_data are data matrices
        """
        if self.need_ages:
            assert train_ages is not None
            assert valid_ages is not None
            # Compute models for removing age trends. Do this on the train set to avoid any data leakage. 
            self.age_adjusted_models = self.model_features_as_function_of_age(train_data, train_ages)
            self.age_adjusted_train_data = self.decorrelate_data_with_age(train_data, train_ages)
            self.age_adjusted_valid_data = self.decorrelate_data_with_age(valid_data, valid_ages)
        else:
            self.age_adjusted_models = None
            self.age_adjusted_train_data = None
            self.age_adjusted_valid_data = None
            
        
        self.train_data = train_data
        self.valid_data = valid_data
        self.train_ages = train_ages
        self.valid_ages = valid_ages
        
        print("Train size %i; valid size %i" % (
            self.train_data.shape[0], self.valid_data.shape[0]))
        
        if self.uses_longitudinal_data:
            self.train_longitudinal_X0 = train_longitudinal_X0
            self.train_longitudinal_X1 = train_longitudinal_X1
            self.train_longitudinal_ages0 = train_longitudinal_ages0
            self.train_longitudinal_ages1 = train_longitudinal_ages1
            print("LONGITUDINAL train data size %i" % self.train_longitudinal_X0.shape[0])
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            np.random.seed(self.random_seed)

            self.X = tf.placeholder("float32", [None, len(self.feature_names)])
            self.age_adjusted_X = tf.placeholder("float32", [None, len(self.feature_names)])
            self.ages = tf.placeholder("float32", None)
            self.regularization_weighting = tf.placeholder("float32")
            self.init_network()
            if self.include_age_in_encoder_input:
                self.Z = self.encode(self.X, self.ages)
            else:
                self.Z = self.encode(self.X)
                
            self.Xr = self.decode(self.Z)
            self.combined_loss, self.binary_loss, self.continuous_loss, self.reg_loss = self.get_loss()
            self.optimizer = self.optimization_method(learning_rate=self.learning_rate).minimize(self.combined_loss)
            if self.uses_longitudinal_data:
                self.combined_longitudinal_loss, self.binary_longitudinal_loss, self.continuous_longitudinal_loss, self.reg_longitudinal_loss = self.get_longitudinal_loss()
                self.longitudinal_optimizer = self.optimization_method(learning_rate=self.learning_rate).minimize(self.combined_longitudinal_loss)
                
            
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
                t0 = time.time()
                regularization_weighting_for_epoch = self.get_regularization_weighting_for_epoch(epoch)
 
                self._train_epoch(regularization_weighting_for_epoch)
                    
                if (epoch % self.num_epochs_before_eval == 0) or (epoch == self.max_epochs - 1):
                    
                    train_mean_combined_loss, train_mean_binary_loss, \
                        train_mean_continuous_loss, train_mean_reg_loss = \
                        self.minibatch_mean_eval(self.train_data, 
                                                 self.train_ages, 
                                                 self.age_adjusted_train_data,
                                                 regularization_weighting_for_epoch)
                    valid_mean_combined_loss, valid_mean_binary_loss, \
                        valid_mean_continuous_loss, valid_mean_reg_loss = \
                        self.minibatch_mean_eval(self.valid_data, 
                                                 self.valid_ages, 
                                                 self.age_adjusted_valid_data,
                                                 regularization_weighting_for_epoch)    

                    print('Epoch %i:\nTrain: mean loss %2.3f (%2.3f + %2.3f + %2.3f * %2.3f).  '
                        'Valid: mean loss %2.3f (%2.3f + %2.3f + %2.3f * %2.3f)' % (
                        epoch, 
                        train_mean_combined_loss, 
                        train_mean_binary_loss,
                        train_mean_continuous_loss,
                        regularization_weighting_for_epoch,
                        train_mean_reg_loss,
                        valid_mean_combined_loss,
                        valid_mean_binary_loss,
                        valid_mean_continuous_loss,
                        regularization_weighting_for_epoch,
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
                                                     
                    if self.learn_continuous_variance:
                        continuous_variance = np.exp(self.sess.run(self.log_continuous_variance)[0])
                        print("Continuous variance is %2.3f" % continuous_variance)
                    if 'encoder_h0_sigma' in self.weights:
                        # make sure latent state for VAE looks ok by printing out diagnostics
                        if self.include_age_in_encoder_input:
                            sampled_Z, mu, sigma = self.sess.run([self.Z, self.Z_mu, self.Z_sigma], feed_dict = {self.X:self.train_data, self.ages:self.train_ages})
                        else:
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
                        
                        if self.can_calculate_Z_mu:
                            print('mean value of Z_mu')
                            print(mu.mean(axis = 0))
                            print("standard deviation of Z_mu (if this is super-close to 0, that's bad)")
                            print(mu.std(axis = 0, ddof=1))
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
                    
    def fill_feed_dict(self, data, regularization_weighting, ages=None, idxs=None, age_adjusted_data=None):
        """
        Returns a dictionary that has two keys:
            self.ages: ages[idxs]
            self.X: data[idxs, :]
        and handles various parameters being set to None.
        """
        if idxs is not None:
            # if idxs is not None, we want to take subsets of the data using boolean indices
            
            # if we pass in ages, subset appropriately; otherwise, just set to None to avoid an error. 
            if ages is not None:
                ages_to_use = ages[idxs]
            else:
                ages_to_use = None
            
            # similarly, if we pass in age_adjusted_data, subset appropriately
            # otherwise, just set to None to avoid an error.
            if age_adjusted_data is not None:
                age_adjusted_data_to_use = age_adjusted_data[idxs, :]
            else:
                age_adjusted_data_to_use = None
                
            # data will always be not None, so we can safely subset it. 
            data_to_use = data[idxs, :]
        else:
            # if we don't pass in indices, we just want to use all the data. 
            ages_to_use = ages
            data_to_use = data
            age_adjusted_data_to_use = age_adjusted_data

       
        if self.need_ages:
            feed_dict = {
                self.ages:ages_to_use, 
                self.X:data_to_use, 
                self.age_adjusted_X:age_adjusted_data_to_use,
                self.regularization_weighting:regularization_weighting}
        else:
            feed_dict = {self.X:data_to_use, 
                        self.regularization_weighting:regularization_weighting}
        return feed_dict

    def minibatch_mean_eval(self, data, ages, age_adjusted_data, regularization_weighting):
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
            feed_dict = self.fill_feed_dict(data, 
                                            regularization_weighting=regularization_weighting, 
                                            ages=ages, 
                                            idxs=idxs, 
                                            age_adjusted_data=age_adjusted_data)
            
            combined_loss, binary_loss, continuous_loss, reg_loss = self.sess.run(
                [self.combined_loss, self.binary_loss, self.continuous_loss, self.reg_loss], 
                feed_dict=feed_dict)
            mean_combined_loss += combined_loss * len(idxs) / data.shape[0]
            mean_binary_loss += binary_loss * len(idxs) / data.shape[0]
            mean_continuous_loss += continuous_loss * len(idxs) / data.shape[0]
            mean_reg_loss += reg_loss * len(idxs) / data.shape[0]

        return mean_combined_loss, mean_binary_loss, mean_continuous_loss, mean_reg_loss


    def _train_epoch(self, regularization_weighting):
        # This function takes very few input arguments because we assume it just uses train data, 
        # which is already stored as fields of the object. 
        data = self.train_data
        ages = self.train_ages
        age_adjusted_data = self.age_adjusted_train_data
        
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
            feed_dict = self.fill_feed_dict(data, 
                                            regularization_weighting=regularization_weighting, 
                                            ages=ages, 
                                            idxs=idxs, 
                                            age_adjusted_data=age_adjusted_data)
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

    def _get_projections_from_processed_data(self, data, ages, project_onto_mean, rotation_matrix=None):
        """
        if project_onto_mean=True, projects onto the mean value of Z. Otherwise, samples Z.  
        If rotation_matrix is passed in, rotates Z by multiplying by the rotation matrix after projecting it. 
        """
        if rotation_matrix is not None:
            print("Rotating Z by the rotation matrix!")
        chunk_size = 10000 # break into chunks so GPU doesn't run out of memory BOOO. 
        start = 0
        Zs = []
        while start < len(data):
            data_i = data[start:(start + chunk_size),]
            ages_i = ages[start:(start + chunk_size)]
            start += chunk_size
            if project_onto_mean:
                if self.can_calculate_Z_mu:
                    # if we have a closed form for Z_mu, use this for Z. 
                    Z = self.sess.run(self.Z_mu, feed_dict = {self.X:data_i, self.ages:ages_i})
                else:
                    # otherwise, compute 100 replicates, take mean. 
                    n_replicates = 100
                    print('number of replicates to compute Z_mu: %i' % n_replicates)
                    for replicate_idx in range(n_replicates):
                        replicate_Z = self.sess.run(self.Z, feed_dict = {self.X:data_i, self.ages:ages_i})
                        if replicate_idx == 0:
                            Z = replicate_Z
                        else:
                            Z += replicate_Z
                    Z = Z / n_replicates
                    
                        
            else:
                Z = self.sess.run(self.Z, feed_dict = {self.X:data_i, self.ages:ages_i})
            if rotation_matrix is not None:
                Z = np.dot(Z, rotation_matrix)
            Zs.append(Z)    
        Z = np.vstack(Zs)
        print("Shape of autoencoder projections is", Z.shape)
        return Z





