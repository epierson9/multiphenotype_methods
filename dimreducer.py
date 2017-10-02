import numpy as np
import scipy.linalg as slin
from multiphenotype_utils import cluster_and_plot_correlation_matrix, get_continuous_features_as_matrix, assert_zero_mean, add_id, remove_id_and_get_mat, make_age_bins, compute_column_means_with_incomplete_data, compute_correlation_matrix_with_incomplete_data, partition_dataframe_into_binary_and_continuous, divide_idxs_into_batches
from IPython import embed
from sklearn.linear_model import LinearRegression, LogisticRegression
import sklearn.decomposition as decomp
import pandas as pd
from sklearn.covariance import EmpiricalCovariance
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import time, random, os

"""
This file contains classes to compute multi-phenotypes. 
"""
class DimReducer(object):
    """
    Base class. 
    """
    def __init__(self, **init_kwargs):
        pass

    def data_preprocessing_function(self, df):
        """
        This function is applied to dataframes prior to applying fit or get_projections. 
        In general, it converts a dataframe to a matrix. 
        """
        print("Extracting continuous features as matrix.")
        X, cols = get_continuous_features_as_matrix(df, return_cols = True)
        self.feature_names = cols
        return X
        
    def fit(self, df, **fit_kwargs):
        """
        fit a model using df. 
        """
        print("Fitting model using method %s." % self.__class__.__name__)
        
        X = self.data_preprocessing_function(df)

        if self.need_ages:
            ages = df.loc[:, 'age_sex___age']
            self._fit_from_processed_data(X, ages, **fit_kwargs)
        else:
            self._fit_from_processed_data(X, **fit_kwargs)

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

    def reconstruct_data(self, Z_df):        
        raise NotImplementedError

    def _fit_from_processed_data(self, X):
        raise NotImplementedError

    def _get_projections_from_processed_data(self, X):
        raise NotImplementedError


class LinearDimReducer(DimReducer):
    """
    Inherits from DimReducer: this is for the special case where we get directions and want to 
    compute projections on those directions. 
    """
    def __init__(self, k, plot_correlation_matrix = True):
        self.k = k
        self.need_ages = False
        self.plot_correlation_matrix = plot_correlation_matrix

    def data_preprocessing_function(self, df):
        print("Extracting continuous features as matrix and zero-meaning.")
        X, cols = get_continuous_features_as_matrix(df, return_cols = True)
        self.feature_names = cols
        X = X - compute_column_means_with_incomplete_data(X)
        return X

    def _get_projections_from_processed_data(self, X):
        """
        U is a d x k matrix where k is the number of eigenvectors
        Returns n x k matrix of projections
        """
        assert(X.shape[1] == self.U.shape[0])
        assert(self.U.shape[1] == self.k)
        assert(X.shape[1] == len(self.feature_names))
        return X.dot(self.U) 

    def get_loading_matrix(self):
        """
        Special method for this subclass: returns a dataframe L where L_ij is the loading of the ith feature, jth component. 
        index is feature names, column names are Z0, ... Z(k-1). 
        """
        loadings_df = pd.DataFrame(self.U)
        loadings_df.columns = ['Z%i' % i for i in range(self.k)]
        loadings_df.index = self.feature_names
        return(loadings_df)

    def get_sorted_loadings(self, z_idx):
        """
        For a given z_idx, prints out the features contributing to that z in sorted order.
        """
        u = self.U[:, z_idx]
        sort_index = np.argsort(u)
        u_sorted = u[sort_index]
        feature_names_sorted = np.array(self.feature_names)[sort_index]

        for feature_name, coef in zip(feature_names_sorted, u_sorted):
            print("%6.3f   %s" % (coef, feature_name))

        return feature_names_sorted, u_sorted

    def reconstruct_data(self, Z_df):
        """
        Input: n x (k+1) data frame with ID column and k latent components
        Output: n x (d+1) data frame with ID column and data projected into the original (post-processed) space
        """
        Z = remove_id_and_get_mat(Z_df) 
        X = Z.dot(self.U.T)
        df = add_id(Z=X, df_with_id=Z_df)
        df.columns = ['individual_id'] + self.feature_names
        return df


class PCA(LinearDimReducer):
    def _fit_from_processed_data(self, X):
        
        if np.isnan(X).sum() > 0:
            print("Warning: X contains fraction %2.3f missing entries. Fitting PCA with incomplete data." % np.isnan(X).mean())
            fit_with_incomplete_data = True
        else:
            fit_with_incomplete_data = False
        
        if fit_with_incomplete_data:
            X_zeroed = X - compute_column_means_with_incomplete_data(X)
            cov, _ = compute_correlation_matrix_with_incomplete_data(X, correlation_type = 'covariance')
        else:
            X_zeroed = X - np.mean(X, axis=0)
            cov = X_zeroed.T.dot(X_zeroed) / X_zeroed.shape[0]
        if self.plot_correlation_matrix: 
            cluster_and_plot_correlation_matrix(cov, column_names = self.feature_names, how_to_sort = 'hierarchical')

        s, U = np.linalg.eig(cov) # Returns eigenvalues s and eigenvectors U
        
        idx = np.argsort(s)[::-1]
        s = s[idx]
        U = U[:, idx]        
        U = U[:, :self.k]

        print('Distribution of eigenvalues:')    
        sns.distplot(s)
        plt.show()
        print('Taking eigenvalues: %s' % s[:self.k])
        print('Total sum of eigenvalues          : %.3f' % np.sum(s))
        print('Total sum of eigenvalues taken    : %.3f' % np.sum(s[:self.k]))
        print('Total sum of eigenvalues not taken: %.3f' % np.sum(s[self.k:]))

        self.U = U
        self.s = s


class CPCA(LinearDimReducer):
    """
    Requires dataframes passed in to have a column foreground and a column background. 
    """
    def __init__(self, k, alpha):        
        self.k = k
        self.alpha = alpha
        self.need_ages = False

    def _fit_from_processed_data(self, X, foreground, background, take_abs):
        # Must pass in matrix X with a boolean column foreground and a boolean column background. 
        # Require both columns in case they are not mutually exhaustive (ie, there are some rows we don't want to use at all). 
        # Stores U = d x k matrix of k eigenvectors where U[:, 0] is first eigenvector
        # and s = vector of eigenvalues
        # take_abs is a boolean that determines whether we take the top k eigenvalues 
        # by absolute or signed value.
        if np.isnan(X).sum() > 0:
            print("Warning: X contains fraction %2.3f missing entries. Fitting CPCA with incomplete data." % np.isnan(X).mean())
            fit_with_incomplete_data = True
        else:
            fit_with_incomplete_data = False
            
        fg_mat = X[foreground,:]
        bg_mat = X[background,:]
        
        if fit_with_incomplete_data:
            fg_mat = fg_mat - compute_column_means_with_incomplete_data(fg_mat)
            bg_mat = bg_mat - compute_column_means_with_incomplete_data(bg_mat)
            fg_cov, _ = compute_correlation_matrix_with_incomplete_data(fg_mat, correlation_type = 'covariance')
            bg_cov, _ = compute_correlation_matrix_with_incomplete_data(bg_mat, correlation_type = 'covariance')
        else:
            fg_mat = fg_mat - np.mean(fg_mat, axis=0)
            bg_mat = bg_mat - np.mean(bg_mat, axis=0)
            fg_cov = fg_mat.T.dot(fg_mat) / fg_mat.shape[0]
            bg_cov = bg_mat.T.dot(bg_mat) / bg_mat.shape[0]
            
        
        assert fg_mat.shape[1] == bg_mat.shape[1]  
        diff_cov = fg_cov - self.alpha * bg_cov
        cluster_and_plot_correlation_matrix(diff_cov, column_names = self.feature_names, how_to_sort = 'hierarchical')
        
        s, U = np.linalg.eig(diff_cov) # Returns eigenvalues s and eigenvectors U
        
        if take_abs:
            idx = np.argsort(np.abs(s))[::-1]
        else:
            idx = np.argsort(s)[::-1]
        s = s[idx]
        U = U[:, idx]        
        U = U[:, :self.k]

        print('Distribution of eigenvalues:')    
        sns.distplot(s)
        plt.show()
        print('Taking eigenvalues: %s' % s[:self.k])
        print('Total sum of eigenvalues          : %.3f' % np.sum(s))
        print('Total sum of eigenvalues taken    : %.3f' % np.sum(s[:self.k]))
        print('Total sum of eigenvalues not taken: %.3f' % np.sum(s[self.k:]))

        self.U = U
        self.s = s


class TibshiraniMixedCriterion(LinearDimReducer):
    """
    6.4 in https://web.stanford.edu/~hastie/Papers/spca_JASA.pdf
    Compromise criterion: explain variance in X while also correlating with an external variable. 
    While we pass in age, this can also be used for eg a genetic matrix. 
    """
    def __init__(self, k, age_weighting):
        self.k = k
        self.age_weighting = age_weighting
        assert(self.age_weighting >= 0) 
        assert(self.age_weighting <= 1)
        self.need_ages = True

    def _fit_from_processed_data(self, X, ages):
        y = np.array(ages).reshape([len(ages), 1]) 
        y = y / np.linalg.norm(y)
        top_block = np.sqrt(1 - self.age_weighting) * X
        bottom_block = np.sqrt(self.age_weighting) * (y.T).dot(X)
        X_a = np.vstack([top_block, bottom_block])
        u, s, v = np.linalg.svd(X_a, full_matrices = 0)
        self.U = v[:self.k,].transpose()
        for i in range(self.k):
            assert(np.abs(np.linalg.norm(self.U[:, i]) - 1) < 1e-8)


class LinearAgePredictor(LinearDimReducer):
    """
    Does a linear regression of age on phenotypes. 
    """
    def __init__(self):        
        self.k = 1
        self.need_ages = True
        
    def data_preprocessing_function(self, df):
        # TODO: get_matrix_for_age_prediction needs to be implemented. 
        X, self.feature_names = get_matrix_for_age_prediction(df, return_cols = True)
        return X

    def _fit_from_processed_data(self, X, ages):
        self.linear_regression_model = LinearRegression(fit_intercept = True)
        self.linear_regression_model.fit(X, ages)
        self.U = self.linear_regression_model.coef_.reshape([-1, 1])

    def _get_projections_from_processed_data(self, X):
        return self.linear_regression_model.predict(X).reshape([len(X), 1])   

class NeuralNetAgePredictor(DimReducer):
    """
    Uses a neural net to predict age given phenotypes. 
    """
    def __init__(self, n_units_per_hidden_layer = 30, n_hidden_layers = 3):
        self.n_units_per_hidden_layer = n_units_per_hidden_layer
        self.n_hidden_layers = n_hidden_layers
        self.max_epochs = 100
        self.train_set_frac = .9

        tf.logging.set_verbosity(tf.logging.INFO) # lots of annoying messages but this prints out loss.  
        self.k = 1
        self.need_ages = True
    def data_preprocessing_function(self, df):
        X, self.feature_names = get_matrix_for_age_prediction(df, return_cols = True)
        return X
            
    def _fit_from_processed_data(self, X, ages):
        t0 = time.time()
        Y = np.array(ages)
        feature_columns = [tf.feature_column.numeric_column('x', shape=np.array(X).shape[1:])]
        hidden_unit_layers = [self.n_units_per_hidden_layer for layer in range(self.n_hidden_layers)]
        # save checkpoints in a scratch dir so they don't fill up the disk. 
        tf_model_dir = '/scratch/tensorflow_model_checkpoints/'
        os.system('rm -rf %s' % tf_model_dir)
        
        self.model = tf.contrib.learn.DNNRegressor(feature_columns = feature_columns, 
                                                   hidden_units = hidden_unit_layers, 
                                                   model_dir = tf_model_dir,
                                                   optimizer=tf.train.AdamOptimizer(learning_rate=0.001), 
                                                config=tf.contrib.learn.RunConfig(save_checkpoints_secs=3))

        # Train. 
        train_idxs = np.random.random(X.shape[0]) < self.train_set_frac # need a validation set to assess whether loss is improving. 
        train_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': X[train_idxs,]}, y=Y[train_idxs], batch_size = 100, num_epochs = self.max_epochs, shuffle = True)
        validation_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': X[~train_idxs,]}, y=Y[~train_idxs], batch_size = 100, shuffle = False, num_epochs = 1)
        
        validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(input_fn = validation_input_fn, every_n_steps = 1000) # this doesn't actually stop us early; it just prints out a validation loss so we can make sure we're not undertraining. 
        
        self.model.fit(input_fn = train_input_fn, monitors = [validation_monitor])
        print("Total time to train: %2.3f seconds" % (time.time() - t0))

    def _get_projections_from_processed_data(self, X):
        test_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': X}, y = None, batch_size=100, num_epochs = 1, shuffle=False)
        predictions = self.model.predict_scores(input_fn = test_input_fn)
        y_predicted = np.array([a for a in predictions])
        return y_predicted.reshape([len(y_predicted), 1])
    

class MahalanobisDistance(DimReducer):
    """
    Computes a person's Mahalanobis distance 
    using the mean and covariance estimated from a set of young people.
    Uses sklearn; verified this matches up with the normal matrix computation.
    """
    def __init__(self, age_lower, age_upper):
        self.age_lower = age_lower
        self.age_upper = age_upper
        self.need_ages = True
        self.k = 1
        
    def _fit_from_processed_data(self, X, ages):
        young_people = (ages >= self.age_lower) & (ages <= self.age_upper)
        print("%i people between %s and %s used for mean/cov calculation" % (
            young_people.sum(), 
            self.age_lower,
            self.age_upper))
        assert young_people.sum() > 1000
        self.model = EmpiricalCovariance(assume_centered=False)
        self.model.fit(X[young_people, :])

    def _get_projections_from_processed_data(self, X):
        md = np.sqrt(self.model.mahalanobis(X)).reshape([-1, 1])
        return md

class Autoencoder(DimReducer):
    def glorot_init(self, shape):
        return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))
    
    def __init__(self, k, max_epochs=300, variational=False):
        self.need_ages = False
        
        self.k = k # innermost hidden state
        self.max_epochs = max_epochs
        
        self.validation_frac = .2
        if not variational:
            self.batch_size = 100
            self.learning_rate = .005
            self.optimization_method = tf.train.AdamOptimizer
            self.initialization_function = tf.random_normal
            self.max_epochs_without_improving = 20
            self.encoder_layer_sizes = [50, 20, 20] # encoder layers prior to innermost hidden state
            self.decoder_layer_sizes = [20, 20, 50] # decoder layers after innermost hidden state
        else:
            self.learning_rate = .001
            self.batch_size = 100
            self.optimization_method = tf.train.RMSPropOptimizer
            self.initialization_function = self.glorot_init
            self.max_epochs_without_improving = 50
            self.encoder_layer_sizes = [50, 20, 20] # encoder layers prior to innermost hidden state
            self.decoder_layer_sizes = [] # decoder layers after innermost hidden state
            self.kl_weighting = 0
            
        self.variational = variational
        print("Creating autoencoder. Variational: %s" % variational)
        
    def data_preprocessing_function(self, df):
        X, self.binary_feature_idxs, self.continuous_feature_idxs, self.feature_names = partition_dataframe_into_binary_and_continuous(df)
        print("Number of continuous features: %i; binary features %i" % (len(self.continuous_feature_idxs), len(self.binary_feature_idxs)))
        return X
    
    def encode(self, X):  
        foward_prop = X
        for idx in range(len(self.encoder_layer_sizes)):
            forward_prop = tf.nn.sigmoid(tf.matmul(forward_prop, self.weights['encoder_h%i' % (idx)]) + self.biases['encoder_b%i' % (idx)])
        Z = tf.nn.sigmoid(tf.matmul(forward_prop, self.weights['encoder_to_hidden_state']) + self.biases['hidden_state'])
        return Z

    def variational_encode(self, X, use_sigma_encoder):
        # if use_sigma_encoder is True, use the neural net which encodes sigma. 
        # otherwise, do not. This just means we reference different layers. 
        # two differences from function above: first, takes an extra argument to reference sigma or not
        # and second, doesn't pass the final Z through a sigmoid because Z can be negative. 
        # (this is still ok even if we're encoding sigma because it gets passed through an exponent, so it's non-negative).
        # see https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/variational_autoencoder.py
        if use_sigma_encoder:
            layer_suffix = '_sigma'
        else:
            layer_suffix = ''
        forward_prop = X
        for idx in range(len(self.encoder_layer_sizes)):
            forward_prop = tf.nn.sigmoid(tf.matmul(forward_prop, self.weights['encoder_h%i%s' % (idx, layer_suffix)]) + self.biases['encoder_b%i%s' % (idx, layer_suffix)])
        gaussian_params = tf.matmul(forward_prop, self.weights['encoder_to_hidden_state%s' % layer_suffix]) + self.biases['hidden_state%s' % layer_suffix]
        return gaussian_params
        
    def decode(self, Z):
        forward_prop = Z
        for idx in range(len(self.decoder_layer_sizes)):
            forward_prop = tf.nn.sigmoid(tf.matmul(forward_prop, self.weights['decoder_h%i' % idx]) + self.biases['decoder_b%i' % idx])
        # no sigmoid on the final layer because it doesn't have to be positive. 
        continuous_X = tf.matmul(forward_prop, self.weights['continuous_output_h']) + self.biases['continuous_output_b']
        logit_X = tf.matmul(forward_prop, self.weights['logit_output_h']) + self.biases['logit_output_b']
        return continuous_X, logit_X   
    
    def _fit_from_processed_data(self, X):
        n_examples = len(X)
        continuous_X = X[:, self.continuous_feature_idxs]
        binary_X = X[:, self.binary_feature_idxs]
        
        # set up train and validation set. 
        shuffled_idxs = list(range(n_examples))
        random.shuffle(shuffled_idxs)
        train_cutoff = int(n_examples * (1 - self.validation_frac))
        self.train_idxs = shuffled_idxs[:train_cutoff]
        self.validation_idxs = shuffled_idxs[train_cutoff:]
        print("Total dataset size %i; train size %i; validation size %i" % (n_examples, len(self.train_idxs), len(self.validation_idxs)))
        
        
        # Set up tensorflow graph
        # placeholders 
        self.input_X = tf.placeholder("float32", [None, len(self.feature_names)])
        self.continuous_output_X = tf.placeholder("float32", [None, len(self.continuous_feature_idxs)])
        self.binary_output_X = tf.placeholder("float32", [None, len(self.binary_feature_idxs)])
        
        # weights and biases
        self.weights = {}
        self.biases = {}
        
        # encoder layers. 
        for encoder_layer_idx, encoder_layer_size in enumerate(self.encoder_layer_sizes):
            if encoder_layer_idx == 0:
                input_dim = len(self.feature_names)
            else:
                input_dim = self.encoder_layer_sizes[encoder_layer_idx - 1]
            output_dim = self.encoder_layer_sizes[encoder_layer_idx]
            print("Added encoder layer with input dimension %i and output dimension %i" % (input_dim, output_dim))
            self.weights['encoder_h%i' % encoder_layer_idx] = tf.Variable(self.initialization_function([input_dim, output_dim]))
            self.biases['encoder_b%i' % encoder_layer_idx] = tf.Variable(self.initialization_function([output_dim]))
            if self.variational: # then we also need a sigma layer. 
                print("Adding sigma layer of same size")
                self.weights['encoder_h%i_sigma' % encoder_layer_idx] = tf.Variable(self.initialization_function([input_dim, output_dim]))
                self.biases['encoder_b%i_sigma' % encoder_layer_idx] = tf.Variable(self.initialization_function([output_dim]))
                
        self.weights['encoder_to_hidden_state'] =  tf.Variable(self.initialization_function([self.encoder_layer_sizes[-1], self.k]))
        self.biases['hidden_state'] =  tf.Variable(self.initialization_function([self.k]))
        print("Added encoder-to-hidden state layer with input dimension %i and output dimension %i" % (self.encoder_layer_sizes[-1], self.k))
        if self.variational:
            self.weights['encoder_to_hidden_state_sigma'] =  tf.Variable(self.initialization_function([self.encoder_layer_sizes[-1], self.k]))
            self.biases['hidden_state_sigma'] =  tf.Variable(self.initialization_function([self.k]))
        
        
        # decoder layers. 
        for decoder_layer_idx, decoder_layer_size in enumerate(self.decoder_layer_sizes):
            if decoder_layer_idx == 0:
                input_dim = self.k
            else:
                input_dim = self.decoder_layer_sizes[decoder_layer_idx - 1]
            output_dim = self.decoder_layer_sizes[decoder_layer_idx]
            print("Added decoder layer with input dimension %i and output dimension %i" % (input_dim, output_dim))
            self.weights['decoder_h%i' % decoder_layer_idx] = tf.Variable(self.initialization_function([input_dim, output_dim]))
            self.biases['decoder_b%i' % decoder_layer_idx] = tf.Variable(self.initialization_function([output_dim]))
        
        if len(self.decoder_layer_sizes) == 0: # trivial decoder -- feeds directly to output
            last_decoder_layer_size = self.k
        else:
            last_decoder_layer_size = self.decoder_layer_sizes[-1]
            
        
        self.weights['continuous_output_h'] =  tf.Variable(self.initialization_function([last_decoder_layer_size, len(self.continuous_feature_idxs)]))
        self.biases['continuous_output_b'] =  tf.Variable(self.initialization_function([len(self.continuous_feature_idxs)]))
        
        self.weights['logit_output_h'] =  tf.Variable(self.initialization_function([last_decoder_layer_size,len(self.binary_feature_idxs)]))
        self.biases['logit_output_b'] =  tf.Variable(self.initialization_function([len(self.binary_feature_idxs)]))
        
        if self.variational:
            self.hidden_mu = self.variational_encode(self.input_X, use_sigma_encoder = False)
            self.hidden_sigma = tf.exp(self.variational_encode(self.input_X, use_sigma_encoder = True))
            self.eps = tf.random_normal(tf.shape(self.hidden_sigma), dtype=tf.float32, mean=0., stddev=1.0)
            self.hidden_state = self.hidden_mu + self.hidden_sigma * self.eps
        else:
            self.hidden_state = self.encode(self.input_X)
        
        self.continuous_reconstruction, self.logit_reconstruction = self.decode(self.hidden_state)
        self.continuous_cost = .5 * tf.reduce_sum(tf.square(self.continuous_output_X - self.continuous_reconstruction))
        self.binary_cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.logit_reconstruction, labels = self.binary_output_X))
        if self.variational:
            # need to compute KL divergence loss and add this in. 
            # See eq 10 here. I am not totally sure this is correct. 
            #self.kl_div_cost = 1 + self.hidden_sigma - tf.square(self.hidden_mu) - tf.exp(self.hidden_sigma)
            #self.kl_div_cost = -0.5 * tf.reduce_sum(self.kl_div_cost)
            self.kl_div_cost = -.5 * (1 + 2 * tf.log(self.hidden_sigma) - tf.square(self.hidden_mu) - tf.square(self.hidden_sigma))
            self.kl_div_cost = tf.reduce_sum(self.kl_div_cost) * self.kl_weighting
            self.cost = self.continuous_cost + self.binary_cost + self.kl_div_cost
        else:
            self.cost = self.continuous_cost + self.binary_cost
        
        self.optimizer = self.optimization_method(learning_rate = self.learning_rate).minimize(self.cost)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        
        train_batches = divide_idxs_into_batches(self.train_idxs, self.batch_size)
        validation_batches = divide_idxs_into_batches(self.validation_idxs, self.batch_size)
        n_epochs_without_improvement = 0 # number of epochs in which the validation loss has not improved. 
        validation_losses = []
        for epoch in range(self.max_epochs):
            random.shuffle(self.train_idxs)
            train_batches = divide_idxs_into_batches(self.train_idxs, self.batch_size)
            total_epoch_train_cost = 0
            total_epoch_validation_cost = 0
            total_epoch_continuous_cost = 0
            total_epoch_binary_cost = 0
            total_epoch_kl_div_cost = 0
            for idxs in train_batches:
                feed_dict = {self.input_X:X[idxs, :], self.continuous_output_X:continuous_X[idxs, :], self.binary_output_X:binary_X[idxs,:]}
                _, c = self.sess.run([self.optimizer, self.cost], feed_dict = feed_dict)
                total_epoch_train_cost += c
            for idxs in validation_batches:
                feed_dict = {self.input_X:X[idxs, :], self.continuous_output_X:continuous_X[idxs, :], self.binary_output_X:binary_X[idxs,:]}
                if self.variational:
                    total_c, continuous_c, binary_c, kl_div_c = self.sess.run([self.cost, self.continuous_cost, self.binary_cost, self.kl_div_cost], feed_dict = feed_dict)
                    total_epoch_validation_cost += total_c 
                    total_epoch_continuous_cost += continuous_c
                    total_epoch_binary_cost += binary_c
                    total_epoch_kl_div_cost += kl_div_c
                else:
                    total_epoch_validation_cost += self.sess.run(self.cost, feed_dict = feed_dict)
                
            if len(validation_losses) > 0 and not (total_epoch_validation_cost < min(validation_losses)):
                print('Warning! Validation loss not decreasing this epoch')
                n_epochs_without_improvement += 1
                if n_epochs_without_improvement > self.max_epochs_without_improving:
                    print("No improvement for too long; quitting")
                    break        
            else:
                n_epochs_without_improvement = 0
            
            validation_losses.append(total_epoch_validation_cost)
            if self.variational:
                print('Epoch %i: train cost %2.3f, validation cost %2.3f; kl cost %2.3f' % (epoch, total_epoch_train_cost, total_epoch_validation_cost, total_epoch_kl_div_cost))
            else:
                print('Epoch %i: train cost %2.3f, validation cost %2.3f' % (epoch, total_epoch_train_cost, total_epoch_validation_cost))
            
        plt.plot(validation_losses)
        plt.xlabel("Epoch")
        plt.ylabel("Validation loss")
        plt.show()
            
    def reconstruct_data(self, Z_df):
        """
        Input: n x (k+1) data frame with ID column and k latent components
        Output: n x (d+1) data frame with ID column and data projected into the original (post-processed) space
        """
        Z = remove_id_and_get_mat(Z_df) 
        continuous_X = self.sess.run(self.continuous_reconstruction, feed_dict = {self.hidden_state:Z})
        logit_X = self.sess.run(self.logit_reconstruction, feed_dict = {self.hidden_state:Z})

        continuous_df = add_id(Z=continuous_X, df_with_id=Z_df)
        continuous_df.columns = ['individual_id'] + [self.feature_names[a] for a in self.continuous_feature_idxs]
        
        logit_df = add_id(Z=logit_X, df_with_id=Z_df)
        logit_df.columns = ['individual_id'] + [self.feature_names[a] for a in self.binary_feature_idxs]
        
        df = logit_df.merge(continuous_df, how = 'inner', on = 'individual_id')
        return df
    
    def _get_projections_from_processed_data(self, X):
        chunk_size = 10000 # break into chunks so GPU doesn't run out of memory BOOO. 
        start = 0
        Zs = []
        while start < len(X):
            X_i = X[start:(start + chunk_size),]
            start += chunk_size
            if self.variational:
                # we want to use the MEAN of the hidden state (not introduce additional stochasticity.
                Zs.append(self.sess.run(self.hidden_mu, feed_dict = {self.input_X:X_i}))
            else:
                Zs.append(self.sess.run(self.hidden_state, feed_dict = {self.input_X:X_i}))
        Z = np.vstack(Zs)
        print("Shape of autoencoder projections is", Z.shape)
        return Z
    
