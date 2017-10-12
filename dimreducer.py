import numpy as np
import scipy.linalg as slin
from multiphenotype_utils import get_continuous_features_as_matrix, assert_zero_mean, add_id, remove_id_and_get_mat, make_age_bins, compute_column_means_with_incomplete_data, compute_correlation_matrix_with_incomplete_data, partition_dataframe_into_binary_and_continuous, divide_idxs_into_batches,cluster_and_plot_correlation_matrix
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
from scipy.special import expit


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
