import math
import pandas as pd
import numpy as np

def move_last_col_to_first(df):
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df.loc[:, cols]
    return df

def compute_correlation_matrix_with_incomplete_data(df, correlation_type):
    """
    Given a dataframe or numpy array df and a correlation type (spearman, pearson, or covariance) computes the pairwise correlations between 
    all columns of the dataframe. Dataframe can have missing data; these will simply be ignored. 
    Nan correlations are set to 0 with a warning. 
    Returns the correlation matrix  and a vector of counts of non-missing data. 
    For correlation_type == covariance, identical to np.cov(df.T, ddof = 0) in case of no missing data. 
    """
    X = pd.DataFrame(df) # make sure we are using a dataframe to do computations. 
    assert correlation_type in ['spearman', 'pearson', 'covariance']

    if correlation_type == 'covariance':
        C = X.cov() * (len(df) - 1) /  len(df) # need correction factor so it's consistent with ddof = 0. Makes little difference. 
    else:
        C = X.corr(correlation_type)
    C = np.array(C)

    assert(C.shape[0] == C.shape[1])
    for i in range(len(C)):
        for j in range(len(C)):
            if np.isnan(C[i][j]):
                print("Warning: entry of covariance matrix is nan; setting to 0.")
                C[i][j] = 0
    non_missing_data_counts = (~np.isnan(X)).sum(axis = 0)
    return C, non_missing_data_counts

def partition_dataframe_into_binary_and_continuous(df):
    """
    Partitions a data frame into binary and continuous features. 
    This is used for the autoencoder so we apply the correct loss function. 
    Returns a matrix X of df values along the column indices of binary and continuous features
    and the feature names. 
    """
    print("Partitioning dataframe into binary and continuous columns")
    phenotypes_to_exclude = [
        'individual_id',
        'age_sex___age']
    feature_names = []
    binary_features = []
    continuous_features = []
    for c in df.columns:
        if c in phenotypes_to_exclude:
            continue
        if set(df[c]) == set([False, True]):
            # this binarization should work even if df[c] is eg 1.0 or 1 rather than True. 
            print("Binary column %s" % c)
            binary_features.append(c)
        else:
            print("Continuous column %s" % c)
            continuous_features.append(c)
        feature_names.append(c)
    binary_feature_idxs = [feature_names.index(a) for a in binary_features]
    continuous_feature_idxs = [feature_names.index(a) for a in continuous_features]
    X = df[feature_names].values
    return X, binary_feature_idxs, continuous_feature_idxs, feature_names


def compute_column_means_with_incomplete_data(df):
    """
    Given a dataframe or numpy array df, computes means for each column. 
    Identical to np.array(data.df).mean(axis = 0) in case of no missing data. 
    """
    X = np.array(df)
    return np.nanmean(X, axis = 0)

def cluster_and_plot_correlation_matrix(C, column_names, how_to_sort):
    """
    Given a correlation matrix c and column_names, sorts correlation matrix using hierarchical clustering if
    how_to_sort == hierarchical, otherwise alphabetically. 
    """
    C = copy.deepcopy(C)
    if np.abs(C).max() - 1 > 1e-6:
        print("Warning: maximum absolute value in C is %2.3f, which is larger than 1; this will be truncated in the visualization." % np.abs(C).max())    
    for i in range(len(C)):
        if(np.abs(C[i, i] - 1) > 1e-6):
            print("Warning: correlation matrix diagonal entry is not one (%2.8f); setting to one for visualization purposes." % C[i, i].mean())
        C[i, i] = 1 # make it exactly one so hierarchical clustering doesn't complain. 
            
    assert how_to_sort in ['alphabetically', 'hierarchical']
    assert(len(C) == len(column_names))
    plt.set_cmap('bwr')
    plt.figure(figsize = [15, 15])
    if how_to_sort == 'hierarchical':
        y = squareform(1 - np.abs(C))
        Z = linkage(y, method = 'average')
        clusters = fcluster(Z, t = 0)
        # print(clusters)
        reordered_idxs = np.argsort(clusters)
    else:
        reordered_idxs = np.argsort(column_names)
    
    C = C[:, reordered_idxs]
    C = C[reordered_idxs, :]
    plt.yticks(range(len(column_names)), np.array(column_names)[reordered_idxs])
    plt.xticks(range(len(column_names)), np.array(column_names)[reordered_idxs], rotation = 90)
    plt.imshow(C, vmin = -1, vmax = 1)
    
    plt.colorbar()
    for i in range(len(C)):
        for j in range(len(C)):
            if np.abs(C[i][j]) > .1:
                plt.scatter([i], [j], color = 'black', s = 1)
    plt.show()

def get_continuous_features_as_matrix(df, return_cols=False):
    cols_to_keep = list(col for col in df.columns if (df[col].dtype == 'float64'))

    # Sanity checks
    phenotypes_to_exclude = [
        'individual_id',
        'age_sex___age',
        'age_sex___self_report_female']
    for phenotype in phenotypes_to_exclude: 
        assert phenotype not in cols_to_keep

    reduced_df = df[cols_to_keep]
    if return_cols:
        return reduced_df.values, cols_to_keep
    else:
        return reduced_df.values

def assert_zero_mean(df):
    print(np.mean(get_continuous_features_as_matrix(df), axis=0))
    assert np.all(np.mean(get_continuous_features_as_matrix(df), axis=0) < 1e-8)

def add_id(Z, df_with_id):
    """
    Takes in a matrix Z and data frame df_with_id
    and converts Z into a data frame with individual_id taken from df_with_id.
    Assumes that rows of Z are aligned with rows of df_with_id.
    """
    assert Z.shape[0] == df_with_id.shape[0]
    assert 'individual_id' in df_with_id.columns

    results_df = pd.DataFrame(Z)
    results_df.loc[:, 'individual_id'] = df_with_id.loc[:, 'individual_id'].values
    results_df = move_last_col_to_first(results_df)
    return results_df

def remove_id_and_get_mat(Z_df):
    assert Z_df.columns[0] == 'individual_id'
    return Z_df.drop('individual_id', axis=1).values

def make_age_bins(bin_size=1, lower=40, upper=69):
    """
    Returns bins such that np.digitize(x, bins) does the right thing.
    """
    bins = np.arange(lower, upper+1, bin_size)
    bins = np.append(bins, upper+1)
    print(bins)
    return bins

def compute_column_means_with_incomplete_data(df):
    """
    Given a dataframe or numpy array df, computes means for each column. 
    Identical to np.array(data.df).mean(axis = 0) in case of no missing data. 
    """
    X = np.array(df)
    return np.nanmean(X, axis = 0)

def divide_idxs_into_batches(idxs, batch_size):
    """
    Given a list of idxs and a batch size, divides into batches. 
    """
    n_examples = len(idxs)
    n_batches = math.ceil(n_examples / batch_size)
    batches = []
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        batches.append(idxs[start:end])
    return batches





