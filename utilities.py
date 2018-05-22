from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from fancyimpute import KNN
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import pandas as pd


class ThresholdColDropper(BaseEstimator, TransformerMixin):
    '''Custom class to automatically drop any fields that have
       more null values than the threshold the class is initialized with.
    '''

    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, X):
        nulls = X.apply(func=lambda x: x.isnull().sum() / len(x), axis=0)
        self.dropcols = X.columns[nulls > self.threshold]
        return self

    def transform(self, X):
        return X.drop(columns=self.dropcols)


class KNNImputer(BaseEstimator, TransformerMixin):
    '''Custom imputer class to integrate fancyimputer's KNN imputer
       into the sklearn Pipeline functionality as a transformer.
       Initialized with the number of closes neighbors to compare to.
       Input is pandas dataframe and returns pandas dataframe.
    '''

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        self.cols = X.columns
        return self

    def transform(self, X):
        df = pd.DataFrame(KNN(k=self.k).complete(X))
        df.columns = self.cols
        return df


class CustomScaler(BaseEstimator, TransformerMixin):
    '''Custom scaler class to only apply the standardscaler to
       columns that are not in the initialized variable cat_cols.
    '''

    def __init__(self, cat_cols):
        self.cat_cols = cat_cols
        self.scaler = StandardScaler()

    def fit(self, X):
        self.continuous_cols = [col for col in X.columns if col not in self.cat_cols]
        self.scaler.fit(X[self.continuous_cols])
        return self

    def transform(self, X):
        X[self.continuous_cols] = self.scaler.transform(X[self.continuous_cols])
        return X


class CFSVariableReducer(BaseEstimator, TransformerMixin):
    '''Class to recursively remove features based upon the heuristic "merit"
       which is a measure of the ratio of variable correlation to the target
       vs inter-variable correlation.

       For backward direction, it emoves features recursively until the
       merit score no longer increases. Returns dataframe with only the
       columns left in the subset.

       For forward direction, it adds features recursively, keeping the
       best scoring set of features until the merit score stops increasing.

       Inspired by: https://www.cs.waikato.ac.nz/~mhall/thesis.pdf
    '''

    def __init__(self, direction='forward', min_features=1, max_features=None):
        self.direction = direction
        self.min_features = min_features
        self.max_features = max_features

    def fit(self, X, Y):
        if self.direction == 'backward':
            self.remaining_variables = backward_cfs(X, Y, self.min_features)
        elif self.direction == 'forward':
            if self.max_features is None:
                self.max_features = X.shape[1]
            self.remaining_variables = forward_cfs(X, Y, self.max_features)

    def transform(self, X):
        return X[self.remaining_variables]


def forward_cfs(X, Y, max_features):
    X['target'] = Y
    corrs = X.corr().abs().values
    cols = X.columns
    variables = list(range(corrs.shape[1] - 2))
    dropped = True
    max_score = 0
    feature_set = []
    while dropped:
        dropped = False
        if len(feature_set) < max_features:
            merits = [calculate_merit(corrs, feature_set + [ix]) for ix in range(len(variables))]
            if max(merits) > max_score:
                max_score = max(merits)
                maxloc = merits.index(max(merits))
                feature_set.append(variables[maxloc])
                print('adding \'' + cols[variables][maxloc] + '\' at index: ' + str(maxloc))
                del variables[maxloc]
                dropped = True
    print('Final merit score: ', str(np.round(max_score, 3)))
    print('Remaining variables:')
    print(cols[feature_set])
    return cols[feature_set]


def backward_cfs(X, Y, min_features):
    X['target'] = Y
    corrs = X.corr().abs().values
    cols = X.columns
    variables = list(range(corrs.shape[1] - 2))
    dropped = True
    max_score = 0
    while dropped:
        dropped = False
        if len(variables) > min_features:
            merits = [calculate_merit_exclude(corrs[:,variables], ix) for ix in range(corrs[:,variables].shape[1])]
            if max(merits) > max_score:
                max_score = max(merits)
                minloc = merits.index(min(merits))
                print('dropping \'' + cols[variables][minloc] + '\' at index: ' + str(minloc))
                del variables[minloc]
                dropped = True
    print('Remaining variables:')
    print(cols[variables])
    return cols[variables]


def calculate_merit_exclude(X, skipcol):
    variables = list(v for v in range(X.shape[1] - 1) if v != skipcol)
    return calculate_merit(X, variables)


def calculate_merit(X, variables):
    k = len(variables)
    rcf = np.nanmean(X[0:X.shape[1] - 1, X.shape[1] - 1])
    rff = np.nanmean([X[v, :] for v in variables])
    return (k * rcf) / (k + k * (k - 1) * rff) ** 0.5


class VIFVariableReducer(BaseEstimator, TransformerMixin):
    '''Class to recursively remove features based upon VIF values
       above a certain threshold. Returns the reduced dataframe with
       only the remaining variables.
    '''

    def __init__(self, threshold=5.0):
        self.threshold = threshold

    def fit(self, X):
        self.remaining_variables = calculate_vif_cols(X, self.threshold, self.num_cores)
        return self

    def transform(self, X):
        return X[self.remaining_variables]


def calculate_vif_cols(X, thresh):
    cols = X.columns
    variables = list(range(X.shape[1]))
    X = X.values
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X[:,variables], ix) for ix in range(X[:,variables].shape[1])]
        if max(vif) > thresh:
            maxloc = vif.index(max(vif))
            print('dropping \'' + cols[variables][maxloc] + '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True
    print('Remaining variables:')
    print(cols[variables])
    return cols[variables]
