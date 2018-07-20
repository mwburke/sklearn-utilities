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

    @property
    def means_(self):
        return self.scaler.means_

    @property
    def scales_(self):
        return self.scaler.scales_


class CustomImputer(BaseEstimator, TransformerMixin):
    '''Custom imputer class to allow the user to define a dictionary
       of columns and the values to impute.
    '''

    def __init__(self, imp_dict):
        self.imp_dict = imp_dict

    def fit(self, X):
        return self

    def transform(self, X):
        for col in self.imp_dict.keys():
            X[col].fillna(self.imp_dict[col], inplace=True)
        return X

    @property
    def statistics_(self):
        return list(self.imp_dict.values())


class CorrelationFeatureSelector(BaseEstimator, TransformerMixin):
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

    def __init__(self, direction='forward', feature_limit=None):
        self.direction = direction
        self.feature_limit = feature_limit

    def fit(self, X, Y):
        if self.direction == 'backward':
            if self.feature_limit is None:
                self.feature_limit = 1
            self.remaining_variables = backward_cfs(X, Y, self.feature_limit)
        elif self.direction == 'forward':
            if self.feature_limit is None:
                self.feature_limit = X.shape[1]
            self.remaining_variables = forward_cfs(X, Y, self.feature_limit)
        return self

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
            merits = [calculate_merit(corrs, feature_set + [ix]) for ix in variables]
            if max(merits) > max_score:
                max_score = max(merits)
                print('Current merit score: ', str(np.round(max_score, 3)))
                maxloc = merits.index(max(merits))
                feature_set.append(variables[maxloc])
                print('adding \'' + cols[variables][maxloc] + '\' at index: ' + str(maxloc))
                del variables[maxloc]
                dropped = True
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
        self.remaining_variables = calculate_vif_cols(X, self.threshold)
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


class IQROutlierRemover(BaseEstimator, TransformerMixin):
    '''Remove outliers from the dataset using the
       interquartile range (IQR) method:

       For each column:
        Q1 = 25% quartile
        Q3 = 75% quartile
        Calculate the IQR = Q3 - Q1
        Set min = Q1 - (IQR x multiplier)
        Set max = Q3 + (IQR x multiplier)
        Remove rows with points above min/max

       Params:
        multiplier: IQR multiplier
        minmax: takes in 'min', 'max', or 'both' to determine
                which sides to limit data on
        replace: boolean field whether to replace outliers
                 with NaNs or to drop rows with outlier values
    '''

    def __init__(self, multiplier=1.5, minmax='both', replace=False):
        self.multiplier = multiplier
        self.minmax = minmax
        self.replace = replace

    def fit(self, X):
        # calculate the outliers here and store in matrix
        self.outliers = X.apply(detect_outliers, axis=0, multiplier=self.multiplier, minmax=self.minmax)
        return self

    def transform(self, X):
        if self.replace:
            for c in X.columns:
                X[c][self.outliers[c]] = np.nan
        else:
            X = X.drop(X.index[self.outliers.apply(sum, axis=1) > 0])
        return X


def detect_outliers(arr, multiplier, minmax):
    q1, q3 = np.nanpercentile(arr, [25, 75])
    iqr = q3 - q1
    arr_min = q1 - iqr * multiplier
    arr_max = q3 + iqr * multiplier

    outlier_min = np.array([False] * len(arr))
    outlier_max = np.array([False] * len(arr))

    if (minmax == 'min') | (minmax == 'both'):
        outlier_min = arr < arr_min
    if (minmax == 'max') | (minmax == 'both'):
        outlier_max = arr > arr_max

    return outlier_min | outlier_max


class ClassMaxLimiter(BaseEstimator, TransformerMixin):
    '''Limit the number of values in the dataset for particular classes
       to a specified number by downsampling categories that have more
       entries than the limit.
    '''

    def __init__(self, columns, limit):
        self.columns = columns
        self.limit = limit

    def fit(self, X):
        return self

    def transform(self, X):
        return X.groupby([self.columns]).apply(limit_class, limit=self.limit)


def limit_class(df, limit):
    if df.shape[0] > limit:
        return df.sample(n=limit)
    return df
