# Sklearn Utilities

This is a collection of various ML utilities based on the scikit-learn fit/transform paradigm for general integration into the ecosystem and pipelines in particular.

Most are focused on preprocessing and feature selection.

Most have dependencies and could do with a bit of optimization.

## Classes:

`ThresholdColDropper`: drops all columns with null values greater than initialized threshold

`KNNImputer`: wrapper around `fancyimpute`'s KNN imputation method

`CustomScaler`: wrapper around `scikit-learn`'s `StandardScaler` but with the option to omit fields from the scaling (such as binary integer fields)

`PercentileScaler`: similar to `scikit-learn`'s `StandardScaler` but specify the upper and lower percentiles to use as the max/min values with which to scaler for each column rather than using the min/max.

`CustomImputer`: allows the user to specify a dictionary of columns and their associated imputation values, replaes all NA values with the imputation values

`CorrelationFeatureSelector`: selects features recursively in a forward or backward manner based on the ratio of feature-target correlation to a mean within-feature correlation heuristic. Based off of this paper (chapter 4): [Correlation-based Feature Selection for
Machine Learning](https://www.cs.waikato.ac.nz/~mhall/thesis.pdf)

`VIFVariableReducer`: recursively drops features with the lowest VIF (variance inflation factor) until the scores fall below an initialized threshold

`IQROutlierRemover`: identifies outliers based on the internal quartile range (IQR) and allows the user to either remove rows with outliers or replace outliers with null values. More details are in the function comment

`ClassMaxLimiter`: Limit the number of values in the dataset for particular fields by downsampling to a specified limit if the number of rows for that class exceed it
