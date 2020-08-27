# 1-D-classifier

This program performs the following tasks on data consisting of age and height of snowfolks.
1. Exploratory Data Analysis
2. 1D Clustering using Otsu's method
3. 1D Clustering with Regularization

Otsu’s method of one dimensional clustering is used to binarize the data. The following graph gives the point showing the best threshold value i.e. the value used to segment the data into two clusters.

![](https://github.com/aishwaryasontakke/1-D-classifier/blob/master/Mixed%20variance%20vs%20Age.png?raw=true)

Thus, 42 years is the best age to separate both clusters since minimum variance i.e. 56.28 was found at this point.

No change is noticed along with regularization which might be because we use regularization to get data of almost equal split whereas the data in this case is already pretty balanced.
