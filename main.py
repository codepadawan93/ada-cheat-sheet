import matplotlib.pyplot as plot
import pandas as pd
import numpy as np
import seaborn as sb

from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.cross_decomposition import CCA
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as hiclu

# pandas
data_frame = pd.read_csv("./resources/example_data.csv", index_col=0)
data_frame.sort_values("globalRank")
# print(data_frame.values, data_frame.index)

# numpy
def standardise(ndarray):
    means = np.mean(ndarray)
    stddev = np.std(ndarray)
    ndarray = (ndarray - means) / stddev
    return ndarray

def replace_na(ndarray):
    return ndarray.fillna(np.nanmean(ndarray))

replaced = replace_na(data_frame)
standardised = standardise(replaced)

r = np.corrcoef(standardised, rowvar=False)
correlation_data_frame = pd.DataFrame(r)
eigenvalues, eigenvectors = np.linalg.eigh(correlation_data_frame)
reversed_eigenvalues = [i for i in reversed(np.argsort(eigenvalues))]
a = eigenvectors[reversed_eigenvalues]
alpha = eigenvectors[:,reversed_eigenvalues]

# plot.plot(alpha, correlation_data_frame.values, "r+")
# sb.heatmap(np.corrcoef(alpha), cmap='bwr', vmin=0, vmax=1, annot=True)
# plot.show()

# graphics
# draw a heatmap
x_series = [[21,1,1],
            [1,2,1],
            [2,1,1]]

y_series = [[10,1,6],
            [10,3,0],
            [1,0,0]]

heatmap_data = np.corrcoef(x_series, y_series)
max = np.max(heatmap_data)
min = np.min(heatmap_data)
# sb.heatmap(np.round(heatmap_data, 2), cmap='bwr', vmin=min, vmax=max, annot=True)

# draw a circle
data_x = np.random.uniform(-1, 1, (20))
data_y = np.random.uniform(-1, 1, (20))
plot.figure("Circle", figsize=(6, 6))
c = [x for x in np.arange(0, np.math.pi * 2, 0.01)]
x = [np.sin(x) for x in c]
y = [np.cos(x) for x in c]
plot.plot(x, y)
plot.plot(data_x, data_y, 'r+')
# plot.show()

# draw a line graph
data_xi = [x for x in np.arange(0, 1, 0.1)]
data_yi = [x * np.random.uniform(1, -1) for x in data_xi]
plot.figure("Line chart", figsize=(3, 6))
plot.plot(data_xi, data_yi)
# plot.show()

# pca
components = PCA().fit(data_frame).explained_variance_ratio_
plot.plot([k*100 for k in components], data_frame.columns)
plot.show()

#efa
factors = FactorAnalysis().fit(data_frame).components_
plot.plot(data_frame.columns, factors)
plot.show()

#cca
cca = CCA(n_components=4).fit(data_frame[data_frame.columns[:5]], data_frame[data_frame.columns[5:]])
print(cca.x_scores_, cca.y_scores_, cca.x_loadings_, cca.y_loadings_)

#hca
clusters = AgglomerativeClustering(n_clusters=6, affinity='manhattan', linkage='average').fit(data_frame)
print(clusters.labels_)
hiclu.dendrogram(np.ndarray(clusters.labels_))