

import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"C:\Users\Avinash\Downloads\Mall_Customers.csv")

x = dataset.iloc[:,[3,4]]

import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrograpm(sch.linkage(x, method = 'ward'))

plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Eucliden distance')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, linkage='ward')
y_hc = hc.fit_predict(x)

# visualising a cluster
plt.scatter()(x[y_hc == 0, 0],x[y_hc == 0, 1], s = 100, c = 'red', )
plt.scatter()(x[y_hc == 1, 0],x[y_hc == 1, 1], s = 100, c = 'blue', )
plt.scatter()(x[y_hc == 2, 0],x[y_hc == 2, 1], s = 100, c = 'green', )
plt.scatter()(x[y_hc == 3, 0],x[y_hc == 3, 1], s = 100, c = 'cyan', )
plt.scatter()(x[y_hc == 4, 0],x[y_hc == 4, 1], s = 100, c = 'magneta, label =
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score (1-100')
plt.legend()
plt.show()

dataset['cluster'] = y_hc

# do compare k-mean clustering and hirarchiecal clustering

# clustering: pca,kmeans,hirachiecal,dbscan

