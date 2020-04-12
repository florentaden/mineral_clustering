# -*- coding: utf8 -*-
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rescale
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering

# ##############################################################################
# Read image
image_file = 'images/image.jpg'
img = plt.imread(image_file)

# and resize it to 30% of the original size to speed up the processing
img = rescale(img, 0.3, multichannel=True)

# initial connectivity matrix, tells you which pixel are close to each other
x, y, z = img.shape
connectivity = grid_to_graph(x, y)

# ##############################################################################
# Do the clustering

to = time.time()
n_clusters = 10  # number of clusters we want to find

# AgglomerativeClustering recursively merges the pair of clusters
# that minimally increases a given linkage distance;
# Here the linkage distance is "ward";
# It minimizes the variance of the clusters being merged.
ward = AgglomerativeClustering(
    n_clusters=n_clusters,
    linkage='ward',
    connectivity=connectivity)

X = np.reshape(img, (-1, 3))  # necessary format (Npixels, Nfeatures)
ward.fit(X)
label = np.reshape(ward.labels_, (x, y))
print('Elapsed time: {:.1f} s'.format(time.time() - to))
print('Number of pixels: ', label.size)
print('Number of clusters: ', np.unique(label).size)

# ##############################################################################
# Plot the results
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(img)
for n in range(n_clusters):
    ax.contour(label == n, linewidths=0.5,
        colors=[plt.cm.nipy_spectral(n / float(n_clusters)), ])
ax.tick_params(labelbottom=False, bottom=False, labelleft=False, left=False)
plt.tight_layout()
plt.savefig('figures/mineral_agglo_{}.png'.format(n_clusters), dpi=300)
plt.close()
