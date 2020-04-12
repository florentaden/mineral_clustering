# -*- coding: utf8 -*-
import time
import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import rescale
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

# ##############################################################################
# Read image
image_file = 'images/image.jpg'
img = plt.imread(image_file)

# and resize it to 30% of the original size to speed up the processing
img = rescale(img, 0.3, multichannel=True) # rescaling too accelerate process
X = np.reshape(img, (-1, 3)) # reshape img with RGB as 3 features

# ##############################################################################
# Do the clustering

to = time.time() # start clock
n_clusters = 6 # number of cluster we are looking for
kmeans = KMeans(n_clusters=n_clusters,
                   init='k-means++', # initial centroid seeds
                   n_init=10) # Number of run with different centroid seeds
                              # so kmeans doesnt fall into local minima

# see https://https://scikit-learn.org/stable/modules/clustering.html#k-means
# for more details

prediction = kmeans.fit_predict(X) # apply kmeans to our data-set
centers = kmeans.cluster_centers_ #  the typical color of each cluster
if np.mean(centers) > 1:
    centers = np.uint8(centers)
# depending of the image and preprocess, RGB can be > 1. If so, cant't be float
labels = kmeans.labels_ # the cluster number of each pixel
new_img = np.reshape(centers[labels.flatten()], img.shape) # segmented image

print('Elapsed time: {:.1f} s'.format(time.time() - to))
print('Number of pixels: ', labels.size)
print('Number of clusters: ', np.unique(labels).size)

# ##############################################################################

""" How many cluster are we actually looking for? This can take some time.."""

# see more about silhouette analysis
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html

# silhouette_avg = silhouette_score(X, prediction)
# print("For n_clusters =", n_clusters,
#       "The average silhouette_score is :", silhouette_avg)
#
# sample_silhouette_values = silhouette_samples(X, prediction)

# ##############################################################################
# Plot the results

fig, ax = plt.subplots(2, 3, figsize=(8, 6), sharex=True, sharey=True)
ax[0, 0].imshow(img)
ax[0, 0].set_title(r'Original Image', fontsize=14)
ax[-1, -1].imshow(new_img)
ax[-1, -1].set_title('Segmented Image (N={})'.format(n_clusters), fontsize=12)

index = np.random.choice(range(n_clusters), size=n_clusters, replace=False)
for a, axis in enumerate(ax.flatten()[1:-1]):
    i = index[a]
    clus_img = np.zeros_like(img)
    idx = labels == i
    clus_img[np.reshape(idx, img.shape[:2]), :] = centers[i]
    clus_img[clus_img == 0] = np.nan
    axis.imshow(clus_img)
    axis.set_title('Cluster #{}'.format(i), fontsize=12)

for axis in ax.flatten():
    axis.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

plt.tight_layout()
plt.savefig('figures/mineral_kmeans_{}.png'.format(n_clusters), dpi=300)
# plt.show()
plt.close()

# ##############################################################################
# Silhouette plot
#
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# ax2.imshow(new_img), ax2.tick_params(labelleft=False, left=False, bottom=False,
#     labelbottom=False)
# ax2.set_title(r'Segmented Image (N$_{cluster}$ = %d)' %n_clusters, fontsize=14)
# ax1.set_xlim([-0.1, 1])
# ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
# y_lower = 10
#
# for i in range(n_clusters):
#     ith_cluster_silhouette_values = \
#             sample_silhouette_values[prediction == i]
#
#     ith_cluster_silhouette_values.sort()
#     size_cluster_i = ith_cluster_silhouette_values.shape[0]
#     y_upper = y_lower + size_cluster_i
#
#     color = centers[i]
#     ax1.fill_betweenx(np.arange(y_lower, y_upper),
#                       0, ith_cluster_silhouette_values,
#                       facecolor=color, edgecolor=color)
#
#     # Label the silhouette plots with their cluster numbers at the middle
#     ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
#     y_lower = y_upper + 10  # 10 for the 0 samples
#
# ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
# ax1.set_yticks([])  # Clear the yaxis labels / ticks
# ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
#
# plt.tight_layout()
# plt.savefig('figures/silhouette_kmeans_{}.png'.format(n_clusters), dpi=200)
# plt.close()
