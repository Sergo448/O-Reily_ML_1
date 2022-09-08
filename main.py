import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()

from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA as RandomizedPCA

faces = fetch_lfw_people(min_faces_per_person=60)
print('faces.target_names:    ', faces.target_names)
print('faces.images.shape:    ', faces.images.shape)

pca = RandomizedPCA(n_components=150, svd_solver='randomized', whiten=True)
pca.fit(faces.data)

fig, axes = plt.subplots(3, 8, figsize=(9, 4),
                         subplot_kw={'xticks':[],
                                     'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(pca.components_[i].reshape(62, 47), cmap='bone')
 #   plt.show()