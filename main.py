import anndata
import scanpy as sc
import numpy as np
from sklearn.decomposition import PCA as pca
import argparse
from kmeans import KMeans
import matplotlib.pyplot as plt

import plotly.express as px
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='number of clusters to find')
    parser.add_argument('--n-clusters', type=int,
                        help='number of features to use in a tree',
                        default=2)
    parser.add_argument('--data', type=str, default='Heart-counts.csv',
                        help='data path')

    a = parser.parse_args()
    return(a.n_clusters, a.data)

def read_data(data_path):
    return anndata.read_csv(data_path)

def preprocess_data(adata: anndata.AnnData, scale :bool=True):
    """Preprocessing dataset: filtering genes/cells, normalization and scaling."""
    sc.pp.filter_cells(adata, min_counts=5000)
    sc.pp.filter_cells(adata, min_genes=500)

    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    adata.raw = adata

    sc.pp.log1p(adata)
    if scale:
        sc.pp.scale(adata, max_value=10, zero_center=True)
        adata.X[np.isnan(adata.X)] = 0

    return adata


def PCA(X, num_components: int):
    return pca(num_components).fit_transform(X)

def main():
    print('n_classifiers, data_path = parse_args()')
    n_classifiers, data_path = parse_args()
    print('heart = read_data(data_path)')
    heart = read_data(data_path)
    print('heart = preprocess_data(heart)')
    heart = preprocess_data(heart)
    print('X = PCA(heart.X, 100)')
    X = PCA(heart.X, 200)
    # Your code
    
    #km = KMeans(n_classifiers,'random',300)
    print("km")
    km = KMeans(n_classifiers,'kmeans++',30)
    print("done")
    clustering = km.fit(X)
    
    #print('lol')
    X2 = PCA(heart.X, 3).T
    
    #print(X2.shape)
    #print(numpy.array(clustering))
    ####nd_visual(X2,clustering)
    #visualize_cluster(X2[0],X2[1],clustering)
    
    #km.silhouette(clustering,X)
    
def visualize_cluster(x, y, clustering):
    plt.scatter(x, y, s=None, c=clustering, alpha=0.5)
    plt.show()
       
    pass

def nd_visual(X,clustering):
    df = pd.DataFrame()
    df['x'] = X[0]
    df['y'] = X[1]
    df['z'] = X[2]
    df['clustering'] = clustering

    fig = px.scatter_3d(df, x='x', y='y', z='z',color='clustering',symbol=None)
    fig.show()


if __name__ == '__main__':
    main()
