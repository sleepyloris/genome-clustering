import numpy as np


class KMeans():

    def __init__(self, n_clusters: int, init: str='random', max_iter = 300):
        """

        :param n_clusters: number of clusters
        :param init: centroid initialization method. Should be either 'random' or 'kmeans++'
        :param max_iter: maximum number of iterations
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = None # Initialized in initialize_centroids()

    def fit(self, X: np.ndarray):
        self.initialize_centroids(X)

        clustering = np.zeros(X.shape[0], dtype=int)
        #clustering = np.zeros(len(X[0]), dtype=int)


        iteration = 0
        while iteration < self.max_iter:
            clustering = self.return_clustering(X)
            self.update_centroids(clustering, X)
            iteration +=1

        return clustering
    
    
    def return_clustering(self, X: np.ndarray):
        clustering = []
        
        for datum in X:
            tval = self.euclidean_distance(datum, self.centroids[0])
            tnum = 0
            
            for c in range(0, len(self.centroids)): #cange to 1?
                if self.euclidean_distance(datum, self.centroids[c]) < tval: 
                    tval = self.euclidean_distance(datum, self.centroids[c])
                    tnum = c
            
            clustering.append(tnum)
        return clustering

    
    def update_centroids(self, clustering: np.ndarray, X: np.ndarray):
        clusterSum = np.zeros([self.n_clusters,len(X[0])])
        entries = np.zeros(self.n_clusters)
        for n in range(0,len(X)):
            clusterSum[clustering[n]] += X[n]
            entries[clustering[n]] +=1

        for n in range(0,self.n_clusters):
            self.centroids[n] = np.divide(clusterSum[n],entries[n])

            
    def pDistance(self, X, rnd_indices):#->np.ndarray:
        #X = XT.copy()
        rArray = [0 for _ in range(0,len(X))] #np.ndarray()
       
        for a in range(0,len(X)):
            if a in rnd_indices:
                rArray[a] = 0
                continue
            
            localMin = 0
            
            for b in range( 1,len(rnd_indices) ):
                if self.euclidean_distance(X[a], X[b]) < self.euclidean_distance(X[a], X[localMin]): localMin = b
            #print(str(a) + " is not in" + str(rnd_indices) + " " + str(self.euclidean_distance(X[a],X[localMin])))
            rArray[a] = self.euclidean_distance(X[a],X[localMin])

        #print("rArray")
        #print(rArray)
        nArray = rArray.copy()

        for n in range(0, len(rArray)):
            if(rArray[n] == 0):
                nArray[n] = 0
            else:               
                nArray[n] = ((rArray[n])/sum(rArray)) #1-
        
        #print('nArray')
        #print(nArray)
        #print('rnd')
        #print(rnd_indices)
        return nArray
        
        
    def initialize_centroids(self, X: np.ndarray):
        """
        Initialize centroids either randomly or using kmeans++ method of initialization.
        :param X:
        :return:
        """
        if self.init == 'random':
            # your code
            #self.centroids = np.random.choice(X, size=self.n_clusters, replace=False)
            
            rnd_indices = np.random.choice(len(X), size=self.n_clusters, replace=False)
            self.centroids = X[rnd_indices]

        elif self.init == 'kmeans++':
            # your code
            rnd_indices = []
            rnd_indices.append(np.random.choice(len(X), size=1, replace=False)[0])
            for n in range(1,self.n_clusters):
                rnd_indices.append(np.random.choice( len(X), size=1, replace=False, p=self.pDistance(X,rnd_indices) )[0])
            print(rnd_indices)
            self.centroids = X[rnd_indices]
        else:
            raise ValueError('Centroid initialization method should either be "random" or "k-means++"')

           
    def euclidean_distance(self, X1:np.ndarray, X2:np.ndarray):
        """
        Computes the euclidean distance between all pairs (x,y) where x is a row in X1 and y is a row in X2.
        Tip: Using vectorized operations can hugely improve the efficiency here.
        :param X1:
        :param X2:
        :return: Returns a matrix `dist` where `dist_ij` is the distance between row i in X1 and row j in X2.
        """
        # your code
        distance = abs(np.subtract(X1,X2)) 
        rvalue = np.sqrt(np.dot(distance,distance))
        return rvalue

    def secondBest(self,x: np.ndarray)->np.int:
        if( self.euclidean_distance(x,self.centroids[0]) > self.euclidean_distance(x,self.centroids[1]) ):
            best = 0
            secondB = 1
        else:
            best = 1
            secondB = 0

        for centroid in range(0,len(self.centroids) ): #change to self.n_clusters
            if self.euclidean_distance(centroid, x) > self.euclidean_distance(self.centroids[best], x):
                secondB = best
                best = centroid
        return secondB    

    def silhouette(self, clustering, X):
        both = []
        
        for a in range(0,len(X)):
            intra = self.euclidean_distance(X[a],self.centroids[clustering[a]])
            inter = self.euclidean_distance(X[a],self.centroids[self.secondBest(a)])
            #both.append( (intra - inter) / max(intra,inter))
            both.append( (intra - inter) / max(intra,inter))
        
        rvalue = ( sum(both)/len(both) )
        return rvalue