import numpy as np
from scipy import ndimage
from copy import deepcopy

class Cluster():
    '''
    Object used to gather information on clusters of pixels
    of the same color on an image
    '''
    def __init__(self, clustered, min_points = 200, min_density = 0.3,
                make_boundaries = False, fill_holes = False):
        '''
        Inputs:
        clustered -> uint array in which all pixels of a cluster 
                have the same value (0 is used for the background)
        min_points -> minimum number of pixels for which a cluster
                is saved in the instance
        min_density -> minimum proportion of pixel in the cluster
                with respect to the pixels of the smallest rectangle
                that covers all the cluster
        make_boundaries -> set to True to create 'self.boundaries'
        fill_holes -> set to True the fill the holes of each cluster
                      before saving them             
        Creates:
        self.clustered -> np.uint8 array, similar to 'clustered',
                but includes only clusters that passed both tests
                (ids start from 1 and have no gaps)
        self.info -> dict with information and the position, size,
                etc...(see below) of each cluster that passed tests
        self.boundaries -> array similar to self.clustered, but all
                interior points are treated as background
        
        ! To improve performance only up to 255 clusters are
          saved (almost impossible to have more per image)
        '''
        self.made_boundaries = make_boundaries # save for later
        self.clustered = np.zeros_like(clustered, dtype = np.uint8)
        # Creates dictionary with information on the size
        # of each cluster, its position, etc..
        self.info = {'ids':[], 'height':[], 'width':[], 'area':[],
                    'minY':[], 'maxY':[], 'meanY':[], 'midY':[],
                    'minX':[], 'maxX':[], 'meanX':[], 'midX':[],
                    'points_count':[]}
        if make_boundaries: # defines additional elements
            self.boundaries = self.clustered.copy()
            self.info['b_points_count'] = []
            self.info['b_meanY'] = []
            self.info['b_meanX'] = []
        k = 0 # index used to create ids (after conducting the
              # two tests) without gaps
        # Gets all ids in clustered (zero, which is 1st, is ignored)
        ids_all = np.unique(clustered)[1:] 
        # Checks if cluster with id i:
        # 1) has more than min_points
        # 2) the points as a proportion of the area of the smallest
        #    rectangle that covers the cluster are at least min_prop
        # ! maximum number of clusters that is saved is 255
        for i in ids_all:
            if k <= 255:
                is_cluster = np.equal(clustered, i)
                points_count = np.count_nonzero(is_cluster)
                if points_count > min_points: # pixel count test
                    # Gets coordinates of all pixels in the cluster
                    Y, X = np.nonzero(is_cluster) 
                    minY, maxY = np.min(Y), np.max(Y)
                    minX, maxX = np.min(X), np.max(X)
                    height, width = maxY-minY, maxX-minX
                    area = height*width
                    if points_count/area > min_density: # density test
                        # updates index and saves data
                        k += 1
                        self.info['ids'].append(k)
                        self.info['minY'].append(minY)
                        self.info['maxY'].append(maxY)
                        self.info['minX'].append(minX)
                        self.info['maxX'].append(maxX)
                        self.info['height'].append(height)
                        self.info['width'].append(width)
                        self.info['area'].append(area)
                        midY, midX = (maxY+minY)//2, (maxX+minX)//2
                        self.info['midY'].append(midY)
                        self.info['midX'].append(midX)
                        # Gets 'is_filled' array, after filling the holes
                        # of 'is_cluster' (required to construct the boundary)
                        if make_boundaries or fill_holes:
                            is_filled  = ndimage.binary_fill_holes(is_cluster)
                        # Save 'is_cluster' or 'is_filled'
                        if fill_holes:
                            self.clustered[is_filled]  = k
                            Y, X = np.nonzero(is_filled)
                        else:
                            self.clustered[is_cluster] = k
                        # Saves elements that depend on above choise
                        self.info['points_count'].append(len(Y))
                        meanY, meanX = int(np.mean(Y)), int(np.mean(X))
                        self.info['meanY'].append(meanY)
                        self.info['meanX'].append(meanX)
                        # Finds the boundary of the cluster, and its info
                        if make_boundaries:
                            # Gets the boundary as the subset of pixels
                            # not completed surrounded by other pixels of
                            # this cluster
                            adjusted_pixel_count = ndimage.correlate(
                                is_filled.astype(int),
                                np.ones((3,3),dtype=int), mode='constant')
                            not_inter = np.less(adjusted_pixel_count, 9)
                            is_boundary  = np.logical_and(not_inter, is_filled)
                            self.boundaries[is_boundary] = k
                            Y, X = np.nonzero(is_boundary)
                            self.info['b_points_count'].append(len(Y))
                            meanY, meanX = int(np.mean(Y)), int(np.mean(X))
                            self.info['b_meanY'].append(meanY)
                            self.info['b_meanX'].append(meanX)
        self.n = len(self.info['ids']) # saves number of clusters

    def trim_data(self, to_keep, in_place = False):
        '''
        Inputs: 
            to_keep -> bool 1-d array with lenght equal to 'self.cluster_n'
            in_place -> set to True to update current instance,
                    instead of returing a new one
        1) Gets ids of clusters in rows of self.info['ids'] marked with
            True by to_keep
        2) Deletes remaining clusters from 'self.clustered', 'self.info',
            and 'self.boundaries' (if defined)
        3) Creates new ids for the remaining clusters, so that ids start
            from 1 and have no gaps
        '''
        # Gets row indexes (in .info) of the cluster to be kept
        to_keep_idxs = [i for i in range(self.n) if to_keep[i]]
        new_n = len(to_keep_idxs) # new number of clusters
        # Creates new empty dictionary and arrays
        new_clustered = np.zeros_like(self.clustered, dtype = np.uint8)
        new_info = {}
        for key in self.info.keys():
            new_info[key] = []
        if self.made_boundaries:
            new_boundaries = new_clustered.copy()
        # For each cluster passes information on new elements
        k = 0
        for i in range(new_n):
            k += 1
            j = to_keep_idxs[i]
            id_ = self.info['ids'][j]
            # Passes info on new_*
            is_cluster = np.equal(self.clustered, id_)
            new_clustered[is_cluster] = k
            # Passes data to new_info
            for key in new_info.keys():
                new_info[key].append(self.info[key][j])
            new_info['ids'][-1] = k
            if self.made_boundaries:
                is_boundary = np.equal(self.boundaries, id_)
                new_boundaries[is_boundary] = k
        # Uses 'new_*' to update corresponding elements of instance
        if in_place:
            self.n = new_n
            self.clustered = new_clustered
            self.info = new_info
            if self.made_boundaries:
                self.boundaries = new_boundaries
        else:
            new_self = deepcopy(self)
            new_self.n = new_n
            new_self.clustered = new_clustered
            new_self.info = new_info
            if new_self.made_boundaries:
                new_self.boundaries = new_boundaries
            return new_self
            
    def merge_clusters(self, pairs, in_place = False):
        '''
        Inputs:
            pairs -> list, the 1st and 2nd element of each row have
                    the ids of the clusters to be merged
            in_place -> set to True to update current instance,
                    instead of returing a new one
        Merges pairs into new clusters and deletes the rest of the
        clusters. To keep cluster without merging it, create row
        in which its id appears twice.
        
        Ids are reassigned so that there are no gaps.
        '''
        new_n = len(pairs)
        # Creates new 'clustered'
        new_clustered = np.zeros_like(self.clustered, dtype=np.uint8)
        for i in range(new_n):
            if pairs[i][0] != pairs[i][1]:
                is_cluster = np.logical_or(
                    np.equal(self.clustered, pairs[i][0]),
                    np.equal(self.clustered, pairs[i][1]))
            else:
                is_cluster = np.equal(self.clustered, pairs[i][0])
            new_clustered[is_cluster] = i+1
        # Creates new 'boundaries'
        if self.made_boundaries:
            new_boundaries = np.zeros_like(self.boundaries, dtype=np.uint8)
            for i in range(new_n):
                if pairs[i][0] != pairs[i][1]:
                    is_boundary = np.logical_or(
                        np.equal(self.boundaries, pairs[i][0]),
                        np.equal(self.boundaries, pairs[i][1]))
                else:
                    is_boundary = np.equal(self.boundaries, pairs[i][0])
                new_boundaries[is_boundary] = i+1
        # Creates new 'info'
        new_info = {'ids':[], 'height':[], 'width':[], 'area':[],
                    'minY':[], 'maxY':[], 'meanY':[], 'midY':[],
                    'minX':[], 'maxX':[], 'meanX':[], 'midX':[],
                    'points_count':[]}
        if self.made_boundaries: # defines additional elements
            new_info['b_points_count'] = []
            new_info['b_meanY'] = []
            new_info['b_meanX'] = []
        for i in range(new_n):
            # If single cluster, simple copies entries
            if pairs[i][0] == pairs[i][1]:
                j = self.info['ids'].index(pairs[i][0])
                for key in new_info.keys():
                    new_info[key].append(self.info[key][j])
                new_info['ids'][-1] = i+1
            # Else, gets indexes of the two clusters in self.info
            # and calculates the entry of the merged cluster using
            # those of its components
            else:
                j0 = self.info['ids'].index(pairs[i][0])
                j1 = self.info['ids'].index(pairs[i][1])
                new_info['ids'].append(i+1)
                minY = min(self.info['minY'][j0], self.info['minY'][j1])
                maxY = max(self.info['maxY'][j0], self.info['maxY'][j1])
                minX = min(self.info['minX'][j0], self.info['minX'][j1])
                maxX = max(self.info['maxX'][j0], self.info['maxX'][j1])
                new_info['minY'].append(minY)
                new_info['maxY'].append(maxY)
                new_info['minX'].append(minX)
                new_info['maxX'].append(maxX)
                height, width = maxY-minY, maxX-minX
                area = height*width
                new_info['height'].append(height)
                new_info['width'].append(width)
                new_info['area'].append(area)
                midY, midX = (maxY+minY)//2, (maxX+minX)//2
                new_info['midY'].append(midY)
                new_info['midX'].append(midX)
                points_count = self.info['points_count'][j0] \
                             + self.info['points_count'][j1]
                new_info['points_count'].append(points_count)
                sumY = self.info['meanY'][j0]*self.info['points_count'][j0]\
                     + self.info['meanY'][j1]*self.info['points_count'][j1]
                sumX = self.info['meanX'][j0]*self.info['points_count'][j0]\
                     + self.info['meanX'][j1]*self.info['points_count'][j1]
                meanY, meanX = sumY/points_count, sumX/points_count
                new_info['meanY'].append(meanY)
                new_info['meanX'].append(meanX)
                if self.made_boundaries:
                    b_points_count = self.info['b_points_count'][j0] \
                                   + self.info['b_points_count'][j1]
                    new_info['b_points_count'].append(b_points_count)
                    sumY = self.info['b_meanY'][j0]*\
                                    self.info['b_points_count'][j0]\
                         + self.info['b_meanY'][j1]*\
                                    self.info['b_points_count'][j1]
                    sumX = self.info['b_meanX'][j0]*\
                                    self.info['b_points_count'][j0]\
                         + self.info['b_meanX'][j1]*\
                                    self.info['b_points_count'][j1]
                    b_meanY, b_meanX = sumY/b_points_count, sumX/b_points_count
                    new_info['b_meanY'].append(b_meanY)
                    new_info['b_meanX'].append(b_meanX)
        # Uses 'new_*' to update corresponding elements of instance
        if in_place:
            self.n = new_n
            self.clustered = new_clustered
            self.info = new_info
            if self.made_boundaries:
                self.boundaries = new_boundaries
        else:
            new_self = deepcopy(self)
            new_self.n = new_n
            new_self.clustered = new_clustered
            new_self.info = new_info
            if new_self.made_boundaries:
                new_self.boundaries = new_boundaries
            return new_self
                                
