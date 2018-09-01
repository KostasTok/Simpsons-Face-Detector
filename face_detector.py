import numpy as np
from scipy import ndimage
from skimage import measure
import face_detector_help as fdh
import colors_simplifier as cs
from cluster_info import Cluster

def find_face(img, return_extra = False):
    '''
    Applies a variety of filters in order to detect the faces
    of the Simpson characters on a image
    
    Inputs:
    img -> uint8 array with the RGB values of the image
        (width*height*3)
    return_extra -> Set to true to get two extra images. 
        a) The 1st is the simplified representation of img 
            (with only four colors)
        b) The 2nd is the black everywhere else apart from
            the squares that contain a face
    Returns:
    img_windows -> list with the RGB values of 200*200 squares
        that contain a face cropped out from the original image
    windows -> same with the above, but in this case the simplified
        version of those faces (constructed below) is returned
    extra1, extra2 -> (if return_extra==True) as described above
    '''
    '''
    1) Gets bool array with a simplified representation of the 
        image with only four colors: 
            (Black, White, Yellow, Other)
        A black pixel is indicated by True on the corresponding
        depth, etc..
    '''
    BYWO = cs.getBYWO(img)
    # Depths are fixed by cs.getBYWO() to be of the following form.
    # There are defined as variables here to make future alterations
    # easier.
    blacks_depth  = 0
    yellows_depth = 1
    whites_depth  = 2
    others_depth  = 3
    '''
    2) Constructs clusters of white pixels (color of eyes):
    Uses Cluster() to save info on clusters that have no less than the
    required number of pixels and density. Instance has:
        whites.clustered -> uint array in which all pixels of a cluster 
                have the same value (0 is used for the background)
        whites.info -> dict with information and the position, size,
                etc...(see below) of each cluster that passed tests
        whites.boundaries -> array similar to self.clustered, but all
                interior points are treated as background
    '''
    clustered = measure.label(BYWO[:,:,whites_depth ], background=False)
    whites = Cluster(clustered, min_points = 200, min_density = 0.3,
                     make_boundaries = True, fill_holes = True)
    '''
    3) Gets position of black pixels surrounded by a sufficiently
    big number of other blacks to indicate a pupil
    '''
    black_density = ndimage.correlate(
                        BYWO[:,:,blacks_depth].astype(int),
                        np.ones((5,5), dtype=int), mode='constant')
    maybe_pupil = np.greater_equal(black_density, 12)
    '''
    4) Identifies clusters of whites (or the combination of two):
        a) the shape of which resembles that of a pair of eyes
        b) maybe_pupil indicates that they contain pupils
    Uses result to create copy of BYWO, which has all non-eye white
    pixels and all black pixels in the background
    '''
    eyes = get_eyes(whites, maybe_pupil)
    # Gets new pixel arrays from eyes
    is_eyes = np.greater(eyes.clustered, 0)
    is_yellow = np.logical_and(BYWO[:,:,yellows_depth],
                               np.logical_not(is_eyes))
    is_back = np.logical_not(
                np.logical_or(is_eyes, is_yellow))
    # Creates updated BYWO
    BYWO2 = BYWO.copy()
    BYWO2[:,:,whites_depth]  = is_eyes
    BYWO2[:,:,yellows_depth] = is_yellow
    BYWO2[:,:,blacks_depth]  = False
    BYWO2[:,:,others_depth]  = is_back
    
    '''
    5) Indentifies potential face skin:
        a) Constructs yellows instance with info on clusters of
            yellows (similar to whites)
        b) Deletes those that are either too small, or too far
            from the identified pairs of eyes
    '''
    clustered = measure.label(BYWO[:,:,yellows_depth ], background=False)
    # !1 boundaries of whole face will be found later, so boundaries of
    #    yellows are not needed
    # !2 holes are not filled so that the area of eyes is not covered
    yellows = Cluster(clustered, min_points = 500, min_density = 0,
                      make_boundaries = False, fill_holes = False)
    skin = get_skin(eyes, yellows, max_min_dist = 200)
    '''
    6) Constructs faces:
        a) For each cluster in 'eyes' forms face skin by identifying
            the best match from the 'skin' instance
        b) Reconstructs skin clusters to ensure that each corresponds
            to only one face
    ''' 
    faces, BYWO2, eyes, skin  = get_faces(BYWO2, eyes, skin)
    '''
    7) Gets cover windoes:
        Gets position of windows that cover the constucted faces
        and returns a BYWO like array in which all pixels out of 
        those windows have turned to black
    '''
    window_min_size = 200
    pos, BYWO3, windows = get_windows(
                    faces, BYWO2, eyes, skin,
                    blacks_depth, window_min_size)
    '''
    8) Constructs final output
    '''
    img_windows = get_img_windows(pos, img)
    if return_extra:
        extra1 = cs.BYWO_to_RGB(BYWO)
        extra2 = cs.BYWO_to_RGB(BYWO3)
        return img_windows, windows, extra1, extra2
    else:
        return img_windows, windows
    
def get_eyes(whites, maybe_pupil):
    '''
    Inputs:
        whites -> instance as defined in main code
        maybe_pupil -> bool array of size similar to 
            whites.clustered, with True on points that could be
            part of a pupil (inference based on density of black
            pixels around this points)
    1) Tests if the shape of the boundary of each cluster of whites
        resembles that of one, or two eyes
    2) a) Merges clusters that passed the one (or two) eye tests.
          Each new cluster corresponds to a potential pair of eyes
       b) Saves clusters that passed only the two eyes test and were
          not merged as a pair of eyes
    3) Applies additional tests based on the shape of the merged
        clusters to filter them once more
    Returns: 
        eyes -> instance similar to whites, but with the clusters
            identified as candidates of pair of eyes
    '''
    # 1) CONDUCTS ONE AND TWO EYES SHAPE TESTS
    th1, th2 = 0.7, 0.7 # thresholds for the one and two eyes tests
    # Creates bool vectors to store the results
    passed1 = np.zeros_like(whites.info['ids'], dtype = bool)
    passed2 = np.zeros_like(whites.info['ids'], dtype = bool)
    for i in range(whites.n):
        is_cluster  = np.equal(whites.clustered , whites.info['ids'][i])
        is_boundary = np.equal(whites.boundaries, whites.info['ids'][i])
        Y, X = np.nonzero(is_boundary) 
        score1 = fdh.get_one_eye_score(is_boundary,
                                        maybe_pupil, Y, X)
        score2 = fdh.get_two_eyes_score(is_cluster, is_boundary,
                                        maybe_pupil, Y, X)
        # For each cluster keeps info on which test it passed
        if score1>th1 and score2>th2:
            passed1[i], passed2[i]  = True, True
        elif score1>th1:
            passed1[i] = True
        elif score2 > th2:
            passed2[i] = True
    # 2) MERGES CLUSTERS TO GET PAIRS OF EYES
    is_merged = np.zeros_like(whites.info['ids'], dtype = bool)
    pairs = []
    for i in range(whites.n):
        if not is_merged[i]: # if not merged yet
            id_i = whites.info['ids'][i]
            if (not passed1[i]) and passed2[i]:
                # If it passed only the two eyes test, clasify
                # immediately as pair of eyes
                pairs.append([id_i, id_i])
                is_merged[i] = True
            elif passed1[i]:
                for j in range(i+1, whites.n):
                    check_j = (passed1[j] or passed2[j]) and not is_merged[j]
                    if check_j and fdh.merged_eyes_test(i, j, whites.info):
                        id_j = whites.info['ids'][j]
                        pairs.append([id_i, id_j])
                        # Ensures that those two will not be checked again
                        is_merged[i], is_merged[j] = True, True
                # If cluster i was not merged with other cluster, but it
                # had passed the two eye test, then identify it alone
                # as a merged cluster (pair of eyes)
                if passed2[i] and not is_merged[i]:
                    pairs.append([id_i, id_i])
                    is_merged[i] = True
    # Merges clusters in .clustered, .boundaries, and .info
    # and deletes all the clusters that were not in 'pairs'
    eyes = whites.merge_clusters(pairs, in_place = False)
    # 3) ADDITIONAL TESTS
    # Contacts final tests to ensure that the merged clusters:
    # a) cover big enough area, and have expected width to height ratio
    # b) are not so close to top or bottom of image so that a face would
    #    not fit around them
    s0 = np.shape(eyes.clustered)[0]
    to_keep = np.ones_like(eyes.info['ids'], dtype = bool)
    for i in range(eyes.n):
        # 1) area and width to height ratio tests
        if eyes.info['width'][i] < 50 :
            to_keep[i] = False
        elif eyes.info['height'][i] < 10:
            to_keep[i] = False
        elif eyes.info['width'][i]/eyes.info['height'][i] < 1:
            to_keep[i] = False
        else:
            # 2) Keeps cluster only it is not so close to top or bottom
            #    of the image so that a face wouldn't fit around it
            FminY = eyes.info['minY'][i] - (eyes.info['height'][i]//2)
            FmaxY = eyes.info['maxY'][i] + (eyes.info['height'][i]//2)
            if FminY < 0 or FmaxY > s0:
                to_keep[i] = False
    if False in to_keep:
        eyes.trim_data(to_keep, in_place = True)
    return eyes

def get_skin(eyes, yellows, max_min_dist = 200):
    '''
    Inputs:
        eyes, yellows -> instances as defined in main code
        max_min_dist -> maximun distance of the centroid of a
            yellow cluster from the closest centroid of an
            'eyes' clusters for which the former is kept
    Filters yellow clusters by deleting the ones that are too
    far away from the identified pairs of eyes
    
    Returns: skin -> 'yellows' after applying the above filter
    '''    
    to_keep = np.zeros_like(yellows.info['ids'], dtype = bool) 
    for i in range(yellows.n):
        # Gets distance from closest white cluster
        dy2 = np.square(np.subtract(yellows.info['meanY'][i],
                                    eyes.info['meanY']))
        dx2 = np.square(np.subtract(yellows.info['meanX'][i],
                                    eyes.info['meanX']))
        try: # works if eyes non empty
            min_dist = np.amin(np.sqrt(np.add(dy2,dx2)))
        except: # else signal to stop
            min_dist = max_min_dist + 1
        if min_dist < max_min_dist:
            to_keep[i] = True
    # Gets skin instance
    skin = yellows.trim_data(to_keep, in_place = False)
    return skin

def get_faces(BYWO2, eyes, skin,
              blacks_depth = 0, yellows_depth = 1,
              whites_depth = 2, others_depth  = 3):
    '''
    Inputs: As defined in main code
    
    a) For each cluster in 'eyes' forms face skin by identifying
        the best match from the 'skin' instance
    b) Reconstructs skin clusters to ensure that each corresponds
        to only one face
    
    Returns: 
        faces -> instance of Cluster() with info on the faces
        BYWO2, eyes, skin -> as in main code after deleting
            pixels of eyes and skins that were not used to
            constuct the faces
    '''
    min_skin_points   = 1000
    min_eye_skin_prop = 0.6
    # To save id of paired skin cluster
    skin_of_eyes = np.zeros_like(eyes.info['ids'], dtype=np.uint8)
    skin_clustered = np.zeros_like(skin.clustered, dtype=np.uint8)
    # To ensure that same pixel is not used twice
    skin_matched = np.zeros_like(skin.clustered, dtype= bool)
    k = 0 # to assign ids of reconstructed skin clusters
    for i in range(eyes.n):
        for j in range(skin.n):
            # Checks if a cluster of skin pixels is appropriately placed
            # compared to pair of eyes
            if skin_of_eyes[i] == 0 \
                and fdh.face_test(i, j, eyes.info, skin.info):
                '''
                Skin Face Reconstruction:
                3 vertical lines (placed bellow the eyes) are used to chose
                the subset of skin pixels (of current cluster) that are 
                directly horizontally connected through other skin pixels 
                to one of those lines
                
                ! 'Lines' move left or right if this allows them to continue
                    to gather pixels (see fdh.construct_face_skin)
                '''
                is_skin = np.logical_and(np.logical_not(skin_matched),
                    np.equal(skin.clustered, skin.info['ids'][j]))
                # Start from pixel closest to mid point right bellow the eyes
                y0 = eyes.info['maxY'][i]+1
                x0 = eyes.info['midX'][i]
                y, x, is_face_skin1 = fdh.construct_face_skin(
                                        y0, x0, is_skin)
                while y != -1:
                    y, x, is_face_skin1 = fdh.construct_face_skin(
                                        y, x, is_skin, is_face_skin1)
                # Repeats process starting from point left of the 1st
                x0 = eyes.info['minX'][i]
                y, x, is_face_skin2 = fdh.construct_face_skin(
                                        y0, x0, is_skin)
                while y != -1:
                    y, x, is_face_skin2 = fdh.construct_face_skin(
                                        y, x, is_skin, is_face_skin2)
                # Repeats process starting from point right of the 1st
                x0 = eyes.info['maxX'][i]
                y, x, is_face_skin3 = fdh.construct_face_skin(
                                        y0, x0, is_skin)
                while y != -1:
                    y, x, is_face_skin3 = fdh.construct_face_skin(
                                        y, x, is_skin, is_face_skin3)
                # Combines results to form new cluster of points
                is_face_skin = np.logical_or(
                    np.logical_or(is_face_skin1, is_face_skin2), is_face_skin3)
                # Uses cluster of points bellow the eyes as a guide to get
                # corresponding points above and between
                _, X = np.nonzero(is_face_skin)
                if len(X) > 0:
                    minX = min(np.min(X), eyes.info['minX'][i])
                    maxX = max(np.max(X), eyes.info['maxX'][i])
                    is_face_skin4 = np.zeros_like(is_skin, dtype = bool)
                    is_face_skin4[:y0, minX:maxX] = True
                    is_face_skin4 = np.logical_and(is_skin, is_face_skin4)
                    is_face_skin  = np.logical_or(is_face_skin, is_face_skin4)
                # Keeps cluster only if 
                # 1) it is big enough
                # 2) has enough skin attached to the eyes
                if np.count_nonzero(is_face_skin) > min_skin_points:
                    attached_to_skin = np.greater(
                            ndimage.correlate(is_face_skin.astype(int), 
                            np.ones((3,3),dtype=int), mode='constant'), 0)
                    eye_attached_to_skin = \
                        np.logical_and(attached_to_skin,
                        np.equal(eyes.boundaries, eyes.info['ids'][i]))
                    eye_skin_prop = np.count_nonzero(eye_attached_to_skin)\
                                  / eyes.info['b_points_count'][i]
                    if eye_skin_prop > min_eye_skin_prop:
                        # Saves stats on reconstructed skin cluster
                        k += 1
                        skin_of_eyes[i] = k
                        skin_clustered[is_face_skin] = k
                        skin_matched = np.logical_or(skin_matched,
                                                     is_face_skin)                 
    # Creates intance of Cluster() for the reconstructed skin
    skin = Cluster(skin_clustered, min_points = 0, min_density = 0,
                   make_boundaries = False, fill_holes = False)
    # Deletes non-matched pairs of eyes
    to_keep = np.greater(skin_of_eyes, 0)
    eyes.trim_data(to_keep, in_place = True)
    # Ensures that there is no overlapping between colors
    is_white = np.greater(eyes.clustered, 0)
    skin.clustered[is_white] = 0
    # Gets face clusters (a paired 'eyes' and 'skin' cluster 
    # will have the same rows in info)
    faces_clustered = np.add(eyes.clustered, skin.clustered)
    faces = Cluster(faces_clustered, min_points = 0, min_density = 0,
                     make_boundaries = True, fill_holes = False)
    # Updates BYWO2, because part of the skin may be unused
    is_eyes = np.greater(eyes.clustered, 0)
    is_skin = np.greater(skin.clustered, 0)
    is_face = np.logical_or(is_eyes, is_skin)
    BYWO2 = np.zeros_like(BYWO2, dtype = bool)
    BYWO2[:,:, whites_depth ] = is_eyes
    BYWO2[:,:, yellows_depth] = is_skin 
    BYWO2[:,:, others_depth ] = np.logical_not(is_face)  
    return faces, BYWO2, eyes, skin
                
def get_windows(faces, BYWO2, eyes, skin,
                blacks_depth, window_min_size):
    '''
    Inputs: as defined in main code
    
    Creates windows that crop out the constructed faces.
    
    Returns: 
        pos -> list in which each row has the
               [minY, maxY, minX, maxX] of the window that
               covers the corresponding face
        BYWO3 -> bool array similar BYWO2, but all pixels out
               of the identified windows are turned to black
        windows -> list of np.uint8 arrays, each containing
               the simplified RGB representation of a face
               as reconstructed by BYWO2
               
    '''
    pos = np.zeros((faces.n,4), dtype = np.uint16)
    BYWO3 = np.zeros_like(BYWO2, dtype = bool)
    min_half = window_min_size//2
    s0, s1   = np.shape(BYWO2)[:2]
    windows = []
    for i in range(faces.n):
        # Fixes size to closest even number and
        # ensures that the window is big enough
        size = max(faces.info['height'][i], faces.info['width'][i])
        half = max(size//2, min_half)
        size = half*2
        # Fix vertical position
        if faces.info['midY'][i] < size/2:
            WminY, WmaxY = 0, size
        elif faces.info['midY'][i]+half > s0:
            WminY, WmaxY = s0-size, s0
        else:
            WminY = faces.info['midY'][i]-half
            WmaxY = faces.info['midY'][i]+half
        # Fix horizontal position
        if faces.info['midX'][i] < half:
            WminX, WmaxX = 0, size
        elif faces.info['midX'][i]+half > s1:
            WminX, WmaxX = s1-size, s1
        else:
            WminX = faces.info['midX'][i]-half
            WmaxX = faces.info['midX'][i]+half
        pos[i,0], pos[i,1] = WminY, WmaxY
        pos[i,2], pos[i,3] = WminX, WmaxX
        # Copys windows from BYWO2 to BYWO3
        BYWO3[WminY:WmaxY,WminX:WmaxX,:] =\
                BYWO2[WminY:WmaxY,WminX:WmaxX,:]
        # Gets cropped windows
        id_ = faces.info['ids'][i]
        is_eyes = np.equal(eyes.clustered, id_)
        is_skin = np.equal(skin.clustered, id_)
        is_back = np.logical_not(np.logical_or(is_eyes, is_skin))
        wind = np.zeros((s0,s1,3), dtype=np.uint8)
        wind[is_eyes,:] = [255,255,255]
        wind[is_skin,:] = [255,213,0]
        wind[is_back,:] = [100,170,190]
        # Crops faces
        windows.append(wind[WminY:WmaxY,WminX:WmaxX,:])
    # Turns pixels that have no value assign yet to black
    is_empty = np.equal(np.sum(BYWO3, axis=2), 0)
    BYWO3[is_empty, blacks_depth] = True
    return pos, BYWO3, windows

def get_img_windows(pos, img):
    '''
    Inputs: as in main code
    
    Returns: list of np.uint8 arrays, each containing the
                original RGB values of the corresponding
                areas in windows
    '''
    img_windows = []
    n = np.shape(pos)[0]
    for i in range(n):
        wind = img[pos[i,0]:pos[i,1],pos[i,2]:pos[i,3],:]
        img_windows.append(wind)
    return img_windows
