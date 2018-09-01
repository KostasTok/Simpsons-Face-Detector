'''
Contains low level functions called by functions in face_detector.py
'''

import numpy as np
from scipy import ndimage

def get_one_eye_score(is_boundary, maybe_pupil, Y = None, X = None):
    '''
    Inputs: 
        is_boundary -> bool array marking the boundary of a cluster
        maybe_pupil -> bool array with trues on the pixels that have
                       an area of high enough density of blacks around
                       them to indicate a pupil
        Y, X -> 1-d arrays with the coordinates of the trues in
                is_boundary (if None calcuated here)
    First it tests if there are indications of a pupil. If yes,
    then it tests how well the shape of the boundary resembles
        1) a circle
        2) the bottom half of a cicle
        3) a rectangle
    Returns: the highest of the three scores (in [0,1])
    '''
    # Gets info on the points that constitute the boundary
    if (Y is None) or (X is None):
        Y, X = np.nonzero(is_boundary)
    minY, maxY = np.min(Y), np.max(Y)
    minX, maxX = np.min(X), np.max(X)
    # Check if there are indications of a pupil in the smallest
    # rectangle that covers the eye candidate
    if True not in maybe_pupil[minY:maxY,minX:maxX]:
        return 0
    else:
        # Draws filters
        circle = make_circle(is_boundary, minY, maxY, minX, maxX)
        semi = make_semicircle(is_boundary, minY, maxY, minX, maxX)
        # Adjusted the size of the window that testes if a point
        # of the two filters is close enough to each point of the
        # cluster's boundary
        s = max(maxY-minY, maxX-minX)*0.2
        s = min(max(np.ceil(s).astype(int),5),10)
        window = np.ones((s,s),dtype=int)
        # Marks points that are close enough to these filters
        sums = ndimage.convolve(circle.astype(int), window,
                            mode='constant')
        close2circle = np.not_equal(sums, 0)
        sums = ndimage.convolve(semi.astype(int), window,
                            mode='constant')
        close2semi = np.not_equal(sums, 0)
        # Gets scores
        close2circle = np.logical_and(close2circle, is_boundary)
        circle_score = np.count_nonzero(close2circle) / len(Y)
        close2semi = np.logical_and(close2semi, is_boundary)
        semi_score = np.count_nonzero(close2semi) / len(Y)
        # Close to bounds filter
        dist_minX = np.subtract(X,minX)
        dist_maxX = np.subtract(maxX,X)
        distX = np.minimum(dist_minX, dist_maxX)
        dist_minY = np.subtract(Y,minY)
        dist_maxY = np.subtract(maxY,Y)
        distY = np.minimum(dist_minY, dist_maxY)
        dist = np.minimum(distX, distY)
        dist_score = np.count_nonzero(np.less(dist,6)) / len(Y)
        
        return max(circle_score, semi_score, dist_score)

def get_two_eyes_score(is_cluster, is_boundary, maybe_pupil,
                       Y = None, X = None):
    '''
    Inputs:
        is_cluster  -> bool array marking a cluster
        is_boundary -> bool array marking the boundary of a cluster
        maybe_pupil -> bool array with trues on the pixels that have
                       an area of high enough density of blacks around
                       them to indicate a pupil
        Y, X -> 1-d arrays with the coordinates of the trues in
                is_boundary (if None calcuated here)
    Splits cluster vertically by using the location of pupil
        candidates and conducts two separate one_eye_tests
    Returns: the average of those two tests
    '''
    # Gets info on the points that constitute the boundary
    if (Y is None) or (X is None):
        Y, X = np.nonzero(is_boundary)
    minY, maxY = np.min(Y), np.max(Y)
    minX, maxX = np.min(X), np.max(X)
    try:
        # Attempt to split the cluster vertically by identifying
        # the first and last time indication of a pupil appear
        # on any row, while moving from left to right
        b0 = np.sum(maybe_pupil[minY:maxY,minX:maxX],axis=0)
        bl = np.roll(b0,-1)
        br = np.roll(b0, 1)
        lefts  = np.nonzero(np.logical_and(np.greater(b0,0), np.equal(br,0)))[0]
        rights = np.nonzero(np.logical_and(np.greater(b0,0), np.equal(bl,0)))[0]
        left1 , left2  = lefts[0] , lefts[1]
        right1, right2 = rights[0], rights[1]
        mid = (maxX-minX)//2
        if (right1<mid) and (mid<left2):
            br = minX+mid
        elif abs(left1-mid)<abs(left2-mid):
            br = min(minX+right1+1,maxX)
        else:
            br = min(minX+left2 +1,maxX)
    except:
        # If not possible to use pupils, split in halfs
        br = (maxX+minX)//2
    mid_bound = np.equal(is_cluster[:,br], True)
    # Get score of left eye
    left_bound = is_boundary.copy()
    left_bound[:,br+1:] = False
    left_bound[mid_bound,br] = True
    if np.count_nonzero(left_bound) > 5:
        left_score = get_one_eye_score(left_bound, maybe_pupil)
    else:
        left_score = 0
    # Get score of right eye
    right_bound = is_boundary.copy()
    right_bound[:,:br] = False
    right_bound[mid_bound,br+1] = True
    if np.count_nonzero(right_bound) > 5:
        right_score = get_one_eye_score(right_bound, maybe_pupil)
    else:
        right_score = 0
    return (left_score+right_score) / 2
        
def make_circle_basic(kernel, y0, x0, radius, semi = False):
    '''
    Draws a circle centered around (y0,x0) with the given radius.
    If semi = True, then draws only the bottom half of the circle,
    and a horizontal line that connects the two sides.
    '''
    circle = np.zeros_like(kernel, dtype = bool)
    # Draws the cirlce
    y, dy = radius-1, 1
    x, dx = 0, 1
    err = dy - (radius << 1)
    while y >= x:
        circle[y0 + y, x0 + x] = True
        circle[y0 + x, x0 + y] = True
        circle[y0 - x, x0 + y] = True
        circle[y0 - y, x0 + x] = True
        circle[y0 - y, x0 - x] = True
        circle[y0 - x, x0 - y] = True
        circle[y0 + x, x0 - y] = True
        circle[y0 + y, x0 - x] = True
        if err <= 0:
            x += 1
            err += dx
            dx += 2
        else:
            y -= 1
            dy += 1
            err += dy - (radius << 1)
    # If semi is true, erase the top half of the cirlce
    # and draw a line that conect the two sides
    if semi:
        circle[y0-radius:y0,x0-radius:x0+radius] = False
        circle[y0+1,x0-radius+1:x0+radius] = True
    return circle

def make_circle(kernel, top, bot, left, right):
    '''
    Uses make_circle_basic() to draw a cirlce, the location
    of which it figures using the smallest rectangle that 
    covers the cluster
    '''
    height, width = bot-top, right-left
    radius = min(width, height)//2
    y0 = top + height//2
    x0 = left + width//2
    return make_circle_basic(kernel, y0, x0, radius)

def make_semicircle(kernel, top, bot, left, right):
    '''
    Similar to make_circle(), but it returns the bottom half
    of a circle.
    '''
    [s0,s1] = np.shape(kernel)
    radius = (right-left)//2
    big_kernel = np.zeros([s0+2*radius,s1], dtype = bool)
    y0 = top + radius
    x0 = left + radius
    semicircle = make_circle_basic(big_kernel, y0, x0, radius, True)
    semicircle = semicircle[radius:radius+s0,:]
    return semicircle

def merged_eyes_test(i, j, whites_info):
    '''
    Inputs :Integers i and j corresponding to the rows of the
            clusters under consideration in whites_info
    Returns: True, if clusters:
        1) have approximately the same size
        2) one is placed immediately on the left or right
           of the other   
    '''
    # Size comparison
    h = abs(whites_info['height'][j]-whites_info['height'][i])\
        / whites_info['height'][i]
    w = abs(whites_info['width'][j]-whites_info['width'][i]) \
        / whites_info['width'][i]
    similar_size = (h < 0.65) and (w < 0.9)
    if not similar_size:
        return False
    else:
        # If similar size check that their vertical position
        # is approximately the same
        t = abs(whites_info['minY'][i] - whites_info['minY'][j]) \
            / whites_info['height'][i]
        b = abs(whites_info['maxY'][i] - whites_info['maxY'][j]) \
            / whites_info['height'][i]
        vert_dist = min(t, b)
        if vert_dist > 0.6:
            return False
        else:
            # If similar vertical position check that the one
            # is on the left of right of the other
            r = abs(whites_info['maxX'][i] - whites_info['minX'][j]) \
                / whites_info['width'][i]
            l = abs(whites_info['minX'][i] - whites_info['maxX'][j]) \
                / whites_info['width'][i]
            if min(r, l) < 1:
                return True
            else:
                return False

def face_test(i, j, whites_info, yellows_info):
    '''
    Inputs :Integers i and j indicating rows of the dictionaries
            whites_info and yellows_info
    Returns: True if
        1) yellow cluster has enough points compared to the white
        2) the yellow horizontal contains the white, but its width
           is not match bigger
        3) there is some vertical overlapping between the two
    '''
    # Size comparison
    h = yellows_info['height'][j] / whites_info['height'][i]
    w = yellows_info['width' ][j] / whites_info['width' ][i]
    if h < 2 or w < 1:
        return False
    else:
        # horizontal position comparison
        l = yellows_info['minX'][j] < whites_info['minX'][i]
        r = yellows_info['maxX'][j] > whites_info['maxX'][i]
        if not (l and r):
            return False
        else:
            # vertical position comparison
            t = yellows_info['minY'][j] < whites_info['maxY'][i]
            b = yellows_info['maxY'][j] > whites_info['maxY'][i]
            if not (t and b):
                return False
            else:
                return True
            
def construct_face_skin(y0, x0, is_yellow, is_skin = None):
    '''
    Inputs:
    is_yellow, is_skin -> bool array of same shape, as
                          in main code. If is_skin is not
    given, then this signal that this is the 1st time the
    function was called on the current face
    y0, x0 -> point used to create the face skin cluster 
              of the previous line
    Returns: (y,x) of pixel from which the process of building
             the cluster of row y0+1 starts, and the arr 'is_skin'
             after turning to True the pixels recognized as skin. 
    If no such points, it returns -1 on the coordinates to signal
    the end of the process
    '''
    s0, s1 = np.shape(is_yellow)
    if y0 + 1 >= s0:
        return -1, -1, is_skin
    else:
        if is_skin is None:
            is_skin = np.zeros_like(is_yellow, dtype = bool)
            # If point given is yellow, then construct 1st line below
            # the face from there, otherwise search for closest point
            # on the same line
            if is_yellow[y0,x0]:
                bound1, bound2 = get_line_cluster(is_yellow[y0,:], x0, s1)
                is_skin[y0, bound1:bound2] = True
                return y0, x0, is_skin
            else:
                Xs = np.nonzero(is_yellow[y0,:])[0]
                if len(Xs) > 0:
                    dist = np.absolute(np.subtract(Xs, x0))
                    am = np.argmin(dist)
                    x = Xs[am]
                    bound1, bound2 = get_line_cluster(is_yellow[y0,:], x, s1)
                    is_skin[y0, bound1:bound2] = True
                    return y0, x, is_skin
                else:
                    # signals process to not start because a valid point was
                    # not found on the 1st line
                    return -1, -1, is_skin
        elif is_yellow[y0+1,x0]:
            # If point below of (y0,x0) is skin, then use this one
            # as a starting point to verify the y0+1
            bound1, bound2 = get_line_cluster(is_yellow[y0+1,:], x0, s1)
            is_skin[y0+1, bound1:bound2] = True
            return y0+1, x0, is_skin 
        else:
            # Gets 1st and last pixel of verified skin on row y0
            xs = np.nonzero(is_skin[y0,:])[0]
            if len(xs) <= 5:
                return -1, -1, is_skin
            else:
                minX, maxX = xs[0], xs[-1]
                # Look left of x0 for closest point x such that
                # 1) (y0,x) was verfied using (y0,x0)
                # 2) (y0_1,x) is a skin color pixel
                x, left_score = x0 - 1, 0
                while x >= minX and not is_yellow[y0+1,x]:
                    x -= 1
                if is_yellow[y0+1,x]:
                    # Gets 1st and last index of the cluster that would form
                    # that verified pixels of y0+1 if (y0+1,x) was chosen
                    left1, left2 = get_line_cluster(is_skin[y0+1,:], x, s1)
                    left_score = left2 - left1
                    left_x = x
                # Repeats the above process, but looking right instead
                x, right_score = x0 + 1, 0
                while x < maxX and not is_yellow[y0+1,x]:
                    x += 1
                if is_yellow[y0+1,x]:
                    right1, right2 = get_line_cluster(is_skin[y0+1,:], x, s1)
                    right_score = right2 - right1
                    right_x = x
                # Returns point and corresponding bounds with biggest score,
                # or signals to stop by returning -1,-1,-1
                if left_score + right_score == 0:
                    return -1, -1, is_skin
                elif left_score > right_score:
                    is_skin[y0+1, left1:left2] = True
                    return y0+1, left_x, is_skin
                else:
                    is_skin[y0+1, right1:right2] = True
                    return y0+1, right_x, is_skin   
            
def get_line_cluster(vect, x0, s1):
    '''
    Inputs:
    vect: 1-d bool array
    x0  : int
    s1  : length of vect
    Finds the largest fully connected area of Trues in
    which x0 is in (area not interrupted by False)
    Returns: indexes of first and last point of this area
    '''
    x = x0 - 1
    while x >= 0 and vect[x]:
        x -= 1
    bound1 = x + 1
    x = x0 + 1
    while x < s1 and vect[x]:
        x += 1
    bound2 = x
    return bound1, bound2
