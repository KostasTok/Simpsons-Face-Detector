'''
Provides functions that simplify the colors of an image
in four main categories:
    Black, White, Yellow, Other
'''
import numpy as np
from scipy import ndimage

def getBYWO(img):
    '''
    BYWO -> (Black,Yellow,White,Other)
    Gets image as RGB (height,width,3) array 
    and returns BYWO (height,width,4) boolean:
    0) BYWO(y,x,0)=True <=> RGB(y,x,:) in range of 
                        Blacks used to draw lines
    1) BYWO(y,x,1)=True <=> RGB(y,x,:) in range of 
                        Yellows on Simpsons heads
    2) BYWO(y,x,2)=True <=> RGB(y,x,:) in range of 
                        Whites used on Simpsons eyes
    3) BYWO(y,x,3)=True <=> RGB(y,x,:) none of above
    '''
    BYWO = np.zeros((np.shape(img)[0],
                     np.shape(img)[1],4), dtype=bool)
    BYWO[:,:,0] = isSimpsonBlack(img)
    BYWO[:,:,1] = isSimpsonYellow(img)
    BYWO[:,:,2] = isSimpsonWhite(img)
    # Turns overlappings to zeros (to be
    # classified as 'Other')
    overlappings = np.greater(np.sum(BYWO,axis=2),1)
    BYWO[overlappings,:] = np.zeros(4).astype(bool)
    # Sets others as the ones that have all zeros
    others = np.equal(np.sum(BYWO,axis=2),0)
    BYWO[:,:,3] = others

    # Uses a kernel based approach to smooth the
    # appearance of the simplified image
    # Deletes blacks that do not form a big enough cluster
    BYWO = smootherSqaure(BYWO, 0, side_colors = [1,2,3], 
                   required_trues = 8, square_side = 5)
    # 2) Delete the others between [yellows,whites]
    BYWO = smootherCross(BYWO, 3, [1,2],
            required_trues = 2,cross_length = 15)
    BYWO = smootherCross(BYWO, 3, 1,
            required_trues = 2,cross_length = 15)
    return BYWO

def BYWO_to_RGB(BYWO):
    '''
    Transforms the simplified representation of RGB
    back to an RGB in which:
        Blacks=(0,0,0),Yellows=(255,213,0),
        Whites=(255,255,255),Others=(255,192,203)
    '''
    s0, s1, _ = np.shape(BYWO)
    RGB = np.zeros((s0,s1,3), dtype=np.uint8)
    RGB[BYWO[:,:,0]] = [0,0,0]
    RGB[BYWO[:,:,1]] = [255,213,0]
    RGB[BYWO[:,:,2]] = [255,255,255]
    RGB[BYWO[:,:,3]] = [100,170,190]
    return RGB
    
############# BASIC COLOR FILTERS ################
def isSimpsonYellow(x):
    '''
    Gets part of array in RGB format and returns
    corresponding boolean array with true for each
    pixel painted with the yellow that is used for
    the Simpsons' faces.
    '''
    Red   = np.greater(x[:,:,0],180)
    Green = np.logical_and(np.greater(x[:,:,1],150), np.greater(215,x[:,:,1]))
    Blue  = np.greater(100,x[:,:,2])
    return np.logical_and(np.logical_and(Red,Green), Blue)

def isSimpsonWhite(x):
    '''As above, but for the white of the eyes'''
    Red   = np.greater(x[:,:,0],160)
    Green = np.greater(x[:,:,1],160)
    Blue  = np.greater(x[:,:,2],160)
    above = np.logical_and(np.logical_and(Red,Green), Blue)
    below = np.greater(np.sum(x, axis = 2), 550)
    return np.logical_and(above, below)

def isSimpsonBlack(x):
    '''As above, but for the black of the pupils'''
    return np.logical_and(
        np.logical_and(np.greater(40,x[:,:,0]),np.greater(40,x[:,:,0])), 
        np.greater(40,x[:,:,0]))
##################################################

############ OUTPUT SMOOTHING ###################

def smootherSqaure(arr, center_color, side_colors = None, 
                   required_trues = 5, square_side = 3):
    height, width, depth = np.shape(arr)
    if side_colors is None:
        # Use all colors, including the center_color
        side_colors = np.arange(depth)
    else:
        # If a scaler side_color was given, transform it to array
        side_colors = np.array(side_colors)
        if side_colors.shape is ():
            side_colors = np.expand_dims(side_colors, 0)
    # Constract box with 1s on the sides and 0 on the centre
    kernel = np.ones((square_side,square_side), dtype = int)
    # Constract arrays with the sums of the colors in a
    # sqaure centering around all pixels with the center_color
    sum_max = np.zeros((height, width), dtype = int)
    depth_max = np.zeros_like(sum_max, dtype=int)
    depth_max[:,:] = center_color
    for i in side_colors:
        sum_i = ndimage.correlate(arr[:,:,i].astype(int), kernel,
                            mode='constant')
        depth_max[np.greater(sum_i, sum_max)] = i
        sum_max = np.maximum(sum_i, sum_max)
        if i == center_color:
            sum_center = sum_i
    if center_color not in side_colors:       
        sum_center = ndimage.correlate(arr[:,:,center_color].astype(int),
                                   kernel, mode='constant')
    # Update arrays with simplified colors
    to_change = np.less(sum_center, required_trues)
    for i in side_colors:
        if i != center_color:
            change_i = np.logical_and(
                np.equal(depth_max, i), to_change)
            arr[change_i, center_color] = False
            arr[change_i, i] = True
    return arr

def smootherCross(arr, center_color, side_color,
                  required_trues = 2, cross_length = 11):
    '''
    args:
    arr -> numpy boolean array (height,width,depth),
           built for BYWO but it should work even for
           arrays with bigger depths.
    center_color   -> [0,1,...,depth], the depth of the color
                      that activates the center of the cross
    side_color     -> [0,1,...,depth], the depth of the color
                      that activates the sides of the cross
                ! A list can also be given, in which case 
                  all side_colors are considered and center_color
                  is replace with the first color in the list
    required_trues -> pos int, with number of Trues required
                      on a side for this to be activated
    cross_length   -> odd int >= least_trues*2+1
    
    IF pixel (y,x) 'is center_color' AND IF the
    North-South (OR East-West) sides of a cross centered
    at (y,x) with total length=cross_length have at least
    'least_trues' pixels of 'side_color', THEN (y,x) is
    changed to 'side_color'
    
    Returns: smoothed BYWO
    '''
    # If a scaler side_color was given, transform it to array
    side_color = np.array(side_color)
    if side_color.shape is ():
        side_color = np.expand_dims(side_color, 0)
    # If more than one colors was given accumulate them
    isSideColor = arr[:,:,side_color[0]].copy()
    for i in side_color[1:]:
        isSideColor = np.logical_or(isSideColor, arr[:,:,i])
     # Builds kernels based on the specifactions of the cross
    kernel = np.zeros((cross_length,cross_length),dtype=np.uint32)
    cross_side = cross_length//2
    # Each side is counted different to use only one convolution
    kernel[0:cross_side , cross_side   ] = 1     # north side
    kernel[cross_side+1:, cross_side   ] = 10**3 # south side
    kernel[cross_side   , 0:cross_side ] = 10**6 # west side
    kernel[cross_side   , cross_side+1:] = 10**9 # east side
    # Executes test by decoding the result of the convolution
    sums = ndimage.correlate(isSideColor.astype(np.uint32), 
                            kernel, mode='constant')
    getEast, rest = np.divmod(sums,10**9)
    getWest, rest = np.divmod(rest,10**6)
    getSouth, getNorth = np.divmod(rest,10**3)
    east = np.greater_equal(getEast,required_trues)
    west = np.greater_equal(getWest,required_trues)
    south = np.greater_equal(getSouth,required_trues)
    north = np.greater_equal(getNorth,required_trues) 
    hori = np.logical_and(west,east)
    vert = np.logical_and(north,south)
    sides = np.logical_or(hori, vert)
    sides_center = np.logical_and(sides,arr[:,:,center_color])
    out = arr.copy()
    out[sides_center,side_color[0]] = True
    for i in side_color[1:]:
        out[sides_center,i] = False
    out[sides_center,center_color] = False
    
    return out



    
    
    