#Manual Vehicle Detection
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from scipy.ndimage.measurements import label

class globalVars():
    #This class stores all of the global variables
    def __init__(self):
        #HOG parameters
        self.color_space = None
        self.orient = None
        self.pix_per_cell = None
        self.cell_per_block = None
        self.hog_channel = None
        self.spatial_size = None
        self.hist_bins = None
        self.hist_range = None
        self.spatial_feat = None
        self.hist_feat = None
        self.hog_feat = None
        
        
        #Image Arrays
        self.car_images = None
        self.noncar_images = None
        
        #HOG Features
        self.car_features = None
        self.noncar_features = None
        #Model
        self.svc = None
        self.X_scaler = None


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    draw_img = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return draw_img


def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    # Use cv2.resize().ravel() to create the feature vector
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)  
    features = cv2.resize(feature_image, size).ravel()
    
    # Return the feature vector
    return features

def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=feature_vec)
        return features
    
def extract_features(imgs, globalVariable):

    color_space = globalVariable.color_space
    orient = globalVariable.orient
    pix_per_cell = globalVariable.pix_per_cell
    cell_per_block = globalVariable.cell_per_block
    hog_channel = globalVariable.hog_channel
    spatial_size = globalVariable.spatial_size
    hist_bins = globalVariable.hist_bins
    hist_range = globalVariable.hist_range
    spatial_feat = globalVariable.spatial_feat
    hist_feat = globalVariable.hist_feat
    hog_feat = globalVariable.hog_feat
    
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)      

        #Adding spatial and histogram features!
        #Compute spatial features if flag is set
        
        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            #Append features to list
            #features.append(spatial_features)
        
               
            
        #Compute histogram features if flag is set
        if hist_feat == True:
            hist_features = color_hist(feature_image, nbins=hist_bins)
            #Append features to list
            #features.append(hist_features)
        ######################################

            
        if hog_feat == True:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
               
        features.append(np.concatenate((spatial_features, hist_features, hog_features)))
         
    # Return list of feature vectors
    
    return features

def extract_all_features(globalVariable):
    car_images = globalVariable.car_images
    noncar_images = globalVariable.noncar_images
    
    ##################################################
    # DELETE THIS ON REAL TEST
    ##################################################
    maxSamples = 300
    car_images = car_images[: maxSamples]
    noncar_images = noncar_images[: maxSamples]


    ##################################################

    
    t=time.time()
    globalVariable.car_features = extract_features(car_images, globalVariable)
    globalVariable.noncar_features = extract_features(noncar_images, globalVariable)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to extract ALL features...')

def train_model(globalVariable):
    car_features = globalVariable.car_features
    noncar_features = globalVariable.noncar_features
   
    
    # Create an array stack of feature vectors
    X = np.vstack((car_features, noncar_features)).astype(np.float64)    
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    globalVariable.X_scaler = X_scaler
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(noncar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:',globalVariable.orient,'orientations',globalVariable.pix_per_cell,
        'pixels per cell and', globalVariable.cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts:     ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
    globalVariable.svc = svc




def visualizeImage(name, img):
    '''
    Takes in a name and image, displays it on computer
    '''
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.waitKey(1)
    

def find_cars(img, ystart, ystop, scale, globalVariable):
    svc = globalVariable.svc
    X_scaler = globalVariable.X_scaler
    orient = globalVariable.orient
    pix_per_cell = globalVariable.pix_per_cell
    cell_per_block = globalVariable.cell_per_block
    spatial_size = globalVariable.spatial_size
    hist_bins = globalVariable.hist_bins
    
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')

    #Create an empty list to receive positive detection windows
    on_windows = []
    
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
            
         
            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)
            
            # Scale features and make a prediction
          
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                on_windows.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
    
    return on_windows
    #return draw_img

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

#add heat map
def create_heatmap(image, hot_windows, threshold = 1):
    
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    #Add heat to each box in box list
    heat = add_heat(heat, hot_windows)
    #visualizeImage('initial heat', heat)
    
    #Apply threshold to remove false positives
    heat = apply_threshold(heat, threshold)
    #visualizeImage('after thresh', heat)
    
    #visualize heatmap
    heatmap = np.clip(heat,0, 255)
    plt.clf()
    plt.imshow(heatmap,cmap= 'hot')
    plt.savefig('heatmap.jpg')
    
    #visualizeImage('heatmap', heatmap)
    labels = label(heatmap)
    plt.imshow(labels[0], cmap = 'gray')
    plt.savefig('labels.jpg')
    heatmap_img = draw_labeled_bboxes(np.copy(image), labels)
    
    return heatmap_img


def vehicleDetectionPipeline(image):
    # Pipeline for detecting vehicles, doesn't take data from previous frames into consideration

    #use find cars function to find the cars
    hot_windows_find_cars = find_cars(image, ystart, ystop, scale, globalVariable)

    #add heat map
    final_image = create_heatmap(np.copy(image), hot_windows_find_cars, threshold = 1)

    return final_image



##################################################################
#   Main Code
##################################################################

#Load Training Data
car_images = glob.glob('vehicles/vehicles/**/*.png')
noncar_images = glob.glob('non-vehicles/non-vehicles/**/*.png')
print('car image size ', len(car_images))
print('non-car image size ', len(noncar_images))


#Setting up variables
color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
hist_range = (0,256)
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
ystart = 400
ystop = 656
scale = 1.5

# Define global variables
globalVariable = globalVars()
globalVariable.color_space = color_space
globalVariable.orient = orient
globalVariable.pix_per_cell = pix_per_cell
globalVariable.cell_per_block = cell_per_block
globalVariable.hog_channel = hog_channel
globalVariable.spatial_size = spatial_size
globalVariable.hist_bins = hist_bins
globalVariable.hist_range = hist_range
globalVariable.spatial_feat = spatial_feat
globalVariable.hist_feat = hist_feat
globalVariable.hog_feat = hog_feat
globalVariable.car_images = car_images
globalVariable.noncar_images = noncar_images




#Visualize what a hog feature looks like

carImg = mpimg.imread(car_images[np.random.randint(0, len(car_images)-1)])
carFeature, carImgDst = get_hog_features(carImg[:,:,2], globalVariable.orient, globalVariable.pix_per_cell, globalVariable.cell_per_block, vis = True, feature_vec = True)
noncarImg = mpimg.imread(noncar_images[np.random.randint(0, len(car_images) -1)])
noncarFeature, noncarImgDst = get_hog_features(noncarImg[:,:,2], globalVariable.orient, globalVariable.pix_per_cell, globalVariable.cell_per_block, vis = True, feature_vec = True)
#save plots
figure, ((plt1, plt2), (plt3, plt4)) = plt.subplots(2,2, figsize = (7,7))
figure.subplots_adjust(hspace = .5, wspace =.2)
plt1.imshow(carImg)
plt1.set_title('Car Image', fontsize = 14)
plt2.imshow(carImgDst)
plt2.set_title('Car Image HOG', fontsize = 14)
plt3.imshow(noncarImg)
plt3.set_title('Non-car Image', fontsize = 14)
plt4.imshow(noncarImgDst)
plt4.set_title('Non-car Image HOG', fontsize = 14)
plt.savefig('HOG Example.jpg')



#Extract ALL features from test set
extract_all_features(globalVariable)

#Train model using ALL features
train_model(globalVariable)


##############################################################
#Test image
##############################################################

image = mpimg.imread('test4.jpg')

#use find cars function to find the cars
hot_windows_find_cars = find_cars(image, ystart, ystop, scale, globalVariable)

#add heat map
final_image = create_heatmap(np.copy(image), hot_windows_find_cars, threshold = 1) 
visualizeImage('final img', final_image)


'''
#Video
output_file = 'video_output.mp4'
input_file = VideoFileClip('project_video.mp4')
processedClip = input_file.fl_image(vehicleDetectionPipeline)
#processedClip.write_videofile(processedClip, audio = False)
%time processedClip.write_videofile(output_file, audio=False)
'''


    




























