# ============================================================================
# File: util.py
# Date: 2025-03-11
# Author: TA
# Description: Utility functions to process BoW features and KNN classifier.
# ============================================================================

import numpy as np
from PIL import Image
from tqdm import tqdm
from cyvlfeat.sift.dsift import dsift
from cyvlfeat.kmeans import kmeans
from time import time
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist, squareform
import scipy.spatial.distance as distance
from scipy.stats import mode

CAT = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
       'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
       'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest']

CAT2ID = {v: k for k, v in enumerate(CAT)}

########################################
###### FEATURE UTILS              ######
###### use TINY_IMAGE as features ######
########################################

###### Step 1-a
def get_tiny_images(img_paths: str):
    '''
    Build tiny image features.
    - Args: : 
        - img_paths (N): list of string of image paths
    - Returns: :
        - tiny_img_feats (N, d): ndarray of resized and then vectorized
                                 tiny images
    NOTE:
        1. N is the total number of images
        2. if the images are resized to 16x16, d would be 256
    '''
    
    #################################################################
    # TODO:                                                         #
    # To build a tiny image feature, you can follow below steps:    #
    #    1. simply resize the original image to a very small        #
    #       square resolution, e.g. 16x16. You can either resize    #
    #       the images to square while ignoring their aspect ratio  #
    #       or you can first crop the center square portion out of  #
    #       each image.                                             #
    #    2. flatten and normalize the resized image.                #
    #################################################################

    tiny_img_feats = []
    
    for path in img_paths:
        img = Image.open(path)
        # Gray scale information is enough
        img2D = img.convert("L")
        
        # resize and flatten
        tiny_img2D = img2D.resize((16, 16), Image.BILINEAR)
        tiny_img1D = np.array(tiny_img2D).flatten()
        # normalize to mean = 0, std = 1
        mean = np.mean(tiny_img1D)
        std = np.std(tiny_img1D)
        tiny_img1D = (tiny_img1D - mean) / std 
        
        tiny_img_feats.append(tiny_img1D)

    tiny_img_feats = np.matrix(tiny_img_feats)

    #################################################################
    #                        END OF YOUR CODE                       #
    #################################################################

    return tiny_img_feats

#########################################
###### FEATURE UTILS               ######
###### use BAG_OF_SIFT as features ######
#########################################

###### Step 1-b-1
def build_vocabulary(
        img_paths: list, 
        vocab_size: int = 400,
    ):
    '''
    Args:
        img_paths (N): list of string of image paths (training)
        vocab_size: number of clusters desired
    Returns:
        vocab (vocab_size, sift_d): ndarray of clusters centers of k-means
    NOTE:
        1. sift_d is 128
        2. vocab_size is up to you, larger value will works better
           (to a point) but be slower to compute,
           you can set vocab_size in p1.py
    '''
    
    ##################################################################################
    # TODO:                                                                          #
    # To build vocabularies from training images, you can follow below steps:        #
    #   1. create one list to collect features                                       #
    #   2. for each loaded image, get its 128-dim SIFT features (descriptors)        #
    #      and append them to this list                                              #
    #   3. perform k-means clustering on these tens of thousands of SIFT features    #
    # The resulting centroids are now your visual word vocabulary                    #
    #                                                                                #
    # NOTE:                                                                          #
    # Some useful functions                                                          #
    #   Function : dsift(img, step=[x, x], fast=True)                                #
    #   Function : kmeans(feats, num_centers=vocab_size)                             #
    #                                                                                #
    # NOTE:                                                                          #
    # Some useful tips if it takes too long time                                     #
    #   1. you don't necessarily need to perform SIFT on all images, although it     #
    #      would be better to do so                                                  #
    #   2. you can randomly sample the descriptors from each image to save memory    #
    #      and speed up the clustering, which means you don't have to get as many    #
    #      SIFT features as you will in get_bags_of_sift(), because you're only      #
    #      trying to get a representative sample here                                #
    #   3. the default step size in dsift() is [1, 1], which works better but        #
    #      usually become very slow, you can use larger step size to speed up        #
    #      without sacrificing too much performance                                  #
    #   4. we recommend debugging with the 'fast' parameter in dsift(), this         #
    #      approximate version of SIFT is about 20 times faster to compute           #
    # You are welcome to use your own SIFT feature                                   #
    ##################################################################################
    
    step_sample = 1
    features = []
    
    for path in tqdm(img_paths):
        img = Image.read(path)
        img2D = img.convert("L")
        
        # frames: contains postions for the descriptors --> size = (N, 2)
        # descriptors: contains all the descriptor      --> size = (N, 128)
        _, descriptors = dsift(img2D, step=[step_sample, step_sample], fast=True)
        
        append_or_not = -1
        for row in descriptors:
            append_or_not += 1
            # pick half descriptors to build vocabulary 
            if not append_or_not%2 :
                features.append(row)

    # Perform k-means clustering on each subset of features
    print("Start k-means clustering \n")
    vocab = kmeans(np.matrix(features).astype('float32'), vocab_size)
    print("Finsh k-means clustering \n")
    
    ##################################################################################
    #                                END OF YOUR CODE                                #
    ##################################################################################
    
    return vocab

###### Step 1-b-2
def get_bags_of_sifts(
        img_paths: list,
        vocab: np.array
    ):
    '''
    Args:
        img_paths (N): list of string of image paths
        vocab (vocab_size, sift_d) : ndarray of clusters centers of k-means
    Returns:
        img_feats (N, d): ndarray of feature of images, each row represent
                          a feature of an image, which is a normalized histogram
                          of vocabularies (cluster centers) on this image
    NOTE :
        1. d is vocab_size here
    '''

    ############################################################################
    # TODO:                                                                    #
    # To get bag of SIFT words (centroids) of each image, you can follow below #
    # steps:                                                                   #
    #   1. for each loaded image, get its 128-dim SIFT features (descriptors)  #
    #      in the same way you did in build_vocabulary()                       #
    #   2. calculate the distances between these features and cluster centers  #
    #   3. assign each local feature to its nearest cluster center             #
    #   4. build a histogram indicating how many times each cluster presents   #
    #   5. normalize the histogram by number of features, since each image     #
    #      may be different                                                    #
    # These histograms are now the bag-of-sift feature of images               #
    #                                                                          #
    # NOTE:                                                                    #
    # Some useful functions                                                    #
    #   Function : dsift(img, step=[x, x], fast=True)                          #
    #   Function : cdist(feats, vocab)                                         #
    #                                                                          #
    # NOTE:                                                                    #
    #   1. we recommend first completing function 'build_vocabulary()'         #
    ############################################################################
      
    start_time = time()
    print("Construct bags of sifts...")
    
    step_sample = 1
    img_feats = []
    
    for path in tqdm(img_paths):
        img = Image.read(path)
        img2D = img.convert("L")
        
        _, descriptors = dsift(img2D, step=[step_sample,step_sample], window_size=4, fast=True)
        # redeuce number of total descriptors
        descriptors = descriptors[::5]
        
        #   histogram
        #   for each img:
        #       for each feature: (may different)
        #           find closet feature from vocab 
        dist = distance.cdist(vocab, descriptors)  
        kmin = np.argmin(dist, axis = 0)
        hist, _ = np.histogram(kmin, bins=len(vocab))
        hist_norm = [float(i)/sum(hist) for i in hist]
        img_feats.append(hist_norm)
    
    img_feats = np.matrix(img_feats)
    
    end_time = time()
    print(f"It takes {((end_time - start_time)/60):.2f} minutes to construct bags of sifts.")
    
    ############################################################################
    #                                END OF YOUR CODE                          #
    ############################################################################
    
    return img_feats

################################################
###### CLASSIFIER UTILS                   ######
###### use NEAREST_NEIGHBOR as classifier ######
################################################

###### Step 2
def nearest_neighbor_classify(
        train_img_feats: np.array,
        train_labels: list,
        test_img_feats: list
    ):
    '''
    Args:
        train_img_feats (N, d): ndarray of feature of 
                        training images, and each element contains a histogram of d columns 
        train_labels (N): list of string of ground truth category for each 
                        training image
        test_img_feats (M, d): ndarray of feature of 
                        testing images, and each element contains a histogram of d columns 
    Returns:
        test_predicts (M): list of string of predict category for each testing image
    NOTE:
        1. d is the dimension of the feature representation, depending on using
           'tiny_image' or 'bag_of_sift'
        2. N is the total number of training images
        3. M is the total number of testing images
    '''

    ###########################################################################
    # TODO:                                                                   #
    # KNN predict the category for every testing image by finding the         #
    # training image with most similar (nearest) features, you can follow     #
    # below steps:                                                            #
    #   1. calculate the distance between training and testing features       #
    #   2. for each testing feature, select its k-nearest training features   #
    #   3. get these k training features' label id and vote for the final id  #
    # Remember to convert final id's type back to string, you can use CAT     #
    # and CAT2ID for conversion                                               #
    #                                                                         #
    # NOTE:                                                                   #
    # Some useful functions                                                   #
    #   Function : cdist(feats, feats)                                        #
    #                                                                         #
    # NOTE:                                                                   #
    #   1. instead of 1 nearest neighbor, you can vote based on k nearest     #
    #      neighbors which may increase the performance                       #
    #   2. hint: use 'minkowski' metric for cdist() and use a smaller 'p' may #
    #      work better, or you can also try different metrics for cdist()     #
    ###########################################################################

    start_time = time() 
    print("Construct KNN...")
    
    k = 5
    dist = cdist(test_img_feats, train_img_feats, metric='minkowski', p=0.5) # size = (M,N)

    # For each testing feature, select its k-nearest training features
    k_nearest = np.argpartition(dist, k, axis=1)[:, :k] # size = (M,k)

    # Vote for the final id based on k nearest training features' label id
    test_predicts = []
    for row in k_nearest:
        labels = np.array(train_labels)[row]
        label_ids = np.array([CAT2ID[label] for label in labels])
        final_id = np.bincount(label_ids).argmax()
        final_label = CAT[final_id]
        test_predicts.append(final_label)
        
    end_time = time()
    print(f"It takes {((end_time - start_time)/60):.2f} minutes to construct KNN.")

    ###########################################################################
    #                               END OF YOUR CODE                          #
    ###########################################################################
    
    return test_predicts
