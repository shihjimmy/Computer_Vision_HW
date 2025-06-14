import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)

def matches(img1, img2):
    """
    This function detects and matches ORB keypoints between two images.
    
    :param img1: First input image to be matched.
    :param img2: Second input image to be matched.
    :return: A tuple of two arrays: 
             - The keypoints from img1 that match with img2.
             - The keypoints from img2 that correspond to img1.
    """
    
    # Create an ORB detector object, setting the number of features to detect.
    orb = cv2.ORB_create(nfeatures=1000)

    # Detect keypoints and descriptors in both images.
    # `kp_a`, `kp_b` are keypoints in img1 and img2 respectively.
    # `desc_a`, `desc_b` are the descriptors corresponding to the keypoints.
    kp_a, desc_a = orb.detectAndCompute(img1, None)
    kp_b, desc_b = orb.detectAndCompute(img2, None)

    # Create a Brute Force matcher using Hamming distance, suitable for ORB.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # Perform knnMatch to find the best 2 matches for each descriptor from img1 to img2.
    matches = bf.knnMatch(desc_a, desc_b, k=2)

    # List to hold good matches that satisfy Lowe's ratio test.
    good_matches = []

    # Apply Lowe's ratio test: 
    # keep a match only if the distance of the first match is less than 0.8 times the distance of the second match.
    for match_1, match_2 in matches:
        if match_1.distance < 0.8 * match_2.distance:
            # Add the match to the list if it passes the ratio test.
            good_matches.append(match_1)  

    # Lists to hold the keypoints of good matches from both images.
    good_kp_a = []  # Keypoints from img1 (source image).
    good_kp_b = []  # Keypoints from img2 (target image).

    # Iterate through all the good matches and extract the corresponding keypoints.
    for match in good_matches:
        good_kp_a.append(kp_a[match.queryIdx].pt)  # Get keypoint coordinates ( .pt -> (x,y) ) from img1.
        good_kp_b.append(kp_b[match.trainIdx].pt)  # Get keypoint coordinates ( .pt -> (x,y) ) from img2.

    # Return the good keypoints as numpy arrays for further processing.
    return np.array(good_kp_a).astype(np.int), np.array(good_kp_b).astype(np.int)



def transform_with_homography(H, points):
    """
    This function applies a homography transformation to a set of points.
    
    :param H: The homography matrix (3x3) that represents the transformation.
    :param points: A numpy array of shape (N, 2) containing N points to be transformed.
                   Each point is represented by (x, y) coordinates.
    :return: A numpy array of shape (N, 2) representing the transformed points in the target coordinate system.
    """
    
    # Step 1: Add a column of ones to the points array to make it homogeneous coordinates (x, y, 1)
    ones = np.ones((points.shape[0], 1))  # Create a column of ones with shape (N, 1)
    points = np.concatenate((points, ones), axis=1)  # Append the ones to the points array to make it (N, 3)
    
    # Step 2: Apply the homography matrix to the points (matrix multiplication)
    transformed_points = H.dot(points.T)  # Perform the matrix multiplication: H * points.T (3, N)
    
    # Step 3: Normalize the homogeneous coordinates (divide by the third row to get (x, y, 1))
    transformed_points = transformed_points / (transformed_points[2,:][np.newaxis, :])  # Normalize by the third coordinate
    
    # Step 4: Extract the (x, y) coordinates from the transformed points (drop the third coordinate)
    transformed_points = transformed_points[0:2,:].T  # Take the first two rows (x, y) and transpose back to (N, 2)

    return transformed_points  # Return the transformed points




def compute_outlier(H, points_a, points_b, threshold=3):
    """
    This function computes the number of outliers by comparing the transformed points with the ground truth.
    
    :param H: The homography matrix used to transform points_b to the coordinate system of points_a.
    :param points_a: Ground truth points (N, 2) from the source image.
    :param points_b: Points (N, 2) from the destination image that we want to transform.
    :param threshold: A distance threshold for considering a point as an outlier.
    :return: The number of outliers (points that are too far from their transformed positions).
    """
    
    # Step 1: Transform points_b to points_a's coordinate system using the homography matrix H
    points_img_b_transformed = transform_with_homography(H, points_b)

    # Step 2: Calculate the Euclidean distance between the transformed points and the ground truth points
    x = points_a[:, 0]                      # x-coordinates of points_a
    y = points_a[:, 1]                      # y-coordinates of points_a
    x_hat = points_img_b_transformed[:, 0]  # x-coordinates of the transformed points
    y_hat = points_img_b_transformed[:, 1]  # y-coordinates of the transformed points
    
    # Step 3: Compute the Euclidean distance between the corresponding points in points_a and transformed points
    distance = np.sqrt(np.power((x_hat - x), 2) + np.power((y_hat - y), 2)).reshape(-1)
    
    # Step 4: Count how many points are outliers (distance greater than threshold)
    outliers_count = 0
    for dis in distance:
        if dis > threshold:  # If the distance is greater than the threshold, it's an outlier
            outliers_count += 1

    return outliers_count  # Return the number of outliers




def ransac_for_homography(matches_1, matches_2):
    """
    This function implements the RANSAC algorithm to estimate the best homography matrix.
    RANSAC is used to handle outliers in the matched points and find the most accurate homography.
    
    :param matches_1: Points from the source image (N, 2).
    :param matches_2: Points from the destination image (N, 2).
    :return: The best homography matrix H after applying RANSAC.
    """
    
    # Step 1: Define the total number of matches and RANSAC parameters
    all_matches = matches_1.shape[0]  # Total number of matches
    
    # RANSAC parameters
    prob_success = 0.99     # Probability of successfully finding the correct model
    sample_points_size = 5  # Number of points to sample for each iteration (at least 4 for homography)
    ratio_of_outlier = 0.5  # Expected ratio of outliers in the data
    N = int(np.log(1.0 - prob_success) / np.log(1 - (1 - ratio_of_outlier) ** sample_points_size))  # Number of iterations

    lowest_outlier = all_matches    # Start with the worst case: all the points are outliers
    best_H = None                   # Variable to store the best homography matrix

    # Step 2: Perform RANSAC iterations
    for i in range(N):
        # Step 2.1: Randomly select sample_points_size points to compute the homography
        rand_index = np.random.choice(all_matches, sample_points_size, replace=False)  # Randomly sample 5 points
        H = solve_homography(matches_2[rand_index], matches_1[rand_index])  # Compute homography using these points
        
        # Step 2.2: Compute the number of outliers for the current homography
        outliers_count = compute_outlier(H, matches_1, matches_2)  # Calculate outliers for the current H
        
        # Step 2.3: If the current homography has fewer outliers, update the best_homography matrix
        if outliers_count < lowest_outlier:
            best_H = H
            lowest_outlier = outliers_count  # Update the lowest outlier count
    
    return best_H  # Return the best homography matrix found by RANSAC



def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None

    # for all images to be stitched:
    for idx in tqdm(range(len(imgs)-1)):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # TODO: 1.feature detection & matching
        matches_1, matches_2 = matches(im1, im2)
        
        # TODO: 2. apply RANSAC to choose best H
        H = ransac_for_homography(matches_1, matches_2)
        
        # TODO: 3. chain the homographies
        last_best_H = last_best_H.dot(H)
        
        # TODO: 4. apply warping
        dst = warping(im2, dst, last_best_H, 'b')
        
    out = dst    
    return out 


if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)
    