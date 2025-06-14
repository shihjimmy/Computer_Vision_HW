import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    """
    two equations:
        h11*ux + h12*uy + h13   + h21*0  + h22*0  + h23*0 -h31*ux*vx -h32*uy*vx -h33*vx = 0
        h11*0  + h12*0  + h13*0 + h21*ux + h22*uy + h23   -h31*ux*vy -h32*uy*vy -h33*vy = 0
    """
    A = []
    for i in range(N):
        A.append([u[i][0], u[i][1], 1, 0, 0, 0, -u[i][0]*v[i][0], -u[i][1]*v[i][0], -v[i][0]])
        A.append([0, 0, 0, u[i][0], u[i][1], 1, -u[i][0]*v[i][1], -u[i][1]*v[i][1], -v[i][1]])

    # TODO: 2.solve H with A
    """
        A*h = 0
        solve h --> finding NULL space of A
    """
    
    A = np.array(A)
    _, _, v_t = np.linalg.svd(A)
    H = v_t[-1].reshape(3,3)
     
    return H


def warping(src, dst, H, direction='b'):
    """
    Perform forward or backward warping on an image using a homography matrix.
    The warping process maps source image pixels to destination image pixels, or vice versa, 
    depending on the direction specified.
    
    :param src: The source image that will be warped (input image).
    :param dst: The destination image (output image) where the warped result is stored.
    :param H: The homography matrix (3x3) used for transforming points between images.
    :param direction: The direction of warping ('b' for backward warping, 'f' for forward warping).
    :return: The destination image with the warping applied.
    """
    
    h_src, w_src, ch = src.shape    # Get the height, width, and number of channels of the source image.
    h_dst, w_dst, ch = dst.shape    # Get the height, width, and number of channels of the destination image.
    H_inv = np.linalg.inv(H)        # Compute the inverse of the homography matrix.

    # TODO: 1. Generate (x, y) coordinate pairs using meshgrid for the destination image
    if direction == 'b':
        x = np.arange(0, w_dst, 1)  # Create an array of x coordinates for the destination image.
        y = np.arange(0, h_dst, 1)  # Create an array of y coordinates for the destination image.
        
        # Use meshgrid to generate a grid of (x, y) coordinates for all pixels in the destination image.
        xx, yy = np.meshgrid(x, y)
        
        # Flatten the meshgrid and reshape it to a 1D array of coordinates.
        xx, yy = xx.flatten()[:, np.newaxis], yy.flatten()[:, np.newaxis]
        
        # Add a column of ones to make the coordinates homogeneous (x, y, 1).
        ones = np.ones((len(xx), 1))    
        des_coor = np.concatenate((xx, yy, ones), axis=1).astype(np.int)  # Concatenate the x, y, and 1 to create homogeneous coordinates.

        # TODO: 3. Apply the inverse homography matrix to the destination pixels to get the corresponding source coordinates (u, v)
        Resource_pixel = H_inv.dot(des_coor.T).T  # Apply the inverse homography to get the source coordinates.
        
        # TODO: 4. Normalize the transformed coordinates (to get x and y, divide by the third coordinate)
        Resource_pixel[:, :2] = Resource_pixel[:, :2] / Resource_pixel[:, 2][:, np.newaxis]  # Normalize the x, y by dividing by the third coordinate (homogeneous normalization).
        
        # Mask to remove invalid transformed coordinates (out of source image boundaries)
        out_boundary = []   
        
        if (Resource_pixel[:, 0] < 0).any():  # Check if any x coordinate is less than 0 (out of bounds).
            out_boundary += np.where(Resource_pixel[:, 0] < 0)[0].tolist()
        if (Resource_pixel[:, 1] < 0).any():  # Check if any y coordinate is less than 0 (out of bounds).
            out_boundary += np.where(Resource_pixel[:, 1] < 0)[0].tolist()
        if (Resource_pixel[:, 0] > w_src-1).any():  # Check if any x coordinate exceeds the width of the source image.
            out_boundary += np.where(Resource_pixel[:, 0] > (w_src - 1))[0].tolist()
        if (Resource_pixel[:, 1] > h_src-1).any():  # Check if any y coordinate exceeds the height of the source image.
            out_boundary += np.where(Resource_pixel[:, 1] > (h_src - 1))[0].tolist()
        
        # TODO: 5. Remove invalid coordinates that are out of the bounds of the source image
        if len(out_boundary):
            Resource_pixel = np.delete(Resource_pixel, out_boundary, 0)     # Remove the invalid coordinates.
            des_coor = np.delete(des_coor, out_boundary, 0)                 # Remove the corresponding destination coordinates.

        # TODO: 6. Use advanced array indexing to assign values from source to destination image.
        tx = Resource_pixel[:, 0].astype(np.int)  # Extract the integer x coordinates of the source image.
        ty = Resource_pixel[:, 1].astype(np.int)  # Extract the integer y coordinates of the source image.
        dx = Resource_pixel[:, 0] - tx  # Compute the fractional part of the x coordinate.
        dy = Resource_pixel[:, 1] - ty  # Compute the fractional part of the y coordinate.

        ones = np.ones(len(dx)).astype(np.float)  # Create an array of ones for bilinear interpolation.

        # Apply **Bilinear Interpolation** to assign values from the source image to the destination image.
        dst[des_coor[:, 1], des_coor[:, 0]] = (
            (((ones - dx) * (ones - dy))[:, np.newaxis] * src[ty, tx]) +
            ((dx * (ones - dy))[:, np.newaxis] * src[ty, tx + 1]) +
            ((dx * dy)[:, np.newaxis] * src[ty + 1, tx + 1]) +
            (((ones - dx) * dy)[:, np.newaxis] * src[ty + 1, tx])
        )

    elif direction == 'f':  # Forward warping (from source to destination).
        x = np.arange(0, w_src-1, 1)  # Create an array of x coordinates for the source image.
        y = np.arange(0, h_src-1, 1)  # Create an array of y coordinates for the source image.
        
        # Use meshgrid to generate a grid of (x, y) coordinates for all pixels in the source image.
        xx, yy = np.meshgrid(x, y)
        xx, yy = xx.flatten()[:, np.newaxis], yy.flatten()[:, np.newaxis]
        
        # Add a column of ones to make the coordinates homogeneous (x, y, 1).
        ones = np.ones((len(xx), 1))
        des_coor = np.concatenate((xx, yy, ones), axis=1).astype(np.int)

        # TODO: 3. Apply the homography matrix to the source pixels to get the corresponding destination coordinates (u, v)
        Resource_pixel = H.dot(des_coor.T).T  # Apply the homography matrix to get the destination coordinates.
        
        # TODO: 4. Normalize the transformed coordinates (to get x and y, divide by the third coordinate)
        Resource_pixel[:, :2] = Resource_pixel[:, :2] / Resource_pixel[:, 2][:, np.newaxis]
        
        # Mask to remove invalid transformed coordinates (out of destination image boundaries)
        out_boundary = []
        
        if (Resource_pixel[:, 0] < 0).any():  # Check if any x coordinate is less than 0 (out of bounds).
            out_boundary += np.where(Resource_pixel[:, 0] < 0)[0].tolist()
        if (Resource_pixel[:, 1] < 0).any():  # Check if any y coordinate is less than 0 (out of bounds).
            out_boundary += np.where(Resource_pixel[:, 1] < 0)[0].tolist()
        if (Resource_pixel[:, 0] > w_dst-1).any():  # Check if any x coordinate exceeds the width of the destination image.
            out_boundary += np.where(Resource_pixel[:, 0] > (w_dst - 1))[0].tolist()
        if (Resource_pixel[:, 1] > h_dst-1).any():  # Check if any y coordinate exceeds the height of the destination image.
            out_boundary += np.where(Resource_pixel[:, 1] > (h_dst - 1))[0].tolist()
        
        # TODO: 5. Remove invalid coordinates that are out of the bounds of the destination image
        if len(out_boundary):
            Resource_pixel = np.delete(Resource_pixel, out_boundary, 0)  # Remove the invalid coordinates.
            des_coor = np.delete(des_coor, out_boundary, 0)  # Remove the corresponding source coordinates.

        # TODO: 6. Use advanced array indexing to assign values from source to destination image.
        tx = Resource_pixel[:, 0].astype(np.int)  # Extract the integer x coordinates of the source image.
        ty = Resource_pixel[:, 1].astype(np.int)  # Extract the integer y coordinates of the source image.
        dx = Resource_pixel[:, 0] - tx  # Compute the fractional part of the x coordinate.
        dy = Resource_pixel[:, 1] - ty  # Compute the fractional part of the y coordinate.

        ones = np.ones(len(dx)).astype(np.float)  # Create an array of ones for bilinear interpolation.

        # Apply **Bilinear Interpolation** to assign values from the source image to the destination image.
        dst[ty, tx] = (
            (((ones - dx) * (ones - dy))[:, np.newaxis] * src[des_coor[:, 1], des_coor[:, 0]]) +
            ((dx * (ones - dy))[:, np.newaxis] * src[des_coor[:, 1], des_coor[:, 0] + 1]) +
            ((dx * dy)[:, np.newaxis] * src[des_coor[:, 1] + 1, des_coor[:, 0] + 1]) +
            (((ones - dx) * dy)[:, np.newaxis] * src[des_coor[:, 1] + 1, des_coor[:, 0]])
        )

    return dst


