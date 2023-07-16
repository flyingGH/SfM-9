import cv2 as cv
import numpy as np

# RANSAC function
def ransac_filter(matches, keypoints1, keypoints2, threshold, max_iterations):
    best_model = None
    best_inliers = []

    for _ in range(max_iterations):
        # Randomly sample a minimal subset of matches
        sample = np.random.choice(matches, 100, replace=False)

        src_points = np.float32([keypoints1[m.queryIdx].pt for m in sample])
        dst_points = np.float32([keypoints2[m.trainIdx].pt for m in sample])

        # Estimate the fundamental matrix
        fundamental_matrix, _ = cv.findFundamentalMat(src_points, dst_points, cv.FM_RANSAC, threshold)

        # Calculate the epilines for the sampled source and destination points
        src_lines = cv.computeCorrespondEpilines(src_points.reshape(-1, 1, 2), 1, fundamental_matrix)
        dst_lines = cv.computeCorrespondEpilines(dst_points.reshape(-1, 1, 2), 2, fundamental_matrix)

        # Reshape the epilines to (N, 3)
        src_lines = src_lines.reshape(-1, 3)
        dst_lines = dst_lines.reshape(-1, 3)

        # Compute the distances between points and epilines
        distances = np.abs(np.sum(src_lines * np.hstack((dst_points, np.ones((dst_points.shape[0], 1)))), axis=1)) + \
                    np.abs(np.sum(dst_lines * np.hstack((src_points, np.ones((src_points.shape[0], 1)))), axis=1))

        # Count points as inliers if the distance is below the threshold
        inliers = [match for match, distance in zip(matches, distances) if distance < threshold]

        # Keep track of the model with the highest number of inliers
        if len(inliers) > len(best_inliers):
            best_model = fundamental_matrix
            best_inliers = inliers

    return best_model, best_inliers



# SIFT
sift = cv.SIFT_create()

# Feature Detection and Matching
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

# read images
img1 = cv.imread('P3Data/1.png')
img2 = cv.imread('P3Data/2.png')

# convert images to gray
img1 = cv.cvtColor(img1, 0)
img2 = cv.cvtColor(img2, 0)

# find keypoints and descriptors
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)
# matched_indices = [m.trainIdx for m in matches]

# img3 = cv.drawMatches(img1, keypoints1, img2, keypoints2, matches[300:600], img2, flags=2)
# cv.imshow('SIFT Matches', img3)

# Apply RANSAC to filter out outliers
threshold = 200  # Distance threshold for inliers
max_iterations = 5000  # Maximum iterations for RANSAC
best_model, best_inliers = ransac_filter(matches, keypoints1, keypoints2, threshold, max_iterations)

img4 = cv.drawMatches(img1, keypoints1, img2, keypoints2, best_inliers, None)
cv.imshow('Inlier Matches', img4)


cv.waitKey(0)
cv.destroyAllWindows()
