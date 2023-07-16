import cv2 as cv
import numpy as np

# RANSAC function
def ransac_filter(matches, keypoints1, keypoints2, threshold=10, max_iterations=1000):
    best_model = None
    best_inliers = []

    for _ in range(max_iterations):
        # Randomly sample a minimal subset of matches
        sample = np.random.choice(matches, 4, replace=False)

        src_points = np.float32([keypoints1[m.queryIdx].pt for m in sample]).reshape(-1, 1, 2)
        dst_points = np.float32([keypoints2[m.trainIdx].pt for m in sample]).reshape(-1, 1, 2)

        # Estimate the transformation model (homography matrix)
        model, _ = cv.findHomography(src_points, dst_points, cv.RANSAC, threshold)

        inliers = []
        for match in matches:
            src_point = keypoints1[match.queryIdx].pt
            dst_point = keypoints2[match.trainIdx].pt
            src_point = np.array([src_point[0], src_point[1], 1]).reshape(3, 1)
            transformed_point = np.dot(model, src_point)
            transformed_point /= transformed_point[2]  # Normalize homogeneous coordinates
            distance = np.linalg.norm(transformed_point[:2] - dst_point)

            # Count points as inliers if the distance is within the threshold
            if distance < threshold:
                inliers.append(match)

        # Keep track of the model with the highest number of inliers
        if len(inliers) > len(best_inliers):
            best_model = model
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
img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# find keypoints and descriptors
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)
# print(len(matches))

# img3 = cv.drawMatches(img1, keypoints1, img2, keypoints2, matches[300:600], img2, flags=2)
# cv.imshow('SIFT Matches', img3)

# Apply RANSAC to filter out outliers
threshold = 200  # Distance threshold for inliers
max_iterations = 1000  # Maximum iterations for RANSAC
best_model, best_inliers = ransac_filter(matches, keypoints1, keypoints2, threshold, max_iterations)

img4 = cv.drawMatches(img1, keypoints1, img2, keypoints2, best_inliers, None)
cv.imshow('Inlier Matches', img4)


cv.waitKey(0)
cv.destroyAllWindows()
