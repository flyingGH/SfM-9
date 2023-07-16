import cv2
import numpy as np
# import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools


def SIFT(source_image, target_image):
# Perform SIFT feature detection
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(source_image, None)
    keypoints2, descriptors2 = sift.detectAndCompute(target_image, None)
    
    return keypoints1, keypoints2, descriptors1, descriptors2

def BFMatcher(keypoints1, keypoints2, descriptors1, descriptors2):
    # Perform Brute-Force matching
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Get the matched keypoints
    matched_keypoints1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    matched_keypoints2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
    
    return matches, matched_keypoints1, matched_keypoints2

def FundamentalMatrix(matched_keypoints1, matched_keypoints2):
    # Compute the fundamental matrix
    fundamental_matrix, mask = cv2.findFundamentalMat(matched_keypoints1, matched_keypoints2, cv2.FM_RANSAC)
    # Print the fundamental matrix
    
    # Check rank of fundamental matrix
    U, S, Vt = np.linalg.svd(fundamental_matrix)
    tol = 1e-6
    rank = np.sum(S > tol)
    print("Rank of Fundamental Matrix:", rank)
    
    return fundamental_matrix, mask

def VisualizeInlierOutlierMatches(source_image, target_image, keypoints1, keypoints2, matches, mask):
    
    # Visualize the matches and the inlier matches
    inlier_matches = [match for match, mask_val in zip(matches, mask) if mask_val]
    outlier_matches = [match for match, mask_val in zip(matches, mask) if not mask_val]

    inlier_image = cv2.drawMatches(source_image, keypoints1, target_image, keypoints2, inlier_matches, None)
    outlier_image = cv2.drawMatches(source_image, keypoints1, target_image, keypoints2, outlier_matches, None)

    cv2.imshow('Inlier Matches', inlier_image)
    cv2.imshow('Outlier Matches', outlier_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def EssentialMatrix(fundamental_matrix):
    
    # K is the camera calibration matrix
    # save('P3Data/Calibration/cameraParams.mat', '-v7')
    # K = scipy.io.loadmat('P3Data/Calibration/cameraParams.mat')
    # K = np.loadtxt('P3Data/Calibration/cameraParams.mat')
    K = np.array([[531.122155322710, 0, 407.192550839899],[0, 531.541737503901, 313.308715048366],[0, 0, 1]])
    print(K)

    # Compute the essential matrix
    essential_matrix = np.matmul(np.matmul(np.transpose(K), fundamental_matrix), K)
    
    return essential_matrix

def EstimateCameraPose(essential_matrix):
    U, D, Vt = np.linalg.svd(essential_matrix)  # Decompose the essential matrix into U, D, Vt using SVD
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # Define the matrix W
    
    print("U:", U)
    print("D:", D)
    print("Vt:", Vt)
    
    # four configurations of camera pose
    C1 = U[:, 2]
    R1 = np.matmul(np.matmul(U, W), Vt)
    C2 = -U[:, 2]
    R2 = np.matmul(np.matmul(U, W), Vt)
    C3 = U[:, 2]
    R3 = np.matmul(np.matmul(U, np.transpose(W)), Vt)
    C4 = -U[:, 2]
    R4 = np.matmul(np.matmul(U, np.transpose(W)), Vt)
    

    C = [C1, C2, C3, C4]
    R = [R1, R2, R3, R4]
    
    for i in range(4):
        if np.linalg.det(R[i]) < 0:
            R[i] = -R[i]
            C[i] = -C[i]

    return C, R

# def Triangulation(C, R, matched_keypoints1, matched_keypoints2):
#     C1 = C[0]
#     C2 = C[1]
#     R1 = R[0]
#     R2 = R[1]
    
#     # Compute the projection matrices
#     proj_mat1 = np.hstack((R1, C1.reshape(3, 1)))
#     proj_mat2 = np.hstack((R2, C2.reshape(3, 1)))
    
#     # Triangulate the points (Homogeneous)
#     points_4d = cv2.triangulatePoints(proj_mat1, proj_mat2, matched_keypoints1.T, matched_keypoints2.T)
    
#     # Convert the points from homogeneous to Euclidean
#     points_3d = cv2.convertPointsFromHomogeneous(points_4d.T)
#     triangulated_points = points_3d.reshape(-1, 3)
    
#     # Check Cheirality condition
#     is_cheirality_satisfied = np.all(points_4d[2] > 0)
    
#     print("Points 3D:", triangulated_points)
#     print("Cheirality Satisfied:", is_cheirality_satisfied)
    
#     # Plot the triangulated points
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(triangulated_points[:, 0], triangulated_points[:, 1], triangulated_points[:, 2], c='b', marker='o')
    
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title('Triangulated Points')
    
#     plt.show()
    
#     return triangulated_points, is_cheirality_satisfied

def Triangulation(C, R, matched_keypoints1, matched_keypoints2):
    
    C_combinations = list(itertools.combinations(C, 2))
    R_combinations = list(itertools.combinations(R, 2))
    
    for C_pair, R_pair in zip(C_combinations, R_combinations):
        C_first, C_second = C_pair
        R_first, R_second = R_pair

        print("C_Pair:", C_pair)
        
        # Compute the projection matrices
        proj_mat1 = np.hstack((R_first, C_first.reshape(3, 1)))
        proj_mat2 = np.hstack((R_second, C_second.reshape(3, 1)))
        
        # Triangulate the points (Homogeneous)
        points_4d = cv2.triangulatePoints(proj_mat1, proj_mat2, matched_keypoints1.T, matched_keypoints2.T)
        
        #Convert the points from homogeneous to Euclidean
        points_3d = cv2.convertPointsFromHomogeneous(points_4d.T)
        triangulated_points = points_3d.reshape(-1, 3)
        print("Triangulated Points:", triangulated_points)

        # Append the triangulated points into a list
        triangulated_points_list = []
        triangulated_points_list.append(triangulated_points)
        
        # Check Cheirality condition
        is_cheirality_satisfied = np.all(points_4d[2] > 0)
        
        # Append the cheirality satisfied flag into a list
        is_cheirality_satisfied_list = []
        is_cheirality_satisfied_list.append(is_cheirality_satisfied)
    
        # print("Points 3D:", triangulated_points)
        # print("Cheirality Satisfied:", is_cheirality_satisfied)
  
    return triangulated_points_list, is_cheirality_satisfied_list
    
def Visualize3DPoints(triangulated_points_list):
    
    # Create a 3D plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    fig, ax = plt.subplots()
    colors = ['r', 'g', 'b', 'c']

    # Plot the triangulated points from each combination
    for i, triangulated_points in enumerate(triangulated_points_list):
        # Extract x, y, z coordinates
        x = triangulated_points[:, 0]
        # y = triangulated_points[:, 1]
        z = triangulated_points[:, 2]

        # Plot the triangulated points
        ax.scatter(x, z, color = colors[i], label = 'Combination ' + str(i + 1))

    # Set labels and title
    ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    ax.set_ylabel('Z')
    ax.set_title('Triangulated Points')

    ax.legend()
    
    # Show the plot
    plt.show()
    
    
    


def main():
    # Load the source and target images
    source_image = cv2.imread('P3Data/1.png')
    target_image = cv2.imread('P3Data/2.png')

    # Perform SIFT feature detection
    keypoints1, keypoints2, descriptors1, descriptors2 = SIFT(source_image, target_image)

    # Perform Brute-Force matching
    matches, matched_keypoints1, matched_keypoints2 = BFMatcher(keypoints1, keypoints2, descriptors1, descriptors2)

    # Compute the fundamental matrix
    fundamental_matrix, mask = FundamentalMatrix(matched_keypoints1, matched_keypoints2)
    print("Fundamental Matrix:")
    print(fundamental_matrix)

    # Visualize the matches and the inlier matches
    VisualizeInlierOutlierMatches(source_image, target_image, keypoints1, keypoints2, matches, mask)

    # Estimate Essesntial Matrix from Fundamental Matrix
    essential_matrix = EssentialMatrix(fundamental_matrix)
    print("Essential Matrix:")
    print(essential_matrix)

    # Estimate Camera Pose from Essential Matrix
    C, R = EstimateCameraPose(essential_matrix)

    # Linear Triangulation
    triangulated_points_list, is_cheirality_satisfied_list = Triangulation(C, R, matched_keypoints1, matched_keypoints2)
    print("Triangulated Points 1st Combination:", triangulated_points_list[0])
    
    # Visualize the triangulated points
    Visualize3DPoints(triangulated_points_list)

 
 
 
    
if __name__ == '__main__':
    main()
