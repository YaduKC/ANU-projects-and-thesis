import PIL
from PIL import Image
import matplotlib.pyplot as plt
import random
import numpy as np
import glob
from heapq import nsmallest

def face_detection(test_img_input):
    
    image_list = [] # Array to store all training images
    # Read all train images
    for filename in glob.glob('Yale-FaceA/trainingset/*.png'):
        im=Image.open(filename)
        image_list.append(im)
    
    # Convert images to single 2-dimensional array with 195*231(resolution of images) rows and 144 columns(num of training images)
    im_matrix = []
    for image in image_list:
        im = (np.array(image)).astype(float)
        im_matrix.append(im.flatten('F'))
    im_matrix = np.stack(im_matrix).T
    # Compute mean image
    mean = np.mean(im_matrix,1)
    r_mean = mean.reshape(195*231,1)
#     plt.axis('off')
#     plt.imshow(mean.reshape(195,231).T)
#     plt.savefig('mean_face_135.png')
    
    # Normalize the training images
    normalised_face = im_matrix - r_mean
    # Compute covariance to minimize computation
    cov = (normalised_face.T @ normalised_face) / 135
    # Compute eigen vectors and eigen values
    e_val, e_vec = np.linalg.eig(cov)
    indx = e_val.argsort()[::-1]
    e_val = e_val[indx]
    e_vec = e_vec[:,indx]
    # Find top 10 eigen vectors
    top_k = e_vec[0:10,:]
    # Compute eigen face using eigen vector
    eigen_face = top_k.dot(normalised_face.T)
    # Compute training weight matrix
    weights = (normalised_face.T).dot(eigen_face.T)
    
# Displaying eigenfaces    
#     fig, ax_1 = plt.subplots(1,5,figsize = (15,10))
#     c = 0
#     for j in range(5):
#         ax_1[j].axis('off')
#         ax_1[j].set_title(str(c))
#         ax_1[j].imshow(((eigen_face[c,:].real).reshape(195,231)).T)
#         c+=1
#     fig.savefig('top_k_1.png')
        
#     fig, ax_2 = plt.subplots(1,5,figsize = (15,10))
#     for j in range(5):
#         ax_2[j].axis('off')
#         ax_2[j].set_title(str(c))
#         ax_2[j].imshow(((eigen_face[c,:].real).reshape(195,231)).T)
#         c+=1
        
#     fig.savefig('top_k_2.png')
            
    # Normalise the test image
    test_img = (test_img_input.flatten('F').T).reshape(195*231,1)
    test_norm = test_img - r_mean
    # Compute weight of test image
    test_w = (test_norm.T).dot(eigen_face.T)
    # Compute distance between test weight and training weights
    dist = np.linalg.norm(test_w-weights,axis=1)
    
    # Find index of top 3 training images based on distance 
    top_k = nsmallest(3,dist)
    index = []
    counter = 0
    for i in dist:
        for j in top_k:
            if i == j:
                index.append(counter)
        counter += 1
    
    # Plot outputs
    fig,axes = plt.subplots(1,4,figsize = (15,10) )
    axes[0].axis('off')
    axes[1].axis('off')
    axes[2].axis('off')
    axes[3].axis('off')
    axes[0].set_title("Input Image")
    axes[1].set_title("First image")
    axes[2].set_title("Second image")
    axes[3].set_title("Third image")
    axes[0].imshow(test_img_input)
    axes[1].imshow((im_matrix[:,index[0]].reshape(195,231)).T)
    axes[2].imshow((im_matrix[:,index[1]].reshape(195,231)).T)
    axes[3].imshow((im_matrix[:,index[2]].reshape(195,231)).T)
    #fig.savefig('my_tr.png')

face_detection(np.array(Image.open("Yale-FaceA/testset/subject16.png")))
plt.show()
