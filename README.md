# Binarization using double threshold 


http://felixniklas.com/imageprocessing/binarization

Binarization is the process of converting a pixel image to a binary image:
In the old days binarization was important for sending faxes. These days its still important for things like digitalising text or segmentation.

To make thresholding completely automated, it is necessary for the computer to automatically select the threshold T. Sezgin and Sankur (2004) categorize thresholding methods into the following six groups based on the information the algorithm manipulates (Sezgin et al., 2004)
    Histogram shape-based methods, where, for example, the peaks, valleys and curvatures of the smoothed histogram are analyzed
    Clustering-based methods, where the gray-level samples are clustered in two parts as background and foreground (object), or alternately are modeled as a mixture of two Gaussians
    Entropy-based methods result in algorithms that use the entropy of the foreground and background regions, the cross-entropy between the original and binarized image, etc.[1]
    Object Attribute-based methods search a measure of similarity between the gray-level and the binarized images, such as fuzzy shape similarity, edge coincidence, etc.
    Spatial methods [that] use higher-order probability distribution and/or correlation between pixels
    Local methods adapt the threshold value on each pixel to the local image characteristics. In these methods, a different T is selected for each pixel in the image.

In this code: 
using double threshold. 

