# Computer_Vision_HW
113-2 Spring Computer Vision in NTU GIEE, only three HW in this semester.  
Each assignment covers foundational techniques from feature detection to deep learning and geometric vision.

## 📌 HW1: Scale-Invariant Feature Detection & Image Filtering

- **Part 1: Difference of Gaussian (DoG)**  
  Implemented a DoG-based scale-invariant feature detector. Key steps included Gaussian pyramid construction, local extrema detection, and feature pruning.  

- **Part 2: Advanced Image Filtering**  
  - Designed a better color-to-gray conversion using perceptual similarity.  
  - Implemented a joint bilateral filter from scratch.  
  - Evaluated grayscale conversion through L1 distance in bilateral-filtered outputs.  

## 🖼️ HW2: Scene Recognition and Image Classification

- **Part 1: Bag-of-Words Scene Recognition**  
  - Built a vocabulary using K-means on SIFT descriptors.  
  - Compared tiny image and BoW + SIFT representations with a custom KNN classifier.  

- **Part 2: CNN Image Classification**  
  - Trained a CNN from scratch and evaluated performance.  
  - Compared it with ResNet18 (pretrained weights allowed).  
  - Applied data augmentation and optional semi-supervised techniques.  
- Delivered robust image recognition pipeline from handcrafted to deep models.  


## 🧭 HW3: Projective Geometry and AR

- **Part 1: Homography Estimation**  
  Implemented DLT-based homography solver and forward warping for image transformation.

- **Part 2: Marker-Based Augmented Reality**  
  Used ArUco markers to overlay virtual content via backward warping on video frames.

- **Part 3: Secret Unwarping**  
  Extracted hidden QR codes through homography correction and analyzed geometric variance.

- **Part 4: Panorama Stitching**  
  Used ORB keypoints and RANSAC to estimate multi-image homographies and stitch panorama with backward warping.  
- Core functions like `solve_homography()` and `warping()` were implemented from scratch.  
