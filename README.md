# DDBF
The code of Dispel Darkness for Better Fusion: A Controllable Visual Enhancer based on Cross-modal Conditional Adversarial Learning

#### Recommended Environment:<br>
 - [ ] python = 2.7
 - [ ] tensorflow-gpu = 1.9.0
 - [ ] numpy = 1.15.4
 - [ ] scipy = 1.2.0
 - [ ] pillow = 5.4.1
 - [ ] scikit-image = 0.13.1

#### After testing, the following configuration is also OK:<br>
 - [ ] python = 3.6
 - [ ] tensorflow-gpu = 1.9.0
 - [ ] numpy = 1.19.2
 - [ ] scipy = 1.5.4
 - [ ] pillow = 8.4.0
 - [ ] scikit-image = 0.17.2

 
#### Prepare data :<br>
- [ ] Put low-light images in the "Dataset/Test/Low-light/..." for testing the function of low-light enhancement
- [ ] Put multi-modal images in the "Dataset/Test/Fusion/..." for testing the complete enhancement and fusion capabilities of DDBF.

#### Testing :<br>
- [ ] Run "CUDA_VISIBLE_DEVICES=X python evaluate_Enhance.py" to enhance the provided low-light images.
- [ ] Run "CUDA_VISIBLE_DEVICES=X python evaluate_DDBF.py" to enhance and fuse the provided multi-modal images.
