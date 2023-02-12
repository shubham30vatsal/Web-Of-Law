# Text-To-Image Generation Using Stable Diffusion



## This repositiory aims to fine-tune latest stable diffusion model on CIFAR-10 and Oxford-102 (Flower) dataset. The details of the repository have been listed below:

1. stable_diffusion_cifar.ipynb notebook contains the code to fine-tune CIFAR-10 on stable diffusion. Please download CIFAR-10 dataset and create the format of dataset as instructed in the notebook.
2. stable_diffusion_flower.ipynb notebook contains the code to fine-tune CIFAR-10 on stable diffusion. Please download Oxford-10 (Flower) dataset and create the format of dataset as instructed in the notebook.
3. train_text_to_image.py contains the hugging face wrapper for fine-tuning stable diffusion.
4. fid_cifar10.ipynb calculates the FID between generated test set and ground truth test set of CIFAR-10.
5. fid_flower.ipynb calculates the FID between generated test set and ground truth test set of Oxford-102 (Flower).
6. gan_test_cifar.ipynb evaluates the classification performance of one of the state of the art models on the generated CIFAR-10 dataset.
7. resize.py contains the code for resizing images
8. split_* contain code for splitting images into classes for the use of imageFolder. 



Please contact sv2128@nyu.edu for any issues/bugs.
