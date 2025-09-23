Residual Multi-Scale Attention Network (RMSANet) for Image Denoising
This project implements a novel Convolutional Neural Network (CNN) for image denoising, named the Residual Multi-Scale Attention Network (RMSANet). The model is designed to effectively remove a wide range of noise from images while preserving fine-grained details.

🌟 Key Features
The RMSANet architecture is a hybrid model that combines three advanced techniques to achieve superior denoising performance:

Residual-in-Residual Blocks (RIR): Inspired by the RIDNet architecture, this feature ensures efficient information and gradient flow through the model's deep network.

Multi-Scale Feature Extraction (MSFE): This module, derived from MSDN principles, processes features at different scales to capture both local textures and broader contextual information.

Enhanced Attention Module (EAM): Based on EAMNet, this module dynamically re-weights features across both channel and spatial dimensions, allowing the model to focus on the most critical parts of the image for optimal denoising.

💻 Setup and Installation
Follow these steps to set up the project environment and install all dependencies.

Create and Activate a Conda Environment:

conda create -n rmsanet_env python=3.9
conda activate rmsanet_env


Install Libraries: Use the requirements.txt file to install all necessary packages.

pip install -r requirements.txt


🚀 Usage
1. Prepare Datasets
The model is trained on pre-generated image patches from the BSD4342 and CBSD432 datasets. These patches are generated using the generate_patches.py script. The final model is evaluated on full images to get accurate performance scores.

2. Training the Grayscale Model
The grayscale model is trained on image patches with AWGN noise ranging from to . This creates a versatile "blind" denoiser. To start the training run:

python train.py


The script will train the model for 100 epochs and automatically save the best-performing version to checkpoints/model_best.pth.

3. Training the Color Model
To train the color model, you will need to first generate color patches from the CBSD432 dataset using an updated generate_patches.py script. Then, you will train the model with a modified configuration. The best-performing model will be saved to checkpoints/model_color_best.pth.

4. Evaluating the Model
After training, you can evaluate the model's performance on the test datasets by running the test.py script. The script takes dataset and noise_sigma as arguments.

To test your grayscale model on the BSD68 dataset with a noise level of 25:

python test.py --dataset BSD68 --noise_sigma 25


To test your color model on the CBSD68 dataset with a noise level of 25:

python test.py --dataset CBSD68 --noise_sigma 25

📈 Final Results
Your enhanced RMSANet model achieved excellent results on the standard test datasets.
