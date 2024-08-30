# **High-Quality Medical Image Synthesis Using Transformer-Enhanced Denoising Diffusion Models**

## **Project Overview**
In healthcare, medical imaging is crucial for accurate diagnosis, treatment planning, and monitoring of disease progression. Generating high-quality medical images for training machine learning models is challenging due to data privacy concerns and the need for large annotated datasets. This project proposes the development of a novel transformer-enhanced Super Resolution Denoising Diffusion Probabilistic Model (SRDDPM) to synthesize high-quality 2D medical images. Leveraging transformer technologies within SRDDPMs aims to improve the stability and quality of generated images compared to existing methods like GANs and VAEs. The model will be trained and validated using multiple medical imaging datasets to ensure its ability to generate medically accurate and diverse synthetic images. The expected outcome is a robust generative model capable of producing high-resolution medical images, significantly aiding in training AI systems without compromising data privacy.

## **Contributions**
This project presents a Cascaded Super-Resolution DDPM consisting of three DDPM architectures: one designed to generate a lower-resolution image and the subsequent two cascaded to progressively upscale the image to a high resolution. The model is trained on the MRNet dataset, focusing on knee MRI images related to musculoskeletal abnormalities. The model is enhanced by utilizing a UNet architecture with a Swin-transformer block in the UNet's bottleneck layer for improved local and global feature learning, which is crucial for high-resolution medical images.

Furthermore, this project includes a comprehensive comparison of the performance of the cascaded Super-Resolution DDPM with other approaches. It compares:
- The cascaded Super-Resolution DDPM against a single-stage DDPM architecture trained directly on images of the same size.
- The cascaded Super-Resolution DDPM with and without the Swin-transformer blocks in the UNet architecture.
- The Super-Resolution models with interpolation methods to test the quality of upscaled images.

## **Dataset Information**
- **Dataset**: Stanford MRNet
- **Description**: The dataset consists of 1,370 knee MRI exams from Stanford University Medical Center, including 1,104 exams with ACL and meniscal tear labels manually extracted from clinical reports.
- **Reference**: Bien, Nicholas, et al. "Deep-learning-assisted diagnosis for knee magnetic resonance imaging: development and retrospective validation of MRNet." PLoS medicine 15.11 (2018): e1002699.

## **Dependencies/Libraries**

To install all necessary dependencies, you can run:

```bash
pip install torch torchvision torchmetrics pandas pillow matplotlib numpy scipy tqdm scikit-learn torchio
```
## **Required Libraries:**
- **Python**: Ensure you have Python 3.x installed.
- **PyTorch**: The core library for building and training deep learning models.
- **TorchVision**: A package that provides datasets, models, and transformations specific to computer vision tasks.
- **TorchMetrics**: Provides a set of metrics for evaluating models, including FID and Inception Score.
- **Pandas**: For handling data in CSV files and DataFrames.
- **Pillow**: A library for opening, manipulating, and saving many different image file formats.
- **Matplotlib**: A library for creating static, animated, and interactive visualizations in Python.
- **NumPy**: For numerical computing, handling arrays, and performing mathematical operations.
- **SciPy**: Used for scientific and technical computing, including entropy calculations and matrix operations.
- **Tqdm**: A library for adding progress bars to loops.
- **Scikit-learn**: For calculating pairwise distances and other machine learning utilities.
- **TorchIO** (optional based on imports): A library for medical image preprocessing and augmentation.

## **Usage**
This project includes multiple Jupyter notebooks, each designed for a specific purpose, such as loading datasets, training models, and generating synthetic images. To get started, you can run the `Pipelinemain_MRNet.ipynb` notebook, which guides you through importing and running the model for image generation.

## **Models and Architectures**

### **Proposed Architecture**
This project employs a Cascaded Super-Resolution DDPM approach with a total of three models: the initial DDPM generates 64x64 images, and two subsequent Super-Resolution models (SR1 and SR2) upscale the images to 128x128 and 256x256, respectively.

- **64x64 Image Generator (Base DDPM):** 
  This model uses a UNet architecture with sinusoidal positional embeddings and self-attention mechanisms to generate 64x64 images from noise. The Swin Transformer is integrated into the bottleneck layer to capture both local and global dependencies.
  
  ![UNet for 64x64 Generator](images/unet_64x64.png)
  
- **Super-Resolution UNet (SR1 and SR2):** 
  The SR1 model upscales images from 64x64 to 128x128, and the SR2 model further upscales them to 256x256. These models do not include self-attention or cross-attention layers, focusing instead on the UNet architecture to refine image details effectively.

  ![Super-Resolution UNet](images/sr_unet.png)

- **Pipeline Overview:** 
  The following image depicts the overall pipeline of the proposed Cascaded Super-Resolution DDPM. It starts with the base DDPM generating low-resolution images, which are progressively upscaled using the SR1 and SR2 models.

  ![Proposed Architecture Pipeline](images/pipeline.png)

### **Single-Stage 256x256 DDPM**
This model is a single-stage DDPM trained to generate 256x256 resolution images directly from noise. It incorporates the Swin Transformer within the UNet architecture to enhance the model’s ability to capture image details.

### **SRDDPM Without Swin Transformer**
This model follows the same cascaded approach but excludes the Swin Transformer from the architecture, reducing computational complexity while still performing the image upscaling tasks.


- **SWIN Transformer**
- **SRDDPM**
- **SRDDPM without SWIN**
- **DDPM**
- **DDPM without SWIN**
- **DDPM 256x256**
There are a total of three models in discussion the SRDDPM, SRDDPM without SWIN and 256x256DDPM
## **Project Structure**
The project is organized as follows:

- **Evaluation Code**: Contains generated images and comparison images between SR models and interpolation methods.
- **DDPM_256x256**: Code for the 256x256 DDPM model and its savepoint.
- **Model_Savepoints**: Directory containing savepoints for all trained models.
- **Mrnet**: Contains a subset of dataset files and some preprocessed slices (located in `train_slices_raw` and `valid_slices_raw`).
- **Training Images**: Includes images generated by the model during training.
- **Jupyter Notebooks**:
  - `mrnet_load_trial.ipynb`: For loading and preprocessing the dataset.
  - `DDPM_64x64Generator_(NOSWIN).ipynb`: DDPM without SWIN transformer in its UNet.
  - `DDPM_64x64Generator.ipynb`: DDPM with SWIN transformer in its UNet.
  - `SR1-(64-128).ipynb`: SR DDPM for 64x64 to 128x128 upscaling with SWIN transformer in its UNet.
  - `SR1-(64-128)(NoSWIN).ipynb`: SR DDPM for 64x64 to 128x128 upscaling without SWIN transformer in its UNet.
  - `SR2-(128-256).ipynb`: SR DDPM for 128x128 to 256x256 upscaling with SWIN transformer in its UNet.
  - `SR2-(128-256)(NOSWIN).ipynb`: SR DDPM for 128x128 to 256x256 upscaling without SWIN transformer in its UNet.
  - `Pipelinemain_MRNet.ipynb`: Main pipeline notebook for loading saved models and generating images.

## **Training Information**
All models were trained using an NVIDIA A100 GPU with 40 GB VRAM. 

## **Examples**
### **Generated Images Comparison**
![Generated Image](images/generated_image.png)

## **References**
- **DDPM Implementation**: Adapted from [DDPM by Jonathan Ho](https://github.com/hojonathanho/diffusion/tree/master).
- **Swin Transformer**: Refer to the paper [Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030).

## **Abbreviations**
- **SR**: Super-resolution
- **DDPM**: Denoising Diffusion Probabilistic Model
- **NOSWIN**: Without SWIN transformer
