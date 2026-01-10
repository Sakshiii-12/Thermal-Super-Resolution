## Overview

Autonomous systems such as self driving vehicles, mobile robots, and aerial platforms must operate reliably in environments with poor visibility. These include night time operation, fog, smoke, rain, glare, and low illumination scenarios. In such conditions, thermal infrared cameras are essential because they capture emitted heat rather than reflected visible light.v
Despite their robustness to lighting conditions, thermal cameras present a major deployment challenge. Low cost thermal sensors provide very low spatial resolution, resulting in blurry images with missing structural details. High resolution thermal cameras exist, but they are expensive, power intensive, and unsuitable for large scale deployment.v
This project focuses on improving thermal perception quality through software based enhancement, allowing low cost thermal sensors to achieve significantly better perceptual quality without requiring hardware upgrades.


## Problem Statement

Low resolution thermal images lack the spatial detail required for reliable perception. Important information such as object boundaries, human limb positions, and obstacle contours is degraded or lost. This negatively impacts downstream perception tasks including detection, tracking, and navigation.vRGB cameras provide high resolution spatial and geometric information at low cost. However, RGB sensors are unreliable in low light and adverse weather conditions and cannot be trusted as primary sensors in safety critical systems. Most existing multispectral perception pipelines fuse RGB and thermal information only at late stages, such as object detection. This approach does not address the fundamental limitation of poor thermal image quality at the sensor level. The core problem addressed in this project is how to enhance thermal perception itself without relying on expensive thermal hardware and without compromising safety when RGB information becomes unreliable.


## Our Solution

We propose a perception level enhancement approach that improves the quality of thermal images before they are used by downstream perception modules. The key idea is to use RGB images only as a source of structural guidance. Geometric cues such as edges and object boundaries are extracted from RGB images and used to guide thermal super resolution. RGB data is never used to estimate temperature values. A lightweight convolutional neural network takes a low resolution thermal image and an RGB derived edge map as input and produces an enhanced thermal image that preserves true thermal characteristics while improving spatial clarity. This design improves thermal image quality, reduces reliance on expensive sensors, and maintains safety through strict separation of sensing modalities.


## Dataset

The project uses the LLVIP dataset, which stands for Low Light Visible and Infrared Paired Dataset. The dataset contains paired RGB and thermal infrared images captured under low light and night time conditions. The RGB and thermal images are spatially aligned, making the dataset suitable for learning cross modal structural relationships.

Dataset reference:
LLVIP Dataset  
Project page: https://bupt-ai-cz.github.io/LLVIP/  
GitHub repository: https://github.com/bupt-ai-cz/LLVIP  

For efficient experimentation, a subset of the dataset is used in this project. The dataset itself is not included in the repository and must be downloaded separately.


## Methodology

The proposed pipeline consists of the following stages.

### Thermal Degradation Modeling

High resolution thermal images are downsampled and then upsampled to simulate the output of low cost thermal sensors. This produces realistic low resolution thermal inputs while retaining accurate high resolution thermal images as ground truth during training.

### Structural Guidance Extraction

RGB images are converted to grayscale and processed using edge detection techniques. The resulting edge maps encode scene geometry and object boundaries while remaining independent of color, illumination, and temperature information.

### RGB Guided Thermal Super Resolution

A lightweight convolutional neural network receives the concatenation of the low resolution thermal image and the RGB edge map as input. The network is trained to predict a residual thermal image that refines the low resolution input. The enhanced thermal output is obtained by adding the predicted residual to the original low resolution thermal image. This residual learning formulation improves training stability and reduces the risk of hallucinated thermal patterns.


## Safety and Robustness

Safety is a primary design consideration in this project. RGB information is never used to infer thermal values. It is used only to provide spatial guidance. In scenarios where RGB input becomes unreliable or unavailable, the system can operate in a thermal only mode by disabling RGB guidance. This ensures graceful degradation and avoids failure modes that could compromise safety in autonomous systems.


## Project Structure

```

thermal_super_resolution/
│
├── data/
│   ├── raw/LLVIP/          Original dataset location
│   ├── processed/         Training ready data
│   │   ├── rgb/
│   │   ├── rgb_edges/
│   │   ├── thermal_lr/
│   │   └── thermal_hr/
│
├── models/
│   ├── backbone/           Thermal super resolution network
│   ├── fusion/             Extensions for advanced fusion
│   └── losses/             Custom loss functions
│
├── training/
│   ├── train.py            Model training
│   └── validate.py         Validation utilities
│
├── inference/
│   ├── demo.py             Visualization demo
│   ├── realtime.py         Real time inference
│   └── benchmark.py        Performance evaluation
│
├── utils/
│   ├── prepare_llvip.py
│   ├── create_thermal_lr.py
│   └── extract_rgb_edges.py
│
├── deployment/
│   └── onnx_export.py
│
├── requirements.txt
└── README.md

````


## How to Use

### Installation

Install the required dependencies using:

```bash
pip install -r requirements.txt
````

A CUDA enabled PyTorch installation is recommended for faster training.

### Data Preparation

Download the LLVIP dataset and place it under:

```
data/raw/LLVIP/
```

Prepare the processed training data using the utility scripts provided in the `utils` directory.

### Training

To train the model, run:

```bash
python training/train.py
```

The model learns to enhance low resolution thermal images using structural guidance from RGB edges.

### Demo

To visualize the results, run:

```bash
python inference/demo.py
```

This displays a side by side comparison of the low resolution thermal input, the enhanced thermal output, and the high resolution thermal ground truth.


## Output Interpretation

The output visualization consists of three images. The first image is the low resolution thermal input, representing the output of a low cost thermal sensor. The second image is the enhanced thermal output produced by the proposed model. The third image is the high resolution thermal ground truth used during training. 
A successful enhancement is indicated when the enhanced thermal image exhibits sharper object boundaries and clearer silhouettes while preserving realistic thermal intensity patterns.

Just tell me.
```
