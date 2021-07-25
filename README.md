# deeplearning
This repo contains tutorials and code for getting into the world of deeplearning.

# Setting up the Deep Learning environment
How to setup up a deeplearning environment for a `GPU` enabled PC/Laptop on `windows`

### 1. Uninstall Previous Setup
Uninstall any development setup installed earlier using the `Control Panel` add and remove option `NVIDIA CUDA Development <7/8/9/10/11>`
### 2. Install Visual Studio Code 
Download and install  the `Visual Studio Code` Community Edition with added workloads `C++ development for desktop` using the link [here](https://visualstudio.microsoft.com/downloads/?utm_medium=microsoft&utm_source=docs.microsoft.com&utm_campaign=button+cta&utm_content=download+vs2017)   
### 3. Install CUDA Tookit
This step requires some verification before installation. you can have a look and find out the Your GPU Compute Capability [here](https://developer.nvidia.com/cuda-gpus). After verifying you can download the a specific version. `CUDA Toolkit 10.1` can be downloaded from [here](https://developer.nvidia.com/cuda-10.1-download-archive-update2) 
### 4. Download cuDNN
You need to signup for NVIDIA Developer account to download this from [here](https://developer.nvidia.com/cudnn)
For cuda 10.1 `cudnn-10.1-windows10-x64-v7.6.5.32` version of cuDNN is one of the version that is compatible. Once downloaded unzip it and you will find 3 folders:
- bin
- lib
- include

Now copy the contents of 
1. `cudnn-10.1-windows10-x64-v7.6.5.32\cuda\bin\*` --> `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin`
2. `cudnn-10.1-windows10-x64-v7.6.5.32\cuda\lib\x64*` --> `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib\x64`
3. `cudnn-10.1-windows10-x64-v7.6.5.32\cuda\include\*` --> `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include`

### 4. Install Anaconda 
Indtall anaconda using this [link](https://www.anaconda.com/download/) then create an python environment and install [tensorflow]( https://www.tensorflow.org/install/gpu) using the below mentioned steps:
```
conda create -n tensorflow python=3.8

activate tensorflow

python -m pip install --ignore-installed --upgrade tensorflow-gpu
```
install pytorch:
```
conda create -n pytorch python=3.8

activate pytorch

pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

### 5. Test Installation
`Tensorflow` <br>
Once the installation is successfully completed. Try to test the installation by running the commands mentioned below:
```
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

```
output
```
tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudart64_101.dll

Num GPUs Available:  1
```
`pytorch` <br>
To test pytorch installation
```
import torch

device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")

print('Using device:', device)
```
output
```
Using device: cuda
```

### 6. Troubleshoot:
If you see error in opening  `cudart64_101.dll` Download it from [here](https://www.dll-files.com/download/1d7955354884a9058e89bb8ea34415c9/cudart64_101.dll.html?c=eVJsNVBIa1hvenBxV004Vkl4eDd3dz09)
and copy it to `C:\Windows\System32` this will require Admin Rights.