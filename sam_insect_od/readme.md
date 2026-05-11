## This project is developed using Python3.11
Newer versions will probably work but to minimize problems downgrade to this version!
## Download dataset from Yoda using iBridges
```
pip install ibridges
```
```
mkdir -p ~/.irods
```
Paste the config from https://geo.yoda.uu.nl/user/data_transfer to ~/.irods/irods_environment.json  

Paste password generated from here https://geo.yoda.uu.nl/user/data_access after running command below
```
ibridges init
```
To check if it went successfully
```
ibridges pwd
```
And download a folder for example:
```
ibridges download "irods:~/research-insect-recognizer/T5M7_AT2" .
```

# Installation

### Using locale python environment (for development)
```
python3.11 -m venv .venv
```
```
source .venv/bin/activate
```
```
pip install -r requirements.txt
```
```
pip install torchvision
```
Paste huggingface access token with Write permission after next command:
```
sudo apt install python3-huggingface-hub
```
```
hf auth login
```

### Using docker (stable but slower since it doesn't use GPU power)
```commandline
docker compose up -d
```
Paste huggingface access token with Write permission after next command:
```commandline
docker compose run insect-detector hf auth login
```
Create a folder in projects root directory with photos you want to proces.
```commandline
docker compose run insect-detector python sam3.py --image-folder ./photos --vis-folder ./results --save-crops ./crops
```

## Available commands
Visualize
```
python3 sam3.py --image test_image.jpg --visualize --csv results.csv
```
Crop and visualize
```
python3 sam3.py --image-folder ./photos --vis-folder ./results --save-crops ./crops --csv results.csv
```

# Troubleshooting
You may see an error like this:
```
torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
Search for `cudaErrorNoKernelImageForDevice' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.                                                                                                                                                                       
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.                                                                                                                                                                                             
For debugging consider passing CUDA_LAUNCH_BLOCKING=1                                                                                                                                                                                                                                                               
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions. 
```

This is likely to happen when you have an older GeForce (non-RTX) NVIDIA graphics card. The newer Torchvision has dropped support. Use a older Torchvision to fix:
```
pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```