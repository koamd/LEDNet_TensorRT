## LEDNet: Joint Low-light Enhancement and Deblurring in the Dark (ECCV 2022) - ONNXRuntime / TensorRT Implementation

Original Paper:
[Paper](https://arxiv.org/abs/2202.03373) | [Project Page](https://shangchenzhou.com/projects/LEDNet/) | [Video](https://youtu.be/450dkE-fOMY) | [Replicate Demo](https://replicate.com/sczhou/lednet)

[Shangchen Zhou](https://shangchenzhou.com/), [Chongyi Li](https://li-chongyi.github.io/), [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/)

S-Lab, Nanyang Technological University

### 1. Dependencies and Installation

Tested On:
- Pytorch >= 2.6.0
- CUDA >= 12.4
- Other required packages in `requirements.txt`

#### Installing using python-venv

```
python3.10 -m venv venv
. venv/bin/activate

pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. Source Code Modifications From Original Repository

#### Torchvision Package Functions 
Some minor modifications have been made to adapt to the latest torchvision package in [degradations.py](./basicsr/data/degradations.py). This only affects the rgb_to_grayscale function. 

```python
torchvision.transforms.functional_tensor
```

converted to 

```python
from torchvision.transforms import functional as F
```


#### Model Conversion for ONNX support

The PPM module [lednet_arch.py](./basicsr/archs/lednet_arch.py) has been modified for ONNX support :

```python
class AdaptiveAvgPool2d_ONNX(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = tuple(output_size)

    def forward(self, x):
        # Step 1: Global average pool to 1x1
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # Step 2: Upsample to target size
        return F.upsample(x, size=self.output_size, mode='nearest')

class PPM2(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM2, self).__init__()
        self.features = []
        for bin in bins: #[1,2,3,6]
            self.features.append(nn.Sequential(
                AdaptiveAvgPool2d_ONNX(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.PReLU()
            ))
        self.features = nn.ModuleList(self.features)
        self.fuse = nn.Sequential(
                nn.Conv2d(in_dim+reduction_dim*4, in_dim, kernel_size=3, padding=1, bias=False),
                nn.PReLU())

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        out_feat = self.fuse(torch.cat(out, 1))
        return out_feat
```

The blur kernel module [lednet_submodules.py](./basicsr/archs/lednet_submodules.py) has also been modified. 

```python
class KernelConv2D_V2(nn.Module):
    def __init__(self, ksize=5, act=True):
        super(KernelConv2D_V2, self).__init__()
        self.ksize = ksize
        self.act = act

    def forward(self, feat_in, kernel):
        channels = feat_in.size(1)
        N, kernels, H, W = kernel.size()
        pad = (self.ksize - 1) // 2

        feat_in = F.pad(feat_in, (pad, pad, pad, pad), mode="replicate")
        
        #modified unfold layer to onnx compatible functions
        feat_in = onnx_unfold_alternative_height(feat_in, self.ksize, 1)
        feat_in = onnx_unfold_alternative_width(feat_in, self.ksize, 1)

        feat_in = feat_in.permute(0, 2, 3, 1, 4, 5).contiguous()
        feat_in = feat_in.reshape(N, H, W, channels, -1)

        kernel = kernel.permute(0, 2, 3, 1).reshape(N, H, W, channels, -1)
        feat_out = torch.sum(feat_in * kernel, -1)
        feat_out = feat_out.permute(0, 3, 1, 2).contiguous()
        
        if self.act: 
            feat_out = F.leaky_relu(feat_out, negative_slope=0.2, inplace=True)
        return feat_out
```


#### Initializing Model with new PPM module. 

To use the ONNX compatible modules, set ppm_version = 2
```python
net = ARCH_REGISTRY.get('LEDNet')(channels=[32, 64, 128, 128], connection=False, ppm_version = 2).to(device)
```

### 3. Train the Model

Training is not necessary as we only modified the non-trainable layers. However if you wish to train the model, refer to the [LEDNet](https://github.com/sczhou/LEDNet)

### 4. Quick Inference

- Download the LEDNet pretrained model from [[Release V0.1.0](https://github.com/sczhou/LEDNet/releases/tag/v0.1.0)] to the `weights` folder. You can manually download the pretrained models OR download by runing the following command.
  
  > python scripts/download_pretrained_models.py LEDNet

Note that we are only using lednet_retrain model for this current project. 

For this project, we are using the LOL test dataset whose images are 640x1120. 

```bash
# test original model
python inference_lednet.py --model lednet_retrain --test_path '/home/ubuntu/Data/Lowlight/LOL/test/low_blur/0052' --result_path './output/uncompressed_ppm1/' --ppm 1 

# test onnx compatible model
python inference_lednet.py --model lednet_retrain --test_path '/home/ubuntu/Data/Lowlight/LOL/test/low_blur/0052' --result_path './output/uncompressed_ppm2/' --ppm 2

```

The results will be saved in the `results` folder.

### 5. Model Export to ONNXRuntime

The export has been tested on the following onnx packages:
onnx==1.17
onnxruntime-gpu==1.22

```bash
pip install onnx==1.17 onnxruntime-gpu==1.22
```

Run the following code to export the model to ONNXRuntime. The file will be stored in ./compiled folder. 

```bash
python export_lednet_onnx.py
```

For Inference, run the following:

```bash
python run_onnx_cuda.py --onnx_path './compiled/lednet_retrain.onnx' --test_path '/home/ubuntu/Data/Lowlight/LOL/test/low_blur/0052' --result_path './output/ppm2_onnx_fp32'
```

### 6. Model Export to TensorRT

Install the following packages:
```bash
pip install tensorrt-cu12
pip install polygraphy
```

Create TensorRT Graph and run inference. Do note that the engine graph shape_profiles must be customized for your own range of input data shapes. LEDNet requires the input image width and height to be divisible by 8. Though the recommended input image size is 256x256 and above.    

Refer to [run_trt.py](run_trt.py)

```python
try:
        engine, engine_path = onnx2trt(
            model_path=onnx_path,
            int8=False,
            fp16=use_fp16,  # feel free to experiment. Model did not seem to be trained with FP16 but the INT8s may compensate.
            bf16=False, # feel free to try this also. 
            optimization_level=3,
            shape_profiles=[
                {
                    input_names[0]: {
                        "min": [1, 3, 640, 1120],
                        "opt": [1, 3, 640, 1120],
                        "max": [1, 3, 640, 1120]
                    }
                }
            ]
        )
```


```bash
python run_trt.py --inputs '/home/ubuntu/Data/Lowlight/LOL/test/low_blur/0052' --model './compiled/lednet_retrain.onnx' --outputs "./output/trt_fp32_optim3"

python run_trt.py --inputs '/home/ubuntu/Data/Lowlight/LOL/test/low_blur/0052' --model './compiled/lednet_retrain.onnx' --outputs "./output/trt_fp16_optim3" --use_fp16 
```

### 7. Inference Speed comparisons (Nvidia A10 GPU)

Currently all benchmarks are based on a small sample test of 60 images from [LOL](https://drive.google.com/drive/folders/11HcsiHNvM7JUlbuHIniREdQ2peDUhtwX) dataset low_blur/0052, average SSIM score was 0.995. 


| Implementation| Framework | Inference time (secs) 640x1120 | SSIM | 
|---|---|---|---|
| Original Model  | Pytorch | 0.127 | - | 
| PPM2     | Pytorch | 0.141 | 0.995 |   
| PPM2     | ONNX-CUDA FP32 | 0.379 | 0.995 |  
| PPM2     | TensorRT FP32 Opt3 | 0.087 | 0.9942 |
| PPM2     | TensorRT FP16 Opt3 | 0.028 | 0.9937 |

SSIM Scores are using Original Model output as Reference. 

### 8. Verification using SSIM between Original Model and our ONNX supported model, TensorRT models

```
python ssim_verify.py --img1_folder './output/uncompressed_ppm1/0052' --img2_folder './output/uncompressed_ppm2/0052'
python ssim_verify.py --img1_folder './output/uncompressed_ppm1/0052' --img2_folder './output/ppm2_onnx_fp32/'
python ssim_verify.py --img1_folder './output/uncompressed_ppm1/0052' --img2_folder './output/trt_fp16_optim3/'
python ssim_verify.py --img1_folder './output/uncompressed_ppm1/0052' --img2_folder './output/trt_fp32_optim3/'
```

### License

This project is licensed under <a rel="license" href="https://github.com/sczhou/LEDNet/blob/master/LICENSE">S-Lab License 1.0</a>. Redistribution and use for non-commercial purposes should follow this license.

### Acknowledgements

From the Original Authors:
LEDNet: Joint Low-light Enhancement and Deblurring in the Dark (ECCV 2022)

[Paper](https://arxiv.org/abs/2202.03373) | [Project Page](https://shangchenzhou.com/projects/LEDNet/) | [Video](https://youtu.be/450dkE-fOMY) | [Replicate Demo](https://replicate.com/sczhou/lednet)

[Shangchen Zhou](https://shangchenzhou.com/), [Chongyi Li](https://li-chongyi.github.io/), [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/)

S-Lab, Nanyang Technological University

### Citation

If this work is useful for your research, please consider citing the original authors:

```bibtex
@InProceedings{zhou2022lednet,
    author = {Zhou, Shangchen and Li, Chongyi and Loy, Chen Change},
    title = {LEDNet: Joint Low-light Enhancement and Deblurring in the Dark},
    booktitle = {ECCV},
    year = {2022}
}
```
