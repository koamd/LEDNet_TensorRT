# Modified by Shangchen Zhou from: https://github.com/TencentARC/GFPGAN/blob/master/inference_gfpgan.py

import os
import argparse
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.utils.download_util import load_file_from_url
import torch
import onnx

pretrain_model_url = {
    'lednet': 'https://github.com/sczhou/LEDNet/releases/download/v0.1.0/lednet.pth',
    'lednet_retrain': 'https://github.com/sczhou/LEDNet/releases/download/v0.1.0/lednet_retrain_500000.pth',
    'lednetgan': 'https://github.com/sczhou/LEDNet/releases/download/v0.1.0/lednetgan.pth',
}

def free_model(model):
    del model  # Delete the model object
    torch.cuda.empty_cache()  # Clear GPU memory (if using CUDA)
    torch.cuda.ipc_collect()  # Collect garbage for shared tensors (optional)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    args.model = 'lednet_retrain'
    
    #create output folder
    output_onnx_folder = "./compiled"
    if not os.path.exists(output_onnx_folder):
        os.makedirs(output_onnx_folder)

    dummy_input = torch.randn(1, 3, 640, 1120).to(device=device)  # Adjust dimensions as necessary
    onnx_path = os.path.join(output_onnx_folder,'lednet_retrain.onnx')

    # ------------------ set up LEDNet network -------------------
    down_factor = 8 # check_image_size
    ckpt_path = load_file_from_url(url=pretrain_model_url[args.model], 
                                    model_dir='./weights', progress=True, file_name=None)
    checkpoint = torch.load(ckpt_path)['params']
    net = ARCH_REGISTRY.get('LEDNet')(channels=[32, 64, 128, 128], connection=False, ppm_version=2).to(device)    
    
    net.load_state_dict(checkpoint)
    net.eval()
    output = net(dummy_input)

    print("Completed loading and running new model")
    
    dynamic_axes_dict = {
        'input': {0: 'batch_size', 2:'img_y', 3:'img_x'},
        'output': {0: 'batch_size', 2:'img_y', 3:'img_x'}, 
    }

    torch.onnx.export(net, 
                  dummy_input, 
                  onnx_path, 
                  export_params=True, 
                  opset_version=20,  # Adjust the opset version if needed
                  do_constant_folding=True,
                  dynamic_axes = dynamic_axes_dict,
                  verbose=True,
                  input_names=['input'], 
                  output_names=['output'])

    print('dummy_input_tensor_type: ', dummy_input.dtype)  # Output: torch.float32

    onnx.checker.check_model(onnx_path, full_check=True)

    free_model(net)
    del dummy_input
    
#python export_lednet_onnx.py
