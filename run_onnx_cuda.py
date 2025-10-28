# Modified by Shangchen Zhou from: https://github.com/TencentARC/GFPGAN/blob/master/inference_gfpgan.py
import os
import cv2
import argparse

import torch
import onnxruntime as ort

from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
import torch.nn.functional as F

import numpy as np
from time import perf_counter

pretrain_model_url = {
    'lednet': 'https://github.com/sczhou/LEDNet/releases/download/v0.1.0/lednet.pth',
    'lednet_retrain': 'https://github.com/sczhou/LEDNet/releases/download/v0.1.0/lednet_retrain_500000.pth',
    'lednetgan': 'https://github.com/sczhou/LEDNet/releases/download/v0.1.0/lednetgan.pth',
}

def free_model(model):
    del model  # Delete the model object
    torch.cuda.empty_cache()  # Clear GPU memory (if using CUDA)
    torch.cuda.ipc_collect()  # Collect garbage for shared tensors (optional)


def np_normalize(img_array, mean, std, inplace=True):
    """
    NumPy equivalent of torchvision.transforms.functional.normalize.
    
    Parameters:
        img_array (numpy.ndarray): Image array with shape (C, H, W) or (H, W, C).
        mean (tuple): Mean values for each channel.
        std (tuple): Standard deviation values for each channel.
        inplace (bool): Whether to perform the operation in-place.

    Returns:
        numpy.ndarray: Normalized image array.
    """
    if not inplace:
        img_array = img_array.copy()  # Avoid modifying the input array directly

    # Normalize each channel
    for c in range(img_array.shape[0]):  # Assuming shape (C, H, W)
        img_array[c] = (img_array[c] - mean[c]) / std[c]
    
    return img_array

def list_image_files(folder_path, extensions=['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']):
    """
    Lists all image files in the specified folder with given extensions.
    
    Parameters:
        folder_path (str): Path to the folder to search.
        extensions (list): List of valid image file extensions.
        
    Returns:
        List of image file paths.
    """
    image_files = [
        os.path.join(folder_path, f) 
        for f in os.listdir(folder_path) 
        if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(tuple(extensions))
    ]
    return image_files

def check_image_size(x, down_factor):
    _, _, h, w = x.size()
    mod_pad_h = (down_factor - h % down_factor) % down_factor
    mod_pad_w = (down_factor - w % down_factor) % down_factor
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def load_img_to_tensor(filename, use_float32, down_factor):

    img = cv2.imread(filename, cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError(f"Image not found at path: {filename}")

    img_t = img2tensor(img / 255., bgr2rgb=True, float32=use_float32)

    # # without [-1,1] normalization in lednet model (paper version) 
    normalize(img_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    
    img_t = img_t.unsqueeze(0).to(device=device)

    #pad to multiple of 8
    H, W = img_t.shape[2:]
    img_t = check_image_size(img_t, down_factor)

    return img_t
    
def run_onnx_session_batch_test(onnx_model_file, input_img_folder, output_img_folder, use_float32, device):

    batch_size = 2

    providers = ['CUDAExecutionProvider']

    print("cuDNN version:", torch.backends.cudnn.version())

    down_factor = 8 # check_image_size

    image_file_list = list_image_files(input_img_folder)
    ort_session = ort.InferenceSession(onnx_model_file, providers=providers)

    if not os.path.exists(output_img_folder):
        os.makedirs(output_img_folder)

    print("Starting Onnx Inference")
            
    img_tensor_1 = load_img_to_tensor(image_file_list[0], use_float32=use_float32, down_factor=down_factor)
    img_tensor_2 = load_img_to_tensor(image_file_list[1], use_float32=use_float32, down_factor=down_factor)

    #concat this two image. 
    img_t = torch.cat((img_tensor_1, img_tensor_2), dim=0)
    H, W = img_t.shape[2:]

    
    
    if use_float32:
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_t)}
    else:
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_t).astype(np.float16)}
    
    #print("Inference Input shape:", img_t.shape)
    #print("Running Inference on Image: ", os.path.basename(image_file_list[0]), os.path.basename(image_file_list[1]))

    start_time = perf_counter()

    outputs = ort_session.run(
        None,
        ort_inputs
    )

    end_time = perf_counter()
    elapsed = end_time - start_time 
    onnx_output_tensor = torch.from_numpy(outputs[0])

    for i in range(0, 2):
        output_t = onnx_output_tensor[i,:,:H,:W]
        output = tensor2img(output_t, rgb2bgr=True, min_max=(-1, 1))
        output = output.astype('uint8')

        print("Completed Image: ", os.path.basename(image_file_list[i]))
        print("Completed Inference in {0} seconds".format(elapsed.total_seconds()))

        imwrite(output, os.path.join(output_img_folder, os.path.basename(image_file_list[i])))

    del ort_session


def run_onnx_session_test(onnx_model_file, input_img_folder, output_img_folder, use_float32, device):

    providers = ['CUDAExecutionProvider']

    print("cuDNN version:", torch.backends.cudnn.version())

    down_factor = 8 # check_image_size

    image_file_list = list_image_files(input_img_folder)

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    ort_session = ort.InferenceSession(onnx_model_file, session_options, providers=providers)

    if not os.path.exists(output_img_folder):
        os.makedirs(output_img_folder)

    print("Starting Onnx Inference")
    total_time_taken_seconds = 0.0

    for index, image_file in enumerate(image_file_list):
        
        img = cv2.imread(image_file, cv2.IMREAD_COLOR)

        if img is None:
            raise FileNotFoundError(f"Image not found at path: {image_file}")

        img_t = img2tensor(img / 255., bgr2rgb=True, float32=use_float32)

        # # without [-1,1] normalization in lednet model (paper version) 
        normalize(img_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        
        img_t = img_t.unsqueeze(0).to(device=device)

        #pad to multiple of 8
        H, W = img_t.shape[2:]
        img_t = check_image_size(img_t, down_factor)
        
        print('img_shape', img_t.shape)

        if use_float32:
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_t)}
        else:
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_t).astype(np.float16)}
        print("Running Inference on Image: ", os.path.basename(image_file))

        start_time = perf_counter()
        outputs = ort_session.run(
            None,
            ort_inputs
        )
        end_time = perf_counter()

        elapsed = end_time - start_time

        if index != 0:
            total_time_taken_seconds += elapsed

        onnx_output_tensor = torch.from_numpy(outputs[0])
        output_t = onnx_output_tensor[:,:,:H,:W]

        output = tensor2img(output_t, rgb2bgr=True, min_max=(-1, 1))
        output = output.astype('uint8')

        print("Completed Image: ", os.path.basename(image_file))
        print("Completed Inference in {0} seconds".format(elapsed))

        imwrite(output, os.path.join(output_img_folder, os.path.basename(image_file)))

    total_time_taken_seconds /= (len(image_file_list) - 1)
    print("[Info] Completed Inference - Average Inference Time: {0} seconds".format(total_time_taken_seconds))

    del ort_session

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_path', type=str, default='./inputs')
    parser.add_argument('--result_path', type=str, default='./results')
    parser.add_argument('--onnx_path', type=str, default='./compiled/lednet_retrain.onnx')

    args = parser.parse_args()

    print("Testing Onnx on: ", device)
    print("Available providers:", ort.get_available_providers())

    run_onnx_session_test(args.onnx_path, args.test_path, args.result_path, use_float32 = True, device=device)


#export LD_LIBRARY_PATH=/home/ubuntu/Projects/temp_projects/LEDNet/venv/lib/python3.10/site-packages/tensorrt_libs:$LD_LIBRARY_PATH
#python run_onnx_cuda.py --onnx_path './compiled/lednet_retrain.onnx' --test_path '/home/ubuntu/Data/Lowlight/LOL/test/low_blur/0052' --result_path './pytorch/ppm2_onnx_fp32'

# ORT CUDA fp32 - 0.373 seconds
# ORT CUDA fp16 - 