import sys
import numpy as np
from pathlib import Path
import cv2
import torch
import math
from torchvision.utils import make_grid
from polygraphy.backend.trt import TrtRunner, EngineFromBytes  # pip install polygraphy
from onnx2trt import onnx2trt
import onnx
import os
from time import perf_counter
import argparse

BASE_DIR = Path(__file__).parent

def check_image_size_numpy(x, window_size=128):
    """
    x: numpy array with shape (N, C, H, W)
    """
    _, _, h, w = x.shape

    mod_pad_h = (window_size - h % window_size) % window_size
    mod_pad_w = (window_size - w % window_size) % window_size

    # numpy.pad takes ((before, after), ...) per dimension
    # We only pad H and W, not N and C
    x = np.pad(
        x,
        pad_width=((0, 0), (0, 0), (0, mod_pad_h), (0, mod_pad_w)),
        mode='reflect'
    )
    return x

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

def run_trt(img_path: str, onnx_path: Path, output_img_folder: str, use_fp16: bool = False):

    image_file_list = list_image_files(img_path)

    if not os.path.exists(output_img_folder):
        os.makedirs(output_img_folder)

    model = onnx.load(onnx_path, load_external_data=False)
    input_names = [x.name for x in model.graph.input]
    del model

    #Generate Engine File
    print(f"[Info] Generating Engine File: float16: {use_fp16}") 

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

    except Exception as e:
        print(e, file=sys.stderr)
        return None

    print("[Info] Generated Engine File")

    runner = TrtRunner(engine, name="LEDNet", allocation_strategy="static")

    print("[Info] Started Runner")

    #normalization array. 
    mean = np.array([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1).astype(np.float32)
    std = np.array([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1).astype(np.float32)

    with runner:
        total_time_taken_seconds = 0.0
        for index, sample_img_path in enumerate(image_file_list):

            img = cv2.imread(sample_img_path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Image not found at path: {sample_img_path}")

            h, w, c = img.shape
            
            #apply normalization
            img = img[None, :, :, ::-1]  # BGR -> RGB and unsqueeze
            img = img.astype(np.float32) / 255.0
            
            img = img.transpose(0, 3, 1, 2)
            img_pt_resized = check_image_size_numpy(img, 8) #pad to a multiple of 8
            x_normalized = (img_pt_resized - mean) / std
           
            start_time = perf_counter()
            
            outputs: dict = runner.infer({input_names[0]: x_normalized}, check_inputs=False, copy_outputs_to_host=False)
            
            end_time = perf_counter()
            
            if index != 0:
                total_time_taken_seconds += (end_time - start_time)

            #print(list(outputs.keys()))
            # final = list(outputs.values())[0]
            final = outputs['output'].numpy()
            
            #convert final to image format
            output = final[:, :, :h, :w].copy()  #remove padding
            output = output.squeeze(0).transpose(1, 2, 0) # remove batch and swap to h,w,c
            output_normalized = (output + 1) / 2.0 #normalize to 0 to 1
            output = np.clip(output_normalized, 0, 1) #clamped between 0 to 1. 
            output = (output * 255).astype(np.uint8)  # Convert to uint8
            output_final = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

            saved_path = os.path.join(output_img_folder, f"{os.path.basename(sample_img_path).split('.')[0]}.png")
            cv2.imwrite(saved_path, output_final)

            print(f"Completed Image: {os.path.basename(saved_path)} in {end_time - start_time} seconds ", )

            del final
            del output
            del output_final

    total_time_taken_seconds /= (len(image_file_list) - 1)
    print("[Info] Completed Inference - Average Inference Time: {0} seconds".format(total_time_taken_seconds))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default="./inputs", help="input image folder")
    parser.add_argument("--outputs", type=Path, default="./output_images", help="output image folder")
    parser.add_argument("--model", type=Path, default="lednet_retrain.onnx", help="input onnx model")
    parser.add_argument("--use_fp16", action='store_true', help="use Float16 for tensorRT")

    args = parser.parse_args()

    run_trt(img_path=args.inputs, onnx_path=args.model, output_img_folder=args.outputs, use_fp16=args.use_fp16)
    
    

if __name__ == '__main__':
    main()

#python run_trt.py --inputs '/home/ubuntu/Data/Lowlight/LOL/test/low_blur/0052' --model './compiled/lednet_retrain.onnx' --outputs "./output/trt_fp32_optim3"
#python run_trt.py --inputs '/home/ubuntu/Data/Lowlight/LOL/test/low_blur/0052' --model './compiled/lednet_retrain.onnx' --outputs "./output/trt_fp16_optim3" --use_fp16 

