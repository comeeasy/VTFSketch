import os
import cv2
import argparse
from datetime import datetime
import torch
from glob import glob

from src.models import FPathPredictor, UNetFPathPredictor
from src.utils import inference
from src.preprocesses import VTFPreprocessor, VTFPreprocessorUNet, ImagePreprocessor

def main(args):
    vtf_paths = glob(os.path.join(args.vtf_dir, '*'))
    vtf_paths.sort()
    vtfs = [VTFPreprocessorUNet.get(vtf_path) for vtf_path in vtf_paths]

    img_paths = glob(os.path.join(args.img_dir, "*"))
    img_paths.sort()
    imgs      = [ImagePreprocessor.get(img_path) for img_path in img_paths]

    # model = FPathPredictor()
    model = UNetFPathPredictor()
    model.load_state_dict(torch.load(args.weight_path))

    model = model.to("cuda")
    model.eval()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f"results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    for vtf, img, img_path in zip(vtfs, imgs, img_paths):
        vtf = torch.tensor(vtf).to('cuda')
        img = torch.tensor(img).to('cuda')
        pred = inference(model, vtf.unsqueeze(0), img.unsqueeze(0))

        result = pred.squeeze().detach().cpu().numpy()
        result = result.transpose((1, 0))

        output_path = os.path.join(output_dir, f"{os.path.basename(img_path).split('.')[0]}.png")
        cv2.imwrite(output_path, result * 255)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on VTFs and save results.")
    parser.add_argument('--vtf_dir', type=str, required=True, help='Directory containing VTF files.')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory containing VTF files.')
    parser.add_argument('--weight_path', type=str, required=True, help='Path to the trained model weights.')
    parser.add_argument('--output_dir', type=str, default="results", help='Directory to save the output images.')
    
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    main(args)
