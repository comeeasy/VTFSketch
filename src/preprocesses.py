import numpy as np
from PIL import Image

import torch
from torchvision.transforms import transforms


class TargetPreprocessor:

    target_transforms = transforms.Compose([
        transforms.ToTensor(),    
    ])

    @staticmethod
    def get(target_path):
        target_img = Image.open(target_path).convert("L")
        target = TargetPreprocessor.__preprocess(target_img)
        return target

    @staticmethod
    def __preprocess(target_img):
        target_img = TargetPreprocessor.target_transforms(target_img)
        
        target = torch.ones_like(target_img)
        target[torch.where(target_img < 0.5)] = 0 # 흐릿한 부분 없이 1 or 0 만 남도록 학습
        return target # [C x H x W]

class VTFPreprocessor:

    @staticmethod
    def get(vtf_path):
        vtf = np.load(vtf_path)['data']
        vtf = VTFPreprocessor.__preprocess(vtf)
        return vtf

    @staticmethod
    def __preprocess(vtf):
        # 픽셀중에 0.1% 정도 1 값을 넘는 값들이 있어 normalize 시 회색 이미지로 그려지는
        # 문제가 있어서 이를 해결
        # vtf[np.where(vtf > 1.0)] = 1
        vtf = vtf.transpose((2, 1, 0)) # [W x H x C] -> [C x H x W]
        return np.clip(vtf, 0.0, 1.0)
    
class ImagePreprocessor:
    
    img_transforms = transforms.Compose([
        transforms.ToTensor(),    
    ])

    
    @staticmethod
    def get(img_path):
        img = Image.open(img_path).convert('RGB')
        img = ImagePreprocessor.__preprocess(img)
        return img

    @staticmethod
    def __preprocess(img):
        img = ImagePreprocessor.img_transforms(img=img) # [C x H x W]
        return img

class InfodrawPreprocessor:
    
    infodraw_transforms = transforms.Compose([
        transforms.ToTensor(),    
    ])
    
    @staticmethod
    def get(infodraw_path):
        infodraw = Image.open(infodraw_path).convert("L")
        infodraw = InfodrawPreprocessor.__preprocess(infodraw=infodraw)
        return infodraw
    
    @staticmethod
    def __preprocess(infodraw):
        infodraw = InfodrawPreprocessor.infodraw_transforms(infodraw)
        return infodraw
    
