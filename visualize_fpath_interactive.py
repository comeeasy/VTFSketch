import cv2
import torch

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm import tqdm

from train2 import FCModel, CNNModel, ResCNN1DLayer, FPathDataset
from src.dataloaders import FPathLazyDataset
from src.preprocesses import VTFPreprocessor, ImagePreprocessor, InfodrawPreprocessor, TargetPreprocessor



# Function to handle mouse events
def handle_mouse_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Left button clicked at position ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN:
        print(f"Right button clicked at position ({x}, {y})")    

def on_click(event):
    if event.xdata is not None and event.ydata is not None:
        print(f"Clicked at x={int(event.xdata)}, y={int(event.ydata)}")


def display(result, target_img):
    
    cv2.namedWindow('result')
    cv2.namedWindow('target_img')
    
    cv2.setMouseCallback('result', handle_mouse_event)
    cv2.setMouseCallback('target_img', handle_mouse_event)
    
    while True:
        cv2.imshow('result', result)
        cv2.imshow('target_img', target_img)
        
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            print("ESC pressed, exiting...")
            break
        elif key == ord('c'):
            print("'c' key pressed, exiting...")
            break

def display_image_with_events(result, target_img):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Display the result image
    ax1 = axes[0]
    im1 = ax1.imshow(result, cmap='gray')
    ax1.set_title('Result')
    ax1.axis('off')
    fig.canvas.mpl_connect('button_press_event', on_click)

    # Display the target image
    ax2 = axes[1]
    im2 = ax2.imshow(target_img, cmap='gray')
    ax2.set_title('Target Image')
    ax2.axis('off')
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()

def main():
    
    # CNNModel
    # model = torch.load("/home/work/joono/VTFSketch/weights/best_CNNModel_acc78.31.pth")

    # FCModel
    # model = torch.load("/home/work/joono/VTFSketch/weights/best_FCModel_acc77.46.pth")

    # model = model.to('cuda')
    
    vtf_path = "/home/work/joono/VTFSketch/dataset/simple_data/test/vtfs/color_901_fpath_of_infodraw.npz"
    infodraw_path = "/home/work/joono/VTFSketch/dataset/simple_data/test/infodraws/color_901_out.png"
    target_path = "/home/work/joono/VTFSketch/dataset/simple_data/test/targets/line_901.png"

    fpath       = VTFPreprocessor.get(vtf_path)
    infodraw    = InfodrawPreprocessor.get(infodraw_path)
    target      = TargetPreprocessor.get(target_path)

    fpath_tensor = torch.tensor(fpath).unsqueeze(0).to('cuda')
    infodraw_tensor = torch.tensor(infodraw).unsqueeze(0).to('cuda')
    
    # model = model.to('cuda')
    # model = model.eval()

    _, _, H, W = infodraw_tensor.shape

    result = infodraw_tensor.clone()

    # target_indices = torch.where(infodraw_tensor < 0.99)
    # selected_fpath_tensor = fpath_tensor[target_indices[0], :, target_indices[2], target_indices[3]]
    # print(f"selected_fpath_tensor: {selected_fpath_tensor.shape}")

    # with torch.no_grad():
    #     y_hat = model(selected_fpath_tensor)

    # y_hat = y_hat.squeeze()

    # for i, (b, c, h, w) in enumerate(zip(target_indices[0], target_indices[1], target_indices[2], target_indices[3], )):
    #     result[b, c, h, w] = y_hat[i]

    # display_image_with_events(result.squeeze(), target)
    display_image_with_events(np.random.rand(1024, 1024), target.squeeze())

if __name__ == "__main__":
    main()
    
    