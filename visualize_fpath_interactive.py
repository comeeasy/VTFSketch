import sys
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt



def get_paths_from_cli_args():
    fpath_path, target_path = sys.argv[1], sys.argv[2]
    print(f"[INFO] {fpath_path=}, {target_path=}")
    return fpath_path, target_path

def load_fpath(fpath_path):
    fpath = np.load(fpath_path)["data"]
    print(f"[INFO] Load fpath, {fpath.shape=}")
    return fpath

def load_target_imgs(target_path):
    target_img = cv2.imread(target_path)
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    print(f"[INFO] Load target img as a grayscale, {target_img.shape=}")

def get_infodraw_from_fpath(fpath):
    # fpath[x, y, 10] is infodraw value
    
    W, H, _, _ = fpath.shape
    infodraw = np.zeros((W, H))
    infodraw = fpath[:, :, 10, 0]
    
    print(f"[INFO] extract infodraw from fpath, {infodraw.shape=}")
    
    return infodraw
    
# Function to handle mouse events
def handle_mouse_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Left button clicked at position ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN:
        print(f"Right button clicked at position ({x}, {y})")    

def display(infodraw, target_img):
    
    cv2.namedWindow('infodraw')
    cv2.namedWindow('target_img')
    
    cv2.setMouseCallback('infodraw', handle_mouse_event)
    cv2.setMouseCallback('target_img', handle_mouse_event)
    
    while True:
        cv2.imshow('infodraw', infodraw)
        cv2.imshow('target_img', target_img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            print("ESC pressed, exiting...")
            break
        elif key == ord('c'):
            print("'c' key pressed, exiting...")
            break
    
def main():
    
    fpath_path, target_path = get_paths_from_cli_args()

    fpath = load_fpath(fpath_path)
    target_img = load_target_imgs(target_path)
    infodraw = get_infodraw_from_fpath(fpath)

    display(infodraw, target_img)
    



if __name__ == "__main__":
    main()
    
    