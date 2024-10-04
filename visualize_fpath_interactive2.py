import cv2
import torch

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm import tqdm

from train2 import FCModel, CNNModel, ResCNN1DLayer, FPathDataset
from src.dataloaders import FPathLazyDataset
from src.preprocesses import VTFPreprocessor, ImagePreprocessor, InfodrawPreprocessor, TargetPreprocessor

from copy import deepcopy


#### Global variables ########
scale = 1.0
center_x, center_y = 0, 0
image = None
##############################


def draw_windows(fpath, vtf):
    x, y = g_window_clicked_x, g_window_clicked_y
    
    # Draw zoom image
    offset = 10
    window_width = 20
    canvas = np.ones((2*offset*window_width, 2*offset*window_width, 3), dtype=np.uint8) * 255
    # cv2.rectangle(image, (x-window_width-2, y-window_width-2), (x+window_width+2, y+window_width+2), (0, 0, 255), thickness=2)
    
    for w in range(-window_width, window_width):
        for h in range(-window_width, window_width):
            canvas_w, canvas_h = w + window_width, h + window_width
            pixel_intensity = image[h+y, w+x, :]
            color = (int(pixel_intensity[0]), int(pixel_intensity[1]), int(pixel_intensity[2]))
            cv2.rectangle(canvas, (offset*canvas_w, offset*canvas_h), (offset*(canvas_w+1), offset*(canvas_h+1)), color=color, thickness=-1)
    
    # Draw flow path
    for i, (fpath_x, fpath_y) in enumerate(fpath[x, y, :, :]):
        if fpath_x < 0 or fpath_y < 0: pass
        
        rel_x, rel_y = x-fpath_x, y-fpath_y
        s_rel_x, s_rel_y = offset * rel_x, offset * rel_y
        s_t_rel_x, s_t_rel_y = s_rel_x + window_width*offset, s_rel_y + window_width*offset
        
        cv2.putText(canvas, f"{i}", (round(s_t_rel_x), round(s_t_rel_y)), color=(0,0,0), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.5, thickness=1)
        
        if abs(rel_x) < 0.0001 and abs(rel_y) < 0.0001: # origin point
            cv2.circle(canvas, (round(s_t_rel_x + offset/2), round(s_t_rel_y + offset/2)), radius=1, color=(255, 0, 0), thickness=1)
        else: # the others
            cv2.circle(canvas, (round(s_t_rel_x + offset/2), round(s_t_rel_y + offset/2)), radius=1, color=(0, 0, 255), thickness=1)
    
    # Draw VTF
    VTF_canvas = np.ones((window_width*2, 21*2*window_width), dtype=np.uint8)
    for i, vtf_intensity in enumerate(vtf[:, y, x]):
        cv2.rectangle(VTF_canvas, (2*window_width*i, 0), (2*window_width*(i+1), 2*window_width), color=int(vtf_intensity*255), thickness=-1)
        cv2.putText(VTF_canvas, f"{i}", ((2*i + 1)*window_width, offset), color=(0, 0, 0), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.5, thickness=1)
        cv2.putText(VTF_canvas, f"{vtf_intensity}", ((2*i + 1)*window_width, window_width), color=(0, 0, 0), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.5, thickness=1)
    
    # g_canvas = canvas
    # g_VTF_canvas = VTF_canvas
    cv2.imshow("VTF", VTF_canvas)
    cv2.imshow("window", image)
    cv2.imshow("zoom", canvas)

# Function to handle mouse events
def handle_mouse_event(event, x, y, flags, param):
    global image, g_window_clicked_x, g_window_clicked_y, g_canvas, g_VTF_canvas
    
    fpath, vtf = param
    """
        vtf: [21 x H x W]
        fpath: [W x H x 21 x 2]
    """
    
    if event == cv2.EVENT_LBUTTONDOWN:
        g_window_clicked_x, g_window_clicked_y = x, y
        draw_windows(fpath, vtf)    
        
        
def main():
    global image, g_window_clicked_x, g_window_clicked_y
    
    # target-result
    overlap_target_result_path      = "overlap_target_result.png"
    overlap_infodraw_result_path    = "overlap_infodraw_result.png"
    overlap_infodraw_target_path    = "overlap_infodraw_target.png"
    
    # flowpath & VTF
    vtf_path                        = "dataset\\test\\vtfs\\color_901_fpath_of_infodraw.npz"
    fpath_path                      = "dataset\\test\\fpaths\\color_901_fpath.npz"
    vtf                             = VTFPreprocessor.get(vtf_path)
    fpath                           = np.load(fpath_path)['data']
    
    cv2.namedWindow("window", flags=cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow("zoom", flags=cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow("VTF", flags=cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback("window", handle_mouse_event, (fpath, vtf))
    
    while True:
        key = cv2.waitKey(0)
        
        if key == ord("1"):
            print("display overlap_target_result_path")
            image = cv2.imread(overlap_target_result_path)
            cv2.imshow("window", image)
        elif key == ord("2"):
            print("display overlap_infodraw_result_path")
            image = cv2.imread(overlap_infodraw_result_path)
            cv2.imshow("window", image)
        elif key == ord("3"):
            print("display overlap_infodraw_target_path")
            image = cv2.imread(overlap_infodraw_target_path)
            cv2.imshow("window", image)
            
        elif key == ord("a"): # left arrow
            g_window_clicked_x -= 1
            draw_windows(fpath, vtf)
        elif key == ord("w"): # up array
            g_window_clicked_y -= 1
            draw_windows(fpath, vtf)
        elif key == ord("d"): # right arrow
            g_window_clicked_x += 1
            draw_windows(fpath, vtf)
        elif key == ord("s"): # down arraw
            g_window_clicked_y += 1
            draw_windows(fpath, vtf)    
            
        elif key == ord("c") or key == 27:
            print("Exit.")
            break

if __name__ == "__main__":
    main()
    
    