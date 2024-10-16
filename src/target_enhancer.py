import os
import pathlib
import torch

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from multiprocessing import Pool

from src.preprocesses import VTFPreprocessor, InfodrawPreprocessor, TargetPreprocessor, FlowpathPreprocessor
from src.utils import point_interpolate_from_gray_image
from src.jfa import JFAVoronoiDiagram





class TargetEnhancer:
    model = None
    dataset = None
    iteration = None
    device = None
    
    def __init__(self) -> None:
        pass
    
    
    
    @staticmethod
    def enhance_target(dataset, model, iteration, device, multiprocessing=False):
        """_summary_

        Args:
            model (torch 모델(state_dict, pth file)): target을 VTF->sketch enhance 할 모델
            iteration (int): 몇 번째 반복인지 나타냄.
            device : default: cpu. (e.g. cude:0, etc.)
        """
        
        # 모든 데이터들에 대하여 적용. self.dataset.data 는 vtf, infodraw, flowpath, target들의 path를 담고 있음.
        
        if multiprocessing:
            print(f"use multiprocessing")
            TargetEnhancer.model = model
            TargetEnhancer.dataset = dataset
            TargetEnhancer.iteration = iteration
            TargetEnhancer.device = device
            
            with Pool(10) as p:
                p.map(__process, dataset.data)
        
        else:
            for data_path in tqdm(dataset.data):
                try:
                    # read vtf, infodraw, target, flowpath
                    vtf = VTFPreprocessor.get(data_path["vtf"])
                    infodraw = InfodrawPreprocessor.get(data_path["infodraw"])
                    target = TargetPreprocessor.get(data_path["target"])
                    flowpath = FlowpathPreprocessor.get(data_path["flowpath"])
                    
                    # calculate mask            
                    mask = infodraw < dataset.mask_threshold

                    # enhance target using trained model
                    mask = mask.to(device)
                    vtf_tensor = torch.tensor(vtf).unsqueeze(0).to(device)
                    model = model.to(device)
                    enhanced_target = get_enhanced_target(
                        predictor=model,
                        vtf_tensor=vtf_tensor,
                        infodraw=infodraw,
                        target=target,
                        flowpath=flowpath,
                        mask=mask,
                    ) 

                    # set path to store enhanced target
                    next_data_path = data_path["target"].split("/")
                    next_data_path[-2]= f"targets_{iteration}"
                    next_data_base_path = "/".join(next_data_path[:-1])
                    next_data_file_path = next_data_path[-1]

                    # create target's directory (e.g. targets_1, targets_2, ... so on)
                    target_base_path = pathlib.Path(next_data_base_path)
                    if not target_base_path.exists():
                        target_base_path.mkdir(parents=True, exist_ok=True)
                    enhanced_target_path = os.path.join(next_data_base_path, next_data_file_path)
                    
                    # save enhanced target
                    plt.imsave(enhanced_target_path, 1-enhanced_target, cmap="gray")
                    
                    # update path for target to enhenced one
                    data_path["target"] = enhanced_target_path
                except:
                    print(f"Error occured in enhance_target function! target_path: {data_path['target']}")
        
        return dataset
    
def get_enhanced_target(
    predictor, 
    vtf_tensor, infodraw, target, flowpath, mask,
    PRED_HIGH_CONF_THRESHOLD=0.7,
    PRED_LOW_CONF_THRESHOLD=0.5,
    DISTMAP_ALPHA=10,
    N_CLOSING_FP=5,
    WEAK_SKETCH_CONVERSION_THRESHOLD=0.5,
    WEAK_SKETCH_CONVERSION_COUNT=6, # N_CLOSING_FP + 1
    SPATIAL_SIM_THRESHOLD=0.5,
    verbose=False
):
    """_summary_

    Args:
        predictor : vtf를 입력으로 받아 sketch인지 아닌지 판별하는 모델
        vtf (torch tensor): (1 x 21 x H x W)
        infodraw (torch tensor): (1 x H x W)
        target (torch tensor): (1 x H x W)
        flowpath (numpy tensor): (H x W x 21 x 2)
    """
    
    assert vtf_tensor.shape[2:] == infodraw.shape[1:] == target.shape[1:] == flowpath.shape[:2]
    
    infodraw = infodraw.squeeze(0).numpy()
    target = target.squeeze(0).numpy()
    target = 1 - target # 1이 sketch, 0이 BG
    
    # 1.Inference P     
    H, W = infodraw.shape
    # prediction = np.zeros((H, W))
    # for h in tqdm(range(H)):
    #     for w in range(W):
    #         if infodraw[0, h, w] < 0.99:
    #             selected_vtf = vtf_tensor[:, :, h, w]
                
    #             pred = predictor(selected_vtf)
    #             # pred = 1.0 if pred > 0.5 else 0.0
                
    #             # 현재는 확률이 0에 가까울 수록 sketch,
    #             # 1에 가까울 수록 배경이다.
    #             # sketch를 1로 보자고 했으니 이를 역전 시킴.
    #             prediction[h, w] = 1 - pred.item()
    prediction = predictor(vtf_tensor)
    prediction = mask * (1 - prediction)
    prediction = prediction.squeeze().detach().cpu().numpy()
    
    
    # 2. calculate distance map from target(GT)
    # MAX_Y, MAX_X= target.shape
    # jfa_instance = JFA(MAX_Y, MAX_X)
    
    # jfa_instance.jump_flooding(target)
    # distmap = jfa_instance.draw_distmap()
    
    
    _, distmap = JFAVoronoiDiagram(target)
    
    distmap_inv_alpha = (1 - distmap)**DISTMAP_ALPHA

    # 3. Blend P & Distance map
    # pred_add_dist_inv_alpha = cv2.addWeighted(np.array(prediction, dtype=np.float32), 0.5, distmap_inv_alpha, 0.5, 0)

    # 4. 각 threshold에 대한 mask 계산
    high_conf_mask = prediction >= PRED_HIGH_CONF_THRESHOLD
    high_conf_prediction = prediction * high_conf_mask
    low_conf_mask = prediction < PRED_LOW_CONF_THRESHOLD 
    low_conf_prediction = prediction * low_conf_mask
    middle_conf_mask = (1 - high_conf_mask) * (1 - low_conf_mask)
    middle_conf_prediction = prediction * middle_conf_mask
    
    
    # 5. middle_conf_mask에 대하여 similarity map 을 계산함.
    spatial_similarity_map = np.zeros_like(prediction)
    # Calculate spatial similarity map
    for h in range(H):
        for w in range(W):
            # if nms_step_2_mask[h, w]:
            # middle confidence를 갖는 pixel 중에서만 다시 우열을 가려야지
            if middle_conf_mask[h, w]:
                fp = flowpath[h, w, :, :] # [21 x 2] (21 * (x, y))
                vtf_df = np.zeros(21)
                diff_vtf_dt = np.zeros(21-1)
                # flowpath는 길이가 다를 수 있음.
                # GT와 완벽히 정합되는 길이가 3인 flowpath와 
                # 21인 flowpath 중에서 길이가 21인 flowpath가
                # 더 spatial similarity가 높아야함.
                x_first, y_first = fp[0]
                if W-1 > x_first >= 0 and H-1 > y_first >= 0:
                    v_prev = point_interpolate_from_gray_image(x_first, y_first, distmap_inv_alpha)
                    vtf_df[0] = v_prev
                    
                    for i, (x, y) in enumerate(fp[1:]):
                        if W-1 > x >= 0 and H-1 > y >= 0:
                            v_cur = point_interpolate_from_gray_image(x, y, distmap_inv_alpha)
                            diff_vtf_dt[i] = np.abs(v_cur - v_prev)
                            vtf_df[i+1] = v_cur
                            v_prev = v_cur
                        else:
                            diff_vtf_dt[i] = 1
                            v_prev = 1
                        
                    curvature_similarity = 1 - np.mean(diff_vtf_dt)
                    level_of_misalignment = np.mean(vtf_df)
                    
                    # GT와 가깝고 spatial similarity가 클 수록 similarity가 크다고 볼 수 있음. 
                    # spatial_similarity_map[h, w] = level_of_misalignment * curvature_similarity
                    spatial_similarity_map[h, w] = curvature_similarity

    spatial_sim_mask = spatial_similarity_map > SPATIAL_SIM_THRESHOLD

    # 6. find closest aligned sketch
    AS_strong = np.clip(high_conf_mask + spatial_sim_mask, a_min=0.0, a_max=1.0)
    # step_2_nms_strong_sketch = np.zeros_like(target)
    AS_weak = np.zeros_like(target)
    target_cnt = 0
    sketch_cnt = 0
    for h in range(H):
        for w in range(W):
            if target[h, w] > 0.9: # is sketch
                target_cnt += 1
                # print(f"h, w: {h, w}")
                # 1. get normal gradient from (h, w)
                fp = flowpath[h, w, :, :]
                
                # find cur position
                is_success, cur_pos = flowpath_find_center_point(h, w, fp)
                if not is_success:
                    # flowpath 상에서 현재 기준 픽셀 (h, w)와 같은 값을 못찾음.
                    # => GT이지만 Flowpath가 없는 부분
                    AS_weak[h, w] = 1
                    sketch_cnt += 1
                    continue
                
                # tangent 계산
                tangent = calculate_tangetn_from_fp_with_centor_point(cur_pos, fp)
                
                # Normalized Gradient 계산
                norm_gradient = calculate_normalized_gradient(tangent)
                
                # 2. get N traversed pixel's position through gradient
                pos_traversing_pixels = gradient_path_from_point(h, w, norm_gradient, N=3, do_round=False)
                
                # # 3. pick maximum value and discard the others
                issuccess, closest_aligned_sketch_pos = find_aligned_sketch(h, w, pos_traversing_pixels, AS_strong, H, W)
                if not issuccess: # gradient path 상에 spatial_sim_threshold보다 큰 값이 존재하지 않음. (아무것도 없음)
                    # 그러면 기존 GT 그대로 사용
                    sketch_cnt += 1
                    AS_weak[h, w] = 1
                else:    
                    sy, sx = closest_aligned_sketch_pos
    step_2 = np.clip(AS_weak + AS_strong, a_min=0, a_max=1.0)
    
    
    
    # 7. Flowpath-base closing 수행
    step_3 = np.zeros_like(target)
    for h in range(H):
        for w in range(W):
            if step_2[h, w] == 0: # weak sketch but not choosen
                fp = flowpath[h, w, :, :]
                
                issuccess, cur_pos = flowpath_find_center_point(h, w, fp)
                if not issuccess:
                    continue
                
                sketch_pix_cnt = 0
                for n in range(-N_CLOSING_FP, N_CLOSING_FP+1):
                    if 21 > cur_pos + n > -1:
                        x, y = fp[cur_pos + n]
                        if x > W-2 or y > H-2:
                            continue
                        
                        intensity = point_interpolate_from_gray_image(x=x, y=y, img=step_2)
                        
                        if intensity >= WEAK_SKETCH_CONVERSION_THRESHOLD:
                            sketch_pix_cnt += 1
                            
                if sketch_pix_cnt >= WEAK_SKETCH_CONVERSION_COUNT:
                    step_3[h, w] = 1
    
    # 8. merge step2 & step3
    final_result = np.clip(step_2 + step_3, a_min=0, a_max=1)

    if verbose:
        # prediction 그 자체 출력
        plt.imsave("1_P.png", prediction, vmin=0, vmax=1)
        plt.imsave("1_1_P_high.png", high_conf_prediction, vmin=0, vmax=1)
        plt.imsave("1_2_P_low.png", low_conf_prediction, vmin=0, vmax=1)
        plt.imsave("1_3_P_mid.png", middle_conf_prediction, vmin=0, vmax=1)
        plt.imsave("2_distmap.png", distmap, vmin=0, vmax=1)
        plt.imsave("2_1_distmap_inv_alpha.png", distmap_inv_alpha, vmin=0, vmax=1)
        plt.imsave("3_S_spatial_similarity_map.png", spatial_similarity_map, vmin=0, vmax=1)
        plt.imsave("3_1_S_over_T_ss.png", spatial_sim_mask, vmin=0, vmax=1)
        plt.imsave("3_2_AS_strong.png", AS_strong, vmin=0, vmax=1)
        plt.imsave("3_2_AS_weak.png", AS_weak, vmin=0, vmax=1)
        plt.imsave("4_Enhanced_GT.png", step_2, vmin=0, vmax=1)
        plt.imsave("4_1_Enhanced_closed_GT.png", final_result, vmin=0, vmax=1)

    return final_result

def __process(data_path):
    assert not TargetEnhancer.model is None, "model must be initialized as a class variable to used in multiprocessing"
    assert not TargetEnhancer.dataset is None, "dataset must be initialized as a class variable to used in multiprocessing"
    assert not TargetEnhancer.iteration is None, "iteration must be initialized as a class variable to used in multiprocessing"
    assert not TargetEnhancer.device is None, "device must be initialized as a class variable to used in multiprocessing"
    
    model = TargetEnhancer.model
    dataset = TargetEnhancer.dataset
    iteration = TargetEnhancer.iteration
    device = TargetEnhancer.device
    
    # read vtf, infodraw, target, flowpath
    vtf = VTFPreprocessor.get(data_path["vtf"])
    infodraw = InfodrawPreprocessor.get(data_path["infodraw"])
    target = TargetPreprocessor.get(data_path["target"])
    flowpath = FlowpathPreprocessor.get(data_path["flowpath"])
    
    # calculate mask            
    mask = infodraw < dataset.mask_threshold

    # enhance target using trained model
    mask = mask.to(device)
    vtf_tensor = torch.tensor(vtf).unsqueeze(0).to(device)
    model = model.to(device)
    enhanced_target = get_enhanced_target(
        predictor=model,
        vtf_tensor=vtf_tensor,
        infodraw=infodraw,
        target=target,
        flowpath=flowpath,
        mask=mask,
    ) 

    # set path to store enhanced target
    next_data_path = data_path["target"].split("/")
    next_data_path[-2]= f"targets_{iteration}"
    next_data_base_path = "/".join(next_data_path[:-1])
    next_data_file_path = next_data_path[-1]

    # create target's directory (e.g. targets_1, targets_2, ... so on)
    target_base_path = pathlib.Path(next_data_base_path)
    if not target_base_path.exists():
        target_base_path.mkdir(parents=True, exist_ok=True)
    enhanced_target_path = os.path.join(next_data_base_path, next_data_file_path)
    
    # save enhanced target
    plt.imsave(enhanced_target_path, 1-enhanced_target, cmap="gray")
    
    # update path for target to enhenced one
    data_path["target"] = enhanced_target_path



# NMS step 2 에서 사용하는 함수 정의

def flowpath_find_center_point(h, w, flowpath):
    # flowpath는 기준 픽셀을 기준으로 normalized tangent를 따라가며
    # 저장된 21개 이하의 픽셀 위치 벡터들임. 
    # 만약 flowpath의 벡터들의 갯수가 21개 미만이라면 남은 부분은 (-1, -1) 로 채워짐.
    # 여기서 중심이 항상 11 (index==10)번째가 아닐 수 있음.
    # 따라서 이를 찾아주는 과정이 필요함.
    
    # 입력: h, w: 는 현재 이미지 상에 위치한 기준 픽셀, flowpath: flowpath
    # 출력: center pos가 없는 경우도 존재. 따라서 찾는데 성공 할 수도 못할 수도 있음.
    
    cur_pos = None
    is_success = False
    for i, (x, y) in enumerate(flowpath):
        if x < 0 or y < 0: continue # (-1, -1) vector는 무시
        # 현재 기준 픽셀 (h, w) 와 flowpath 상에 저장된 픽셀 벡터 간의 거리가 1미만이면
        # 기준 픽셀
        if (y-h)*(y-h) + (x-w)*(x-w) < 1.0:
            cur_pos = i
            is_success = True
            break
        
    return is_success, cur_pos

def calculate_tangetn_from_fp_with_centor_point(cur_pos, fp):
    if 20 > cur_pos > 0:
        x_prev, y_prev = fp[cur_pos-1]
        x_next, y_next = fp[cur_pos+1]
        if (x_prev < 0 or y_prev < 0) and (x_next < 0 or y_next < 0):
            raise RuntimeError(f"Gradient를 구하는데 잘못됨. cur_pos: {cur_pos}, flowpath: {fp}")
        elif x_prev < 0 or y_prev < 0: # cur_pos-1 의 flowpath는 없고 cur_pos+1은 있음.
            tangent = fp[cur_pos+1] - fp[cur_pos]
        elif x_next < 0 or y_next < 0: # cur_pos+1의 flowpath는 없고 curpos-1은 있음.
            tangent = fp[cur_pos] - fp[cur_pos-1]
        else: # 모두 있음.
            tangent = fp[cur_pos+1] - fp[cur_pos-1]
    elif cur_pos == 0:
        # print(f"cur_pos: {cur_pos}, x_next: {x_next}, y_next: {y_next}")
        x_next, y_next = fp[cur_pos+1]
        if x_next < 0 or y_next < 0:
            raise RuntimeError(f"Gradient를 구하는데 잘못됨. cur_pos: {cur_pos}, flowpath: {fp}")
        else:
            tangent = fp[cur_pos+1] - fp[cur_pos]
    else: # cur_pos == 20:
        x_prev, y_prev = fp[cur_pos-1]
        if x_prev < 0 or y_prev < 0:
            raise RuntimeError(f"Gradient를 구하는데 잘못됨. cur_pos: {cur_pos}, flowpath: {fp}")
        else:
            tangent = fp[cur_pos] - fp[cur_pos-1]
    
    return tangent

def calculate_normalized_gradient(tangent):
    tangent_xyz = np.array([tangent[0], tangent[1], 0], dtype=np.float64)
    z_axis = np.array([0, 0, 1], np.float32)
    gradient = np.cross(tangent_xyz, z_axis)
    norm_gradient = gradient / np.linalg.norm(gradient)
    
    return norm_gradient

def gradient_path_from_point(h, w, norm_gradient, N=3, do_round=False):
    global SPATIAL_SIM_THRESHOLD
    
    # gradient는 flowpath의 tangent로 계산되어 (x, y) 순서임.
    ngx, ngy = norm_gradient[0], norm_gradient[1] # each x, y values of normalized gradient
    
    if do_round:
        pos_traversing_pixels = np.array([(round(h+n*ngy), round(w+n*ngx)) for n in range(-N, N+1)])
    else:
        pos_traversing_pixels = np.array([(h+n*ngy, w+n*ngx) for n in range(-N, N+1)])
        
    return pos_traversing_pixels

def pick_maxval_through_pixels(pos_traversing_pixels, img, H, W):
    # pos_traversing_pixels: gradient path를 기반으로 구한 pixels들의 위치
    # img: pos_traversing_pixels가 따라갈 이미지 (spatial similarity map)

    intensities_through_pixels = []    
    for pos in pos_traversing_pixels:
        y, x = pos
        if x > W-1 or y > H-1:
            continue
        intensity = point_interpolate_from_gray_image(x, y, img)
        intensities_through_pixels.append(intensity)
    
    maxval = max(intensities_through_pixels)
    if maxval < SPATIAL_SIM_THRESHOLD:
        return False, None
    else:
        return True, maxval
    
def find_aligned_sketch(h, w, pos_traversing_pixels, img, H, W):
    # h, w: current position
    # pos_traversing_pixels: gradient path를 기반으로 구한 pixels들의 위치
    # img: pos_traversing_pixels가 따라갈 이미지 (spatial similarity map)

    aligned_sketch_positions = []    
    for pos in pos_traversing_pixels:
        y, x = pos
        if x >= W-1 or y >= H-1:
            continue
        intensity = point_interpolate_from_gray_image(x, y, img)
        
        # 주위 4개의 pixel 중 하나라도 스케치가 있다면
        if intensity > 0.24: # y, x 에 aligned sketch가 존재함.
            # 가장 가까운 pixel 의 위치를 구하기 위해 distance를 계산함.
            distance = (y-h)*(y-h) + (x-w)*(x-w)
            aligned_sketch_positions.append((distance, y, x))
    
    if not aligned_sketch_positions:
        return False, None
    else:
        closest_aligned_sketch_position = None
        mindist = 1e5
        for distance, y, x in aligned_sketch_positions:
            if distance < mindist:
                mindist = distance
                closest_aligned_sketch_position = (y, x)
        
        # print(f"closest_aligned_sketch_positions: {aligned_sketch_positions}")
        return True, closest_aligned_sketch_position
    