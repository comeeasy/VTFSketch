import math
import numpy as np
import cv2
import torch

import matplotlib.pyplot as plt

from tqdm import tqdm

from src.preprocesses import VTFPreprocessor, InfodrawPreprocessor, TargetPreprocessor, ImagePreprocessor
from src.utils import point_interpolate_from_gray_image






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
    MAX_Y, MAX_X= target.shape
    jfa_instance = JFA(MAX_Y, MAX_X)
    
    jfa_instance.jump_flooding(target)
    distmap = jfa_instance.draw_distmap()
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
    
    # prediction 그 자체 출력
    plt.imsave("1_P.png", prediction, vmin=0, vmax=1)
    plt.imsave("1_1_P_high.png", high_conf_prediction, vmin=0, vmax=1)
    plt.imsave("1_2_P_low.png", low_conf_prediction, vmin=0, vmax=1)
    plt.imsave("1_3_P_mid.png", middle_conf_prediction, vmin=0, vmax=1)
    plt.imsave("2_distmap.png", distmap, vmin=0, vmax=1)
    plt.imsave("2_1_distmap_inv_alpha.png", distmap_inv_alpha, vmin=0, vmax=1)
    
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
                if W-1 > x_first > 0 and H-1 > y_first > 0:
                    v_prev = point_interpolate_from_gray_image(x_first, y_first, distmap_inv_alpha)
                    vtf_df[0] = v_prev
                for i, (x, y) in enumerate(fp[1:]):
                    if W-1 > x > 0 and H-1 > y > 0:
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
    
    plt.imsave("3_S_spatial_similarity_map.png", spatial_similarity_map, vmin=0, vmax=1)
    plt.imsave("3_1_S_over_T_ss.png", spatial_sim_mask, vmin=0, vmax=1)
    plt.imsave("3_2_AS_strong.png", AS_strong, vmin=0, vmax=1)
    plt.imsave("3_2_AS_weak.png", AS_weak, vmin=0, vmax=1)
    plt.imsave("4_Enhanced_GT.png", step_2, vmin=0, vmax=1)
    
    
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
                    print(f"[{h}, {w}] 가 약 스케치에서 강 스케치로 전환됨.")
    
    # 8. merge step2 & step3
    final_result = np.clip(step_2 + step_3, a_min=0, a_max=1)

    plt.imsave("4_1_Enhanced_closed_GT.png", final_result, vmin=0, vmax=1)

    return final_result
    
class SeedInfo:
    def __init__(self):
        self.sid = -1
        self.dist = float('inf')
        self.dist_arc = 1.0
        self.clr = np.zeros(3)
        self.sx, self.sy = 0, 0
        self.isseed = False

    def set(self, sid_, d, clr_, sx_, sy_):
        self.sid = sid_
        self.dist = d
        self.clr = clr_.copy()
        self.sx = sx_
        self.sy = sy_

    def update(self, new_sid, new_dist):
        if new_dist < self.dist:
            self.dist = new_dist
            self.sid = new_sid
            
class JFA:
    def __init__(self, max_x, max_y):
        self.MAX_X = max_x
        self.MAX_Y = max_y
        self.dmap = np.array([[SeedInfo() for _ in range(self.MAX_Y)] for _ in range(self.MAX_X)])
        self.outClr = np.zeros((self.MAX_X, self.MAX_Y, 3))

    def init_jfa(self):
        self.dmap = np.array([[SeedInfo() for _ in range(self.MAX_Y)] for _ in range(self.MAX_X)])
        self.outClr = np.zeros((self.MAX_X, self.MAX_Y, 3))

    def is_zero_vector(self, v):
        return np.all(v == 0)

    def __process_seed(self, seed):
        new_seeds = []
        
        if len(seed) == 2:
            # print(f"seed: {seed}")
            px, py = seed
            k = self.k_parrallel
            
            for r in range(px - k, px + k + 1, k):
                for c in range(py - k, py + k + 1, k):
                    if (r == px and c == py) or not (0 <= r < self.MAX_X and 0 <= c < self.MAX_Y):
                        continue

                    idx = self.dmap[px][py].sid
                    sx, sy = self.seeds[idx]
                    d = math.sqrt((r - sx)**2 + (c - sy)**2)

                    if self.dmap[r][c].dist > d:
                        self.dmap[r][c].set(self.dmap[px][py].sid, d, self.dmap[sx][sy].clr, sx, sy)
                        if not self.dmap[r][c].isseed:
                            self.dmap[r][c].isseed = True
                            new_seeds.append((r, c))
        
        return new_seeds

    def jump_flooding(self, InSeeds):
        print(f"Start jump flooding algorithm..")
        self.seeds = []
        cnt = 0

        # Step 1: Seed initialization
        print(f"\tStep 1: Seed initialization")
        for i in tqdm(range(self.MAX_X)):
            for j in range(self.MAX_Y):
                if not self.is_zero_vector(InSeeds[i][j]):
                    self.seeds.append((i, j))
                    self.dmap[i][j].set(cnt, 0.0, InSeeds[i][j], i, j)
                    self.dmap[i][j].isseed = True
                    cnt += 1

        nseeds = cnt
        k = min(self.MAX_X, self.MAX_Y)

        # Step 2: Jump flooding algorithm
        print(f"\tStep 2: Jump flooding algorithm")
        #### Serial ======================================================
        while k >= 1:
            new_seeds = []
            for px, py in self.seeds:
                for r in range(px - k, px + k + 1, k):
                    for c in range(py - k, py + k + 1, k):
                        if (r == px and c == py) or not (0 <= r < self.MAX_X and 0 <= c < self.MAX_Y):
                            continue

                        idx = self.dmap[px][py].sid
                        sx, sy = self.seeds[idx]
                        d = math.sqrt((r - sx)**2 + (c - sy)**2)

                        if self.dmap[r][c].dist > d:
                            self.dmap[r][c].set(self.dmap[px][py].sid, d, self.dmap[sx][sy].clr, sx, sy)
                            if not self.dmap[r][c].isseed:
                                self.dmap[r][c].isseed = True
                                new_seeds.append((r, c))

            self.seeds.extend(new_seeds)
            nseeds = len(self.seeds)
            k //= 2
        #### Serial ======================================================
        #### parrallel ======================================================
        
        ### 더 느림
        
        # from concurrent.futures import ThreadPoolExecutor
        # with ThreadPoolExecutor() as executor:
        #     while k >= 1:
        #         print(f"k: {k}")
        #         self.k_parrallel = k
        #         new_seeds = executor.map(self.__process_seed, self.seeds)
                
        #         for new_seed in new_seeds:
        #             self.seeds.extend(new_seed)
        #         nseeds = len(self.seeds)
                
        #         k //= 2
        #### parrallel ======================================================


        # Step 3: Copy color information
        print(f"\tStep 3: Copy color information")
        for i in range(self.MAX_X):
            for j in range(self.MAX_Y):
                self.outClr[i][j] = self.dmap[i][j].clr

        # Step 4: Find maximum distance
        print(f"\tStep 4: Find maximum distance")
        maxdist = 0.0
        threshold = float('inf')
        for i in range(self.MAX_X):
            for j in range(self.MAX_Y):
                if self.dmap[i][j].dist <= threshold:
                    maxdist = max(maxdist, self.dmap[i][j].dist)

        # Step 5: Normalize distances
        print(f"\tStep 5: Normalize distances")
        for i in range(self.MAX_X):
            for j in range(self.MAX_Y):
                if self.dmap[i][j].dist > threshold:
                    self.dmap[i][j].dist_arc = 1.0
                else:
                    self.dmap[i][j].dist_arc = self.dmap[i][j].dist / maxdist

    def draw_distmap(self, show=False):
        print(f"Drawing distance map")
        dist_map = np.zeros((self.MAX_X, self.MAX_Y), dtype=np.float32)
        for i in range(self.MAX_X):
            for j in range(self.MAX_Y):
                dist_map[i][j] = self.dmap[i][j].dist_arc
        if show:
            plt.imshow(dist_map, cmap='gray')
            plt.show()
        
        return dist_map

    def draw_voronoi(self, show=False):
        print(f"Drawing Voronoi map")
        voronoi_map = np.zeros((self.MAX_X, self.MAX_Y, 3))
        for i in range(self.MAX_X):
            for j in range(self.MAX_Y):
                voronoi_map[i][j] = self.dmap[i][j].clr
        if show:
            plt.imshow(voronoi_map)
            plt.show()
        
        return voronoi_map
    
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
        if x > W-1 or y > H-1:
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
    