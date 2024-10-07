""" 
Program to compute Voronoi diagram using JFA.

@author yisiox
@version September 2022
"""

import cv2
import numpy as np

import matplotlib.pyplot as plt
import cupy as cp
from random import sample




def initSeed(seed_img: np.ndarray, ping: np.ndarray) -> np.ndarray:
    H, W = seed_img.shape
    for h in range(H):
        for w in range(W):
            if seed_img[h, w] > 0.1:
                ping[h, w] = h * W + w
    pong = np.copy(ping)
    return pong
    


# Elementwise kernel that applies basic hash function for colour mapping.
displayKernel = cp.ElementwiseKernel(
        "int64 x",
        "int64 y",
        f"y = (x < 0) ? x : x % 103",
        "displayTransform")

def displayDiagram(frame, graph):
    """
    Function to display and save the current state of the diagram.

    @param frame The current frame.
    @oaram graph The graph to be displayed.
    """
    output = cp.asnumpy(displayKernel(graph))
    plt.imshow(output, vmin = 0.0, interpolation = "none")
    plt.savefig(f"frames/voronoi_frame_{frame}.png")
    plt.show()


# CUDA Kernel for making 1 pass of JFA
# Python int is long long in C
voronoiKernel = cp.RawKernel(r"""
    extern "C" __global__
    void voronoiPass(const long long step, const long long xDim, const long long yDim, 
                     const long long *ping, long long *pong, long long *distmap) {
        
        /* Index the point being processed */
        long long idx = blockIdx.x * blockDim.x + threadIdx.x;
        long long stp = blockDim.x * gridDim.x;
        long long N   = xDim * yDim;
        for (long long i = idx; i < N; i += stp) {
            
            /* Enumerate neighbours */
            int dydx[] = {-1, 0, 1};
            for (int j = 0; j < 3; ++j) 
            for (int k = 0; k < 3; ++k) {
                
                /* Get index of current neighbour being processed */
                long long dx  = step * dydx[j] * yDim;
                long long dy  = step * dydx[k];
                long long s   = i + dx + dy;

                /* Check if invalid neighbour */
                if (s < 0 || s >= N || ping[s] == -1)
                    continue;

                /* Calculate distances */
                long long x1, y1, x2, y2, x3, y3;
                x1 = i / yDim;
                y1 = i % yDim;
                x2 = pong[i] / yDim;
                y2 = pong[i] % yDim;
                x3 = ping[s] / yDim;
                y3 = ping[s] % yDim;
                long long curr_dist = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
                long long jump_dist = (x1 - x3) * (x1 - x3) + (y1 - y3) * (y1 - y3);

                /* Check if current point is unpopulated and populate if so */
                if (pong[i] == -1) {
                    pong[i] = ping[s];
                    distmap[i] = jump_dist;
                    continue;
                }
                else
                {
                    if (jump_dist < curr_dist)
                    {
                        pong[i] = ping[s];
                        distmap[i] = jump_dist;
                    }    
                }
            }
        }
    }
    """, "voronoiPass")


def JFAVoronoiDiagram(seed_img):
    assert len(seed_img.shape) == 2, f"seed_img must be grayscale image. But got {seed_img.shape} shaped image"
    H, W = seed_img.shape
    
    # print(f"1. init seed...")
    ping = -np.ones_like(seed_img, dtype=np.int64)
    pong = initSeed(seed_img=seed_img, ping=ping)


    # compute initial step size
    step = max(H, W) // 2
    # # initalise frame number and display original state
    frame = 0
    # displayDiagram(frame, ping)
    
    # print(f"2. processing jump flooding algorithm...")
    ping = cp.asarray(ping)
    pong = cp.asarray(pong)
    # print(f"ping: {ping}, pong: {pong}")
    distmap = cp.full((H, W), 0, dtype=int)
    while step:
        #grid size, block size and arguments
        voronoiKernel((min(H, 256),), (min(W, 256),), (step, H, W, ping, pong, distmap))
        #swap read and write graphs and update variables
        ping, pong = pong, ping
        frame += 1
        step //= 2
        #display current state
        # displayDiagram(frame, ping)
    
    # return result
    ping, distmap = ping.get(), distmap.get()
    
    # normalize distmap 0-1
    distmap = np.sqrt(distmap)
    distmap /= np.max(distmap)
    return ping, distmap

# driver code
def main():
    seed_img = np.float64(cv2.imread("target.png", cv2.IMREAD_GRAYSCALE)) / 255.
    seed_img = 1 - seed_img
    voronoiMap, distMap = JFAVoronoiDiagram(seed_img=seed_img)
    
    plt.imsave("voronoiMap.png", voronoiMap)
    plt.imsave("distMap.png", np.sqrt(distMap))

if __name__ == "__main__":
    main()
