from functools import partial
from itertools import product
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Union
import cv2
import numpy as np

import pandas as pd
import torch
from tqdm import tqdm


def load_pts(path_to_points):
    pts = pd.read_csv(path_to_points)
    colors = pts.pop('Colors')
    pts = pts.astype('float32')
    return pts, colors


def mask_to_nans(image, mask) -> np.ndarray:
    d = image.copy().astype(float)
    d[mask] = np.nan
    return d


def find_diff(*, warped, mask, static):
    d = mask_to_nans(image=warped, mask=mask)
    s = mask_to_nans(image=static, mask=mask)
    diff = np.abs(s-d)
    diff /= 100  # 100C to C
    return diff[~np.isnan(diff)].mean(), diff


def warp(*, pts: Union[str, Path], dynamic, static):
    if isinstance(dynamic, torch.Tensor):
        dynamic = dynamic.detach().cpu().numpy().squeeze()
    if isinstance(static, torch.Tensor):
        static = static.detach().cpu().numpy().squeeze()

    M = cv2.getPerspectiveTransform(pts[['STATIC_X', 'STATIC_Y']].values, pts[['DYN_X', 'DYN_Y']].values)
    dst = cv2.warpPerspective(dynamic, M, list(reversed(static.shape[-2:])))
    mask = dst == 0

    # # mask negative values in gt
    # mask = mask | (gt < 0) | (gt > 100)
    # # # mask negative and bigger than 1 values in warped
    # mask = mask | (warped < 0) | (warped > 100)
    return dst, mask


def mp_warp(pts, dynamic, static):
    warped, mask = warp(pts=pts, dynamic=dynamic, static=static)
    return find_diff(warped=warped, mask=mask, static=static)[0]


def optim_best_pts(*, path: Union[str, Path], idx_of_frame: int, n_pixels_to_permute: int = 0, n_runs: int = 100):
    dynamic = np.load(path / f"left_{idx_of_frame}.npy")
    static = np.load(path / f"right_{idx_of_frame}.npy")
    pts, colors = load_pts(path_to_points=path / f'points_{idx_of_frame}.csv')
    loss_init = mp_warp(dynamic=dynamic, static=static, pts=pts)

    if n_runs <= 0:
        return pts

    print('Optimize all points...', flush=True)
    print('Generate permutations...', flush=True)
    pts_list = generate_permutations(SIZE=np.prod(
        pts.shape), N=n_pixels_to_permute+1).reshape(-1, *pts.shape).astype('float32') - 1
    print('Shuffle permutations...')
    np.random.shuffle(pts_list)
    pts_list = pts_list[:n_runs] if n_runs > 0 else pts_list
    print('Add original points...')
    pts_list = [pts + p for p in pts_list]

    warper = partial(mp_warp, dynamic=dynamic, static=static)
    with Pool(cpu_count()) as p:
        losses = list(tqdm(p.imap(warper, pts_list, chunksize=16), total=len(pts_list), desc='Optimize all points'))
    loss = min(losses)
    print(f'Best loss on optimizing all points: {loss:.2f}')
    if loss < loss_init:
        pts_best = pts_list[np.argmin(losses)]
        pts_best.index = colors
        if not all(pts_best >= 0):
            print('Negative values in points!')
            print(pts_best)
            return
        pts_best.astype(int).to_csv(path / f'points_{idx_of_frame}.csv', index=True)
        pts = pts_best
        loss_init = loss


def optim_single_pts(*, path: Union[str, Path],
                     n_pixels_for_single_points: int = 0,
                     idx_of_frame: int):
    dynamic = np.load(path / f"left_{idx_of_frame}.npy")
    static = np.load(path / f"right_{idx_of_frame}.npy")
    pts, colors = load_pts(path_to_points=path / f'points_{idx_of_frame}.csv')
    loss_init = mp_warp(dynamic=dynamic, static=static, pts=pts)
    loss_best = loss_init

    # Optimize all points as single points first
    warper = partial(mp_warp, dynamic=dynamic, static=static)
    permutations = np.arange(- n_pixels_for_single_points, n_pixels_for_single_points+1)
    with tqdm(total=np.prod(pts.shape), desc='Optimize single points') as pbar:
        for i, j in np.ndindex(pts.shape):
            value_of_point = pts.iloc[i, j]
            dim_of_point = pts.columns[j]
            if 'x' in dim_of_point.lower():
                dim_of_point = 1
            elif 'y' in dim_of_point.lower():
                dim_of_point = 0
            permutations_bounded_by_image = permutations[(value_of_point + permutations >= 0) &
                                                         (value_of_point + permutations < dynamic.shape[dim_of_point])]
            pts_ = pts.copy().values[None, ...].repeat(len(permutations_bounded_by_image), 0)
            pts_[..., i, j] += permutations_bounded_by_image
            pts_list = list(pts_)
            pts_list = [pd.DataFrame(p, columns=pts.columns, index=pts.index) for p in pts_list]
            with Pool(cpu_count()) as p:
                losses = list(p.imap(warper, pts_list))
            # losses = list(tqdm(map(warper, pts_list), total=len(pts_list), desc=f'Optimize single points {i},{j}'))
            loss = min(losses)
            if loss < loss_best:
                pts_best = pts_list[np.argmin(losses)]
                pts_best.index = colors
                if not all(pts_best >= 0):
                    print('Negative values in points!')
                    print(pts_best)
                    break
                pts_best.astype(int).to_csv(path / f'points_{idx_of_frame}.csv', index=True)
                pts = pts_best
                loss_best = loss
            pbar.set_postfix_str(f'Loss: Init {loss_init:.2f}, Best {loss_best:.2f}')
            pbar.update()


def generate_permutations(N, SIZE):
    elements = np.arange(N + 1)  # Generate elements from 0 to N
    permutations = list(product(elements, repeat=SIZE))
    return np.array(permutations)


IDX_OF_FRAME = 25
for _ in range(5):
    optim_single_pts(path=Path('rawData'), n_pixels_for_single_points=200, idx_of_frame=IDX_OF_FRAME)
# optim_best_pts(path=Path('rawData'), n_pixels_to_permute=1, idx_of_frame=IDX_OF_FRAME)

# import matplotlib.pyplot as plt

# dynamic = np.load(f'rawData/left_{IDX_OF_FRAME}.npy')
# static = np.load(f'rawData/right_{IDX_OF_FRAME}.npy')
# pts, colors = load_pts(path_to_points=f'rawData/points_{IDX_OF_FRAME}.csv')
# warped, mask = warp(pts=pts, dynamic=dynamic, static=static)
# mask = ~mask

# SIZE_OF_EROSION = 9
# mask = torch.from_numpy(mask.astype(float))[None, None]
# mask = torch.nn.functional.conv2d(
#     mask,
#     torch.ones(1, 1, SIZE_OF_EROSION, SIZE_OF_EROSION).to(mask),
#     padding=SIZE_OF_EROSION // 2).squeeze()
# mask /= SIZE_OF_EROSION**2
# mask = mask == 1

# diff = np.abs(static.astype(float)-warped.astype(float))
# diff /= 100
# vmin = diff[mask].min()
# vmax = diff[mask].max()
# diff[~mask] = 0
# plt.figure()
# plt.imshow(diff, vmin=vmin, vmax=vmax, cmap='bwr')
# plt.title('Difference, Avg error: {:.2f}C'.format(diff[mask].mean()))
# plt.colorbar()
# plt.show()
