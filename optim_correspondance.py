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
    try:
        pts = pd.read_csv(path_to_points)
    except FileNotFoundError:
        return None, None
    colors = pts.pop('Colors')
    pts = pts.astype('float32')
    return pts, colors


def mask_to_nans(image, mask) -> np.ndarray:
    d = image.copy().astype(float)
    d[mask] = np.nan
    return d


def find_diff(*, warped, mask, static):
    d = mask_to_nans(image=warped, mask=~mask)
    s = mask_to_nans(image=static, mask=~mask)
    diff = np.abs(s-d)
    diff /= 100  # 100C to C
    return diff[~np.isnan(diff)].mean(), diff


def get_M(pts):
    return cv2.getPerspectiveTransform(pts[['STATIC_X', 'STATIC_Y']].values, pts[['DYN_X', 'DYN_Y']].values)


def warp(*, pts: Union[str, Path], dynamic, static):
    if isinstance(dynamic, torch.Tensor):
        dynamic = dynamic.detach().cpu().numpy().squeeze()
    if isinstance(static, torch.Tensor):
        static = static.detach().cpu().numpy().squeeze()

    M = get_M(pts)
    dst = cv2.warpPerspective(dynamic, M, list(reversed(static.shape[-2:])))
    mask = cv2.warpPerspective(np.ones_like(dynamic), M, list(reversed(static.shape[-2:])),
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0, flags=cv2.INTER_NEAREST).astype(bool)

    # Erode mask
    kernel = np.eye(3, dtype=np.uint8)
    mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    # # mask negative values in gt
    # mask = mask | (gt < 0) | (gt > 100)
    # # # mask negative and bigger than 1 values in warped
    # mask = mask | (warped < 0) | (warped > 100)
    return dst, mask


def mp_warp(pts, dynamic, static):
    warped, mask = warp(pts=pts, dynamic=dynamic, static=static)
    return find_diff(warped=warped, mask=mask, static=static)[0]


def optim_single_pts(*, path: Union[str, Path],
                     n_pixels_for_single_points: int = 0,
                     idx_of_frame: int):
    dynamic = np.load(path / f"left_{idx_of_frame}.npy")
    static = np.load(path / f"right_{idx_of_frame}.npy")
    pts, colors = load_pts(path_to_points=path / f'points_{idx_of_frame}.csv')
    if not isinstance(pts, pd.DataFrame):
        # Create a pandas dataframe with 4 points:
        # two points at the left edge of the frame, and two points at the right edge of the frame.
        # The points are in the format [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        pts = pd.DataFrame(
            np.array(
                [[DISTANCE_FROM_FRAME_EDGES, DISTANCE_FROM_FRAME_EDGES],
                 [DISTANCE_FROM_FRAME_EDGES, dynamic.shape[0] - DISTANCE_FROM_FRAME_EDGES],
                 [dynamic.shape[1] - DISTANCE_FROM_FRAME_EDGES, DISTANCE_FROM_FRAME_EDGES],
                 [dynamic.shape[1] - DISTANCE_FROM_FRAME_EDGES, dynamic.shape[0] - DISTANCE_FROM_FRAME_EDGES]]),
            columns=['STATIC_X', 'STATIC_Y'])
        pts['DYN_X'] = pts['STATIC_X']
        pts['DYN_Y'] = pts['STATIC_Y']
        pts = pts.astype(np.float32)
        colors = ['red', 'green', 'blue', 'yellow']

    # Calculate initial loss
    loss_init = mp_warp(dynamic=dynamic, static=static, pts=pts)
    loss_best = loss_init
    pts_best = pts.copy()

    # Optimize all points as single points first
    warper = partial(mp_warp, dynamic=dynamic, static=static)
    permutations = np.arange(- n_pixels_for_single_points, n_pixels_for_single_points+1)

    # Optimize only the dynamic points
    shape_of_pts_dynamic = pts[['DYN_X', 'DYN_Y']].shape

    for i, j in np.ndindex(shape_of_pts_dynamic):
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
            pts_best.index.name = 'Colors'
            if not all(pts_best >= 0):
                print('Negative values in points!')
                print(pts_best)
                raise RuntimeError
            pts_best.astype(int).to_csv(path / f'points_{idx_of_frame}.csv', index=True)
            pts = pts_best
            loss_best = loss
    if loss_best <= LOSS_THRESHOLD:  # an empiric threshold for saving the homography
        np.save(arr=get_M(pts=pts_best), file=path / f"M_{idx_of_frame}.npy")
    return loss_best


DISTANCE_FROM_FRAME_EDGES = 10
LOSS_THRESHOLD = 0.55

path_to_files = Path('rawData')
loss_prev = float('inf')
list_of_files = list(path_to_files.glob('left_*.npy'))
list_of_files.sort(key=lambda x: int(x.stem.split('left_')[1]))

# Remove all points files
for path in tqdm(path_to_files.glob('points_*.csv'), desc='Remove all points files'):
    path.unlink()
for path in tqdm(path_to_files.glob('M_*.npy'), desc='Remove all homography files'):
    path.unlink()

count_losses_below_threshold = 0
with tqdm(total=len(list_of_files), desc=f'Optimize single points, loss threshold {LOSS_THRESHOLD:.2f}') as pbar:
    for path in list_of_files:
        idx_of_frame = int(path.stem.split('left_')[1])
        if not (path_to_files / f'right_{idx_of_frame}.npy').exists():
            raise FileNotFoundError(path_to_files / f'right_{idx_of_frame}.npy')
        iteration_total, iteration_no_improvement = 0, 0
        while True:
            loss = optim_single_pts(
                path=Path('rawData'),
                n_pixels_for_single_points=DISTANCE_FROM_FRAME_EDGES,
                idx_of_frame=idx_of_frame)
            pbar.set_postfix_str(f'File: {idx_of_frame}, Saved {count_losses_below_threshold} homography matrices')
            if loss >= loss_prev:
                iteration_no_improvement += 1
            if iteration_no_improvement > 1:
                count_losses_below_threshold += (1 if loss < LOSS_THRESHOLD else 0)
                break
            loss_prev = loss
            iteration_total += 1
        pbar.update(1)
