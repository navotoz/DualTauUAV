from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Union
import cv2
from matplotlib import pyplot as plt
import numpy as np

import pandas as pd
import torch

from utils.tools import load_files_from_dir


def save_each_frame_to_separate_file(*, src, dest, path_to_files):
    # Delete old files
    for p in path_to_files.glob('*.npy'):
        p.unlink()

    for idx, frame in enumerate(src['frames']):
        np.save(path_to_files / f'src_{idx}.npy', frame)
    for idx, frame in enumerate(dest['frames']):
        np.save(path_to_files / f'dest_{idx}.npy', frame)
    np.savez(path_to_files / 'src.npz', **src)
    np.savez(path_to_files / 'dest.npz', **dest)
    print(f"Saved {len(src['frames'])} frames to {path_to_files}")


def load_src_and_dest(path_to_files):
    src = load_files_from_dir(path_to_files / 'pan')
    dest = load_files_from_dir(path_to_files / 'mono')

    print(f"List of keys in dict: {list(dest.keys())}\n")
    print(f"Number of left frames: {len(src['time_ns']):,}")
    print(f"Number of right frames: {len(dest['time_ns']):,}")
    return src, dest


def _get_diff(src, dest):
    if len(src['time_ns']) > len(dest['time_ns']):
        return np.abs(src['time_ns'][:, None] - dest['time_ns'][None, :])
    else:
        return np.abs(src['time_ns'][None, :] - dest['time_ns'][:, None])


def fix_timing_between_left_and_right(*,
                                      src: dict[str, np.ndarray],
                                      dest: dict[str, np.ndarray],
                                      max_time_diff: float) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    src = deepcopy(src)
    dest = deepcopy(dest)
    max_time_diff_ns = max_time_diff * 1e9

    # Remove frames that are not in both cameras
    while abs(len(src['time_ns']) - len(dest['time_ns'])) > 1:
        # Find the difference between the timestamps of the two cameras
        diff = _get_diff(src, dest)

        # Find the indices in the longer array that have no counterpart in the shorter array
        diff_values_of_longer_dict = np.min(diff, axis=1)
        mask_longer_dict = diff_values_of_longer_dict <= max_time_diff_ns
        indices_of_longer_dict = np.where(mask_longer_dict)[0]
        if len(src['time_ns']) > len(dest['time_ns']):
            src = {k: v[indices_of_longer_dict] for k, v in src.items()}
        else:
            dest = {k: v[indices_of_longer_dict] for k, v in dest.items()}
    print(f"Length of left: {len(src['time_ns']):,}")
    print(f"Length of right: {len(dest['time_ns']):,}")

    # Sort both arrays to have the closest timestamps have same index
    diff = np.argmin(np.abs(src['time_ns'][:, None] - dest['time_ns'][None, :]), axis=0)
    src = {k: v[diff] for k, v in src.items()}

    diff = np.abs(src['time_ns'] - dest['time_ns'])*1e-9
    print(f"Max time difference between frames: {np.max(diff):.2g} seconds")
    print(f"Min time difference between frames: {np.min(diff):.2g} seconds")

    plt.figure()
    plt.scatter(range(len(src['time_ns'])), src['time_ns'] * 1e-9, label='Left', s=2)
    plt.scatter(range(len(dest['time_ns'])), dest['time_ns'] * 1e-9, label='Right', s=2)
    plt.title('Timestamps of frames')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()
    plt.scatter(range(len(diff)), diff, c='r', s=2)
    plt.plot(np.ones_like(src['time_ns'])*np.mean(diff))
    plt.grid()
    plt.title('Time difference between frames')
    plt.ylabel('Time difference [s]')
    plt.show()
    return src, dest


def create_permutations(vector):
    N = len(vector)
    permutations = np.empty((N*N, 2), dtype=int)
    index = 0
    for i in range(N):
        for j in range(N):
            permutations[index] = (vector[i], vector[j])
            index += 1
    return permutations


def load_pts(path_to_points):
    try:
        pts = pd.read_csv(path_to_points)
    except FileNotFoundError:
        return None, None
    colors = pts.pop('Colors')
    pts = pts.astype('float32')
    return pts, colors


def find_diff(*, warped, mask, dest):
    diff = np.abs(warped[mask].astype(float)-dest[mask].astype(float))
    return diff.mean()


def get_M(pts):
    return cv2.getPerspectiveTransform(src=pts[['DEST_X', 'DEST_Y']].values,
                                       dst=pts[['SRC_X', 'SRC_Y']].values,)


def warp(*, pts: Union[str, Path], src, dest):
    if isinstance(src, torch.Tensor):
        src = src.detach().cpu().numpy().squeeze()
    if isinstance(dest, torch.Tensor):
        dest = dest.detach().cpu().numpy().squeeze()

    M = get_M(pts)
    dst = cv2.warpPerspective(src, M, list(reversed(dest.shape[-2:])))
    mask = cv2.warpPerspective(np.ones_like(src), M, list(reversed(dest.shape[-2:])),
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0, flags=cv2.INTER_NEAREST).astype(bool)

    # Erode mask
    mask = cv2.erode(mask.astype(np.uint8), kernel=None, iterations=1).astype(bool)

    # # mask negative values in gt
    # mask = mask | (gt < 0) | (gt > 100)
    # # # mask negative and bigger than 1 values in warped
    # mask = mask | (warped < 0) | (warped > 100)
    return dst, mask


def mp_warp(pts, src, dest):
    warped, mask = warp(pts=pts, src=src, dest=dest)
    if not mask.any():
        return np.inf
    return find_diff(warped=warped, mask=mask, dest=dest)


def optim_single_pts(*, path: Union[str, Path],
                     n_pixels_for_single_points: int = 0,
                     idx_of_frame: int,
                     distance_from_frame_edges: int,
                     loss_threshold: float):
    src = np.load(path / f"src_{idx_of_frame}.npy")
    dest = np.load(path / f"dest_{idx_of_frame}.npy")
    pts, colors = load_pts(path_to_points=path / f'points_{idx_of_frame}.csv')
    if not isinstance(pts, pd.DataFrame):
        # Create a pandas dataframe with 4 points:
        # two points at the left edge of the frame, and two points at the right edge of the frame.
        # The points are in the format [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        pts = pd.DataFrame(
            np.array(
                [[distance_from_frame_edges, distance_from_frame_edges],
                 [distance_from_frame_edges, src.shape[0] - distance_from_frame_edges],
                 [src.shape[1] - distance_from_frame_edges, distance_from_frame_edges],
                 [src.shape[1] - distance_from_frame_edges, src.shape[0] - distance_from_frame_edges]]),
            columns=['DEST_X', 'DEST_Y'])
        pts['SRC_X'] = pts['DEST_X']
        pts['SRC_Y'] = pts['DEST_Y']
        pts = pts.astype(np.float32)
        colors = ['red', 'green', 'blue', 'yellow']

    # Calculate initial loss
    loss_init = mp_warp(src=src, dest=dest, pts=pts)
    loss_best = loss_init
    pts_best = pts.copy()

    # Optimize all points as single points first
    # Optimize only the dynamic points
    warper = partial(mp_warp, src=src, dest=dest)
    permutations = np.arange(-n_pixels_for_single_points, n_pixels_for_single_points+1)
    permutations = create_permutations(permutations)
    number_of_pts = pts.shape[0]

    for idx_pt in range(number_of_pts):
        x_pt = pts.iloc[idx_pt].loc['SRC_X']
        y_pt = pts.iloc[idx_pt].loc['SRC_Y']
        permutations_ = permutations.copy()
        # Remove permutations outside the frame
        mask = (x_pt + permutations_ >= 0) & (x_pt + permutations_ < src.shape[1])
        mask = mask[:, 0]
        permutations_ = permutations_[mask]
        mask = (y_pt + permutations_ >= 0) & (y_pt + permutations_ < src.shape[0])
        mask = mask[:, 1]
        permutations_ = permutations_[mask]

        pts_ = pts.copy().values[None, ...]
        pts_ = np.repeat(pts_, len(permutations_), axis=0)
        pts_[..., idx_pt, :2] += permutations_  # add permutations to the dynamic points
        pts_list = list(pts_)
        pts_list = [pd.DataFrame(p, columns=pts.columns, index=pts.index) for p in pts_list]
        losses = list(map(warper, pts_list))
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

    if loss_best <= loss_threshold:  # an empiric threshold for saving the homography
        np.save(arr=get_M(pts=pts_best), file=path / f"M_{idx_of_frame}.npy")
    return loss_best


def mp_optimize_homography(index_of_frame, distance_from_frame_edges: int, loss_threshold: float,
                           path_to_files: Union[str, Path], verbose: bool = False):
    path_to_files = Path(path_to_files)
    if not (path_to_files / f'dest_{index_of_frame}.npy').exists():
        raise FileNotFoundError(path_to_files / f'dest_{index_of_frame}.npy')
    iteration_no_improvement, loss_prev = 0, float('inf')
    for iterations in range(30):
        loss = optim_single_pts(
            path=path_to_files,
            n_pixels_for_single_points=distance_from_frame_edges,
            idx_of_frame=index_of_frame,
            loss_threshold=loss_threshold,
            distance_from_frame_edges=distance_from_frame_edges)
        if loss >= loss_prev:
            iteration_no_improvement += 1
        if iteration_no_improvement > 1:
            break
        loss_prev = loss
    if verbose:
        print(f'Frame {index_of_frame}: loss = {loss:.2g}, iterations = {iterations:d}', flush=True)
    return loss
