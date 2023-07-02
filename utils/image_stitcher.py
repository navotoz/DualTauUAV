from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import cv2
import numpy as np
from PIL import Image
import torch

import torch.nn as nn
from kornia.feature import LocalFeatureMatcher, LoFTR
from kornia.geometry.homography import find_homography_dlt_iterated
from kornia.geometry.ransac import RANSAC
from kornia.geometry.transform import warp_perspective
from tqdm import tqdm


def make_gif_of_sequence(images_list, full_save_path: (str, Path)) -> None:
    """
    Saves a .gif for a sequence of images.

    Parameters
    ----------
    images_list: List[np.ndarray]
        A list of numpy arrays with dimensions (h,w), all with the same dimensions. The must be at least 2 images.
    full_save_path: str, Path
        The path of the .gif to save.
    """
    if len(images_list) < 2:
        raise ValueError(f'Cannot make GIF with {len(images_list)} images.')
    if len(images_list[0].shape) != 2:
        raise ValueError(f'Images must be of shape 2. Got shape {images_list[0].shape}.')
    if isinstance(images_list[0], torch.Tensor):
        images_list = [img.detach().cpu().numpy() for img in images_list]
    full_save_path = Path(full_save_path)
    if full_save_path.is_dir():
        raise IsADirectoryError(f"{full_save_path} is a directory.")
    if not full_save_path.parent.is_dir():
        raise IOError(f"{full_save_path.parent} does not exists.")

    # normalize all frames
    images = [p.copy() for p in (images_list.values() if isinstance(images_list, dict) else images_list)]
    min_val = min([p.min() for p in images])
    images = [(p - min_val) for p in images]
    max_val = max([p.max() for p in images])
    images = [p / max_val for p in images]
    images = [p * 255 for p in images]
    images = [p.astype('uint8') for p in images]

    frames = []
    for idx, img in enumerate(images):
        frame = Image.fromarray(img).convert('RGB')
        img = np.array(frame)
        img = cv2.putText(img, f'{idx}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        frame = Image.fromarray(img)
        # Saving/opening is needed for better compression and quality
        io_obj = BytesIO()
        frame.save(io_obj, 'GIF')
        frame = Image.open(io_obj)
        frames.append(frame)

    # Save the frames as animated GIF to BytesIO
    animated_gif = BytesIO()
    frames[0].save(animated_gif,
                   format='GIF',
                   save_all=True,
                   append_images=frames[1:],  # Pillow >= 3.4.0
                   duration=1000,
                   loop=0)
    animated_gif.seek(0, 2)
    animated_gif.seek(0)
    open(full_save_path.with_suffix('.gif'), 'wb').write(animated_gif.read())


class ImageStitcher(nn.Module):
    """Stitch two images with overlapping fields of view.

    Args:
        matcher: image feature matching module.
        estimator: method to compute homography, either "vanilla" or "ransac".
            "ransac" is slower with a better accuracy.
        blending_method: method to blend two images together.
            Only "naive" is currently supported.

    Note:
        Current implementation requires strict image ordering from left to right.

    .. code-block:: python

        IS = ImageStitcher(KF.LoFTR(pretrained='outdoor'), estimator='ransac').cuda()
        # Compute the stitched result with less GPU memory cost.
        with torch.inference_mode():
            out = IS(img_left, img_right)
        # Show the result
        plt.imshow(K.tensor_to_image(out))
    """

    def __init__(self, matcher: nn.Module, estimator: str = 'ransac', blending_method: str = "naive") -> None:
        super().__init__()
        self.matcher = matcher
        self.estimator = estimator
        self.blending_method = blending_method
        if estimator not in ['ransac', 'vanilla']:
            raise NotImplementedError(f"Unsupported estimator {estimator}. Use ‘ransac’ or ‘vanilla’ instead.")
        if estimator == "ransac":
            self.ransac = RANSAC('homography')

    def _estimate_homography(self, keypoints1: torch.Tensor, keypoints2: torch.Tensor) -> torch.Tensor:
        """Estimate homography by the matched keypoints.

        Args:
            keypoints1: matched keypoint set from an image, shaped as :math:`(N, 2)`.
            keypoints2: matched keypoint set from the other image, shaped as :math:`(N, 2)`.
        """
        homo: torch.Tensor
        if self.estimator == "vanilla":
            homo = find_homography_dlt_iterated(
                keypoints2[None], keypoints1[None], torch.ones_like(keypoints1[None, :, 0])
            )
        elif self.estimator == "ransac":
            homo, _ = self.ransac(keypoints2, keypoints1)
            homo = homo[None]
        else:
            raise NotImplementedError(f"Unsupported estimator {self.estimator}. Use ‘ransac’ or ‘vanilla’ instead.")
        return homo

    def estimate_transform(self, **kwargs) -> torch.Tensor:
        """Compute the corresponding homography."""
        homos: List[torch.Tensor] = []
        kp1, kp2, idx = kwargs['keypoints0'], kwargs['keypoints1'], kwargs['batch_indexes']
        for i in range(len(idx.unique())):
            homos.append(self._estimate_homography(kp1[idx == i], kp2[idx == i]))
        if len(homos) == 0:
            raise RuntimeError("Compute homography failed. No matched keypoints found.")
        return torch.cat(homos)

    def blend_image(self, src_img: torch.Tensor, dst_img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Blend two images together."""
        out: torch.Tensor
        if self.blending_method == "naive":
            out = torch.where(mask == 1, src_img, dst_img)
        else:
            raise NotImplementedError(f"Unsupported blending method {self.blending_method}. Use ‘naive’.")
        return out

    def preprocess(self, image_1: torch.Tensor, image_2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Preprocess input to the required format."""
        # TODO: probably perform histogram matching here.
        if isinstance(self.matcher, LoFTR) or isinstance(self.matcher, LocalFeatureMatcher):
            input_dict: Dict[str, torch.Tensor] = {  # LofTR works on grayscale images only
                "image0": image_1,
                "image1": image_2,
            }
        else:
            raise NotImplementedError(f"The preprocessor for {self.matcher} has not been implemented.")
        return input_dict

    def postprocess(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # NOTE: assumes no batch mode. This method keeps all valid regions after stitching.
        mask_: torch.Tensor = mask.sum((0, 1))
        index: int = int(mask_.bool().any(0).long().argmin().item())
        if index == 0:  # If no redundant space
            return image
        return image[..., :index]

    def on_matcher(self, data) -> dict:
        return self.matcher(data)

    def stitch_pair(
            self,
            images_left: torch.Tensor,
            images_right: torch.Tensor,
            mask_left: Optional[torch.Tensor] = None,
            mask_right: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute the transformed images
        input_dict: Dict[str, torch.Tensor] = self.preprocess(images_left, images_right)
        # out_shape: Tuple[int, int] = (images_left.shape[-2], images_left.shape[-1] + images_right.shape[-1])
        correspondences: dict = self.on_matcher(input_dict)
        homo: torch.Tensor = self.estimate_transform(**correspondences)
        src_img = warp_perspective(images_right, homo, images_right.shape[-2:], align_corners=True, mode='bicubic')
        # dst_img = torch.cat([images_left, torch.zeros_like(images_right)], dim=-1)

        # Compute the transformed masks
        # if mask_left is None:
        #     mask_left = torch.ones_like(images_left)
        # if mask_right is None:
        #     mask_right = torch.ones_like(images_right)
        # 'nearest' to ensure no floating points in the mask
        src_mask = warp_perspective(torch.ones_like(images_right), homo, images_right.shape[-2:], mode='nearest')
        # src_mask = warp_perspective(mask_right, homo, out_shape, mode='nearest')
        # dst_mask = torch.cat([mask_left, torch.zeros_like(mask_right)], dim=-1)
        return src_img, src_mask, homo

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.stitch_pair(imgs[:, [0]], imgs[:, [1]])
