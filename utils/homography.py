from dataclasses import dataclass

import kornia.feature as kf
import torch
from einops import rearrange, repeat
from kornia.geometry import warp_perspective
from torch import Tensor
from torch.nn import Module

from utils.image_stitcher import ImageStitcher
POISSON_RATE = torch.ones(1)


@dataclass(frozen=False)
class HomographyResult:
    warped: Tensor
    mask: Tensor = None
    homography: Tensor = None
    shift: Tensor = None
    src: Tensor = None
    dest: Tensor = None

    def __post_init__(self):
        if self.homography is None:
            self.homography = torch.eye(3, device=self.warped.device, dtype=self.warped.dtype).unsqueeze(0)
            self.homography = repeat(self.homography, '1 hh wh -> b hh wh', b=self.warped.shape[0])
        if self.mask is None:
            self.mask = torch.ones_like(self.warped)
        self.mask = self.mask.bool()

    def cpu(self):
        self.warped = self.warped.cpu() if isinstance(self.warped, Tensor) else self.warped
        self.mask = self.mask.cpu() if isinstance(self.mask, Tensor) else self.mask
        self.homography = self.homography.cpu() if isinstance(self.homography, Tensor) else self.homography
        self.src = self.src.cpu() if isinstance(self.src, Tensor) else self.src
        self.dest = self.dest.cpu() if isinstance(self.dest, Tensor) else self.dest
        self.shift = self.shift.cpu() if isinstance(self.shift, Tensor) else self.shift
        return self


def exclude_reference_frame(burst) -> Tensor:
    reference_idx = burst.shape[1] // 2
    return torch.cat((burst[:, :reference_idx], burst[:, reference_idx + 1:]), dim=1)


class AbstractHomography(Module):
    def __init__(self):
        super(AbstractHomography, self).__init__()


class DummyHomography(AbstractHomography):
    def __init__(self) -> None:
        super(DummyHomography, self).__init__()

    @staticmethod
    def forward(x: Tensor, m: Tensor, mask: Tensor) -> HomographyResult:
        results = HomographyResult(homography=m, warped=x, mask=mask)
        return results


class HomographyGroundTruth(AbstractHomography):
    def __init__(
            self, *, limit_shift: int, limit_perspective,
            limit_rotation, with_reference_frame: bool, sigma: float = 0.5,
            poisson_rate: float = 0.0):
        super(HomographyGroundTruth, self).__init__()
        assert limit_shift >= 0, f'shift_limit must be >= 0, got {limit_shift}'
        self._with_reference_frame = with_reference_frame
        self._shifts = limit_shift
        self._perspective = limit_perspective
        self._rotation = limit_rotation
        self._poisson_rate = poisson_rate

    @torch.no_grad()
    def forward(self, x: Tensor, m: Tensor, mask: Tensor, mid_idx: Tensor) -> HomographyResult:
        n_batch, n_frames = x.shape[:2]
        x_ = rearrange(x, 'batch frames h w -> (batch frames) 1 h w')
        m_ = rearrange(m, 'batch frames h w -> (batch frames) h w')

        # create a 2-axis random shift for each frame
        if self._shifts != 0:
            random_shift = torch.randint(low=-self._shifts, high=self._shifts+1,
                                         size=(n_batch * n_frames, 2), device=m_.device, dtype=m_.dtype)
        else:
            random_shift = torch.zeros((n_batch * n_frames, 2), device=x.device, dtype=x.dtype)
        idx = random_shift + self._shifts

        # render some frames invalid by adding a huge shift
        if self._poisson_rate > 0:
            rate = torch.poisson(input=self._poisson_rate) / n_frames
            mask_invalid_frames = torch.rand_like(random_shift) < rate
            random_shift[mask_invalid_frames] = 16

        # Add perspective shift to the homography
        perspective_shift = torch.zeros_like(random_shift)
        if self._perspective:
            perspective_shift = torch.rand((n_batch * n_frames, 2), device=m_.device, dtype=m_.dtype)
            perspective_shift = perspective_shift * 2 - 1
            perspective_shift *= self._perspective

        # Add perspective shift to the homography
        rotations = torch.zeros_like(random_shift)
        if self._rotation:
            raise NotImplementedError

        # remove the shift in the reference frame
        if self._with_reference_frame:
            remove_shift_in_ref = torch.ones((x.shape[0], x.shape[1], 2), device=m.device, dtype=m.dtype)
            remove_shift_in_ref[:, n_frames // 2, :] = 0
            remove_shift_in_ref = rearrange(remove_shift_in_ref, 'batch frames s -> (batch frames) s', s=2)
            random_shift *= remove_shift_in_ref
            perspective_shift *= remove_shift_in_ref

        # apply the shift
        m_[:, :2, 2] += random_shift
        m_[:, 2, :2] += perspective_shift

        # warp the frames
        warped = warp_perspective(x_, m_, (x.shape[-2], x.shape[-1]), mode='bicubic', align_corners=True)
        warped = rearrange(warped, '(batch frames) 1 h w -> batch frames h w', frames=x.shape[1])
        mask_warped = warp_perspective(torch.ones_like(x_), m_, (x.shape[-2], x.shape[-1]), mode='nearest')
        mask_warped = rearrange(mask_warped, '(batch frames) 1 h w -> batch frames h w', frames=x.shape[1])

        # clip negative values
        mask_warped = mask_warped >= 0.5
        warped = torch.clamp(warped * mask_warped, 0, 1)
        return HomographyResult(homography=m, warped=warped, mask=mask_warped, shift=idx)


class HomographySIFT(AbstractHomography):
    def __init__(self):
        super(HomographySIFT, self).__init__()
        self._model = ImageStitcher(matcher=kf.LocalFeatureMatcher(
            kf.GFTTAffNetHardNet(),  # KF.KeyNetAffNetHardNet(),
            # kf.SIFTFeature(num_features=256),  # KF.KeyNetAffNetHardNet(),
            kf.DescriptorMatcher('mnn', 0.9)),
            estimator='vanilla')  # 'vanilla' | 'ransac'

    def forward(self, x: Tensor, m: Tensor, mask: Tensor, verbose: bool = True) -> HomographyResult:
        warped, mask, homography = self._model(x)
        return HomographyResult(src=x[:, 0], dest=x[:, 1],
                                homography=homography, warped=warped[0], mask=mask[0], shift=None)
