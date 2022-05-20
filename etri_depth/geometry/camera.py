import torch
import torch.nn.functional as F
from torch import nn


def pixel2cam(depth, inv_K):
    B, _, H, W = depth.shape
    depth_flat = depth.view(B, 1, -1)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )

    pix_coords = torch.stack([grid_x.flatten(), grid_y.flatten(), torch.ones(H * W)], 0).to(
        depth.device
    )

    cam_points = torch.matmul(inv_K, pix_coords)
    cam_points = depth_flat * cam_points
    cam_points = torch.cat([cam_points, torch.ones_like(depth_flat)], 1)
    cam_points = cam_points.view(B, 4, H, W)

    return cam_points


def cam2pixel(points, K, T, eps=1e-7):
    B, _, H, W = points.shape

    P = torch.matmul(K, T[:, :3, :])  # [B, 3, 4]
    pix_points = torch.matmul(P, points.view(B, 4, -1))  # [B, 3, H*W]

    Z = pix_points[:, 2].clamp(min=eps)
    X = pix_points[:, 0] / Z
    Y = pix_points[:, 1] / Z

    # Normalize to [-1, 1]
    X_norm = 2 * (X / (W - 1)) - 1
    Y_norm = 2 * (Y / (H - 1)) - 1

    pix_grid = torch.stack([X_norm, Y_norm], dim=-1)
    pix_grid = pix_grid.reshape(B, H, W, 2)
    return pix_grid, Z.reshape(B, 1, H, W)


class GridSampleWithoutExtrinsic(nn.Module):
    """Synthesize new view for the same camera (no extrinsic)"""

    def __init__(self, tgtH: int, tgtW: int, tgtK: list[float]):
        super().__init__()

        # Prepare pix_coords [3, tgtH*tgtW]
        grid_y, grid_x = torch.meshgrid(
            torch.arange(tgtH, dtype=torch.float32),
            torch.arange(tgtW, dtype=torch.float32),
            indexing="ij",
        )

        pix_coords = torch.stack([grid_x.flatten(), grid_y.flatten(), torch.ones(tgtH * tgtW)], 0)
        self.register_buffer("pix_coords", pix_coords, persistent=False)

        # Prepare tgt_invK [3, 3]
        tgt_invK = torch.tensor(tgtK, dtype=torch.float32).reshape(3, 3).inverse()
        self.register_buffer("tgt_invK", tgt_invK, persistent=False)

        #
        self.tgtH = tgtH
        self.tgtW = tgtW

    def forward(
        self,
        src_img: torch.Tensor,
        srcK: torch.Tensor,  # [B, 3, 3]
    ):
        # [B, 3, tgtH*tgtW]
        pix_points = (srcK @ self.tgt_invK) @ self.pix_coords

        Z = pix_points[:, 2].clamp(min=1e-7)
        X = pix_points[:, 0] / Z
        Y = pix_points[:, 1] / Z
        # Normalize to [-1, 1]
        B, _, srcH, srcW = src_img.shape
        X_norm = 2 * (X / (srcW - 1)) - 1
        Y_norm = 2 * (Y / (srcH - 1)) - 1

        pix_grid = torch.stack([X_norm, Y_norm], dim=-1)
        pix_grid = pix_grid.reshape(B, self.tgtH, self.tgtW, 2)

        return F.grid_sample(
            src_img,
            pix_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )


def grid_sample_with_extrinsic(
    tgt_depth: torch.Tensor,  # [B, 3, tgtH, tgtW]
    tgt_invK: torch.Tensor,  # [B, 3, 3]
    src_img: torch.Tensor,  # [B, 3, srcH, srcW]
    srcK: torch.Tensor,  # [B, 3, 3]
    T: torch.Tensor,  # [B, 4, 4]
):
    B, _, tgtH, tgtW = tgt_depth.shape
    B, _, srcH, srcW = src_img.shape
    depth_flat = tgt_depth.view(B, 1, -1)  # [B, 1, tgtH*tgtW]

    grid_y, grid_x = torch.meshgrid(
        torch.arange(tgtH, dtype=torch.float32),
        torch.arange(tgtW, dtype=torch.float32),
        indexing="ij",
    )

    pix_coords = torch.stack([grid_x.flatten(), grid_y.flatten(), torch.ones(tgtH * tgtW)], 0).to(
        tgt_depth.device
    )  # [3, tgtH*tgtW]

    # pixel2cam
    cam_points = tgt_invK @ pix_coords  # [B, 3, tgtH*tgtW]
    cam_points = depth_flat * cam_points  # [B, 3, tgtH*tgtW]
    cam_points = torch.cat([cam_points, torch.ones_like(depth_flat)], 1)  # [B, 4, tgtH*tgtW]

    # cam2cam, cam2pixel
    P = torch.bmm(srcK, T[:, :3, :])  # [B, 3, 4]
    pix_points = torch.bmm(P, cam_points)  # [B, 3, tgtH*tgtW]

    Z = pix_points[:, 2].clamp(min=1e-7)
    X = pix_points[:, 0] / Z  # [B, tgtH*tgtW]
    Y = pix_points[:, 1] / Z  # [B, tgtH*tgtW]

    # Normalize to [-1, 1]
    X_norm = 2 * (X / (srcW - 1)) - 1
    Y_norm = 2 * (Y / (srcH - 1)) - 1

    # grid_sample
    pix_grid = torch.stack([X_norm, Y_norm], dim=-1)  # [B, tgtH*tgtW, 2]
    pix_grid = pix_grid.reshape(B, tgtH, tgtW, 2)  # [B, tgtH, tgtW, 2]

    return F.grid_sample(
        src_img,
        pix_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
