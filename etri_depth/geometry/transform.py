import numpy as np
import torch


def transformation_from_parameters(angle_params, transl_params):
    """Convert the network's (angle_params, transl_params) output into a 4x4 matrix"""
    #
    R = rot_from_axisangle(angle_params)
    # R = euler2mat(angle_params)
    # R = exp_map(angle_params)

    #
    transl_params = transl_params.clamp(-3.0, 3.0)

    # Bx4x4
    M = torch.zeros(
        (transl_params.shape[0], 4, 4),
        dtype=transl_params.dtype,
        device=transl_params.device,
    )
    M[:, 3, 3] = 1
    M[:, :3, 3] = transl_params
    M[:, :3, :3] = R

    return M


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 3x3 rot matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx3
    """
    angle = torch.norm(vec, 2, dim=1)
    angle = angle.clamp(max=np.pi / 4)
    axis = vec / (angle.unsqueeze(1) + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[:, 0]
    y = axis[:, 1]
    z = axis[:, 2]

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 3, 3), dtype=vec.dtype, device=vec.device)

    rot[:, 0, 0] = x * xC + ca
    rot[:, 0, 1] = xyC - zs
    rot[:, 0, 2] = zxC + ys
    rot[:, 1, 0] = xyC + zs
    rot[:, 1, 1] = y * yC + ca
    rot[:, 1, 2] = yzC - xs
    rot[:, 2, 0] = zxC - ys
    rot[:, 2, 1] = yzC + xs
    rot[:, 2, 2] = z * zC + ca

    return rot


def euler2mat(vec: torch.Tensor):
    """
    Converts euler angles to rotation matrix
    Args:
        vec: rotation angle (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """

    # vec = vec.clamp(-np.pi, np.pi)
    vec = vec.clamp(-np.pi / 4, np.pi / 4)

    x = vec[:, 0]
    y = vec[:, 1]
    z = vec[:, 2]

    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    # Construct RotationMatrix
    rot_x = torch.stack(
        [
            torch.stack([ones, zeros, zeros], axis=-1),
            torch.stack([zeros, x.cos(), -x.sin()], axis=-1),
            torch.stack([zeros, x.sin(), x.cos()], axis=-1),
        ],
        axis=1,
    )

    rot_y = torch.stack(
        [
            torch.stack([y.cos(), zeros, y.sin()], axis=-1),
            torch.stack([zeros, ones, zeros], axis=-1),
            torch.stack([-y.sin(), zeros, y.cos()], axis=-1),
        ],
        axis=1,
    )

    rot_z = torch.stack(
        [
            torch.stack([z.cos(), -z.sin(), zeros], axis=-1),
            torch.stack([z.sin(), z.cos(), zeros], axis=-1),
            torch.stack([zeros, zeros, ones], axis=-1),
        ],
        axis=1,
    )
    return rot_x.bmm(rot_y).bmm(rot_z)
