# Meshes and batch transforms handled here
import sys
import pdb
import cv2
import math
import torch
import imageio
import trimesh
import numpy as np


HAND_LENGTH = 0.4  # 40cm
HAND_RADIUS = 0.04 # 4cm

def angle_axis_to_rotation_matrix(angle_axis: torch.Tensor) -> torch.Tensor:
    """Convert 3d vector of axis-angle rotation to 4x4 rotation matrix
    SOURCE:
    https://torchgeometry.readthedocs.io/en/latest/_modules/kornia/geometry/conversions.html#angle_axis_to_rotation_matrix

    Args:
        angle_axis (torch.Tensor): tensor of 3d vector of axis-angle rotations.

    Returns:
        torch.Tensor: tensor of 4x4 rotation matrices.

    Shape:
        - Input: :math:`(N, 3)`
        - Output: :math:`(N, 4, 4)`

    Example:
        >>> input = torch.rand(1, 3)  # Nx3
        >>> output = kornia.angle_axis_to_rotation_matrix(input)  # Nx4x4
    """

    def _compute_rotation_matrix(angle_axis, theta2, eps=1e-6):
        # We want to be careful to only evaluate the square root if the
        # norm of the angle_axis vector is greater than zero. Otherwise
        # we get a division by zero.
        k_one = 1.0
        theta = torch.sqrt(theta2)
        wxyz = angle_axis / (theta + eps)
        wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = torch.cat(
            [r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    def _compute_rotation_matrix_taylor(angle_axis):
        rx, ry, rz = torch.chunk(angle_axis, 3, dim=1)
        k_one = torch.ones_like(rx)
        rotation_matrix = torch.cat(
            [k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    # stolen from ceres/rotation.h

    _angle_axis = torch.unsqueeze(angle_axis, dim=1)
    theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
    theta2 = torch.squeeze(theta2, dim=1)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)

    # create mask to handle both cases
    eps = 1e-6
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = (mask).type_as(theta2)
    mask_neg = (mask == False).type_as(theta2)  # noqa

    # create output pose matrix
    batch_size = angle_axis.shape[0]
    rotation_matrix = torch.eye(4).to(angle_axis.device).type_as(angle_axis)
    rotation_matrix = rotation_matrix.view(1, 4, 4).repeat(batch_size, 1, 1)
    # fill output matrix with masked values
    rotation_matrix[..., :3, :3] = \
        mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    return rotation_matrix  # Nx4x4


def nr_fixed_transform(B=1, device='cpu'):
    """Flips axes to make it look proper after rendering since
    mesh coordinate system and NR systems are slightly different
        new x: left-right, right is positive
        new y: in depth, farther is positive
        new z: up-down, up is positive

    NOTE: camera azimuth measured from y-axis
    """
    T = torch.Tensor([[1, 0, 0, 0],
                      [0, 0, 1, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1]])
    # T = torch.eye(4)
    return T.unsqueeze(0).repeat(B, 1, 1).to(device)


def batch_apply_transform(vertices, transforms, device='cpu'):
    """Computes the transform given 3D mesh vertices and 4x4 transform matrix.
    vertices: BS x N x 3
    transforms: BS x 4 x 4
    """

    BS, N, _ = vertices.size()
    append_ones = torch.ones(BS, N, 1).to(device)
    vertices4 = torch.cat((vertices, append_ones), dim=2)
    tvertices = vertices4.bmm(transforms.transpose(1, 2))[:, :, :3]
    return tvertices


def batch_object_transform(obj_size, obj_pos, obj_rot_vec=None, device='cpu'):
    """Create 4D transform matrix for object, based on size and position.
    obj_size: BS x 3 dimensional for x,y,z
    obj_pos: BS x 2 dimensional for x,y,z
    obj_rot_vec: BS x 3 dimensional
    output: BS x 4x4 transformation matrix
    """

    B = obj_size.size(0)

    if obj_rot_vec is None:
        # create new transform matrix
        transform = torch.eye(4).unsqueeze(0).repeat(B, 1, 1).to(device)

    else:
        # initialize transformation matrix from rotation vector
        transform = angle_axis_to_rotation_matrix(obj_rot_vec)

    # set sizes by scaling diagonal elements
    osize4 = torch.cat((obj_size, torch.ones(B, 1).to(device)), dim=1).unsqueeze(2)  # B x 4 x 1
    osize44 = torch.diag_embed(osize4.squeeze(2))  # B x 4 x 4 with loaded diagonals
    transform = transform.bmm(osize44)
    # is almost the same as multiplying osize4 directly
    #transform *= osize4
    # set position by adding translation component
    transform[:, 0:3, 3] = obj_pos

    return transform


def batch_hand_transform(azimuth, elevation, hand_pos, hand_len=0.4, device='cpu'):
    """Create 4D transform matrix for hand based on angles and position
    azimuth, elevation: BS dimensional torch vectors
    hand_pos: BS x 3 dimensional torch matrices
    output: BS x 4x4 transformation matrix

    hand azimuth:
        0 -- pointing towards x-axis (right)
        90 -- pointing towards y-axis (far depth)
        -90 -- pointing towards negative y-axis (towards us)
    """

    B = azimuth.size(0)
    ty = math.pi/2 + elevation
    tz = azimuth
    tzeros = torch.zeros(B).to(device)
    tones = torch.ones(B).to(device)
    # rotation y-matrix: [[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]]
    ry = torch.stack([
            torch.stack([torch.cos(ty), tzeros.clone(), torch.sin(ty)], dim=1),
            torch.stack([tzeros.clone(), tones.clone(), tzeros.clone()], dim=1),
            torch.stack([-torch.sin(ty), tzeros.clone(), torch.cos(ty)], dim=1)
        ], dim=2).transpose(1, 2)

    # rotation z-matrix: [[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]]
    rz = torch.stack([
            torch.stack([torch.cos(tz), -torch.sin(tz), tzeros.clone()], dim=1),
            torch.stack([torch.sin(tz), torch.cos(tz), tzeros.clone()], dim=1),
            torch.stack([tzeros.clone(), tzeros.clone(), tones.clone()], dim=1)
        ], dim=2).transpose(1, 2)

    rot = rz.bmm(ry)

    # translation vectors for the batch
    hand_len_vec = - hand_len * tones.clone() / 2
    translate = rot.bmm(torch.stack([tzeros.clone(), tzeros.clone(), hand_len_vec], dim=1).unsqueeze(2)).squeeze(2)

    # create transform matrix
    transform = torch.eye(4).unsqueeze(0).repeat(B, 1, 1).to(device)
    transform[:, :3, :3] = rot
    transform[:, 0:3, 3] = translate + hand_pos
    return transform


def mesh_to_tensor(mesh, B, device='cpu'):
    """Convert mesh to torch tensors
    vertices: B x N x 3 (3D points)
    faces: B x M x 3 (3D object faces)
    """

    # convert vertices to torch tensor, repeat B times
    vertices = np.array(mesh.vertices)[None, :, :].astype('float32')
    vertices = torch.from_numpy(vertices).repeat(B, 1, 1).to(device)
    # convert faces to torch tensor, repeat B times
    faces = np.array(mesh.faces)[None, :, :].astype('float32')
    faces = torch.from_numpy(faces).repeat(B, 1, 1).to(device)

    return vertices, faces


def create_hand_mesh(radius=0.04, length=0.40):
    """Hand mesh
    radius = 4cm cylinder radius
    length = 40cm hand length
             15cm hand wrist
    """
    return trimesh.creation.cylinder(radius=radius, height=length)


def create_object_mesh(x=1., y=1., z=1.):
    """Cuboid object mesh
    Size in m (x, y, z) coordinates
    """
    return trimesh.creation.box(extents=(x, y, z))


def proc_im(torch_images):
    """Converts input BS x H x W into a list of numpy images to visualize
    """
    im_list = []
    np_images = torch_images.detach().cpu().numpy()
    for im in np_images:
        if np.ndim(im) == 3:
            im_list.append(np.ascontiguousarray((255 * im.transpose(1, 2, 0)).astype(np.uint8)))
        else:
            im_list.append(np.ascontiguousarray((255 * im).astype(np.uint8)))
    return im_list


def render(renderer, vertices, faces, silhouette=True):
    """Create images given vertices and faces
    Assumes no textures

    Outputs:
    images: BS x H x W
    vis_image_list: [H x W, ..., H x W]
    """

    # Render
    if silhouette:
        images = renderer(vertices, faces, mode='silhouettes')
        # [batch_size, image_size, image_size]

    else:
        # create texture [batch_size=1, num_faces, texture_size, texture_size, texture_size, RGB]
        texture_size = 2
        B, N, _ = faces.size()
        textures = torch.ones(B, N, texture_size, texture_size, texture_size, 3, dtype=torch.float32).cuda()
        images, _, _ = renderer(vertices, faces, textures)
        # [batch_size, RGB, image_size, image_size]

    # generate image list for visualization
    vis_image_list = proc_im(images)

    return images, vis_image_list


def save_image(images, fnames):
    """Save a list of images as JPG filenames
    """

    for im, fn in zip(images, fnames):
        cv2.imwrite(fn, im)


if __name__ == '__main__':
    sys.path.append('..')
    from utils import sth_dataset

    device = 'cuda:0'

    # Initialize a renderer
    import neural_renderer as nr

    camera_distance = 2  # metre :/
    camera_elevation = 30
    camera_azimuth = 0
    renderer = nr.Renderer(camera_mode='look_at', perspective=True, image_size=512)
    # renderer eye default position
    renderer.eye = nr.get_points_from_angles(camera_distance, camera_elevation, 0)

    # Hand transformation
    B = 50
    nrT = nr_fixed_transform(B, device)
    print('Hand in and out')
    hm = create_hand_mesh()
    hv, hf = mesh_to_tensor(hm, B, device)
    # renderer eye position for the actual real2sim
    cdis = torch.Tensor([camera_distance])
    cazi = torch.Tensor([camera_azimuth])
    cele = torch.Tensor([camera_elevation])
    renderer.eye = torch.cat([ cdis * torch.cos(cele) * torch.sin(cazi),
                               cdis * torch.sin(cele),
                              -cdis * torch.cos(cele) * torch.cos(cazi)])
    hazi = torch.ones(B) * 60
    hele = torch.ones(B) * 30
    # hpos as hand enters in for first 25 steps
    hpos_x = torch.cat([-1*torch.ones(5),
                        torch.linspace(-1, 0, steps=20),
                        torch.linspace(0, 0, steps=5),
                        torch.linspace(0, -1, steps=10),
                        -1*torch.ones(10)], dim=0)
    hpos_y = torch.cat([0.2*torch.ones(5),
                        torch.linspace(0.2, 0.2, steps=20),
                        torch.linspace(0.2, 0, steps=5),
                        torch.linspace(0, 0, steps=10),
                        0*torch.ones(10)], dim=0)
    hpos = torch.stack((hpos_x, hpos_y, 0.2*torch.ones(B)), dim=1)
    # hpos as hand leaves the scene for last 25 steps
    hT = batch_hand_transform(hazi.to(device), hele.to(device), hpos.to(device), device=device)
    hvT = batch_apply_transform(hv, hT, device)
    hvT = batch_apply_transform(hvT, nrT, device)
    # render some images
    _, th_images = render(renderer, hvT, hf)
    sth_dataset.save_gif(th_images, 'hand_inout.gif', (240,360))
    sys.exit(0)

    # Example hand modification
    B = 10
    nrT = nr_fixed_transform(B, device)
    print('Hand transforms')
    hm = create_hand_mesh()
    hv, hf = mesh_to_tensor(hm, B, device)
    hazi = torch.arange(0, B) * 2 * math.pi / B  # B steps full circle rotation
    hele = torch.ones(B) * (math.pi * 30. / 180)
    hpos = torch.ones(B, 3) * torch.Tensor([0, 0.5, 0.2])
    # hpos = torch.ones(B, 3) * torch.Tensor([-1.5, 1.5, 1.5])
    hT = batch_hand_transform(hazi.to(device), hele.to(device), hpos.to(device), device=device)
    hvT = batch_apply_transform(hv, hT, device)
    hvT = batch_apply_transform(hvT, nrT, device)
    # render some images
    _, th_images = render(renderer, hvT, hf)
    sth_dataset.save_gif(th_images, 'th.gif', (240,360))

    # Example of object modification
    print('Object transforms')
    om = create_object_mesh()
    ov, of = mesh_to_tensor(om, B, device)
    # osize = torch.ones(B, 3) * torch.Tensor([0.1, 0.1, 0.4])
    # opos = torch.ones(B, 2) * torch.Tensor([-1.2, 1.2])
    osize = torch.ones(B, 3) * torch.Tensor([0.2, 0.5, 0.1])  # object size
    opos = torch.ones(B, 2) * torch.Tensor([0.4, 0.3])  # xy translation
    opos *= torch.arange(0, 1, 1./B).unsqueeze(1) # B steps to reach point [0.4, 0.3]
    oT = batch_object_transform(osize.to(device), opos.to(device), device)
    ovT = batch_apply_transform(ov, oT, device)
    ovT = batch_apply_transform(ovT, nrT, device)
    # render some images
    _, to_images = render(renderer, ovT, of)
    sth_dataset.save_gif(to_images, 'to.gif', (240,360))

