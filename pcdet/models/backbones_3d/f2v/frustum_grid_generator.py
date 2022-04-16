import torch
import torch.nn as nn
import kornia

from pcdet.utils import transform_utils, grid_utils, depth_utils


class FrustumGridGenerator(nn.Module):

    def __init__(self, grid_size, pc_range, disc_cfg):
        """
        Initializes Grid Generator for frustum features
        Args:
            grid_size [np.array(3)]: Voxel grid shape [X, Y, Z]
            pc_range [list]: Voxelization point cloud range [X_min, Y_min, Z_min, X_max, Y_max, Z_max]
            disc_cfg [int]: Depth discretiziation configuration
        """
        super().__init__()
        self.dtype = torch.float32
        self.grid_size = torch.as_tensor(grid_size)
        self.pc_range = pc_range
        self.out_of_bounds_val = -2
        self.disc_cfg = disc_cfg

        # Calculate voxel size
        pc_range = torch.as_tensor(pc_range).reshape(2, 3)
        self.pc_min = pc_range[0]
        self.pc_max = pc_range[1]
        self.voxel_size = (self.pc_max - self.pc_min) / self.grid_size

        # Create voxel grid
        # self.depth, self.width, self.height = self.grid_size.int()
        self.width, self.depth, self.height = self.grid_size.int()
        self.voxel_grid = kornia.utils.create_meshgrid3d(depth=self.depth,
                                                         height=self.height,
                                                         width=self.width,
                                                         normalized_coordinates=False)

        # self.voxel_grid = self.voxel_grid.permute(0, 1, 3, 2, 4)  # XZY-> XYZ
        self.voxel_grid = self.voxel_grid.permute(0, 3, 1, 2, 4)  # depth 1Y height 2Z width 3X-> XYZ
        print ('voxel_grid ', self.voxel_grid.shape)

        # Add offsets to center of voxel
        self.voxel_grid += 0.5
        self.grid_to_lidar = self.grid_to_lidar_unproject(pc_min=self.pc_min,
                                                          voxel_size=self.voxel_size)

    def grid_to_lidar_unproject(self, pc_min, voxel_size):
        """
        Calculate grid to LiDAR unprojection for each plane
        Args:
            pc_min [torch.Tensor(3)]: Minimum of point cloud range [X, Y, Z] (m)
            voxel_size [torch.Tensor(3)]: Size of each voxel [X, Y, Z] (m)
        Returns:
            unproject [torch.Tensor(4, 4)]: Voxel grid to LiDAR unprojection matrix
        """
        x_size, y_size, z_size = voxel_size
        x_min, y_min, z_min = pc_min
        unproject = torch.tensor([[x_size, 0, 0, x_min],
                                  [0, y_size, 0, y_min],
                                  [0,  0, z_size, z_min],
                                  [0,  0, 0, 1]],
                                 dtype=self.dtype)  # (4, 4)

        return unproject

    def transform_grid(self, voxel_grid, grid_to_lidar, lidar_to_cam, cam_to_img):
        """
        Transforms voxel sampling grid into frustum sampling grid
        Args:
            grid [torch.Tensor(B, X, Y, Z, 3)]: Voxel sampling grid
            grid_to_lidar [torch.Tensor(4, 4)]: Voxel grid to LiDAR unprojection matrix
            lidar_to_cam [torch.Tensor(B, 4, 4)]: LiDAR to camera frame transformation
            cam_to_img [torch.Tensor(B, 3, 4)]: Camera projection matrix
        Returns:
            frustum_grid [torch.Tensor(B, X, Y, Z, 3)]: Frustum sampling grid
        """
        B = lidar_to_cam.shape[0]

        # Create transformation matricies
        V_G = grid_to_lidar  # Voxel Grid -> LiDAR (4, 4)
        C_V = lidar_to_cam  # LiDAR -> Camera (B, 4, 4)
        I_C = cam_to_img  # Camera -> Image (B, 3, 4)
        trans = C_V @ V_G

        # pointVeh = (0, 50, 0, 1)
        # pointCam = lidar_to_cam.cpu().numpy()[0] @ pointVeh
        # cam_intrinsic = cam_to_img.cpu().numpy()[0]
        # print (cam_intrinsic)
        # pointuv = cam_intrinsic @ pointCam
        # print ('pointVeh ', pointVeh)
        # print ('pointCam ', pointCam)
        # print (pointuv / pointuv[2])
        # print (pointuv)

        # Reshape to match dimensions
        trans = trans.reshape(B, 1, 1, 4, 4)
        voxel_grid = voxel_grid.repeat_interleave(repeats=B, dim=0)

        # Transform to camera frame
        camera_grid = kornia.transform_points(trans_01=trans, points_1=voxel_grid)

        # Project to image
        I_C = I_C.reshape(B, 1, 1, 3, 4)
        image_grid, image_depths = transform_utils.project_to_image(project=I_C, points=camera_grid)
        print ('image_grid.shape ', image_grid.shape)

        # Convert depths to depth bins
        image_depths = depth_utils.bin_depths(depth_map=image_depths, **self.disc_cfg)
        print ('image_depths.shape ', image_depths.shape)
        # Stack to form frustum grid
        image_depths = image_depths.unsqueeze(-1)
        frustum_grid = torch.cat((image_grid, image_depths), dim=-1)
        print ('frustum_grid.shape ', frustum_grid.shape)
        return frustum_grid

    def forward(self, lidar_to_cam, cam_to_img, image_shape):
        """
        Generates sampling grid for frustum features
        Args:
            lidar_to_cam [torch.Tensor(B, 4, 4)]: LiDAR to camera frame transformation
            cam_to_img [torch.Tensor(B, 3, 4)]: Camera projection matrix
            image_shape [torch.Tensor(B, 2)]: Image shape [H, W]
        Returns:
            frustum_grid [torch.Tensor(B, X, Y, Z, 3)]: Sampling grids for frustum features
        """

        frustum_grid = self.transform_grid(voxel_grid=self.voxel_grid.to(lidar_to_cam.device),
                                           grid_to_lidar=self.grid_to_lidar.to(lidar_to_cam.device),
                                           lidar_to_cam=lidar_to_cam,
                                           cam_to_img=cam_to_img)

        # Normalize grid
        image_shape, _ = torch.max(image_shape, dim=0)
        # print ('image_shape ', image_shape)
        image_depth = torch.tensor([self.disc_cfg["num_bins"]], device=image_shape.device, dtype=image_shape.dtype)
        frustum_shape = torch.cat((image_depth, image_shape))
        # print ('frustum_shape ', frustum_shape)
        # print ('frustum_grid1 ', frustum_grid.shape)
        frustum_grid = grid_utils.normalize_coords(coords=frustum_grid, shape=frustum_shape)
        # print ('frustum_grid2 ', frustum_grid)

        # Replace any NaNs or infinites with out of bounds
        mask = ~torch.isfinite(frustum_grid)
        frustum_grid[mask] = self.out_of_bounds_val
        # print ('frustum_grid3 ', frustum_grid.shape)
        return frustum_grid
