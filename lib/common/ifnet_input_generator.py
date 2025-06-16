import open3d as o3d
import numpy as np
from typing import List


class IFNetsInputGenerator:
    def __init__(self, voxel_resolution=256, voxel_length=0.005, sdf_trunc=0.02):
        self.resolution = voxel_resolution
        self.voxel_length = voxel_length
        self.sdf_trunc = sdf_trunc

    def fuse_depth_to_mesh(self, depth_list: List[np.ndarray],
                           extrinsics: List[np.ndarray],
                           intrinsic: o3d.camera.PinholeCameraIntrinsic) -> o3d.geometry.TriangleMesh:

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_length,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor
        )

        for depth_np, extrinsic in zip(depth_list, extrinsics):
            depth_image = o3d.geometry.Image(depth_np.astype(np.float32))
            dummy_color = o3d.geometry.Image(np.full((*depth_np.shape, 3), 128, dtype=np.uint8))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                dummy_color, depth_image, convert_rgb_to_intensity=False)

            volume.integrate(rgbd, intrinsic, extrinsic.astype(np.float64))

        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        return mesh

    def voxelize_mesh(self, mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
            mesh,
            voxel_size=1.0 / self.resolution,
            min_bound=(-1, -1, -1),
            max_bound=(1, 1, 1)
        )

        voxels = np.zeros((self.resolution, self.resolution, self.resolution), dtype=np.float32)
        for voxel in voxel_grid.get_voxels():
            x, y, z = voxel.grid_index
            if 0 <= x < self.resolution and 0 <= y < self.resolution and 0 <= z < self.resolution:
                voxels[x, y, z] = 1.0
        return voxels

    def process(self, depth_list: List[np.ndarray], extrinsics: List[np.ndarray],
                intrinsic: o3d.camera.PinholeCameraIntrinsic) -> np.ndarray:
        fused_mesh = self.fuse_depth_to_mesh(depth_list, extrinsics, intrinsic)
        return self.voxelize_mesh(fused_mesh)
