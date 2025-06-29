import xatlas
import numpy as np
from PIL import Image
import cv2
import trimesh
from termcolor import colored

class TextureGenerator:
    """
    A callable class for generating texture maps for reconstructed meshes by projecting
    colors from multiple input views.
    """
    
    def __init__(self, texture_size=2048, debug=False, debug_dir=None):
        """
        Initialize the TextureGenerator.
        
        Args:
            texture_size (int): The resolution of the output texture map (e.g., 2048x2048).
            debug (bool): Whether to save intermediate debugging outputs.
            debug_dir (str): Directory to save debug outputs. If None, uses out_dir/debug_textures.
        """
        self.texture_size = texture_size
        self.debug = debug
        self.debug_dir = debug_dir
    
    def __call__(self, final_mesh, multi_view_data, transform_manager, out_dir, subject_name):
        """
        Generate a texture map for the final reconstructed mesh.

        Args:
            final_mesh (trimesh.Trimesh): The final watertight mesh.
            multi_view_data (dict): Dictionary containing data for all views,
                                    including image paths, masks, and intrinsics.
            transform_manager (TransformManager): Object to get camera extrinsics.
            out_dir (str): The main output directory.
            subject_name (str): The name of the subject for file naming.
            
        Returns:
            trimesh.Trimesh: The textured mesh with UV coordinates.
        """
        print(colored("\n🎨 Starting Multi-view Texture Generation...", "cyan"))
        
        # Setup debug directory
        if self.debug:
            if self.debug_dir is None:
                self.debug_dir = f"{out_dir}/debug_textures"
            import os
            os.makedirs(self.debug_dir, exist_ok=True)
            print(colored(f"🐛 Debug mode enabled. Saving intermediate outputs to: {self.debug_dir}", "yellow"))
        
        # Step 1: UV Unwrapping
        uv_mesh = self._unwrap_mesh(final_mesh, subject_name)
        
        # Step 2: Rasterize UVs to get per-pixel 3D information
        pixel_to_3d_point, pixel_to_3d_normal, valid_pixel_mask = self._rasterize_uvs(uv_mesh, subject_name)
        
        # Step 3: Project and blend colors from all views
        texture_image_np, total_weight = self._project_and_blend_colors(
            uv_mesh, pixel_to_3d_point, pixel_to_3d_normal, valid_pixel_mask,
            multi_view_data, transform_manager, final_mesh, subject_name
        )
        
        # Step 4: Inpaint missing areas
        inpainted_texture = self._inpaint_texture(texture_image_np, total_weight, subject_name)
        
        # Step 5: Save textured mesh
        textured_mesh = self._save_textured_mesh(uv_mesh, inpainted_texture, out_dir, subject_name)
        
        return textured_mesh
    
    def _unwrap_mesh(self, final_mesh, subject_name):
        """UV unwrap the mesh using xatlas."""
        print("1. Unwrapping mesh with xatlas...")
        
        if self.debug:
            # Save original mesh
            debug_original_path = f"{self.debug_dir}/{subject_name}_01_original_mesh.obj"
            final_mesh.export(debug_original_path)
            print(f"🐛 Saved original mesh: {debug_original_path}")
        
        atlas = xatlas.Atlas()
        atlas.add_mesh(final_mesh.vertices, final_mesh.faces)
        chart_options = xatlas.ChartOptions()
        pack_options = xatlas.PackOptions()
        pack_options.resolution = self.texture_size
        atlas.generate(chart_options=chart_options, pack_options=pack_options)
        
        # Get the new mesh data with UV coordinates
        vmapping, indices, uvs = atlas[0]  # Get first (and only) mesh from atlas
        
        print(f"📊 UV Unwrapping stats:")
        print(f"   Original vertices: {len(final_mesh.vertices)}")
        print(f"   UV mapped vertices: {len(vmapping)}")
        print(f"   UV faces: {len(indices)}")
        print(f"   UV coordinate range: [{uvs.min():.3f}, {uvs.max():.3f}]")
        
        # Create mesh with UV coordinates
        uv_mesh = trimesh.Trimesh(
            vertices=final_mesh.vertices[vmapping], 
            faces=indices, 
            process=False
        )
        
        # Set UV coordinates properly
        uv_mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
        
        if self.debug:
            # Save UV unwrapped mesh
            debug_uv_path = f"{self.debug_dir}/{subject_name}_02_uv_mesh.obj"
            uv_mesh.export(debug_uv_path)
            print(f"🐛 Saved UV unwrapped mesh: {debug_uv_path}")
            
            # Save UV layout visualization
            self._save_uv_debug_image(uvs, indices, subject_name, "02_uv_layout")
        
        return uv_mesh
    
    def _rasterize_uvs(self, uv_mesh, subject_name):
        """Rasterize UV layout to create pixel-to-3D-point map."""
        print("2. Rasterizing UV layout to create pixel-to-3D-point map...")
        
        # Get the 3D position and normal for each pixel in the texture map
        pixel_coords = np.indices((self.texture_size, self.texture_size)).transpose(1, 2, 0)[:, :, ::-1]
        pixel_coords = (pixel_coords + 0.5) / self.texture_size # Normalize to [0, 1]
        
        # Get UV coordinates from the mesh
        if hasattr(uv_mesh.visual, 'uv') and uv_mesh.visual.uv is not None:
            mesh_uvs = uv_mesh.visual.uv
            print(f"✅ Found UV coordinates: {len(mesh_uvs)} vertices")
        else:
            # Fallback: create simple UV coordinates if not available
            mesh_uvs = np.random.rand(len(uv_mesh.vertices), 2)
            print("⚠️ Warning: No UV coordinates found, using random UV mapping")
        
        # Find the 3D position and normal for each pixel in the texture
        pixel_to_3d_point, pixel_to_3d_normal = self._interpolate_from_uv(
            uv_mesh, mesh_uvs, pixel_coords.reshape(-1, 2)
        )
        pixel_to_3d_point = pixel_to_3d_point.reshape(self.texture_size, self.texture_size, 3)
        pixel_to_3d_normal = pixel_to_3d_normal.reshape(self.texture_size, self.texture_size, 3)

        # Create a mask of valid pixels (where a 3D point was found)
        valid_pixel_mask = ~np.isnan(pixel_to_3d_point).any(axis=-1)
        
        print(f"📊 Rasterization stats:")
        print(f"   Total texture pixels: {self.texture_size * self.texture_size}")
        print(f"   Valid pixels: {valid_pixel_mask.sum()}")
        print(f"   Coverage: {valid_pixel_mask.sum() / (self.texture_size * self.texture_size) * 100:.1f}%")
        
        if self.debug:
            # Save pixel validity mask
            validity_image = (valid_pixel_mask * 255).astype(np.uint8)
            debug_mask_path = f"{self.debug_dir}/{subject_name}_03_valid_pixels.png"
            cv2.imwrite(debug_mask_path, validity_image)
            print(f"🐛 Saved validity mask: {debug_mask_path}")
            
            # Save 3D position visualization (depth map)
            if valid_pixel_mask.sum() > 0:
                valid_depths = pixel_to_3d_point[valid_pixel_mask][:, 2]  # Z-coordinates
                depth_image = np.zeros((self.texture_size, self.texture_size))
                depth_image[valid_pixel_mask] = valid_depths
                
                # Normalize to 0-255 range
                if valid_depths.max() > valid_depths.min():
                    depth_normalized = (depth_image - valid_depths.min()) / (valid_depths.max() - valid_depths.min())
                    depth_normalized = (depth_normalized * 255).astype(np.uint8)
                    debug_depth_path = f"{self.debug_dir}/{subject_name}_03_depth_map.png"
                    cv2.imwrite(debug_depth_path, depth_normalized)
                    print(f"🐛 Saved depth map: {debug_depth_path}")
        
        return pixel_to_3d_point, pixel_to_3d_normal, valid_pixel_mask
    
    def _interpolate_from_uv(self, uv_mesh, mesh_uvs, query_uvs):
        """
        Interpolate 3D positions and normals from UV coordinates.
        
        Args:
            uv_mesh (trimesh.Trimesh): The 3D mesh
            mesh_uvs (np.ndarray): UV coordinates for each vertex (N, 2)
            query_uvs (np.ndarray): UV coordinates to query (M, 2)
            
        Returns:
            tuple: (positions, normals) both with shape (M, 3)
        """
        try:
            # Create a 2D triangulation in UV space
            from scipy.spatial import Delaunay
            
            # Build triangulation in UV space
            triangulation = Delaunay(mesh_uvs)
            
            # Find which triangle each query point belongs to
            simplex_indices = triangulation.find_simplex(query_uvs)
            
            # Initialize output arrays
            num_queries = len(query_uvs)
            positions = np.full((num_queries, 3), np.nan)
            normals = np.full((num_queries, 3), np.nan)
            
            # Get vertices and vertex normals
            vertices = uv_mesh.vertices
            vertex_normals = uv_mesh.vertex_normals
            
            # For each valid query point
            valid_mask = simplex_indices >= 0
            valid_indices = np.where(valid_mask)[0]
            
            for i in valid_indices:
                simplex_idx = simplex_indices[i]
                query_uv = query_uvs[i]
                
                # Get the triangle vertices in UV space
                triangle_vertices_2d = triangulation.simplices[simplex_idx]
                triangle_uvs = mesh_uvs[triangle_vertices_2d]
                
                # Calculate barycentric coordinates
                bary_coords = self._calculate_barycentric_2d(query_uv, triangle_uvs)
                
                # Interpolate 3D position
                triangle_3d_vertices = vertices[triangle_vertices_2d]
                positions[i] = np.sum(bary_coords[:, np.newaxis] * triangle_3d_vertices, axis=0)
                
                # Interpolate normal
                triangle_normals = vertex_normals[triangle_vertices_2d]
                normals[i] = np.sum(bary_coords[:, np.newaxis] * triangle_normals, axis=0)
                # Normalize the interpolated normal
                norm = np.linalg.norm(normals[i])
                if norm > 0:
                    normals[i] /= norm
            
            return positions, normals
            
        except Exception as e:
            print(f"⚠️ UV interpolation failed, using simplified approach: {e}")
            # Fallback: use nearest neighbor approach
            return self._interpolate_from_uv_fallback(uv_mesh, mesh_uvs, query_uvs)
    
    def _calculate_barycentric_2d(self, point, triangle):
        """Calculate barycentric coordinates for a point in a 2D triangle."""
        v0, v1, v2 = triangle
        v0v1 = v1 - v0
        v0v2 = v2 - v0
        v0p = point - v0
        
        dot00 = np.dot(v0v2, v0v2)
        dot01 = np.dot(v0v2, v0v1)
        dot02 = np.dot(v0v2, v0p)
        dot11 = np.dot(v0v1, v0v1)
        dot12 = np.dot(v0v1, v0p)
        
        inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        w = 1.0 - u - v
        
        return np.array([w, v, u])
    
    def _interpolate_from_uv_fallback(self, uv_mesh, mesh_uvs, query_uvs):
        """Fallback interpolation using nearest neighbors."""
        try:
            from scipy.spatial.distance import cdist
            
            # Find nearest UV coordinates for each query
            distances = cdist(query_uvs, mesh_uvs)
            nearest_indices = np.argmin(distances, axis=1)
            
            # Use the 3D positions and normals of nearest vertices
            positions = uv_mesh.vertices[nearest_indices]
            normals = uv_mesh.vertex_normals[nearest_indices]
            
            return positions, normals
            
        except Exception as e:
            print(f"⚠️ All UV interpolation methods failed: {e}")
            # Ultimate fallback: return zeros
            num_queries = len(query_uvs)
            return (
                np.zeros((num_queries, 3)), 
                np.zeros((num_queries, 3))
            )
    
    def _project_and_blend_colors(self, uv_mesh, pixel_to_3d_point, pixel_to_3d_normal, 
                                  valid_pixel_mask, multi_view_data, transform_manager, final_mesh, subject_name):
        """Project and blend colors from all views."""
        print("3. Projecting and blending colors from all views...")
        
        aggregated_color = np.zeros((self.texture_size, self.texture_size, 3))
        total_weight = np.zeros((self.texture_size, self.texture_size, 1))
        
        # Convert tensors to numpy arrays for processing
        images = []
        masks = []
        
        for view_data in multi_view_data:
            # Convert img_icon tensor to PIL Image
            img_tensor = view_data['img_icon'].squeeze(0)  # Remove batch dimension if present
            if img_tensor.shape[0] == 3:  # (C, H, W) format
                img_tensor = img_tensor.permute(1, 2, 0)  # Convert to (H, W, C)
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            images.append(Image.fromarray(img_np))
            
            # Convert img_mask tensor to PIL Image
            mask_tensor = view_data['img_mask'].squeeze()  # Remove extra dimensions
            mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
            masks.append(Image.fromarray(mask_np, mode='L'))
        
        for i, view_data in enumerate(multi_view_data):
            print(f"   - Processing view {i}...")
            
            colors, weights, pixel_coords = self._process_single_view(
                view_data, i, images[i], masks[i], transform_manager, final_mesh,
                pixel_to_3d_point, pixel_to_3d_normal, valid_pixel_mask, subject_name
            )
            
            if self.debug and len(colors) > 0:
                # Save per-view contribution
                view_texture = np.zeros((self.texture_size, self.texture_size, 3), dtype=np.uint8)
                view_texture[pixel_coords] = colors
                debug_view_path = f"{self.debug_dir}/{subject_name}_04_view_{i:02d}_contribution.png"
                cv2.imwrite(debug_view_path, cv2.cvtColor(view_texture, cv2.COLOR_RGB2BGR))
                print(f"🐛 Saved view {i} contribution: {debug_view_path}")
                
                # Save weight map for this view
                view_weights = np.zeros((self.texture_size, self.texture_size))
                view_weights[pixel_coords] = weights
                weight_normalized = (view_weights * 255).astype(np.uint8)
                debug_weight_path = f"{self.debug_dir}/{subject_name}_04_view_{i:02d}_weights.png"
                cv2.imwrite(debug_weight_path, weight_normalized)
                print(f"🐛 Saved view {i} weights: {debug_weight_path}")
            
            # Add weighted colors to the aggregate maps
            if len(colors) > 0:
                aggregated_color[pixel_coords] += colors * weights[:, np.newaxis]
                total_weight[pixel_coords] += weights[:, np.newaxis]
                print(f"     ✅ Added {len(colors)} pixels from view {i}")
            else:
                print(f"     ⚠️ No valid pixels from view {i}")

        # Normalize colors by total weight
        texture_image_np = np.divide(aggregated_color, total_weight + 1e-8, where=total_weight > 0)
        texture_image_np = np.clip(texture_image_np, 0, 255).astype(np.uint8)
        
        if self.debug:
            # Save intermediate blended texture
            debug_blended_path = f"{self.debug_dir}/{subject_name}_05_blended_texture.png"
            cv2.imwrite(debug_blended_path, cv2.cvtColor(texture_image_np, cv2.COLOR_RGB2BGR))
            print(f"🐛 Saved blended texture: {debug_blended_path}")
            
            # Save total weight map
            weight_vis = np.clip(total_weight.squeeze() * 255, 0, 255).astype(np.uint8)
            debug_total_weight_path = f"{self.debug_dir}/{subject_name}_05_total_weights.png"
            cv2.imwrite(debug_total_weight_path, weight_vis)
            print(f"🐛 Saved total weights: {debug_total_weight_path}")
        
        print(f"📊 Color blending stats:")
        print(f"   Pixels with color data: {(total_weight > 0).sum()}")
        print(f"   Average weight per pixel: {total_weight[total_weight > 0].mean():.3f}")
        
        return texture_image_np, total_weight
    
    def _process_single_view(self, view_data, view_idx, image, mask, transform_manager, 
                           final_mesh, pixel_to_3d_point, pixel_to_3d_normal, valid_pixel_mask, subject_name):
        """Process a single view to get colors and weights for visible pixels."""
        # Get camera parameters
        frame_id = int(view_data["name"].split("_")[1])
        T_world_to_cam = transform_manager.get_transform_to_target(frame_id).cpu().numpy()
        
        # Get camera intrinsic matrix
        if 'calib' in view_data:
            # Calib contains the full projection matrix (intrinsic @ extrinsic)
            # We need to extract just the intrinsic part
            calib_full = view_data['calib'].cpu().numpy()
            # For now, use a default intrinsic matrix assuming 512x512 images
            img_h, img_w = 512, 512  # Default image size from the renderer
            focal_length = 512  # Reasonable default focal length
            K = np.array([
                [focal_length, 0, img_w/2],
                [0, focal_length, img_h/2], 
                [0, 0, 1]
            ])
        else:
            # Fallback: construct default camera matrix
            img_h, img_w = 512, 512
            focal_length = 512
            K = np.array([
                [focal_length, 0, img_w/2],
                [0, focal_length, img_h/2], 
                [0, 0, 1]
            ])
        
        # Get image dimensions
        if hasattr(view_data, 'img_size') and view_data['img_size'] is not None:
            img_h, img_w = view_data['img_size']
        else:
            # Use the actual image dimensions from the loaded tensor
            if hasattr(view_data['img_icon'], 'shape'):
                img_tensor = view_data['img_icon']
                if len(img_tensor.shape) == 4:  # (B, C, H, W)
                    img_h, img_w = img_tensor.shape[2], img_tensor.shape[3]
                elif len(img_tensor.shape) == 3:  # (C, H, W)
                    img_h, img_w = img_tensor.shape[1], img_tensor.shape[2]
                else:
                    img_h, img_w = 512, 512  # Default fallback
            else:
                img_h, img_w = 512, 512  # Default fallback

        # Create a scene for occlusion testing for this specific view
        scene = trimesh.Scene(final_mesh)
        cam = trimesh.scene.Camera(
            resolution=(img_w, img_h), 
            K=K
        )
        scene.camera_transform = T_world_to_cam
        
        # Get occlusion map: depth buffer from the camera's point of view
        depth_map = scene.camera_rays()[2]

        # Project all valid 3D points into the current camera view
        points_3d_h = np.concatenate([pixel_to_3d_point[valid_pixel_mask], np.ones_like(pixel_to_3d_point[valid_pixel_mask][:, :1])], axis=1)
        points_cam_h = (T_world_to_cam @ points_3d_h.T).T
        points_proj_h = (K @ points_cam_h[:, :3].T).T
        
        # Normalize to get 2D image coordinates
        uv_coords = points_proj_h[:, :2] / points_proj_h[:, 2:]
        depth_values = points_cam_h[:, 2]

        # Check visibility (img_h, img_w already determined above)
        
        # 1. Check if inside image bounds
        in_bounds_mask = (uv_coords[:, 0] >= 0) & (uv_coords[:, 0] < img_w) & \
                         (uv_coords[:, 1] >= 0) & (uv_coords[:, 1] < img_h)
        
        # 2. Check against foreground mask
        uv_int = uv_coords[in_bounds_mask].astype(int)
        mask_vals = np.array(mask)[uv_int[:, 1], uv_int[:, 0]]
        in_fg_mask = (mask_vals > 128)
        
        # 3. Check occlusion using the pre-rendered depth map
        rendered_depth = depth_map[uv_int[in_fg_mask][:, 1], uv_int[in_fg_mask][:, 0]]
        is_visible_mask = np.isclose(depth_values[in_bounds_mask][in_fg_mask], rendered_depth, atol=0.05) # 5cm tolerance
        
        # Combine all masks to find the final set of visible points for this view
        final_visibility_mask = np.zeros(valid_pixel_mask.sum(), dtype=bool)
        temp_mask = np.zeros(in_bounds_mask.sum(), dtype=bool)
        temp_mask[in_fg_mask] = is_visible_mask
        final_visibility_mask[in_bounds_mask] = temp_mask

        # Calculate view-based weights for visible points
        cam_pos = np.linalg.inv(T_world_to_cam)[:3, 3]
        view_vectors = cam_pos - pixel_to_3d_point[valid_pixel_mask][final_visibility_mask]
        view_vectors /= np.linalg.norm(view_vectors, axis=1, keepdims=True)
        normals = pixel_to_3d_normal[valid_pixel_mask][final_visibility_mask]
        
        # Weight = (normal dot view_vector)^2
        weights = np.maximum(0, np.sum(normals * view_vectors, axis=1))**2
        
        # Sample colors
        visible_uvs = uv_coords[final_visibility_mask].astype(int)
        sampled_colors = np.array(image)[visible_uvs[:, 1], visible_uvs[:, 0]]
        
        # Get the original pixel indices for updating the aggregated maps
        original_indices = np.where(valid_pixel_mask)[0][final_visibility_mask]
        original_pixel_coords = np.unravel_index(original_indices, (self.texture_size, self.texture_size))
        
        return sampled_colors, weights, original_pixel_coords
    
    def _inpaint_texture(self, texture_image_np, total_weight, subject_name):
        """Inpaint missing areas in the texture map."""
        print("4. Inpainting holes in the texture map...")
        
        # Create inpaint mask from areas with no weight (no color data)
        inpaint_mask = (total_weight.squeeze() == 0).astype(np.uint8)
        
        print(f"📊 Inpainting stats:")
        print(f"   Pixels to inpaint: {inpaint_mask.sum()}")
        print(f"   Inpaint coverage: {inpaint_mask.sum() / (self.texture_size * self.texture_size) * 100:.1f}%")
        
        if self.debug:
            # Save inpaint mask
            debug_inpaint_mask_path = f"{self.debug_dir}/{subject_name}_06_inpaint_mask.png"
            cv2.imwrite(debug_inpaint_mask_path, inpaint_mask * 255)
            print(f"🐛 Saved inpaint mask: {debug_inpaint_mask_path}")
        
        # Convert to BGR for OpenCV inpainting
        texture_bgr = cv2.cvtColor(texture_image_np, cv2.COLOR_RGB2BGR)
        inpainted_bgr = cv2.inpaint(texture_bgr, inpaint_mask, 3, cv2.INPAINT_NS)
        
        # Convert back to RGB
        inpainted_texture = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
        
        if self.debug:
            # Save final inpainted texture
            debug_final_path = f"{self.debug_dir}/{subject_name}_07_final_texture.png"
            cv2.imwrite(debug_final_path, cv2.cvtColor(inpainted_texture, cv2.COLOR_RGB2BGR))
            print(f"🐛 Saved final texture: {debug_final_path}")
        
        return inpainted_texture
    
    def _save_uv_debug_image(self, uvs, faces, subject_name, step_name):
        """Save UV layout as a debug image."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            
            # Draw UV triangles
            for face in faces:
                triangle_uvs = uvs[face]
                triangle = patches.Polygon(triangle_uvs, fill=False, edgecolor='blue', linewidth=0.5)
                ax.add_patch(triangle)
            
            # Draw UV points
            ax.scatter(uvs[:, 0], uvs[:, 1], c='red', s=1, alpha=0.5)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.set_title(f'UV Layout - {subject_name}')
            ax.grid(True, alpha=0.3)
            
            debug_uv_image_path = f"{self.debug_dir}/{subject_name}_{step_name}.png"
            plt.savefig(debug_uv_image_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"🐛 Saved UV layout: {debug_uv_image_path}")
            
        except Exception as e:
            print(f"⚠️ Could not save UV debug image: {e}")
    
    def _save_textured_mesh(self, uv_mesh, inpainted_texture, out_dir, subject_name):
        """Save the final textured mesh."""
        print("5. Saving final textured mesh...")
        
        # Create the texture visual object
        texture_visual = trimesh.visual.TextureVisuals(
            uv=uv_mesh.visual.uv,
            image=Image.fromarray(inpainted_texture)
        )
        uv_mesh.visual = texture_visual

        # Export the final textured mesh as OBJ with MTL file
        textured_obj_path = f"{out_dir}/{subject_name}_textured.obj"
        uv_mesh.export(textured_obj_path)
        print(colored(f"✅ Saved final textured mesh to {textured_obj_path}", "green"))
        return uv_mesh


# For backward compatibility, provide the original function interface
def generate_texture_map(final_mesh, multi_view_data, transform_manager, out_dir, subject_name, 
                        texture_size=2048, debug=False, debug_dir=None):
    """
    Legacy function interface for generating texture maps.
    
    Args:
        final_mesh (trimesh.Trimesh): The final watertight mesh.
        multi_view_data (dict): Dictionary containing data for all views.
        transform_manager (TransformManager): Object to get camera extrinsics.
        out_dir (str): The main output directory.
        subject_name (str): The name of the subject for file naming.
        texture_size (int): The resolution of the output texture map.
        debug (bool): Whether to save intermediate debugging outputs.
        debug_dir (str): Directory to save debug outputs.
    
    Returns:
        trimesh.Trimesh: The textured mesh with UV coordinates.
    """
    generator = TextureGenerator(texture_size=texture_size, debug=debug, debug_dir=debug_dir)
    return generator(final_mesh, multi_view_data, transform_manager, out_dir, subject_name)