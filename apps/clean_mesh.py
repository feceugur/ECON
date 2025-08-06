import pymeshlab
import trimesh
import numpy as np
from scipy.spatial import cKDTree  
import os 
import tempfile 
from typing import List, Optional, Tuple
from collections import Counter
from dataclasses import dataclass


import logging
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FaceGraftingConfig:
    """Configuration for the FaceGraftingPipeline."""

    # === Core methods ===
    stitch_method: str = "body_driven_loft"
    smplx_face_neck_loop_strategy: str = "full_face_silhouette"

    # === Geometry thresholds & precisions ===
    projection_footprint_threshold: float = 0.01
    footprint_dilation_rings: int = 0
    squashed_triangle_drop_fraction: float = 0.7
    vertex_coordinate_precision_digits: Optional[int] = 5

    # === Body mesh simplification ===
    body_simplification_target_faces: int = 12000

    # === Hole–seam handling ===
    close_smplx_face_holes: bool = True
    max_seam_hole_fill_vertices: int = 250
    final_polish_max_hole_edges: int = 100

    # === Boundary smoothing ===
    hole_boundary_smoothing_iterations: int = 25
    hole_boundary_smoothing_factor: float = 0.1
    pre_hole_cut_smoothing_iterations: int = 0
    pre_hole_cut_smoothing_lambda: float = 0.5

    # === Iterative repair & regularization ===
    iterative_repair_s1_iters: int = 5
    regularize_hole_loop_iterations: int = 5
    regularize_hole_loop_relaxation: float = 0.35

    # === Cleaning & debug options ===
    use_open3d_cleaning: bool = False
    save_debug_body_with_hole: bool = True
    debug_smply_integrity_trace: bool = True


class FaceGraftingPipeline:
    """
    Pipeline for grafting an SMPLX face mesh onto a full body mesh.
    """
    INTERNAL_VERSION_TRACKER = "FinalFaceGraftingPipeline_V6.0"

    def __init__(self,
                 full_body_mesh_path: str,
                 smplx_face_mesh_path: str,
                 output_path: str,
                 config: FaceGraftingConfig):
        self.full_body_mesh_path = full_body_mesh_path
        self.smplx_face_mesh_path = smplx_face_mesh_path
        self.output_path = output_path
        self.config = config

        self.temp_files_to_clean: List[str] = []
        self.original_smplx_face_geom_tri: Optional[trimesh.Trimesh] = None
        self.simplified_body_trimesh: Optional[trimesh.Trimesh] = None
        self.body_with_hole_trimesh: Optional[trimesh.Trimesh] = None
        self.stitched_mesh_intermediate: Optional[trimesh.Trimesh] = None
        self.final_processed_mesh: Optional[trimesh.Trimesh] = None
        self.faces_to_remove_mask_on_body: Optional[np.ndarray] = None
        self.ordered_s_vidx_loop: Optional[np.ndarray] = None
        self.s_loop_coords_ordered: Optional[np.ndarray] = None
        self.b_loop_coords_aligned: Optional[np.ndarray] = None
        self.debug_tracked_vertex_coords: Optional[np.ndarray] = None # <-- ADD THIS LINE

    def _make_temp_path(self, suffix_label: str, use_ply: bool = False) -> str:
        """Helper to create and track temporary file paths."""
        actual_suffix = suffix_label if suffix_label.startswith('_') else '_' + suffix_label
        file_ext = ".ply" if use_ply else ".obj"
        fd, path = tempfile.mkstemp(suffix=actual_suffix + file_ext, dir=None, prefix="graft_")
        os.close(fd)
        self.temp_files_to_clean.append(path)
        return path

    def _cleanup_temp_files(self):
        """Removes all temporary files created during the process."""
        for temp_path in self.temp_files_to_clean:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError as e:
                    logger.warning(f"Could not remove temp file {temp_path}: {e}")
        self.temp_files_to_clean.clear()
            
    @staticmethod
    def _thoroughly_clean_trimesh_object(mesh: Optional[trimesh.Trimesh], mesh_name_for_log: str, 
                                         version_tracker: str, merge_verts: bool = True) -> Optional[trimesh.Trimesh]:
        if mesh is None: 
            return None
        
        if not MeshCleanProcess._is_mesh_valid_for_concat(mesh, f"{mesh_name_for_log}_PreClean_V{version_tracker}"):
            return mesh 

        try:
            mesh.remove_duplicate_faces()
            if mesh.is_empty: return None 
            
            if merge_verts:
                mesh.merge_vertices(merge_tex=False, merge_norm=False)
                if mesh.is_empty: return None

            mesh.remove_unreferenced_vertices()

            # --- DEPRECATION FIX ---
            non_degenerate_face_mask = mesh.nondegenerate_faces()
            if np.any(~non_degenerate_face_mask):
                mesh.update_faces(non_degenerate_face_mask)
            # --- END FIX ---
            
            if mesh.is_empty: return None
                
            # mesh.fix_normals(multibody=True) 
            if not MeshCleanProcess._is_mesh_valid_for_concat(mesh, f"{mesh_name_for_log}_PostClean_V{version_tracker}"):
                return None 

            return mesh
        except Exception as e:
            logger.error(f"Exception during thorough cleaning of {mesh_name_for_log}: {e}", exc_info=True)
            return None

    @staticmethod
    def _close_smplx_holes_by_fan_fill(input_mesh: trimesh.Trimesh) -> Optional[trimesh.Trimesh]:
        """
        Closes all but the largest boundary loop on a mesh using fan triangulation
        WITHOUT creating new vertices. This is for filling mouth/nostril holes.
        """
        if not MeshCleanProcess._is_mesh_valid_for_concat(input_mesh, "FanFillInput"):
            return input_mesh

        logger.info("Attempting to fan-fill internal SMPLX holes (No New Vertices Method)...")
        
        # This is critical for the orientation check later
        # input_mesh.fix_normals()

        all_loops_vidx = MeshCleanProcess.get_all_boundary_loops(input_mesh, min_loop_len=3)

        if not all_loops_vidx or len(all_loops_vidx) <= 1:
            logger.info("No internal holes found to fill or only one boundary loop exists.")
            return input_mesh

        # Identify the largest loop (the neck hole) and exclude it from filling.
        all_loops_vidx.sort(key=len, reverse=True)
        neck_hole_loop = all_loops_vidx[0]
        loops_to_fill = all_loops_vidx[1:]
        
        logger.info(f"Found {len(loops_to_fill)} internal hole(s) to fill. Skipping largest loop with {len(neck_hole_loop)} vertices.")

        if not loops_to_fill:
            return input_mesh

        current_faces = list(input_mesh.faces)
        added_any_faces = False

        for loop in loops_to_fill:
            if len(loop) < 3:
                continue
            
            try:
                # --- FAN TRIANGULATION (ROOT VERTEX METHOD) ---
                # This method uses an existing vertex and adds NO new geometry.

                # 1. Find the average normal of faces surrounding the hole for orientation check.
                neighbor_face_indices_nested = input_mesh.vertex_faces[loop]
                if len(neighbor_face_indices_nested) > 0:
                    valid_faces = [f for f_list in neighbor_face_indices_nested for f in f_list if f != -1]
                    if not valid_faces: continue
                    unique_face_indices = np.unique(valid_faces)
                    target_normal = trimesh.util.unitize(np.mean(input_mesh.face_normals[unique_face_indices], axis=0))
                else:
                    continue

                # 2. Choose a root vertex for the fan. The first vertex is a stable choice.
                root_vidx = loop[0]

                # 3. Create the fan of faces from the root vertex.
                new_patch_faces = []
                for i in range(1, len(loop) - 1):
                    v1_idx = loop[i]
                    v2_idx = loop[i + 1]
                    new_patch_faces.append([root_vidx, v1_idx, v2_idx])
                
                # 4. Check and fix the orientation of the new patch.
                if new_patch_faces and np.any(np.abs(target_normal) > 1e-6):
                    v0_coords, v1_coords, v2_coords = input_mesh.vertices[new_patch_faces[0]]
                    test_normal = trimesh.util.unitize(np.cross(v1_coords - v0_coords, v2_coords - v0_coords))
                    
                    if np.dot(test_normal, target_normal) < 0.0:
                        new_patch_faces = [face[::-1] for face in new_patch_faces]

                current_faces.extend(new_patch_faces)
                added_any_faces = True
            except Exception as e:
                logger.warning(f"Failed to fill an internal hole: {e}", exc_info=True)

        if not added_any_faces:
            return input_mesh

        # Create the new mesh with the added faces. Vertices are unchanged.
        filled_mesh = trimesh.Trimesh(vertices=input_mesh.vertices, faces=np.array(current_faces, dtype=int), process=False)
        # filled_mesh.fix_normals()

        if MeshCleanProcess._is_mesh_valid_for_concat(filled_mesh, "FanFillResult"):
            logger.info(f"Successfully filled internal holes without adding new vertices.")
            return filled_mesh
        else:
            logger.warning("Fan-filling (root vertex) resulted in an invalid mesh. Reverting to original.")
            return input_mesh
                        
    def _clean_with_open3d(self, input_mesh: trimesh.Trimesh, mesh_name_for_log: str) -> trimesh.Trimesh:
        """
        Applies a robust cleaning pipeline using Open3D.
        Returns the cleaned mesh, or the original mesh if cleaning fails or is skipped.
        """
        if not MeshCleanProcess._is_mesh_valid_for_concat(input_mesh, f"PreO3D_{mesh_name_for_log}"):
            return input_mesh
        
        try:
            import open3d as o3d
        except ImportError:
            logger.warning("Open3D is not installed. Skipping Open3D cleaning step.")
            return input_mesh

        logger.info(f"--- Applying aggressive Open3D cleaning for '{mesh_name_for_log}' ---")
        temp_in = self._make_temp_path("o3d_in", use_ply=True)
        temp_out = self._make_temp_path("o3d_out", use_ply=True)

        try:
            input_mesh.export(temp_in)
            mesh_o3d = o3d.io.read_triangle_mesh(temp_in)
            
            if not mesh_o3d.has_triangles():
                logger.warning("Mesh became empty after loading into Open3D. Aborting O3D clean.")
                return input_mesh

            mesh_o3d.remove_duplicated_vertices()
            mesh_o3d.remove_degenerate_triangles()
            mesh_o3d.remove_duplicated_triangles()
            mesh_o3d.remove_non_manifold_edges()
            # The 'remove_self_intersecting_triangles' method call has been removed to avoid warnings.
            mesh_o3d.orient_triangles()
            mesh_o3d.compute_vertex_normals()
            
            o3d.io.write_triangle_mesh(temp_out, mesh_o3d)
            cleaned_mesh = trimesh.load_mesh(temp_out, process=True)
            
            if MeshCleanProcess._is_mesh_valid_for_concat(cleaned_mesh, f"PostO3D_{mesh_name_for_log}"):
                logger.info(f"--- Open3D cleaning successful for '{mesh_name_for_log}' ---")
                return cleaned_mesh
            else:
                logger.warning(f"Open3D cleaning for '{mesh_name_for_log}' resulted in an invalid mesh. Reverting.")
                return input_mesh

        except Exception as e:
            logger.error(f"Error during Open3D cleaning process for '{mesh_name_for_log}': {e}", exc_info=True)
            return input_mesh

    def _load_and_simplify_meshes(self) -> bool:
        logger.info("=== STEP 1 & 2: Load meshes, clean face, simplify and clean body ===")
        
        self.original_smplx_face_geom_tri = trimesh.load_mesh(self.smplx_face_mesh_path, process=False)
        if self.original_smplx_face_geom_tri is None or \
           not MeshCleanProcess._is_mesh_valid_for_concat(self.original_smplx_face_geom_tri, "InitialSMPLXFace"):
            logger.critical(f"Failed to load or invalid SMPLX Face mesh: '{self.smplx_face_mesh_path}'. Aborting.")
            return False
        
        cleaned_smplx_face = FaceGraftingPipeline._thoroughly_clean_trimesh_object(
            self.original_smplx_face_geom_tri.copy(), "InitialSMPLXFace", self.INTERNAL_VERSION_TRACKER, merge_verts=False
        )
        if cleaned_smplx_face is not None: self.original_smplx_face_geom_tri = cleaned_smplx_face
        else: logger.warning("Thorough cleaning of initial SMPLX face failed. Using original loaded version.")

        if self.config.close_smplx_face_holes:
            filled_face = FaceGraftingPipeline._close_smplx_holes_by_fan_fill(self.original_smplx_face_geom_tri)
            if filled_face is not None:
                self.original_smplx_face_geom_tri = filled_face
            else:
                logger.warning("SMPLX hole filling failed, continuing with original face.")

        self.original_smplx_v_pre_graft = self.original_smplx_face_geom_tri.vertices.copy()
        self.original_smplx_edges_pre_graft = {tuple(sorted(edge)) for edge in self.original_smplx_face_geom_tri.edges_unique}
        logger.info(f"Stored original SMPLX face integrity baseline: {len(self.original_smplx_v_pre_graft)} vertices, {len(self.original_smplx_edges_pre_graft)} unique edges.")

        self.simplified_body_trimesh = trimesh.load_mesh(self.full_body_mesh_path, process=True)
        if self.simplified_body_trimesh is None or \
           not MeshCleanProcess._is_mesh_valid_for_concat(self.simplified_body_trimesh, "InitialFullBody"):
            logger.critical(f"Failed to load or invalid Full Body mesh: '{self.full_body_mesh_path}'. Aborting.")
            return False

        # if self.original_smplx_face_geom_tri: self.original_smplx_face_geom_tri.fix_normals()
        # if self.simplified_body_trimesh: self.simplified_body_trimesh.fix_normals()

        if self.simplified_body_trimesh.faces.shape[0] > self.config.body_simplification_target_faces:
            logger.info(f"Simplifying body mesh to target {self.config.body_simplification_target_faces} faces...")
            body_state_before_pml_simplification = self.simplified_body_trimesh.copy() 
            temp_in = self._make_temp_path("body_simp_in", use_ply=True)
            temp_out = self._make_temp_path("body_simp_out", use_ply=True)
            loaded_s_from_pml = None
            ms_s = None
            try:
                if self.simplified_body_trimesh.export(temp_in): 
                    ms_s = pymeshlab.MeshSet()
                    ms_s.load_new_mesh(temp_in)
                    if ms_s.current_mesh_id() != -1 and ms_s.current_mesh().face_number() > 0:
                        ms_s.meshing_decimation_quadric_edge_collapse(
                            targetfacenum=self.config.body_simplification_target_faces, 
                            preservenormal=True, preservetopology=True, optimalplacement=True,
                            planarquadric=True, qualitythr=0.7)
                        ms_s.meshing_repair_non_manifold_edges(method='Remove Faces')
                        ms_s.meshing_repair_non_manifold_edges(method='Split Vertices') 
                        try: ms_s.meshing_repair_non_manifold_vertices_by_splitting(threshold=pymeshlab.Percentage(0.01))
                        except (AttributeError, pymeshlab.PyMeshLabException): pass
                        ms_s.meshing_remove_unreferenced_vertices()
                        ms_s.save_current_mesh(temp_out)
                        loaded_s_from_pml = trimesh.load_mesh(temp_out, process=True)
                    else: logger.warning("Mesh in PyMeshLab for simplification was empty or invalid.")
                
                if loaded_s_from_pml is not None and MeshCleanProcess._is_mesh_valid_for_concat(loaded_s_from_pml, "SimpPML"):
                    self.simplified_body_trimesh = loaded_s_from_pml
                    logger.info(f"Body simplification via PyMeshLab successful. New F={self.simplified_body_trimesh.faces.shape[0]}")
                else:
                    logger.warning("Body simplification (PML) failed or resulted in invalid mesh. Reverting.")
                    self.simplified_body_trimesh = body_state_before_pml_simplification
            except Exception as e_simp_block: 
                logger.warning(f"Exception during body simplification (PML): {e_simp_block}. Reverting.", exc_info=True) 
                self.simplified_body_trimesh = body_state_before_pml_simplification 
            finally:
                if ms_s is not None: del ms_s
        
        if self.simplified_body_trimesh is None:
            logger.critical("Simplified body is None before final cleaning. Aborting.")
            return False
            
        # --- MODIFIED FINAL CLEANING STEP ---
        if self.config.use_open3d_cleaning:
            self.simplified_body_trimesh = self._clean_with_open3d(
                self.simplified_body_trimesh, "SimplifiedBodyFinalClean_O3D"
            )
        else:
            cleaned_simplified_body = FaceGraftingPipeline._thoroughly_clean_trimesh_object(
                self.simplified_body_trimesh.copy(), "SimplifiedBodyFinalClean", self.INTERNAL_VERSION_TRACKER
            )
            if cleaned_simplified_body is not None:
                self.simplified_body_trimesh = cleaned_simplified_body
            else:
                logger.warning("Final thorough cleaning of simplified body failed. Using pre-clean version.")
        # --- END MODIFIED STEP ---

        if not MeshCleanProcess._is_mesh_valid_for_concat(self.simplified_body_trimesh, "FinalSimplifiedBody"):
            logger.critical("Simplified body mesh is invalid after all steps in _load_and_simplify. Aborting.")
            return False
            
        if self.simplified_body_trimesh is not None and not self.simplified_body_trimesh.is_empty:
            try:
                _ = self.simplified_body_trimesh.edges
                if len(self.simplified_body_trimesh.boundary_edges) > 0:
                    logger.warning(f"FINAL SIMPLIFIED BODY MESH has boundary edges. Not watertight.")
            except Exception: pass
        
        logger.info(f"Finished loading and simplifying. Final body: V={len(self.simplified_body_trimesh.vertices)}, F={len(self.simplified_body_trimesh.faces)}")
        return True

    @staticmethod
    def _regularize_loop_to_equal_edge_lengths(
        verts: np.ndarray,
        loop_vidx: np.ndarray,
        iters: int,
        relax: float
    ) -> None:
        """
        Modifies vertex positions IN-PLACE to make all boundary edges
        approximately equal length without adding or removing vertices.
        """
        if len(loop_vidx) < 3:
            return

        for _ in range(iters):
            # Get current vertex positions for the loop
            p = verts[loop_vidx]

            # Calculate edge vectors and their lengths
            edge_vecs = np.roll(p, -1, axis=0) - p
            lengths = np.linalg.norm(edge_vecs, axis=1)
            
            # Avoid division by zero if a loop collapses
            if np.any(lengths < 1e-9): break 
            
            # Calculate the target average edge length
            L_avg = lengths.mean()

            # Error for each edge (how much longer/shorter it is than the average)
            err_current_edge = lengths - L_avg
            err_previous_edge = np.roll(err_current_edge, 1)

            # Tangent vectors (direction of each edge)
            tangents = edge_vecs / (lengths[:, None] + 1e-12)

            # For each vertex, the move is influenced by the error of the two edges connected to it.
            # A vertex is "pulled" from its longer-edge side and "pushed" from its shorter-edge side.
            move = (
                -(err_current_edge[:, None] * tangents) +
                 (err_previous_edge[:, None] * np.roll(tangents, 1, axis=0))
            )
            
            # Apply the displacement, scaled by the relaxation factor.
            # This is an in-place modification of the original `verts` array.
            verts[loop_vidx] += relax * move * 0.5
                
    def _smooth_main_hole_boundary(self) -> bool:
        if not self.body_with_hole_trimesh or \
            not MeshCleanProcess._is_mesh_valid_for_concat(self.body_with_hole_trimesh, "BodyForHoleSmooth"):
            logger.warning("Cannot smooth hole boundary: body_with_hole_trimesh is invalid or None.")
            return False

        all_loops_vidx = MeshCleanProcess.get_all_boundary_loops(self.body_with_hole_trimesh, min_loop_len=3)
        if not all_loops_vidx: 
            logger.info("Hole smoothing: No boundary loops found. Skipping.")
            return True # Not an error, just nothing to do.

        largest_loop_vidx = max(all_loops_vidx, key=len, default=None)
        if largest_loop_vidx is None or len(largest_loop_vidx) < 3: 
            logger.info("Hole smoothing: No sufficiently large boundary loop found. Skipping.")
            return True

        # NOTE: The problematic 'before' copy and squashed-face-removal logic have been removed.

        if self.config.hole_boundary_smoothing_iterations > 0:
            logger.info(f"Smoothing main hole boundary ({self.config.hole_boundary_smoothing_iterations} iter)...")
            vertices_copy = self.body_with_hole_trimesh.vertices.copy()
            num_loop_verts = len(largest_loop_vidx)
            for _ in range(self.config.hole_boundary_smoothing_iterations):
                new_positions_for_loop = np.zeros((num_loop_verts, 3))
                for i in range(num_loop_verts):
                    curr_v, prev_v, next_v = largest_loop_vidx[i], largest_loop_vidx[(i - 1 + num_loop_verts) % num_loop_verts], largest_loop_vidx[(i + 1) % num_loop_verts]
                    lap_target = (vertices_copy[prev_v] + vertices_copy[next_v]) / 2.0
                    new_positions_for_loop[i] = vertices_copy[curr_v] + self.config.hole_boundary_smoothing_factor * (lap_target - vertices_copy[curr_v])
                for i in range(num_loop_verts): vertices_copy[largest_loop_vidx[i]] = new_positions_for_loop[i]
            self.body_with_hole_trimesh.vertices = vertices_copy
        
        if self.config.regularize_hole_loop_iterations > 0:
            logger.info(f"Regularizing hole boundary edge lengths ({self.config.regularize_hole_loop_iterations} iter)...")
            FaceGraftingPipeline._regularize_loop_to_equal_edge_lengths(
                verts=self.body_with_hole_trimesh.vertices,
                loop_vidx=largest_loop_vidx,
                iters=self.config.regularize_hole_loop_iterations,
                relax=self.config.regularize_hole_loop_relaxation
            )

        # The aggressive face removal and vertex merging have been removed from this function
        # to prevent the hole from being prematurely closed. We only modify vertex positions here.
        
        if hasattr(self.body_with_hole_trimesh, '_cache'): self.body_with_hole_trimesh._cache.clear()
        return True

    def _determine_hole_faces_and_create_body_with_hole(self) -> bool:
        if self.simplified_body_trimesh is None or self.original_smplx_face_geom_tri is None:
            logger.error("Missing meshes for hole creation. Aborting.")
            return False

        logger.info(f"=== STEP 3: Determining Hole Faces & Creating Body With Hole (Proximity Method) ===")
        
        self.faces_to_remove_mask_on_body = np.zeros(len(self.simplified_body_trimesh.faces), dtype=bool)

        if self.original_smplx_face_geom_tri.vertices.shape[0] > 0 and \
           hasattr(self.original_smplx_face_geom_tri, 'vertex_normals'):
            
            logger.info("Using proximity search to determine hole footprint...")
            try:
                _, d_cp, t_cp = trimesh.proximity.closest_point(self.simplified_body_trimesh, self.original_smplx_face_geom_tri.vertices)
                if d_cp is not None and t_cp is not None:
                    valid_mask = (d_cp < self.config.projection_footprint_threshold) & (t_cp < len(self.faces_to_remove_mask_on_body))
                    if np.any(valid_mask): self.faces_to_remove_mask_on_body[np.unique(t_cp[valid_mask])] = True
                
                offset = self.config.projection_footprint_threshold * 0.5
                p_f = self.original_smplx_face_geom_tri.vertices + self.original_smplx_face_geom_tri.vertex_normals * offset
                _, _, t_f = trimesh.proximity.closest_point(self.simplified_body_trimesh, p_f)
                if t_f is not None:
                    valid_mask = t_f < len(self.faces_to_remove_mask_on_body)
                    if np.any(valid_mask): self.faces_to_remove_mask_on_body[np.unique(t_f[valid_mask])] = True
                
                p_b = self.original_smplx_face_geom_tri.vertices - self.original_smplx_face_geom_tri.vertex_normals * offset
                _, dists_b, t_b = trimesh.proximity.closest_point(self.simplified_body_trimesh, p_b)
                if dists_b is not None and t_b is not None:
                    valid_mask = (dists_b < self.config.projection_footprint_threshold) & (t_b < len(self.faces_to_remove_mask_on_body))
                    if np.any(valid_mask): self.faces_to_remove_mask_on_body[np.unique(t_b[valid_mask])] = True
            except Exception as e_p: 
                logger.warning(f"Error during proximity-based hole determination: {e_p}", exc_info=True)

        if self.config.footprint_dilation_rings > 0 and np.any(self.faces_to_remove_mask_on_body):
            logger.info(f"Dilating footprint by {self.config.footprint_dilation_rings} ring(s)...")
            adj = self.simplified_body_trimesh.face_adjacency
            current_wavefront = self.faces_to_remove_mask_on_body.copy()
            for _ in range(self.config.footprint_dilation_rings):
                wave_indices = np.where(current_wavefront)[0]
                if not wave_indices.size: break
                all_neighbors_this_ring = adj[np.isin(adj, wave_indices).any(axis=1)].flatten()
                unique_neigh = np.unique(all_neighbors_this_ring)
                new_to_add = unique_neigh[~self.faces_to_remove_mask_on_body[unique_neigh]]
                if not new_to_add.size: break
                self.faces_to_remove_mask_on_body[new_to_add] = True
                current_wavefront.fill(False); current_wavefront[new_to_add] = True

        # --- MODIFIED LOGIC: REMOVE FACES AND IMMEDIATELY ISOLATE LARGEST COMPONENT ---
        current_mesh = self.simplified_body_trimesh.copy()
        if np.any(self.faces_to_remove_mask_on_body):
            # 1. Apply the (potentially messy) removal mask
            current_mesh.update_faces(~self.faces_to_remove_mask_on_body)
            current_mesh.remove_unreferenced_vertices()
            
            # 2. Split the result into all its parts (main body + floating islands)
            logger.info("Isolating main body component to remove floating face islands...")
            components = current_mesh.split(only_watertight=False)
            
            # 3. Robustly find the largest component and discard the rest
            actual_components_list = [c for c in np.atleast_1d(components) if c is not None and not c.is_empty]
            if len(actual_components_list) > 1:
                logger.info(f"Found {len(actual_components_list)} components after removal. Keeping largest.")
                current_mesh = max(actual_components_list, key=lambda c: len(c.faces))
            elif actual_components_list:
                current_mesh = actual_components_list[0]
            else:
                logger.warning("Splitting after face removal resulted in no valid components.")
                current_mesh = trimesh.Trimesh() # Create empty mesh if all else fails
        else:
            logger.warning("No faces were marked for removal.")

        self.body_with_hole_trimesh = current_mesh
        
        if not MeshCleanProcess._is_mesh_valid_for_concat(self.body_with_hole_trimesh, "FinalBodyWHole"):
            logger.critical("Final Body-with-hole mesh is invalid. Aborting.")
            return False

        logger.info(f"Finished body_with_hole creation. V={len(self.body_with_hole_trimesh.vertices)}, F={len(self.body_with_hole_trimesh.faces)}")
        return True
    
    def _extract_smplx_face_loop(self) -> bool:
        if not self.original_smplx_face_geom_tri: return False
        # logger.info("Extracting SMPLX face loop.") 
        if self.config.smplx_face_neck_loop_strategy == "full_face_silhouette":
            face_boundary_edges = MeshCleanProcess._get_boundary_edges_manually_from_faces(self.original_smplx_face_geom_tri)
            if face_boundary_edges is not None and len(face_boundary_edges) >= 3:
                all_face_boundary_components_vidx = trimesh.graph.connected_components(face_boundary_edges, min_len=3)
                valid_comps = [c for c in all_face_boundary_components_vidx if c is not None and len(c) >= 3]
                if valid_comps:
                    s_unord = max(valid_comps, key=len, default=np.array([],dtype=int))
                    if len(s_unord) >=3:
                        s_set = set(s_unord)
                        s_edges = [e for e in face_boundary_edges if e[0] in s_set and e[1] in s_set]
                        if s_edges:
                            self.ordered_s_vidx_loop = MeshCleanProcess._order_loop_vertices_from_edges(
                                "SMPLXFaceLoop", s_unord, np.array(s_edges))
        
        if self.ordered_s_vidx_loop is not None and len(self.ordered_s_vidx_loop) >= 3:
            if self.ordered_s_vidx_loop.max() < len(self.original_smplx_face_geom_tri.vertices):
                self.s_loop_coords_ordered = self.original_smplx_face_geom_tri.vertices[self.ordered_s_vidx_loop]
                # logger.info(f"Extracted SMPLX face loop ({len(self.s_loop_coords_ordered)} verts).")
                return True
            else: logger.warning("SMPLX face loop indices out of bounds.")
        logger.warning("Failed to extract a valid SMPLX face loop.")
        return False

    def _align_body_loop_to_smplx_loop(self, b_loop_coords_ordered_pre_align: np.ndarray) -> bool:
        if self.s_loop_coords_ordered is None or len(self.s_loop_coords_ordered) < 2 or \
           b_loop_coords_ordered_pre_align is None or len(b_loop_coords_ordered_pre_align) < 2: 
            logger.warning("Not enough points in loops for alignment.")
            return False

        # logger.info("Aligning body loop to SMPLX face loop (KD-Tree Method).")
        
        # Step 1: Align the starting points by rolling the body loop.
        s_start_pt = self.s_loop_coords_ordered[0]
        _, closest_idx_on_b = cKDTree(b_loop_coords_ordered_pre_align).query(s_start_pt, k=1)
        b_rolled = np.roll(b_loop_coords_ordered_pre_align, -closest_idx_on_b, axis=0)
        
        # Step 2: Determine the correct orientation (forward vs. reversed) without resampling.
        # We create two candidates and see which one is a better overall match to the smplx loop.
        b_candidate_forward = b_rolled
        b_candidate_reversed = b_rolled[::-1].copy()

        # Build KD-Trees for both candidate orientations
        kdt_forward = cKDTree(b_candidate_forward)
        kdt_reversed = cKDTree(b_candidate_reversed)

        # Query both trees using the SMPLX loop points to find the sum of nearest-neighbor distances.
        distances_forward, _ = kdt_forward.query(self.s_loop_coords_ordered, k=1)
        distances_reversed, _ = kdt_reversed.query(self.s_loop_coords_ordered, k=1)
        
        total_dist_forward = np.sum(distances_forward)
        total_dist_reversed = np.sum(distances_reversed)

        # The orientation with the smaller total distance is the correct one.
        if total_dist_reversed < total_dist_forward:
            self.b_loop_coords_aligned = b_candidate_reversed
            # logger.info("Body loop orientation was reversed for best fit.")
        else:
            self.b_loop_coords_aligned = b_candidate_forward
            # logger.info("Body loop orientation kept forward for best fit.")

        return self.b_loop_coords_aligned is not None and len(self.b_loop_coords_aligned) >= 3

    # -------------------------------------------------------------------------
    #  2) MAIN: watertight strip where *only* body rim is resampled
    # -------------------------------------------------------------------------
    def _create_loft_stitch_mesh_resample_body(self) -> Optional[trimesh.Trimesh]:
        """
        Builds a topologically robust stitch strip by creating a 1-to-1 correspondence
        between the face loop and an index-sampled body loop. This version uses
        plane projection and signed area for robust orientation matching.
        """
        if (self.s_loop_coords_ordered is None or len(self.s_loop_coords_ordered) < 3 or
            self.b_loop_coords_aligned is None or len(self.b_loop_coords_aligned) < 3):
            return None

        face_loop = self.s_loop_coords_ordered
        body_loop_full = self.b_loop_coords_aligned
        Ns, Nb = len(face_loop), len(body_loop_full)

        # 1. Resample the body loop BY INDEX to match the number of vertices on the face loop.
        step = Nb / Ns
        body_idx = np.floor(np.arange(Ns) * step).astype(int)
        body_idx[-1] = Nb - 1 # Ensure the loop closes properly
        body_loop_resampled = body_loop_full[body_idx]

        # 2. Use 2D polygon signed area to robustly determine and match loop orientation.
        def _get_signed_area(points_3d: np.ndarray) -> float:
            if len(points_3d) < 3: return 0.0
            # Project points onto their best-fit plane to get 2D coordinates
            plane_origin, plane_normal = trimesh.points.plane_fit(points_3d)
            transform = trimesh.geometry.plane_transform(plane_origin, plane_normal)
            points_2d = trimesh.transform_points(points_3d, transform)[:, :2]
            # Shoelace formula for signed area
            x, y = points_2d.T
            return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))
        
        # 3. If winding orders differ (signs of area are different), reverse the body loop
        if np.sign(_get_signed_area(body_loop_resampled)) != np.sign(_get_signed_area(face_loop)):
            logger.info("Reversing body loop orientation to match face loop winding order.")
            body_loop_resampled = body_loop_resampled[::-1]

        # 4. Final alignment of start points after orientation is fixed
        k0 = np.argmin(np.linalg.norm(body_loop_resampled - face_loop[0], axis=1))
        body_loop_final = np.roll(body_loop_resampled, -k0, axis=0)

        # 5. Build the strip vertices and faces
        strip_verts = np.vstack([body_loop_final, face_loop])
        faces = []
        for i in range(Ns):
            j = (i + 1) % Ns
            # Quad: (i, j, j+Ns, i+Ns) -> Triangles: (i, j, j+Ns) and (i, j+Ns, i+Ns)
            faces.extend([[i, j, j + Ns], [i, j + Ns, i + Ns]])

        strip = trimesh.Trimesh(vertices=strip_verts, faces=np.array(faces, dtype=int), process=False)
        logger.info("Built a topologically robust stitch strip (body-resample method).")
        return strip

    def _create_loft_stitch_mesh_projection(self) -> Optional[trimesh.Trimesh]:
        """Creates a stitch strip using the original, simpler projection method. (FALLBACK)"""
        if (self.s_loop_coords_ordered is None or len(self.s_loop_coords_ordered) < 3 or
                self.b_loop_coords_aligned   is None or len(self.b_loop_coords_aligned)   < 3):
            return None
        b_loop, s_loop = self.b_loop_coords_aligned, self.s_loop_coords_ordered
        Nb = len(b_loop)
        
        # Project body loop vertices onto the SMPLX loop to find correspondences
        nearest_s_idx = cKDTree(s_loop).query(b_loop, k=1)[1]
        
        strip_verts = np.vstack((b_loop, s_loop))
        faces = []
        
        # Create faces based on the projection. This can sometimes skip vertices.
        for i in range(Nb):
            i_next = (i + 1) % Nb
            v0, v1 = i, i_next # Indices in the first part of strip_verts (b_loop)
            
            # Indices in the second part of strip_verts (s_loop), offset by Nb
            s_curr = nearest_s_idx[i] + Nb
            s_next = nearest_s_idx[i_next] + Nb
            
            # Create two triangles to form a quad
            faces.extend([[v0, v1, s_next], [v0, s_next, s_curr]])
            
        strip = trimesh.Trimesh(vertices=strip_verts, faces=np.asarray(faces, dtype=int), process=False)
        if not MeshCleanProcess._is_mesh_valid_for_concat(strip, "LoftStripProjection"):
            logger.warning("Projection-based loft strip failed validation.")
            return None
            
        logger.info("Fallback loft stitch mesh (projection-based) built.")
        return strip

    def _create_loft_stitch_mesh_zipper(self) -> Optional[trimesh.Trimesh]:
        """Creates a stitch strip using the robust zipper method. (PRIMARY)"""
        if (self.s_loop_coords_ordered is None or len(self.s_loop_coords_ordered) < 3 or
                self.b_loop_coords_aligned   is None or len(self.b_loop_coords_aligned)   < 3):
            logger.warning("Loft strip (zipper): missing or too-short loops for creation.")
            return None
        
        b_loop, s_loop = self.b_loop_coords_aligned, self.s_loop_coords_ordered
        Nb, Ns = len(b_loop), len(s_loop)
        strip_verts = np.vstack((b_loop, s_loop))
        faces = []
        i, j = 0, 0

        for _ in range(Nb + Ns + 5):
            b_vidx_curr = i
            s_vidx_curr = j + Nb
            b_loop_idx_next = (i + 1) % Nb
            s_loop_idx_next = (j + 1) % Ns
            b_vidx_next = b_loop_idx_next
            s_vidx_next = s_loop_idx_next + Nb
            p_b_curr, p_s_curr = b_loop[i], s_loop[j]
            p_b_next, p_s_next = b_loop[b_loop_idx_next], s_loop[s_loop_idx_next]
            diag_len_sq_1 = np.sum((p_b_next - p_s_curr)**2)
            diag_len_sq_2 = np.sum((p_b_curr - p_s_next)**2)

            if diag_len_sq_1 < diag_len_sq_2:
                faces.append([b_vidx_curr, b_vidx_next, s_vidx_curr])
                i = b_loop_idx_next
            else:
                faces.append([b_vidx_curr, s_vidx_next, s_vidx_curr])
                j = s_loop_idx_next
            
            if i == 0 and j == 0:
                break
        
        if not faces:
            logger.warning("Zipper stitch failed to generate any faces.")
            return None

        strip = trimesh.Trimesh(vertices=strip_verts, faces=np.asarray(faces, dtype=int), process=False)
        strip.remove_degenerate_faces()
        if not MeshCleanProcess._is_mesh_valid_for_concat(strip, "ZipperStitchMesh"):
            logger.warning("Zipper stitch mesh failed validation.")
            return None
            
        strip.fix_normals()
        if strip.face_normals is not None and len(strip.face_normals) > 0:
            face_center = strip.triangles_center[0]
            vec_to_center = strip.centroid - face_center
            if np.dot(strip.face_normals[0], vec_to_center) > 0:
                strip.invert()
                
        return strip

    @staticmethod
    def _is_stitch_strip_valid(strip_mesh: Optional[trimesh.Trimesh]) -> bool:
        """
        Validates if a generated stitch strip is topologically sound.
        A valid strip must be a single component with exactly two boundary loops.
        """
        if not MeshCleanProcess._is_mesh_valid_for_concat(strip_mesh, "StitchStripValidation"):
            return False
        
        # A valid strip must be one single connected component.
        if len(strip_mesh.split(only_watertight=False)) > 1:
            logger.warning("Stitch strip validation failed: contains multiple components.")
            return False
            
        # A valid strip must have exactly two boundary loops (top and bottom).
        # Any other number indicates holes in the strip or other topological errors.
        boundary_loops = MeshCleanProcess.get_all_boundary_loops(strip_mesh, min_loop_len=3)
        if len(boundary_loops) != 2:
            logger.warning(f"Stitch strip validation failed: found {len(boundary_loops)} boundary loops instead of 2.")
            return False
            
        return True
    def _extract_main_boundary_loop_from_body_with_hole(self) -> Optional[np.ndarray]:
        if not self.body_with_hole_trimesh or \
           not MeshCleanProcess._is_mesh_valid_for_concat(self.body_with_hole_trimesh, "BodyForLoopExtract"):
            logger.warning("EXTRACT_MAIN_BWH_LOOP: body_with_hole_trimesh is invalid or None.")
            return None
        if self.s_loop_coords_ordered is None or len(self.s_loop_coords_ordered) < 3:
            logger.warning("EXTRACT_MAIN_BWH_LOOP: s_loop (SMPLX face) missing. Fallback to largest loop on body_with_hole.")
            all_loops_vidx = MeshCleanProcess.get_all_boundary_loops(self.body_with_hole_trimesh, min_loop_len=2)
            if not all_loops_vidx: return None
            largest_loop_vidx = max(all_loops_vidx, key=len, default=None)
            if largest_loop_vidx is not None and len(largest_loop_vidx) >=3 and largest_loop_vidx.max() < len(self.body_with_hole_trimesh.vertices):
                return self.body_with_hole_trimesh.vertices[largest_loop_vidx]
            return None

        all_loops_vidx = MeshCleanProcess.get_all_boundary_loops(self.body_with_hole_trimesh, min_loop_len=2)
        if not all_loops_vidx: return None
        best_loop_coords, min_avg_distance = None, float('inf')
        s_loop_kdtree = cKDTree(self.s_loop_coords_ordered)
        for loop_vidx in all_loops_vidx:
            if loop_vidx is None or len(loop_vidx) < 3 or loop_vidx.max() >= len(self.body_with_hole_trimesh.vertices): continue
            current_loop_coords = self.body_with_hole_trimesh.vertices[loop_vidx]
            distances, _ = s_loop_kdtree.query(current_loop_coords, k=1)
            avg_distance = np.mean(distances)
            if avg_distance < min_avg_distance: min_avg_distance, best_loop_coords = avg_distance, current_loop_coords
        
        if best_loop_coords is not None: return best_loop_coords
        logger.warning("EXTRACT_MAIN_BWH_LOOP: Failed to select main boundary loop.")
        return None

    def _attempt_body_driven_loft(self) -> bool:
        """
        Builds a watertight composite mesh by trying a cascade of methods.
        This version now uses a "Keep and Weld" assembly strategy that preserves
        all original SMPLX face vertices.
        """
        logger.info("=== STEP 4 & 5: Attempting Hybrid 'BODY_DRIVEN_LOFT' (Keep and Weld Strategy) ===")
        
        # Step 1: Extract the SMPLX face loop
        if not self._extract_smplx_face_loop():
            logger.error("LOFT_FAIL: Prep Step 1/3 failed. Could not extract a valid boundary loop from the SMPLX face mesh.")
            return False
        logger.info(f"LOFT_OK: Prep Step 1/3 - Extracted SMPLX face loop with {len(self.ordered_s_vidx_loop)} vertices.")

        # Step 2: Extract the main boundary loop from the body with the hole
        b_loop_pre_align = self._extract_main_boundary_loop_from_body_with_hole()
        if b_loop_pre_align is None or len(b_loop_pre_align) < 3:
            num_verts_found = len(b_loop_pre_align) if b_loop_pre_align is not None else 0
            num_loops_total = 0
            if self.body_with_hole_trimesh:
                all_loops = MeshCleanProcess.get_all_boundary_loops(self.body_with_hole_trimesh, min_loop_len=0)
                num_loops_total = len(all_loops)

            logger.error("LOFT_FAIL: Prep Step 2/3 failed. Could not extract a valid main hole loop from the body mesh.")
            logger.error(f"  - Diagnostic: Found {num_loops_total} total boundary loop(s) on the 'body_with_hole_trimesh' object.")
            logger.error(f"  - Diagnostic: The best candidate loop had {num_verts_found} vertices (minimum required is 3).")
            logger.error("  - ACTION: Please inspect the saved '..._debug_body_with_hole.obj' file. It likely has no hole or the hole is malformed.")
            return False
        logger.info(f"LOFT_OK: Prep Step 2/3 - Extracted main body hole loop with {len(b_loop_pre_align)} vertices.")

        # Step 3: Align the body and face loops
        if not self._align_body_loop_to_smplx_loop(b_loop_pre_align):
            logger.error("LOFT_FAIL: Prep Step 3/3 failed. Could not align the body and face loops for stitching.")
            return False
        logger.info("LOFT_OK: Prep Step 3/3 - Successfully aligned body and face loops.")

        base_name, ext = os.path.splitext(self.output_path)

        def _assemble(strip: trimesh.Trimesh) -> Optional[trimesh.Trimesh]:
            """
            Assembles the body, the original face, and the stitch strip by
            concatenating all three and then merging vertices at the seams.
            This preserves all original SMPLX face vertices.
            """
            try:
                body_h = self.body_with_hole_trimesh
                smplx_f = self.original_smplx_face_geom_tri

                # 1. Create a list of all three mesh parts to combine.
                #    Crucially, we are now including the *entire* SMPLX face.
                all_parts = [body_h, smplx_f, strip]
                
                # 2. Concatenate all parts into a single (disconnected) object.
                #    No vertices have been lost at this stage.
                final_mesh = trimesh.util.concatenate(all_parts)
                
                # 3. Perform the "weld". merge_vertices finds vertices that occupy the
                #    same 3D space (within a tolerance) and merges them into one.
                #    This is what connects the boundaries.
                final_mesh.merge_vertices()
                
                # 4. Standard cleanup.
                final_mesh.remove_unreferenced_vertices()
                final_mesh.remove_degenerate_faces()

                return final_mesh if MeshCleanProcess._is_mesh_valid_for_concat(final_mesh, "WeldedLoftResult") else None
            except Exception as e:
                logger.error(f"Welded assembly failed: {e}", exc_info=False)
                return None
        # --- END OF MODIFIED LOGIC ---
        
        # --- Method 1: Zipper ---
        logger.info("Attempting stitch method 1/3: Zipper")
        strip = self._create_loft_stitch_mesh_zipper()
        if self._is_stitch_strip_valid(strip):
            stitched = _assemble(strip)
            if stitched and self._verify_seam_closure_on_mesh(stitched, "ZipperResult"):
                self.stitched_mesh_intermediate = stitched
                return True
            logger.warning("Zipper stitch was valid, but resulted in an open seam or invalid assembly. Falling back.")
        else:
            logger.warning("Zipper strip was topologically invalid. Falling back.")

        # --- Method 2: Body Resample ---
        logger.info("Attempting stitch method 2/3: Body-Resample")
        strip = self._create_loft_stitch_mesh_resample_body()
        if self._is_stitch_strip_valid(strip):
            stitched = _assemble(strip)
            if stitched and self._verify_seam_closure_on_mesh(stitched, "BodyResampleResult"):
                self.stitched_mesh_intermediate = stitched
                return True
            logger.warning("Body-Resample stitch was valid, but resulted in an open seam or invalid assembly. Falling back.")
        else:
             logger.warning("Body-Resample strip was topologically invalid. Falling back.")
        
        # --- Method 3: Projection (Last Resort) ---
        logger.info("Attempting stitch method 3/3: Projection (Last Resort)")
        strip = self._create_loft_stitch_mesh_projection()
        if strip:
            stitched = _assemble(strip)
            if stitched:
                logger.info("SUCCESS: Fallback Projection method assembled a mesh.")
                self.stitched_mesh_intermediate = stitched
                return True

        logger.critical("All stitching strategies failed.")
        return False

    def _verify_seam_closure_on_mesh(self, mesh_to_check: trimesh.Trimesh, mesh_name_for_log: str) -> bool:
        # logger.info(f"--- Verifying Seam Closure on: {mesh_name_for_log} ---")
        if not MeshCleanProcess._is_mesh_valid_for_concat(mesh_to_check, f"SeamCheck_{mesh_name_for_log}"): return True 
        if self.s_loop_coords_ordered is None or len(self.s_loop_coords_ordered) < 3 or \
           self.b_loop_coords_aligned is None or len(self.b_loop_coords_aligned) < 3: return True 
        try:
            orig_seam_coords = np.vstack((self.s_loop_coords_ordered, self.b_loop_coords_aligned))
            if len(orig_seam_coords) == 0: return True
            orig_seam_kdtree = cKDTree(orig_seam_coords)
        except Exception: return True
        boundary_edges = MeshCleanProcess._get_boundary_edges_manually_from_faces(mesh_to_check)
        if boundary_edges is None or len(boundary_edges) == 0:
            logger.info(f"Seam check ({mesh_name_for_log}): No boundary edges. Seam closed."); return True 
        gap_thresh = getattr(self.config, 'seam_gap_verification_threshold', 1e-4) 
        gap_edges_indices = [] 
        for v_a, v_b in boundary_edges:
            if v_a >= len(mesh_to_check.vertices) or v_b >= len(mesh_to_check.vertices): continue
            try:
                d_a, _ = orig_seam_kdtree.query(mesh_to_check.vertices[v_a], k=1, distance_upper_bound=gap_thresh*1.1) 
                d_b, _ = orig_seam_kdtree.query(mesh_to_check.vertices[v_b], k=1, distance_upper_bound=gap_thresh*1.1)
            except Exception: continue
            if d_a < gap_thresh and d_b < gap_thresh: gap_edges_indices.append((v_a, v_b))
        if not gap_edges_indices:
            logger.info(f"Seam check ({mesh_name_for_log}): No gap edges. Seam effectively closed."); return True 
        else:
            logger.warning(f"Seam check ({mesh_name_for_log}): Found {len(gap_edges_indices)} potential gap edges."); return False 
        
    def _attempt_direct_weld(self) -> bool:
        """
        Builds a composite mesh by directly 'welding' the body hole loop to the
        SMPLX face loop. This strategy preserves ALL original SMPLX vertices.
        """
        logger.info("=== STEP 4 & 5: Attempting 'DIRECT_WELD' Strategy ===")
        
        # 1. Get the ordered vertex loops from the body hole and the SMPLX face.
        if not self._extract_smplx_face_loop(): return False
        b_loop_pre_align = self._extract_main_boundary_loop_from_body_with_hole()
        if b_loop_pre_align is None or len(b_loop_pre_align) < 3: return False
        if not self._align_body_loop_to_smplx_loop(b_loop_pre_align): return False
        
        try:
            body_h = self.body_with_hole_trimesh
            smplx_f = self.original_smplx_face_geom_tri

            # 2. Concatenate the two meshes. ALL original vertices are now preserved in a single object.
            combined_mesh = trimesh.util.concatenate(body_h, smplx_f)
            
            # 3. We need to find the vertex indices of our loops *within the new combined mesh*.
            num_body_verts = len(body_h.vertices)
            
            # The body loop indices are easy to find as they don't need an offset.
            kdtree_combined = cKDTree(combined_mesh.vertices)
            _, body_loop_indices_in_combined = kdtree_combined.query(self.b_loop_coords_aligned, k=1)
            
            # The SMPLX loop indices must be offset by the number of vertices in the body mesh.
            # We use the original index loop (`self.ordered_s_vidx_loop`) plus the offset.
            smplx_loop_indices_in_combined = self.ordered_s_vidx_loop + num_body_verts
            
            # 4. Use trimesh's utility to "zip" the two loops together by creating new faces.
            # We must ensure both loops have the same winding order for this to work.
            # We already aligned them in `_align_body_loop_to_smplx_loop`, but we double-check.
            
            # Resample the body loop to match the length of the SMPLX loop for trimesh.graph.stitch
            resampled_body_indices = trimesh.graph.resample_path(
                combined_mesh.vertices, body_loop_indices_in_combined, count=len(smplx_loop_indices_in_combined)
            )

            # Generate the faces for the stitch strip
            new_faces = trimesh.graph.stitch(
                combined_mesh, resampled_body_indices, smplx_loop_indices_in_combined
            )
            
            # 5. Add the new faces to the combined mesh
            combined_mesh.faces = np.vstack([combined_mesh.faces, new_faces])
            
            # 6. Final cleanup
            combined_mesh.remove_unreferenced_vertices()
            combined_mesh.remove_degenerate_faces()
            
            if MeshCleanProcess._is_mesh_valid_for_concat(combined_mesh, "DirectWeldResult"):
                self.stitched_mesh_intermediate = combined_mesh
                logger.info("SUCCESS: Direct Weld method created a valid stitched mesh.")
                return True
            else:
                logger.error("Direct Weld failed to produce a valid mesh.")
                return False

        except Exception as e:
            logger.error(f"An exception occurred during direct weld: {e}", exc_info=True)
            return False

    def _stitch_components(self) -> bool:
        """
        Dispatches to the chosen stitching method. The chosen method MUST succeed,
        otherwise the pipeline fails. There are no fallbacks.
        """
        if self.config.stitch_method == "direct_weld":
            if self._attempt_direct_weld():
                return True
        elif self.config.stitch_method == "body_driven_loft":
            if self._attempt_body_driven_loft():
                return True

        logger.critical(
            f"The chosen stitch method '{self.config.stitch_method}' failed to produce a valid result. "
            "The pipeline cannot continue."
        )
        return False

    def _apply_final_polish(self) -> bool:
        if not self.final_processed_mesh or \
           not MeshCleanProcess._is_mesh_valid_for_concat(self.final_processed_mesh, "MeshBeforeFinalPolish"):
            logger.warning("Skipping final polish: mesh invalid/missing."); return True 

        logger.info("=== STEP 5.5: Applying AGGRESSIVE Final Polish and Repair ===")
        temp_in = self._make_temp_path("polish_in", use_ply=True)
        temp_out = self._make_temp_path("polish_out", use_ply=True)
        polished_loaded = None
        ms_polish = None
        try:
            self.final_processed_mesh.export(temp_in)
            ms_polish = pymeshlab.MeshSet()
            ms_polish.load_new_mesh(temp_in)
            
            if ms_polish.current_mesh_id() != -1 and ms_polish.current_mesh().vertex_number() > 0:
                logger.info("...merging duplicate vertices...")
                ms_polish.meshing_remove_duplicate_vertices()

                logger.info("...repairing non-manifold edges...")
                try:
                    ms_polish.meshing_repair_non_manifold_edges(method='Split Vertices')
                except Exception:
                    try: 
                        ms_polish.meshing_repair_non_manifold_edges(method='Remove Faces')
                    except Exception: 
                        logger.warning("Could not repair non-manifold edges.")

                # --- The problematic degenerate face removal step has been REMOVED. ---
                # Trimesh will handle this more robustly upon final loading.
                
                logger.info("...final unreferenced vertex removal...")
                ms_polish.meshing_remove_unreferenced_vertices()

                is_mani = False
                try:
                    if ms_polish.get_topological_measures().get('non_manifold_edges', -1) == 0:
                        is_mani = True
                except Exception:
                    pass

                if is_mani:
                    logger.info("...mesh is now manifold, attempting to close remaining small holes...")
                    try:
                        ms_polish.meshing_close_holes(maxholesize=self.config.final_polish_max_hole_edges)
                    except Exception as e_ch:
                        logger.info(f"Polish: PML close_holes failed: {e_ch}")
                else:
                    logger.warning("...mesh is still not manifold after repairs, skipping hole filling.")

                ms_polish.compute_normal_per_face() 
                ms_polish.save_current_mesh(temp_out)
                
                # Trimesh's `process=True` flag will automatically handle the removal of
                # any degenerate faces that PyMeshLab might have missed or created.
                polished_loaded = trimesh.load_mesh(temp_out, process=True)
            
            if MeshCleanProcess._is_mesh_valid_for_concat(polished_loaded, "PolishedMeshPML"):
                self.final_processed_mesh = polished_loaded
                self.final_processed_mesh.fix_normals() 
                logger.info("Aggressive final polish applied.")
            else:
                logger.warning("Aggressive final polish resulted in invalid mesh. Keeping pre-polish version.")
        
        except Exception as e_polish:
            logger.warning(f"Error in final polish step: {e_polish}", exc_info=True)
        finally:
            if ms_polish is not None:
                del ms_polish
        
        return True
        
    def _verify_seam_closure(self) -> bool:
        logger.info(f"=== STEP 6 (Verify): Verifying Seam Closure ===")
        if not MeshCleanProcess._is_mesh_valid_for_concat(self.final_processed_mesh, "FinalMeshForSeamVerify"):
            logger.warning("Seam verify: Final mesh invalid/missing."); return True 
        if self.s_loop_coords_ordered is None or len(self.s_loop_coords_ordered) < 3 or \
           self.b_loop_coords_aligned is None or len(self.b_loop_coords_aligned) < 3: return True 
        try:
            orig_seam_coords = np.vstack((self.s_loop_coords_ordered, self.b_loop_coords_aligned))
            if len(orig_seam_coords) == 0: return True
            orig_seam_kdtree = cKDTree(orig_seam_coords)
        except Exception: logger.warning("Seam verify: KDTree build failed."); return True
        boundary_edges = MeshCleanProcess._get_boundary_edges_manually_from_faces(self.final_processed_mesh)
        if boundary_edges is None or len(boundary_edges) == 0:
            logger.info("Seam verify: Final mesh no boundary edges. Seam closed."); return True
        gap_thresh = getattr(self.config, 'seam_gap_verification_threshold', 1e-8) 
        gap_edges_found = []
        for v_a, v_b in boundary_edges:
            if v_a >= len(self.final_processed_mesh.vertices) or v_b >= len(self.final_processed_mesh.vertices): continue
            try:
                d_a, _ = orig_seam_kdtree.query(self.final_processed_mesh.vertices[v_a],k=1,distance_upper_bound=gap_thresh*1.1) 
                d_b, _ = orig_seam_kdtree.query(self.final_processed_mesh.vertices[v_b],k=1,distance_upper_bound=gap_thresh*1.1)
            except Exception: continue
            if d_a < gap_thresh and d_b < gap_thresh: gap_edges_found.append((v_a, v_b))
        if not gap_edges_found:
            logger.info(f"Seam verify: No gap edges found. Seam effectively closed (thresh {gap_thresh:.1e})."); return True 
        logger.warning(f"Seam verify: Found {len(gap_edges_found)} potential gap edges (thresh {gap_thresh:.1e})."); return False 

    def _verify_smplx_face_integrity(self) -> bool:
        """
        Checks the final mesh to ensure the original SMPLX face geometry has not been altered.
        If debug_smply_integrity_trace is True, it will capture lost vertices for tracing.
        """
        logger.info("=== STEP 7 (Verify): Verifying SMPLX Face Integrity ===")
        if self.final_processed_mesh is None or not MeshCleanProcess._is_mesh_valid_for_concat(self.final_processed_mesh, "FinalMeshForIntegrityCheck"):
            logger.warning("SMPLX Integrity Check: Final mesh is invalid or missing. Skipping.")
            return True

        if self.original_smplx_v_pre_graft is None:
            logger.warning("SMPLX Integrity Check: Original geometry baseline not found. Skipping.")
            return True

        try:
            final_mesh_kdtree = cKDTree(self.final_processed_mesh.vertices)
            distances, new_indices = final_mesh_kdtree.query(self.original_smplx_v_pre_graft, k=1)

            # Check 1: Were any vertices lost or moved significantly?
            lost_vertices_mask = distances > 1e-5
            if np.any(lost_vertices_mask):
                lost_count = np.sum(lost_vertices_mask)
                lost_indices_original = np.where(lost_vertices_mask)[0]
                logger.warning(f"SMPLX Integrity FAIL: {lost_count} original vertices seem to be lost or moved.")
                logger.warning(f"  - Original indices of lost vertices: {lost_indices_original[:10]}...")

                # --- INITIATE DEBUG TRACING ---
                if self.config.debug_smply_integrity_trace:
                    self.debug_tracked_vertex_coords = self.original_smplx_v_pre_graft[lost_vertices_mask]
                    logger.warning(f"*** Initiating integrity trace for {len(self.debug_tracked_vertex_coords)} lost vertices. ***")
                # --- END ---
                return False

            # ... (rest of the function for merge/topology checks remains the same) ...
            num_original_verts = len(self.original_smplx_v_pre_graft)
            num_unique_found_verts = len(np.unique(new_indices))
            if num_unique_found_verts < num_original_verts:
                merged_count = num_original_verts - num_unique_found_verts
                logger.warning(f"SMPLX Integrity FAIL: {merged_count} original vertices were merged into others.")
                return False
                
            if self.original_smplx_edges_pre_graft is not None:
                original_vidx_to_new_vidx_map = dict(enumerate(new_indices))
                reconstructed_edges = set()
                for v1_orig, v2_orig in self.original_smplx_edges_pre_graft:
                    if v1_orig in original_vidx_to_new_vidx_map and v2_orig in original_vidx_to_new_vidx_map:
                        reconstructed_edges.add(tuple(sorted((original_vidx_to_new_vidx_map[v1_orig], original_vidx_to_new_vidx_map[v2_orig]))))
                missing_edges = reconstructed_edges - {tuple(sorted(e)) for e in self.final_processed_mesh.edges_unique}
                if missing_edges:
                    logger.warning(f"SMPLX Integrity FAIL: {len(missing_edges)} topological edges were broken.")
                    return False
                    
            logger.info("SMPLX Integrity PASS: Original face geometry and topology preserved in the final mesh.")
            return True

        except Exception as e:
            logger.error(f"An exception occurred during SMPLX integrity check: {e}", exc_info=True)
            return False

    def _fill_seam_holes_by_fan(self) -> bool:
        """
        Identifies boundary loops corresponding to the grafting seam and fills them using fan triangulation.
        This version uses an existing vertex on the loop as the fan's origin (no new vertices).
        """
        if not self.stitched_mesh_intermediate or \
           not MeshCleanProcess._is_mesh_valid_for_concat(self.stitched_mesh_intermediate, "MeshBeforeSeamFill"):
            logger.warning("Skipping seam hole fill: stitched mesh invalid/missing.")
            self.final_processed_mesh = self.stitched_mesh_intermediate
            return True
        
        if self.config.stitch_method not in ["trimesh_dynamic_loft", "body_driven_loft"]:
            self.final_processed_mesh = self.stitched_mesh_intermediate
            return True

        logger.info("Attempting FAN TRIANGULATION SEAM HOLE FILL post-loft (Root Vertex Method)...")
        mesh_to_fill = self.stitched_mesh_intermediate.copy()
        
        mesh_to_fill.fix_normals()

        # We will modify the faces list in place
        current_faces = list(mesh_to_fill.faces)
        
        all_loops = MeshCleanProcess.get_all_boundary_loops(mesh_to_fill, min_loop_len=3)
        if not all_loops:
            self.final_processed_mesh = self.stitched_mesh_intermediate
            return True

        kdt_s = cKDTree(self.s_loop_coords_ordered) if self.s_loop_coords_ordered is not None else None
        kdt_b = cKDTree(self.b_loop_coords_aligned) if self.b_loop_coords_aligned is not None else None
        prox_thresh = 0.025

        added_any_faces = False
        for loop_idx, loop_vidx in enumerate(all_loops):
            if not (3 <= len(loop_vidx) <= self.config.max_seam_hole_fill_vertices):
                continue

            loop_coords = mesh_to_fill.vertices[loop_vidx]
            
            is_seam = False
            if kdt_s and kdt_b:
                d_s, _ = kdt_s.query(loop_coords, k=1, distance_upper_bound=prox_thresh * 2)
                d_b, _ = kdt_b.query(loop_coords, k=1, distance_upper_bound=prox_thresh * 2)
                finite_d_s, finite_d_b = d_s[np.isfinite(d_s)], d_b[np.isfinite(d_b)]
                if finite_d_s.size > 0 and finite_d_b.size > 0 and np.mean(finite_d_s) < prox_thresh and np.mean(finite_d_b) < prox_thresh:
                    is_seam = True
            
            if not is_seam:
                continue

            try:
                # --- FAN TRIANGULATION (ROOT VERTEX METHOD) ---
                
                # 1. Find the average normal of faces surrounding the hole for orientation check.
                neighbor_face_indices_nested = mesh_to_fill.vertex_faces[loop_vidx]
                if len(neighbor_face_indices_nested) > 0:
                    valid_faces = [f for f_list in neighbor_face_indices_nested for f in f_list if f != -1]
                    if not valid_faces: continue
                    unique_face_indices = np.unique(valid_faces)
                    target_normal = trimesh.util.unitize(np.mean(mesh_to_fill.face_normals[unique_face_indices], axis=0))
                else:
                    continue

                # 2. Choose a root vertex for the fan. The first vertex in the loop is a stable choice.
                root_vidx = loop_vidx[0]

                # 3. Create the fan of faces from the root vertex.
                # We iterate from the second vertex to the one before the last, creating triangles.
                new_patch_faces = []
                for i in range(1, len(loop_vidx) - 1):
                    v1_idx = loop_vidx[i]
                    v2_idx = loop_vidx[i + 1]
                    new_patch_faces.append([root_vidx, v1_idx, v2_idx])
                
                # 4. Check and fix the orientation of the new patch.
                if new_patch_faces and np.any(np.abs(target_normal) > 1e-6):
                    # We can check the normal of the first new face to determine the whole patch's orientation.
                    v0, v1, v2 = mesh_to_fill.vertices[new_patch_faces[0]]
                    test_normal = trimesh.util.unitize(np.cross(v1 - v0, v2 - v0))
                    
                    if np.dot(test_normal, target_normal) < 0.0:
                        logger.info(f"Flipping fan patch for loop {loop_idx} to match surrounding geometry.")
                        # Reverse the winding of all new faces
                        new_patch_faces = [face[::-1] for face in new_patch_faces]

                current_faces.extend(new_patch_faces)
                added_any_faces = True
                logger.info(f"Filled seam hole (loop {loop_idx}) with {len(new_patch_faces)} fan faces.")

            except Exception as e:
                logger.warning(f"Failed to fan-fill loop {loop_idx}: {e}", exc_info=True)

        if added_any_faces:
            # Rebuild the mesh with the added faces. No new vertices were added.
            updated_mesh = trimesh.Trimesh(vertices=mesh_to_fill.vertices, faces=np.array(current_faces, dtype=int), process=True)
            
            if MeshCleanProcess._is_mesh_valid_for_concat(updated_mesh, "MeshAfterFanFill"):
                self.final_processed_mesh = updated_mesh
                logger.info("Fan-fill seam hole filling applied.")
            else:
                logger.warning("Mesh invalid after fan-fill. Reverting.")
                self.final_processed_mesh = self.stitched_mesh_intermediate
        else:
            self.final_processed_mesh = self.stitched_mesh_intermediate
            
        return True

    def _perform_global_orientation_fix(self) -> bool:
        """
        Performs a robust, global orientation of the entire mesh using Trimesh's
        built-in graph-based algorithm. This is the definitive way to ensure
        all face normals are consistent.
        """
        logger.info("=== Performing Global and Final Mesh Orientation Fix ===")
        if not self.final_processed_mesh or self.final_processed_mesh.is_empty:
            logger.warning("Skipping global orientation fix: final mesh is missing or empty.")
            return True

        mesh_before_fix = self.final_processed_mesh.copy()
        try:
            # fix_normals() uses a robust traversal algorithm to make all face windings consistent.
            # multibody=True is important as it handles disconnected components gracefully.
            self.final_processed_mesh.fix_normals(multibody=True)

            if not MeshCleanProcess._is_mesh_valid_for_concat(self.final_processed_mesh, "PostGlobalOrientation"):
                logger.warning("Global orientation fix resulted in an invalid mesh. REVERTING this step.")
                self.final_processed_mesh = mesh_before_fix
            else:
                logger.info("Global orientation fix completed successfully.")
            
            return True

        except Exception as e:
            logger.error(f"Error during global orientation fix: {e}", exc_info=True)
            self.final_processed_mesh = mesh_before_fix # Revert on any error
            return False

    def _perform_aggressive_final_repair(self) -> bool:
        if not self.final_processed_mesh or \
           not MeshCleanProcess._is_mesh_valid_for_concat(self.final_processed_mesh, "MeshBeforeFinalRepair"):
            logger.warning("Skipping final aggressive repair: mesh invalid/missing."); return True

        logger.info("=== STEP 5.5: Applying ULTIMATE Final Repair and Polish ===")
        temp_in = self._make_temp_path("agg_repair_in", use_ply=True)
        temp_out = self._make_temp_path("agg_repair_out", use_ply=True)
        repaired_mesh = None
        ms_repair = None
        try:
            self.final_processed_mesh.export(temp_in)
            ms_repair = pymeshlab.MeshSet()
            ms_repair.load_new_mesh(temp_in)

            if ms_repair.current_mesh_id() != -1 and ms_repair.current_mesh().vertex_number() > 0:
                logger.info("...merging duplicate vertices as a pre-pass...")
                ms_repair.meshing_remove_duplicate_vertices()

                # --- ULTIMATE REPAIR SEQUENCE ---
                logger.info("...Step 1/2: Splitting non-manifold vertices (the root of the problem)...")
                try:
                    # This is the most powerful filter for fixing "bad contiguous edges."
                    ms_repair.meshing_repair_non_manifold_vertices_by_splitting()
                except (AttributeError, pymeshlab.PyMeshLabException):
                    logger.warning("Could not run `repair_non_manifold_vertices_by_splitting`. Your PyMeshLab version might be too old. Proceeding with edge-based repair.")

                logger.info("...Step 2/2: Iteratively repairing non-manifold edges...")
                # After splitting vertices, we clean up the resulting edges.
                for i in range(3): # Run multiple passes to be thorough
                    num_nm_edges = ms_repair.get_topological_measures().get('non_manifold_edges', -1)
                    if num_nm_edges == 0:
                        logger.info(f"...non-manifold edge repair complete after {i} iterations.")
                        break
                    ms_repair.meshing_repair_non_manifold_edges(method='Split Vertices')
                
                # --- DEGENERATE FACE REMOVAL IS NOW HANDLED BY TRIMESH ON LOAD ---
                # This makes the pipeline robust against older PyMeshLab versions.
                logger.info("...topological repair complete. Removing unreferenced vertices...")
                ms_repair.meshing_remove_unreferenced_vertices()
                # --- END OF REPAIR SEQUENCE ---

                # Now, attempt final hole filling on the topologically repaired mesh.
                if ms_repair.get_topological_measures().get('non_manifold_edges', 0) == 0:
                    logger.info("...mesh is now manifold, closing final holes.")
                    try:
                        ms_repair.meshing_close_holes(maxholesize=self.config.final_polish_max_hole_edges)
                    except Exception as e_ch:
                        logger.info(f"Final hole filling failed: {e_ch}")
                else:
                    logger.warning("Mesh is still non-manifold after ultimate repair. Cannot fill holes.")
                
                ms_repair.save_current_mesh(temp_out)
                
                # Load with `process=True` to let Trimesh handle final geometric cleanup,
                # including the removal of any zero-area faces.
                repaired_mesh = trimesh.load_mesh(temp_out, process=True)

            if repaired_mesh and MeshCleanProcess._is_mesh_valid_for_concat(repaired_mesh, "AggressiveRepairResult"):
                self.final_processed_mesh = repaired_mesh
                self.final_processed_mesh.fix_normals()
                logger.info("Ultimate final repair and polish completed successfully.")
            else:
                logger.warning("Ultimate repair resulted in invalid mesh. Using pre-repair version.")

        except Exception as e:
            logger.error(f"An exception occurred during the ultimate repair step: {e}", exc_info=True)
        finally:
            if ms_repair: del ms_repair
        
        return True

    def _fill_internal_face_holes_on_final_mesh(self) -> bool:
        """
        Intelligently finds and fills the internal holes of the face (e.g., mouth, nostrils)
        on the final, fully assembled mesh.
        """
        if not self.config.close_face_holes_at_end:
            return True # Feature is turned off

        logger.info("=== Attempting to fill internal face holes on final mesh ===")
        if self.final_processed_mesh is None or self.final_processed_mesh.is_empty:
            logger.warning("Skipping: final mesh is missing or empty.")
            return True
            
        if self.original_smplx_v_pre_graft is None or self.b_loop_coords_aligned is None:
            logger.warning("Skipping: Missing reference geometry for face/body loops.")
            return True

        mesh = self.final_processed_mesh
        all_loops = MeshCleanProcess.get_all_boundary_loops(mesh, min_loop_len=3)
        if not all_loops:
            logger.info("No boundary loops found to fill.")
            return True
        
        # Build KDTrees from our original reference geometry
        face_ref_kdtree = cKDTree(self.original_smplx_v_pre_graft)
        body_seam_ref_kdtree = cKDTree(self.b_loop_coords_aligned)

        # A distance threshold to decide if a loop "belongs" to a reference
        proximity_threshold = 0.05 
        loops_to_fill = []
        
        for loop_vidx in all_loops:
            loop_coords = mesh.vertices[loop_vidx]
            
            # Check distance to original face vertices
            dist_to_face, _ = face_ref_kdtree.query(loop_coords, k=1)
            avg_dist_to_face = np.mean(dist_to_face)
            
            # Check distance to original body seam vertices
            dist_to_body, _ = body_seam_ref_kdtree.query(loop_coords, k=1)
            avg_dist_to_body = np.mean(dist_to_body)
            
            # A hole is an "internal face hole" if it is VERY close to the face reference
            # points but NOT close to the body seam reference points.
            if avg_dist_to_face < proximity_threshold and avg_dist_to_body > proximity_threshold:
                loops_to_fill.append(loop_vidx)

        if not loops_to_fill:
            logger.info("No holes were identified as internal face holes.")
            return True

        logger.info(f"Identified {len(loops_to_fill)} internal face hole(s) to fill.")
        
        # Now, fill the identified loops using the robust fan-fill method
        current_faces = list(mesh.faces)
        for loop_vidx in loops_to_fill:
            root_vidx = loop_vidx[0]
            for i in range(1, len(loop_vidx) - 1):
                v1_idx = loop_vidx[i]
                v2_idx = loop_vidx[i + 1]
                current_faces.append([root_vidx, v1_idx, v2_idx])
                
        # Create a new mesh with the filled holes
        filled_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=current_faces)
        
        if MeshCleanProcess._is_mesh_valid_for_concat(filled_mesh, "FinalMeshWithHolesFilled"):
            self.final_processed_mesh = filled_mesh
            logger.info("Successfully filled internal face holes on final mesh.")
        else:
            logger.warning("Mesh became invalid after filling face holes. Reverting.")

        return True

    def _smooth_simplified_body_pre_hole_cut(self) -> bool:
        """
        Applies Laplacian smoothing to the entire simplified body mesh before cutting the hole.
        This is an optional step controlled by the config.
        """
        if self.config.pre_hole_cut_smoothing_iterations <= 0:
            return True

        if not self.simplified_body_trimesh or not MeshCleanProcess._is_mesh_valid_for_concat(self.simplified_body_trimesh, "PreBodySmooth"):
            logger.warning("Simplified body mesh is invalid or None, skipping pre-hole smoothing.")
            return True 
    
        logger.info(f"=== STEP 2.75: Smoothing simplified body ({self.config.pre_hole_cut_smoothing_iterations} iter, lambda={self.config.pre_hole_cut_smoothing_lambda}) ===")
        body_before_smooth = self.simplified_body_trimesh.copy()
        
        try:
            smoothed_body = trimesh.smoothing.filter_laplacian(
                self.simplified_body_trimesh,
                iterations=self.config.pre_hole_cut_smoothing_iterations,
                lamb=self.config.pre_hole_cut_smoothing_lambda
            )

            if MeshCleanProcess._is_mesh_valid_for_concat(smoothed_body, "PostBodySmooth"):
                if self.config.use_open3d_cleaning:
                    smoothed_body = self._clean_with_open3d(smoothed_body, "PostBodySmoothing_O3D")
                else:
                    logger.info("...ensuring consistent face orientation after smoothing (Trimesh).")
                    smoothed_body.fix_normals()

                self.simplified_body_trimesh = smoothed_body
                logger.info("Successfully smoothed and cleaned simplified body.")
            else:
                logger.warning("Smoothing resulted in an empty or invalid mesh. Reverting to pre-smooth state.")
                self.simplified_body_trimesh = body_before_smooth

        except Exception as e:
            logger.error(f"An exception occurred during simplified body smoothing: {e}. Reverting to pre-smooth state.", exc_info=True)
            self.simplified_body_trimesh = body_before_smooth
    
        return True

    def _split_and_save_main_and_secondary_components(self) -> bool:
        """
        Splits the mesh into its disconnected components. It saves the largest
        component (the body) as the main output and intelligently combines and
        saves any smaller components (like eyeballs) to a separate file.
        This version ensures the resulting mesh is fully processed.
        """
        logger.info("--- Splitting mesh into main (body) and secondary (eyeballs) components ---")
        if not self.final_processed_mesh or self.final_processed_mesh.is_empty:
            return True

        mesh_before_split = self.final_processed_mesh.copy()
        
        try:
            components = self.final_processed_mesh.split(only_watertight=False)
            
            if len(components) > 1:
                logger.info(f"Found {len(components)} disconnected components.")
                components.sort(key=lambda c: len(c.faces), reverse=True)
                largest_component = components[0]
                secondary_components = components[1:]

                if MeshCleanProcess._is_mesh_valid_for_concat(largest_component, "LargestComponent"):
                    
                    # --- THE DEFINITIVE FIX ---
                    # Re-create the Trimesh object from its core data with process=True.
                    # This forces a full build of all internal graph data, including
                    # any version of boundary properties that may or may not exist.
                    # This is the most robust way to solve the raw mesh problem.
                    logger.info("Re-processing the largest component to ensure graph integrity...")
                    self.final_processed_mesh = trimesh.Trimesh(
                        vertices=largest_component.vertices,
                        faces=largest_component.faces,
                        process=True
                    )
                    logger.info("Successfully isolated and processed the main body component.")
                    # --- END OF FIX ---

                else:
                    logger.error("The largest component was invalid after splitting. Reverting.")
                    self.final_processed_mesh = mesh_before_split
                    return False

                if secondary_components:
                    logger.info(f"Found {len(secondary_components)} secondary component(s) to save separately.")
                    secondary_mesh = trimesh.util.concatenate(secondary_components)
                    base_name, ext = os.path.splitext(self.output_path)
                    secondary_path = f"{base_name}_eyeballs{ext}"
                    secondary_mesh.export(secondary_path)
                    logger.info(f"Saved secondary components (eyeballs) to: {secondary_path}")
            else:
                logger.info("Mesh is already a single connected component. No splitting needed.")

            return True

        except Exception as e:
            logger.error(f"An error occurred during component splitting: {e}", exc_info=True)
            self.final_processed_mesh = mesh_before_split
            return False

    def _final_watertightness_check_and_save(self) -> bool:
        """
        Saves the final processed mesh unconditionally, then performs a
        watertightness check and returns the result of that check.
        A False return indicates the pipeline produced a non-watertight mesh,
        even though the file was saved.
        """
        if not self.final_processed_mesh or not MeshCleanProcess._is_mesh_valid_for_concat(self.final_processed_mesh, "FinalMeshForSave"):
            logger.critical("Cannot save final mesh: Mesh is invalid or missing.")
            return False

        # --- STEP 1: Unconditionally save the file ---
        logger.info(f"--- Saving final processed mesh to: {self.output_path} ---")
        try:
            self.final_processed_mesh.export(self.output_path)
            logger.info("File saved successfully.")
        except Exception as e:
            logger.critical(f"Failed to export the final mesh: {e}", exc_info=True)
            return False # The save operation itself failed.

        # --- STEP 2: Perform the quality check and report on it ---
        logger.info("--- Performing Final Watertightness Check on Saved Mesh ---")
        
        if self.final_processed_mesh.is_watertight:
            logger.info("SUCCESS: The saved mesh is watertight.")
            return True # The pipeline's goal was met.
        else:
            logger.warning("--- WATERTIGHTNESS CHECK FAILED ---")
            logger.warning("The saved mesh is NOT watertight, but the file was saved as requested.")
            
            # Use a robust try-except block for diagnostics
            try:
                # This block will ONLY run if the necessary attributes exist.
                is_manifold = self.final_processed_mesh.is_manifold
                boundary_edge_count = len(self.final_processed_mesh.boundary_edges)
                
                if boundary_edge_count > 0:
                    logger.warning(f"  - Diagnostics: Found {boundary_edge_count} boundary edges (gaps in the mesh).")
                if not is_manifold:
                    logger.warning("  - Diagnostics: Mesh is not manifold (e.g., contains edges shared by more than two faces).")
            
            except AttributeError:
                # If ANY attribute is missing, fall back to a generic message.
                logger.warning("  - Detailed diagnostics unavailable for this Trimesh version.")

            # Return False to indicate the watertightness GOAL was not met.
            return False
                        
    def _retain_largest_component_only(self) -> bool:
        """
        Ensures the mesh consists of only one single connected component by
        discarding any smaller, disconnected "islands" of geometry.
        """
        logger.info("--- Verifying mesh consists of a single connected component ---")
        if not self.final_processed_mesh or self.final_processed_mesh.is_empty:
            return True # Nothing to do

        mesh_before_split = self.final_processed_mesh.copy()
        
        try:
            # The split operation returns a list of Trimesh objects
            components = self.final_processed_mesh.split(only_watertight=False)
            
            if len(components) > 1:
                logger.warning(f"Found {len(components)} disconnected components. Retaining only the largest.")
                
                # Find the largest component by number of faces
                largest_component = max(components, key=lambda c: len(c.faces))
                
                if MeshCleanProcess._is_mesh_valid_for_concat(largest_component, "LargestComponent"):
                    self.final_processed_mesh = largest_component
                    logger.info("Successfully isolated the largest mesh component.")
                else:
                    logger.error("The largest component was invalid after splitting. Reverting.")
                    self.final_processed_mesh = mesh_before_split
                    return False
            else:
                logger.info("Mesh is already a single connected component. No action needed.")

            return True

        except Exception as e:
            logger.error(f"An error occurred during component splitting: {e}", exc_info=True)
            self.final_processed_mesh = mesh_before_split
            return False

    def process(self) -> Optional[trimesh.Trimesh]:
        """
        Executes the full face grafting pipeline.
        Returns the processed Trimesh object if successful, None otherwise.
        """
        logger.info(f"--- Starting Face Grafting Pipeline (V{self.INTERNAL_VERSION_TRACKER}) ---")
        pipeline_ok = False
        final_mesh_for_return = None
        try:
            if not self._load_and_simplify_meshes(): return None
            if not self._perform_iterative_body_repair(): return None
            if not self._smooth_simplified_body_pre_hole_cut(): return None
            if not self._determine_hole_faces_and_create_body_with_hole(): return None
            if not self._stitch_components(): return None 
            
            self.final_processed_mesh = self.stitched_mesh_intermediate
            
            self._fill_seam_holes_by_fan() 
            self._apply_final_polish()

            # --- REORDERED VERIFICATION STEPS (AS REQUESTED) ---
            # 1. Verify SMPLX integrity on the fully assembled mesh BEFORE component splitting.
            self._verify_smplx_face_integrity()

            # 2. Retain only the largest component.
            self._retain_largest_component_only()
            # --- END OF REORDERED STEPS ---
            
            if self._final_watertightness_check_and_save():
                pipeline_ok = True
                final_mesh_for_return = self.final_processed_mesh
            else:
                return None
            
            # The seam closure check is still performed on the final, saved mesh.
            self._verify_seam_closure()
            
            return final_mesh_for_return
        except Exception as e_main_pipeline: 
            logger.error(f"--- Pipeline V{self.INTERNAL_VERSION_TRACKER} Failed with unhandled exception: {e_main_pipeline}", exc_info=True)
            return None
        finally:
            self._cleanup_temp_files()
            if not pipeline_ok: 
                logger.error(f"Pipeline V{self.INTERNAL_VERSION_TRACKER} did not complete successfully.")
                                
    @staticmethod
    def _iterative_non_manifold_repair_pml_aggressive(
        input_mesh: trimesh.Trimesh,
        pml_hole_fill_max_edges: int,
        max_main_iterations: int = 3,
        max_edge_repair_iters: int = 3,
        max_hole_fill_iters: int = 2,
        min_component_faces_to_keep: int = 50,
        merge_vertices_at_end: bool = True,
        temp_file_prefix: str = "nm_repair_agg"        
    ) -> Optional[trimesh.Trimesh]:

        if not MeshCleanProcess._is_mesh_valid_for_concat(
            input_mesh, f"InputAggRepair_{temp_file_prefix}"
        ):
            logger.warning(
                f"Aggressive Repair: Input invalid '{temp_file_prefix}'. Skipping."
            )
            return input_mesh

        logger.info(f"--- Starting AGGRESSIVE Iterative Repair: '{temp_file_prefix}' ---")

        current_mesh = input_mesh.copy()
        local_temps: List[str] = []

        def mk_tmp(sfx: str) -> str:
            fd, path = tempfile.mkstemp(
                suffix=f"_{sfx}.ply", prefix=f"{temp_file_prefix}_"
            )
            os.close(fd)
            local_temps.append(path)
            return path

        for main_iter in range(max_main_iterations):

            if not MeshCleanProcess._is_mesh_valid_for_concat(
                current_mesh, f"PreMainIter{main_iter}"
            ):
                logger.error(
                    f"AggRepair: Mesh invalid before main iter {main_iter + 1}. Aborting."
                )
                break

            tmp_in = mk_tmp(f"iter{main_iter}_in")
            current_mesh.export(tmp_in)

            ms: Optional[pymeshlab.MeshSet] = None
            try:
                ms = pymeshlab.MeshSet()
                ms.load_new_mesh(tmp_in)

                if ms.current_mesh_id() == -1 or ms.current_mesh().vertex_number() == 0:
                    logger.error("AggRepair: Mesh empty/invalid in PML. Aborting iter.")
                    break

                # ---- edge-repair stage -------------------------------------------------
                for _ in range(max_edge_repair_iters):
                    try:
                        ms.set_selection_none()
                        ms.apply_filter("compute_selection_by_non_manifold_edges_per_face")
                        if ms.current_mesh().selected_face_number() == 0:
                            break
                        ms.meshing_repair_non_manifold_edges(method="Split Vertices")
                    except pymeshlab.PyMeshLabException:
                        try:
                            ms.set_selection_none()
                            ms.apply_filter(
                                "compute_selection_by_non_manifold_edges_per_face"
                            )
                            if ms.current_mesh().selected_face_number() > 0:
                                ms.meshing_repair_non_manifold_edges(method="Remove Faces")
                        except pymeshlab.PyMeshLabException:
                            logger.warning("AggRepair: Edge repair (remove) failed.")
                            break

                # ---- basic clean-ups ---------------------------------------------------
                try:
                    ms.meshing_remove_duplicate_faces()
                    ms.meshing_remove_duplicate_vertices()
                    ms.meshing_remove_unreferenced_vertices()
                except Exception:
                    pass

                # ---- drop tiny islands --------------------------------------------------
                if min_component_faces_to_keep > 0:
                    try:
                        ms.meshing_remove_connected_component_by_face_number(
                            mincomponentsize=min_component_faces_to_keep
                        )
                    except Exception:
                        pass

                # ---- decide if hole-fill is safe ---------------------------------------
                is_mani_for_fill = False
                try:
                    topo = ms.get_topological_measures()
                    is_mani_for_fill = topo.get("non_manifold_edges", -1) == 0
                except Exception:
                    pass

                # ---- hole-fill loop -----------------------------------------------------
                if is_mani_for_fill:
                    for _ in range(max_hole_fill_iters):
                        try:
                            topo = ms.get_topological_measures()
                            be_before = topo.get("boundary_edges", -1)
                        except Exception:
                            be_before = -1

                        if be_before == 0:
                            break

                        try:
                            ms.meshing_close_holes(maxholesize=pml_hole_fill_max_edges)
                        except Exception:
                            break

                        try:
                            topo = ms.get_topological_measures()
                            be_after = topo.get("boundary_edges", -1)
                        except Exception:
                            be_after = -1

                        if (be_after == be_before and be_before not in (0, -1)) or be_after == 0:
                            break

                # ---- reload back to trimesh --------------------------------------------
                tmp_out = mk_tmp(f"iter{main_iter}_out")
                ms.save_current_mesh(tmp_out)
                reloaded = trimesh.load_mesh(tmp_out, process=False)

                if MeshCleanProcess._is_mesh_valid_for_concat(
                    reloaded, f"ReloadPMLIter{main_iter}"
                ):
                    current_mesh = reloaded
                else:
                    logger.error(
                        f"AggRepair: Mesh invalid after PML reload iter {main_iter + 1}."
                    )
                    break

            except Exception as e:
                logger.error(
                    f"AggRepair: PML processing iter {main_iter + 1} error: {e}",
                    exc_info=True,
                )
                break
            finally:
                if ms is not None:
                    del ms

        # ──────────────────────────────────────────────────────────────────────
        # Final clean-up in trimesh
        # ──────────────────────────────────────────────────────────────────────
        if merge_vertices_at_end and current_mesh is not None:
            cleaned = current_mesh.copy()
            cleaned.merge_vertices()                     # default tolerance
            cleaned.remove_unreferenced_vertices()
            
            # --- DEPRECATION FIX ---
            non_degenerate_face_mask = cleaned.nondegenerate_faces()
            if np.any(~non_degenerate_face_mask):
                cleaned.update_faces(non_degenerate_face_mask)
            # --- END FIX ---
            
            cleaned.fix_normals()

            if MeshCleanProcess._is_mesh_valid_for_concat(cleaned, "AggRepairFinalClean"):
                current_mesh = cleaned
            else:
                logger.warning("AggRepair: Final Trimesh clean made mesh invalid.")

        # remove temp files
        for f in local_temps:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

        if not MeshCleanProcess._is_mesh_valid_for_concat(
            current_mesh, f"FinalAggRepair_{temp_file_prefix}"
        ):
            logger.error("AggRepair: Mesh invalid at end. Returning None.")
            return None

        # logger.info(f"--- Finished AGGRESSIVE Iterative Repair for '{temp_file_prefix}' ---")
        return current_mesh

    def _perform_iterative_body_repair(self) -> bool:
        if self.simplified_body_trimesh is None: logger.error("Iterative repair: body None."); return False
        if self.config.iterative_repair_s1_iters > 0:
            logger.info("=== STEP 2.5: AGGRESSIVE Iterative Repair on Simplified Body ===")
            body_before = self.simplified_body_trimesh.copy()
            repaired = FaceGraftingPipeline._iterative_non_manifold_repair_pml_aggressive(
                self.simplified_body_trimesh, 
                pml_hole_fill_max_edges=self.config.final_polish_max_hole_edges, 
                max_main_iterations=self.config.iterative_repair_s1_iters,
                temp_file_prefix=f"body_s1_agg_repair_v{self.INTERNAL_VERSION_TRACKER}"
            )
            if repaired is not None and MeshCleanProcess._is_mesh_valid_for_concat(repaired, "BodyAfterAggRepairS1"):
                self.simplified_body_trimesh = repaired
            else:
                logger.warning("Aggressive body repair (S1) failed/invalid. Reverting."); self.simplified_body_trimesh = body_before
                if not MeshCleanProcess._is_mesh_valid_for_concat(self.simplified_body_trimesh, "BodyAfterFailedAggRepair"):
                    logger.error("Body mesh invalid after reverting from failed agg repair. Critical."); return False
        return True

class MeshCleanProcess:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.ms = pymeshlab.MeshSet()

    def load_mesh(self):
        try:
            self.ms.load_new_mesh(self.input_path)
            if self.ms.current_mesh().vertex_number() == 0:
                 logger.warning(f"Loaded mesh {self.input_path} is empty."); return False
            return True
        except pymeshlab.PyMeshLabException as e: logger.error(f"Error loading {self.input_path}: {e}"); return False

    def clean_mesh(self):
        if self.ms.current_mesh().vertex_number() == 0: return 
        try:
            self.ms.meshing_remove_duplicate_faces(); self.ms.meshing_remove_duplicate_vertices()
            try: self.ms.meshing_repair_non_manifold_edges()
            except Exception: logger.warning("Could not repair non-manifold edges (PML).")
            try: self.ms.meshing_repair_non_manifold_vertices()
            except Exception: logger.warning("Could not repair non-manifold vertices (PML).")
            self.ms.meshing_remove_unreferenced_vertices()
        except pymeshlab.PyMeshLabException as e: logger.error(f"Error cleaning (PML): {e}")

    def fill_holes(self):
        if self.ms.current_mesh().vertex_number() == 0: return 
        try: self.ms.meshing_close_holes()
        except pymeshlab.PyMeshLabException as e: logger.warning(f"Could not close holes (PML): {e}")

    def reconstruct_surface(self, method='poisson', **kwargs):
        if self.ms.current_mesh().vertex_number() == 0: return False
        if method == 'poisson':
            depth = kwargs.get('depth', 10) 
            try:
                self.ms.generate_surface_reconstruction_screened_poisson(depth=depth, preclean=True)
                if self.ms.current_mesh().vertex_number() == 0:
                    logger.warning("Poisson reconstruction resulted in empty mesh."); return False
                return True
            except pymeshlab.PyMeshLabException as e: logger.error(f"Poisson recon error: {e}"); return False
        raise ValueError("Unsupported recon method.")

    def check_watertight(self, engine: str = "trimesh") -> bool:
        """
        Return True if the current PyMeshLab mesh is watertight,
        as determined by the selected *engine*.

        Parameters
        ----------
        engine : {"trimesh", "open3d"}, default "trimesh"
            Library used for the test.

        Notes
        -----
        • Requires trimesh ≥ 2.38.0 when *engine* == "trimesh".
        • Requires open3d ≥ 0.18 when *engine* == "open3d".
        """
        cm = self.ms.current_mesh()                          # PyMeshLab mesh
        if cm.vertex_number() == 0:
            return False                                     # empty set → not watertight

        try:
            verts: np.ndarray = cm.vertex_matrix()           # (N, 3) float64
            faces: np.ndarray = cm.face_matrix()             # (M, 3) int32

            if engine == "trimesh":
                import trimesh
                mesh = trimesh.Trimesh(
                    vertices=verts,
                    faces=faces,
                    process=False            # keep topology untouched
                )
                is_wt = mesh.is_watertight

                # --- ADDED DIAGNOSTIC BLOCK ---
                if not is_wt:
                    reasons = []
                    # This is the check for "every single edge that is only part of one face"
                    if len(mesh.boundary_edges) > 0:
                        reasons.append(f"{len(mesh.boundary_edges)} boundary edges (gaps) found")
                    
                    # Trimesh's `is_watertight` also checks for other non-manifold conditions.
                    if not mesh.is_manifold:
                         reasons.append("mesh is not manifold (e.g., edges shared by >2 faces)")
                    
                    if not reasons:
                        reasons.append("unknown reason (topology may be complex)")
                    
                    logger.info(f"Watertight check (trimesh) failed. Reasons: {'; '.join(reasons)}.")
                # --- END DIAGNOSTIC BLOCK ---

            elif engine == "open3d":
                import open3d as o3d
                mesh = o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(verts),
                    o3d.utility.Vector3iVector(faces)
                )
                is_wt = mesh.is_watertight()

                # --- ADDED DIAGNOSTIC BLOCK ---
                if not is_wt:
                    # In Open3D, boundary edges are a type of non-manifold edge.
                    non_manifold_edges = mesh.get_non_manifold_edges()
                    logger.info(
                        f"Watertight check (open3d) failed. Found {len(non_manifold_edges)} non-manifold/boundary edges."
                    )
                # --- END DIAGNOSTIC BLOCK ---

            else:
                raise ValueError(f"Unknown engine '{engine}'")

            return is_wt

        except Exception as exc:
            logger.error("Error checking watertight (%s): %s", engine, exc)
            return False

    def save_mesh(self):
        if self.ms.current_mesh().vertex_number() == 0:
            logger.warning(f"Skipping save of empty mesh to {self.output_path}"); return False
        try: self.ms.save_current_mesh(self.output_path); return True
        except pymeshlab.PyMeshLabException as e: logger.error(f"Error saving {self.output_path}: {e}"); return False

    def process(self, reconstruction_method='poisson', **kwargs): # This is the original watertightness utility
        if not self.load_mesh(): return False 
        self.clean_mesh(); self.fill_holes() 
        if not self.reconstruct_surface(method=reconstruction_method, **kwargs):
             logger.warning("Watertightness util: stopping as surface recon failed."); return False 
        self.clean_mesh()
        return self.check_watertight() and self.save_mesh()
    
    @staticmethod
    def get_all_boundary_loops(mesh: trimesh.Trimesh, min_loop_len: int = 2) -> List[np.ndarray]:
        all_loops = []
        if not MeshCleanProcess._is_mesh_valid_for_concat(mesh, "InputMeshForAllLoops"): return all_loops
        boundary_edges = MeshCleanProcess._get_boundary_edges_manually_from_faces(mesh)
        if boundary_edges is None or len(boundary_edges) < min_loop_len: return all_loops
        loop_components = trimesh.graph.connected_components(boundary_edges, min_len=min_loop_len) 
        if not loop_components or not any(c is not None and len(c) >= min_loop_len for c in loop_components): return all_loops
        for i, comp_unord in enumerate(loop_components):
            if comp_unord is None or len(comp_unord) < min_loop_len: continue
            comp_set = set(comp_unord)
            edges_comp = [e for e in boundary_edges if e[0] in comp_set and e[1] in comp_set]
            if not edges_comp or len(edges_comp) < len(comp_unord) -1 : continue
            ordered = MeshCleanProcess._order_loop_vertices_from_edges(f"Loop_{i}", comp_unord, np.array(edges_comp))
            if ordered is not None and len(ordered) >= min_loop_len: all_loops.append(ordered)
        return all_loops
                                    
    @staticmethod
    def _is_mesh_valid_for_concat(mesh: Optional[trimesh.Trimesh], mesh_name: str) -> bool: # mesh_name for debug only if logs re-enabled
        return not (mesh is None or mesh.is_empty or \
                    not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces') or \
                    mesh.vertices.ndim != 2 or mesh.vertices.shape[1] != 3 or \
                    mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3)
                                    
    @staticmethod
    def _get_boundary_edges_manually_from_faces(mesh: trimesh.Trimesh) -> Optional[np.ndarray]:
        if not (mesh and not mesh.is_empty and hasattr(mesh,'vertices') and mesh.vertices is not None and len(mesh.vertices) > 0 and \
                hasattr(mesh,'faces') and mesh.faces is not None and len(mesh.faces) > 0 ): return None
        try:
            edges = []
            for f in mesh.faces:
                if len(f)!=3: continue
                edges.extend([tuple(sorted((f[0],f[1]))), tuple(sorted((f[1],f[2]))), tuple(sorted((f[2],f[0])))])
            if not edges: return np.array([],dtype=int).reshape(-1,2)
            counts = Counter(edges)
            b_edges = [list(e) for e,c in counts.items() if c==1]
            return np.array(b_edges,dtype=int) if b_edges else np.array([],dtype=int).reshape(-1,2)
        except Exception as e: logger.error(f"Err in _get_boundary_edges: {e}", exc_info=True); return None

    @staticmethod
    def _order_loop_vertices_from_edges(mesh_name_for_debug: str, loop_vidx_unique: np.ndarray, all_edges_loop: np.ndarray) -> Optional[np.ndarray]:
        if loop_vidx_unique is None or len(loop_vidx_unique) < 3 or all_edges_loop is None or len(all_edges_loop) < max(0, len(loop_vidx_unique)-1): return None
        adj = {v:[] for v in loop_vidx_unique}; degrees = {v:0 for v in loop_vidx_unique}
        for u,v in all_edges_loop:
            if u in adj and v in adj: adj[u].append(v); adj[v].append(u); degrees[u]+=1; degrees[v]+=1
        if not loop_vidx_unique.size: return None
        start_node = loop_vidx_unique[0] # Default start
        deg1 = [v for v,d in degrees.items() if d==1]; deg2 = [v for v,d in degrees.items() if d==2]
        if deg1: start_node = deg1[0]
        elif deg2: start_node = deg2[0]
        path = [start_node]; visited_edges = set(); curr = start_node
        for _ in range(len(all_edges_loop) + 2): # Max possible path length
            found_next = False
            for neighbor in adj.get(curr, []):
                edge = tuple(sorted((curr, neighbor)))
                if edge not in visited_edges:
                    path.append(neighbor); visited_edges.add(edge); curr = neighbor; found_next = True; break
            if not found_next: break
        if len(path) > 1 and path[0] == path[-1]: path = path[:-1] # Close loop
        unique_in_path = len(np.unique(path)); expected_unique = len(loop_vidx_unique)
        if unique_in_path == expected_unique and len(path) == expected_unique: return np.array(path,dtype=int)
        # The problematic line that caused the NameError has been removed.
        return None

    @staticmethod
    def get_dominant_plane(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if points is None or points.ndim != 2 or points.shape[1]!=3 or len(points)<3:
            return np.array([0.,0.,1.]), (np.mean(points,axis=0) if (points is not None and points.ndim==2 and points.shape[1]==3 and len(points)>0) else np.array([0.,0.,0.]))
        center = np.mean(points,axis=0); centered = points-center
        try: _,_,vh = np.linalg.svd(centered,full_matrices=False); return trimesh.util.unitize(vh[-1,:]), center
        except Exception: return np.array([0.,0.,1.]), center

    @staticmethod
    def run_face_grafting_pipeline(
        full_body_mesh_path: str, output_path: str, smplx_face_mesh_path: str,
        **pipeline_params
    ) -> Optional[trimesh.Trimesh]:
        """
        Initializes and runs the grafting pipeline.
        
        Parameters
        ----------
        full_body_mesh_path : str
            Path to the full body mesh file.
        output_path : str
            Path where the final grafted mesh will be saved.
        smplx_face_mesh_path : str
            Path to the SMPLX face mesh file.
        **pipeline_params : dict
            A dictionary of optional parameters to override the defaults in FaceGraftingConfig.
        """
        # Create the configuration by unpacking the provided dictionary.
        # Any parameters not in the dictionary will use the defaults from the dataclass.
        config = FaceGraftingConfig(**pipeline_params)
        
        # Initialize and run the pipeline instance with this config.
        pipeline = FaceGraftingPipeline(
            full_body_mesh_path=full_body_mesh_path,
            smplx_face_mesh_path=smplx_face_mesh_path,
            output_path=output_path,
            config=config
        )
        return pipeline.process()
