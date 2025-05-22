import pymeshlab
import trimesh
import numpy as np
from scipy.spatial import cKDTree  
import os 
import traceback
import tempfile 
from typing import List, Optional, Tuple
from collections import Counter
from dataclasses import dataclass


import logging

# Setup basic logging for the module if not already configured
# In a real application, the calling code would configure logging.
# For this standalone refactor, this ensures logs are visible.
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class FaceGraftingConfig:
    """Configuration for the FaceGraftingPipeline."""
    projection_footprint_threshold: float = 0.01
    footprint_dilation_rings: int = 1
    body_simplification_target_faces: int = 12000
    stitch_method: str = "body_driven_loft"
    smplx_face_neck_loop_strategy: str = "full_face_silhouette"
    alignment_resample_count: int = 1000
    loft_strip_resample_count: int = 100
    max_seam_hole_fill_vertices: int = 250
    final_polish_max_hole_edges: int = 100
    iterative_repair_s1_iters: int = 5  # Body repair before hole cutting
    iterative_repair_s2_iters: int = 5  # Placeholder, original had s2, but it wasn't used distinctly
                                        # If needed for post-stitch, we'd add a call.
    iterative_repair_s2_remesh_percent: Optional[float] = None # Placeholder
    spider_filter_area_factor: Optional[float] = 250.0
    spider_filter_max_edge_len_factor: Optional[float] = 0.15
    debug_dir: Optional[str] = None


class FaceGraftingPipeline:
    """
    Pipeline for grafting an SMPLX face mesh onto a full body mesh.
    Refactored from MeshCleanProcess.process_mesh_graft_smplx_face_v5.
    """
    INTERNAL_VERSION_TRACKER = "5.79_refactored_pipeline"

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

        # Intermediate mesh states
        self.original_smplx_face_geom_tri: Optional[trimesh.Trimesh] = None
        self.simplified_body_trimesh: Optional[trimesh.Trimesh] = None
        self.body_with_hole_trimesh: Optional[trimesh.Trimesh] = None
        self.stitched_mesh_intermediate: Optional[trimesh.Trimesh] = None # Mesh after stitching/concatenation
        self.final_processed_mesh: Optional[trimesh.Trimesh] = None # Mesh after all processing steps

        # Intermediate data for stitching/filling
        self.faces_to_remove_mask_on_body: Optional[np.ndarray] = None
        self.ordered_s_vidx_loop: Optional[np.ndarray] = None
        self.s_loop_coords_ordered: Optional[np.ndarray] = None
        self.b_loop_coords_aligned: Optional[np.ndarray] = None

        if self.config.debug_dir:
            os.makedirs(self.config.debug_dir, exist_ok=True)

    def _make_temp_path(self, suffix_label: str, use_ply: bool = False) -> str:
        """Helper to create and track temporary file paths."""
        actual_suffix = suffix_label if suffix_label.startswith('_') else '_' + suffix_label
        file_ext = ".ply" if use_ply else ".obj"
        # Ensure tempfile has a unique prefix related to the pipeline instance if needed, or rely on mkstemp uniqueness
        fd, path = tempfile.mkstemp(suffix=actual_suffix + file_ext, dir=self.config.debug_dir, prefix="graft_")
        os.close(fd)
        self.temp_files_to_clean.append(path)
        return path

    def _cleanup_temp_files(self):
        """Removes all temporary files created during the process."""
        logger.debug(f"Cleaning up {len(self.temp_files_to_clean)} temporary files.")
        for temp_path in self.temp_files_to_clean:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError as e:
                    logger.warning(f"Could not remove temp file {temp_path}: {e}")
        self.temp_files_to_clean.clear()

    def _load_and_simplify_meshes(self) -> bool:
        """STEP 1 & 2: Load meshes and simplify body."""
        logger.info("=== STEP 1 & 2: Load meshes and simplify body ===")
        self.original_smplx_face_geom_tri = trimesh.load_mesh(self.smplx_face_mesh_path, process=False)
        if not MeshCleanProcess._is_mesh_valid_for_concat(self.original_smplx_face_geom_tri, f"InitialSMPLXFace_V{self.INTERNAL_VERSION_TRACKER}"):
            logger.critical(f"Initial SMPLX Face mesh ('{self.smplx_face_mesh_path}') is invalid. Aborting.")
            return False
        
        self.simplified_body_trimesh = trimesh.load_mesh(self.full_body_mesh_path, process=False)
        if not MeshCleanProcess._is_mesh_valid_for_concat(self.simplified_body_trimesh, f"InitialFullBody_V{self.INTERNAL_VERSION_TRACKER}"):
            logger.critical(f"Initial Full Body mesh ('{self.full_body_mesh_path}') is invalid. Aborting.")
            return False

        self.original_smplx_face_geom_tri.fix_normals()
        self.simplified_body_trimesh.fix_normals()

        if self.simplified_body_trimesh.faces.shape[0] > self.config.body_simplification_target_faces:
            logger.info(f"Simplifying body mesh to target {self.config.body_simplification_target_faces} faces.")
            body_to_simplify = self.simplified_body_trimesh
            temp_in = self._make_temp_path(f"b_v{self.INTERNAL_VERSION_TRACKER}_s_in", use_ply=True)
            temp_out = self._make_temp_path(f"b_v{self.INTERNAL_VERSION_TRACKER}_s_out", use_ply=True)
            loaded_s = None
            ms_s = None
            try:
                if body_to_simplify.export(temp_in):
                    ms_s = pymeshlab.MeshSet()
                    ms_s.load_new_mesh(temp_in)
                    if ms_s.current_mesh().face_number() > 0:
                        ms_s.meshing_decimation_quadric_edge_collapse(targetfacenum=self.config.body_simplification_target_faces, preservenormal=True)
                        ms_s.save_current_mesh(temp_out)
                        loaded_s = trimesh.load_mesh(temp_out, process=False)
                if MeshCleanProcess._is_mesh_valid_for_concat(loaded_s, f"SimpPML_V{self.INTERNAL_VERSION_TRACKER}"):
                    self.simplified_body_trimesh = loaded_s
                    self.simplified_body_trimesh.fix_normals()
                    logger.info("Body simplification successful.")
                else:
                    logger.warning("Body simplification failed or resulted in invalid mesh. Using original.")
            except Exception as e_simp:
                logger.warning(f"Exception during body simplification: {e_simp}. Using original.")
            finally:
                if ms_s is not None: del ms_s
        
        if not MeshCleanProcess._is_mesh_valid_for_concat(self.simplified_body_trimesh, f"FinalSimpBody_V{self.INTERNAL_VERSION_TRACKER}"):
            logger.critical("Simplified body mesh is invalid. Aborting.")
            return False
        return True

    def _perform_iterative_body_repair(self) -> bool:
        """STEP 2.5: Iterative Non-Manifold Repair on Simplified Body."""
        if not self.simplified_body_trimesh: return False # Should be caught by previous step
        
        if self.config.iterative_repair_s1_iters > 0:
            logger.info("=== STEP 2.5: Iterative Non-Manifold Repair on Simplified Body ===")
            repaired_body = MeshCleanProcess._iterative_non_manifold_repair_pml(
                self.simplified_body_trimesh,
                max_iterations=self.config.iterative_repair_s1_iters,
                debug_dir=self.config.debug_dir,
                temp_file_prefix=f"body_v{self.INTERNAL_VERSION_TRACKER}_s1_repair"
            )
            if not MeshCleanProcess._is_mesh_valid_for_concat(repaired_body, f"BodyAfterIterativeRepairS1_V{self.INTERNAL_VERSION_TRACKER}"):
                logger.critical("Body mesh invalid after iterative repair (S1). Aborting.")
                return False
            self.simplified_body_trimesh = repaired_body
            logger.info("Iterative body repair (S1) completed.")
        else:
            logger.info("Skipping iterative body repair (S1) as iterations are 0.")
        return True

    def _determine_hole_faces_and_create_body_with_hole(self) -> bool:
        """STEP 3: Determining Hole Faces and Creating Body with Hole."""
        if not self.simplified_body_trimesh or not self.original_smplx_face_geom_tri: return False
        logger.info("=== STEP 3: Determining Hole Faces and Creating Body with Hole ===")
        
        self.faces_to_remove_mask_on_body = np.zeros(len(self.simplified_body_trimesh.faces), dtype=bool)
        
        # Robust hole determination logic (copied and adapted)
        if self.original_smplx_face_geom_tri.vertices.shape[0] > 0 and \
           hasattr(self.original_smplx_face_geom_tri, 'vertex_normals') and \
           self.original_smplx_face_geom_tri.vertex_normals.shape == self.original_smplx_face_geom_tri.vertices.shape:
            try:
                _, d_cp, t_cp = trimesh.proximity.closest_point(self.simplified_body_trimesh, self.original_smplx_face_geom_tri.vertices)
                if d_cp is not None and t_cp is not None and len(d_cp) == len(t_cp):
                    h_cp = t_cp[d_cp < self.config.projection_footprint_threshold]
                    if len(h_cp) > 0: self.faces_to_remove_mask_on_body[np.unique(h_cp)] = True
                
                offset = self.config.projection_footprint_threshold * 0.5
                p_f = self.original_smplx_face_geom_tri.vertices + self.original_smplx_face_geom_tri.vertex_normals * offset
                p_b = self.original_smplx_face_geom_tri.vertices - self.original_smplx_face_geom_tri.vertex_normals * offset
                
                _, _, t_f = trimesh.proximity.closest_point(self.simplified_body_trimesh, p_f)
                if t_f is not None and len(t_f) > 0: self.faces_to_remove_mask_on_body[np.unique(t_f)] = True
                
                _, actual_dists_behind, t_b = trimesh.proximity.closest_point(self.simplified_body_trimesh, p_b)
                if actual_dists_behind is not None and t_b is not None and len(actual_dists_behind) == len(t_b):
                    h_b = t_b[actual_dists_behind < self.config.projection_footprint_threshold]
                    if len(h_b) > 0: self.faces_to_remove_mask_on_body[np.unique(h_b)] = True
                logger.debug(f"Initial faces to remove from projection: {np.sum(self.faces_to_remove_mask_on_body)}")
            except Exception as e_p:
                logger.warning(f"Error during robust hole determination: {e_p}")

            if self.config.footprint_dilation_rings > 0 and np.any(self.faces_to_remove_mask_on_body):
                logger.debug(f"Dilating footprint by {self.config.footprint_dilation_rings} rings.")
                adj = self.simplified_body_trimesh.face_adjacency
                current_wavefront_mask = self.faces_to_remove_mask_on_body.copy()
                for ring_idx in range(self.config.footprint_dilation_rings):
                    faces_indices_in_wavefront = np.where(current_wavefront_mask)[0]
                    if not faces_indices_in_wavefront.size: break
                    all_neighbors_this_ring_list = []
                    for face_idx_in_wavefront in faces_indices_in_wavefront:
                        mask_adj_contains_face = np.any(adj == face_idx_in_wavefront, axis=1)
                        neighboring_pairs_for_face = adj[mask_adj_contains_face]
                        for pair_val in neighboring_pairs_for_face:
                            all_neighbors_this_ring_list.append(pair_val[0] if pair_val[1] == face_idx_in_wavefront else pair_val[1])
                    if not all_neighbors_this_ring_list: break
                    unique_neighbors_this_ring = np.unique(all_neighbors_this_ring_list)
                    truly_new_face_indices_to_add = unique_neighbors_this_ring[~self.faces_to_remove_mask_on_body[unique_neighbors_this_ring]]
                    if not truly_new_face_indices_to_add.size: break
                    self.faces_to_remove_mask_on_body[truly_new_face_indices_to_add] = True
                    current_wavefront_mask = np.zeros_like(self.faces_to_remove_mask_on_body)
                    current_wavefront_mask[truly_new_face_indices_to_add] = True
                logger.debug(f"Faces to remove after dilation: {np.sum(self.faces_to_remove_mask_on_body)}")
        
        temp_body_with_hole = self.simplified_body_trimesh.copy()
        if np.any(self.faces_to_remove_mask_on_body):
            faces_to_keep_mask = ~self.faces_to_remove_mask_on_body
            if np.any(faces_to_keep_mask):
                temp_body_with_hole = trimesh.Trimesh(vertices=self.simplified_body_trimesh.vertices, faces=self.simplified_body_trimesh.faces[faces_to_keep_mask])
                temp_body_with_hole.remove_unreferenced_vertices()
                temp_body_with_hole.fix_normals()
            else: # All faces marked for removal, means something went wrong or face is fully occluded
                logger.warning("All faces on body mesh were marked for removal. This might be an issue.")
        
        self.body_with_hole_trimesh = temp_body_with_hole
        if MeshCleanProcess._is_mesh_valid_for_concat(temp_body_with_hole, f"TempBodyWHolePreSplit_V{self.INTERNAL_VERSION_TRACKER}"):
            if hasattr(temp_body_with_hole, 'split') and callable(temp_body_with_hole.split):
                components = temp_body_with_hole.split(only_watertight=False)
                if components:
                    largest_comp = max(components, key=lambda c: len(c.faces if hasattr(c, 'faces') and c.faces is not None else []))
                    if MeshCleanProcess._is_mesh_valid_for_concat(largest_comp, f"LargestCompBodyWHole_V{self.INTERNAL_VERSION_TRACKER}"):
                        self.body_with_hole_trimesh = largest_comp
                        logger.debug("Kept largest component of body with hole.")
        
        if not MeshCleanProcess._is_mesh_valid_for_concat(self.body_with_hole_trimesh, f"FinalBodyWHole_V{self.INTERNAL_VERSION_TRACKER}"):
            logger.warning("Body with hole is invalid. Attempting to use simplified body directly for concatenation.")
            self.body_with_hole_trimesh = self.simplified_body_trimesh.copy() # Fallback
            if not self.body_with_hole_trimesh: return False # if simplified_body_trimesh itself was problematic

        return True

    def _extract_smplx_face_loop(self) -> bool:
        if not self.original_smplx_face_geom_tri: return False
        logger.debug("Extracting SMPLX face loop.")
        
        if self.config.smplx_face_neck_loop_strategy == "full_face_silhouette":
            face_boundary_edges = MeshCleanProcess._get_boundary_edges_manually_from_faces(self.original_smplx_face_geom_tri)
            if face_boundary_edges is not None and len(face_boundary_edges) >= 3:
                all_face_boundary_components_vidx = trimesh.graph.connected_components(face_boundary_edges, min_len=3)
                valid_face_components = [comp for comp in all_face_boundary_components_vidx if comp is not None and len(comp) >= 3]
                if valid_face_components:
                    s_silhouette_comp_vidx_unord = max(valid_face_components, key=len, default=np.array([], dtype=int))
                    if len(s_silhouette_comp_vidx_unord) >= 3:
                        silhouette_set = set(s_silhouette_comp_vidx_unord)
                        edges_for_silhouette_comp = [e for e in face_boundary_edges if e[0] in silhouette_set and e[1] in silhouette_set]
                        if edges_for_silhouette_comp:
                            self.ordered_s_vidx_loop = MeshCleanProcess._order_loop_vertices_from_edges(
                                f"SMPLX_Full_Silhouette_Order_V{self.INTERNAL_VERSION_TRACKER}", 
                                s_silhouette_comp_vidx_unord, 
                                np.array(edges_for_silhouette_comp)
                            )
        
        if self.ordered_s_vidx_loop is not None and len(self.ordered_s_vidx_loop) >= 3:
            if self.ordered_s_vidx_loop.max() < len(self.original_smplx_face_geom_tri.vertices) and self.ordered_s_vidx_loop.min() >= 0:
                self.s_loop_coords_ordered = self.original_smplx_face_geom_tri.vertices[self.ordered_s_vidx_loop]
                logger.debug(f"Extracted SMPLX face loop with {len(self.s_loop_coords_ordered)} vertices.")
                return True
            else:
                logger.warning("SMPLX face loop vertex indices out of bounds.")
                self.ordered_s_vidx_loop = None
        
        logger.warning("Failed to extract a valid SMPLX face loop.")
        return False

    def _extract_and_order_body_hole_loop(self) -> Optional[np.ndarray]:
        if not self.simplified_body_trimesh or self.faces_to_remove_mask_on_body is None or \
           not self.s_loop_coords_ordered is not None: # s_loop_coords_ordered needed for proximity selection
            return None
        logger.debug("Extracting and ordering body hole loop.")

        body_hole_defining_edges = MeshCleanProcess._get_hole_boundary_edges_from_removed_faces(
            self.simplified_body_trimesh, self.faces_to_remove_mask_on_body
        )
        unord_b_comp_selected_vidx = np.array([], dtype=int)

        if body_hole_defining_edges is not None and len(body_hole_defining_edges) >= 3:
            hole_boundary_components_vidx_list = trimesh.graph.connected_components(body_hole_defining_edges, min_len=3)
            hole_comps_filtered = [comp for comp in hole_boundary_components_vidx_list if comp is not None and len(comp) >= 3]
            
            if hole_comps_filtered:
                best_loop_prox_vidx = None
                best_score_prox = np.inf
                for comp_idx, current_comp_vidx in enumerate(hole_comps_filtered):
                    if current_comp_vidx.max() >= len(self.simplified_body_trimesh.vertices) or current_comp_vidx.min() < 0:
                        continue
                    loop_coords_candidate = self.simplified_body_trimesh.vertices[current_comp_vidx]
                    if len(loop_coords_candidate) < 3: continue
                    try:
                        tree_candidate = cKDTree(loop_coords_candidate)
                        dists_to_candidate, _ = tree_candidate.query(self.s_loop_coords_ordered, k=1)
                        current_score_prox = np.mean(dists_to_candidate)
                        if current_score_prox < best_score_prox:
                            best_score_prox = current_score_prox
                            best_loop_prox_vidx = current_comp_vidx
                    except Exception: continue
                
                if best_loop_prox_vidx is not None:
                    unord_b_comp_selected_vidx = best_loop_prox_vidx
                elif hole_comps_filtered: # Fallback to largest if proximity failed
                    unord_b_comp_selected_vidx = max(hole_comps_filtered, key=len, default=np.array([], dtype=int))
        
        ordered_b_vidx_footprint = None
        if len(unord_b_comp_selected_vidx) >= 3:
            b_comp_set = set(unord_b_comp_selected_vidx)
            if body_hole_defining_edges is not None and len(body_hole_defining_edges) > 0:
                b_edges_for_selected_comp = [e for e in body_hole_defining_edges if e[0] in b_comp_set and e[1] in b_comp_set]
                if b_edges_for_selected_comp:
                    ordered_b_vidx_footprint = MeshCleanProcess._order_loop_vertices_from_edges(
                        f"BodyHoleFootprint_Order_V{self.INTERNAL_VERSION_TRACKER}", 
                        unord_b_comp_selected_vidx, 
                        np.array(b_edges_for_selected_comp)
                    )
        
        if ordered_b_vidx_footprint is not None and len(ordered_b_vidx_footprint) >= 3:
            if ordered_b_vidx_footprint.max() < len(self.simplified_body_trimesh.vertices) and ordered_b_vidx_footprint.min() >= 0:
                logger.debug(f"Extracted body hole loop with {len(ordered_b_vidx_footprint)} vertices.")
                return self.simplified_body_trimesh.vertices[ordered_b_vidx_footprint]
            else:
                logger.warning("Body hole loop vertex indices out of bounds.")
        
        logger.warning("Failed to extract a valid body hole loop.")
        return None

    def _align_body_loop_to_smplx_loop(self, b_loop_coords_ordered_pre_align: np.ndarray) -> bool:
        if not self.s_loop_coords_ordered is not None or len(self.s_loop_coords_ordered) < 2 or \
           len(b_loop_coords_ordered_pre_align) < 2:
            logger.warning("Not enough points in loops for alignment.")
            return False

        logger.debug("Aligning body loop to SMPLX face loop.")
        b_loop_coords_to_align = b_loop_coords_ordered_pre_align.copy()
        
        s_start_pt = self.s_loop_coords_ordered[0]
        kdt_b_align = cKDTree(b_loop_coords_to_align)
        _, closest_idx_on_b_to_s_start = kdt_b_align.query(s_start_pt, k=1)
        
        b_loop_coords_rolled = np.roll(b_loop_coords_to_align, -closest_idx_on_b_to_s_start, axis=0)
        
        resample_count = self.config.alignment_resample_count
        if len(self.s_loop_coords_ordered) >= 2 and len(b_loop_coords_rolled) >= 2 and resample_count >= 2:
            s_r_a = MeshCleanProcess.resample_polyline_to_count(self.s_loop_coords_ordered, resample_count)
            b_r_f = MeshCleanProcess.resample_polyline_to_count(b_loop_coords_rolled, resample_count) # Forward
            b_r_b = MeshCleanProcess.resample_polyline_to_count(b_loop_coords_rolled[::-1], resample_count) # Backward (reversed)

            if s_r_a is not None and b_r_f is not None and b_r_b is not None and \
               len(s_r_a) == resample_count and len(b_r_f) == resample_count and len(b_r_b) == resample_count:
                dist_fwd = np.sum(np.linalg.norm(s_r_a - b_r_f, axis=1))
                dist_bwd = np.sum(np.linalg.norm(s_r_a - b_r_b, axis=1))
                if dist_bwd < dist_fwd:
                    self.b_loop_coords_aligned = b_loop_coords_rolled[::-1].copy()
                    logger.debug("Aligned body loop (reversed orientation found better).")
                else:
                    self.b_loop_coords_aligned = b_loop_coords_rolled.copy()
                    logger.debug("Aligned body loop (forward orientation).")
            else:
                logger.warning("Resampling for alignment failed, using rolled loop.")
                self.b_loop_coords_aligned = b_loop_coords_rolled.copy()
        else:
            logger.debug("Using rolled loop directly (not enough points or resample count too low for full alignment).")
            self.b_loop_coords_aligned = b_loop_coords_rolled.copy()
            
        return self.b_loop_coords_aligned is not None and len(self.b_loop_coords_aligned) >=3

    def _create_loft_stitch_mesh(self) -> Optional[trimesh.Trimesh]:
        if not self.s_loop_coords_ordered is not None or len(self.s_loop_coords_ordered) < 2 or \
           not self.b_loop_coords_aligned is not None or len(self.b_loop_coords_aligned) < 2:
            logger.warning("Not enough data for loft stitch mesh creation.")
            return None

        logger.debug("Creating loft stitch mesh.")
        new_stitch_triangles_list = []
        
        resampled_s_target = MeshCleanProcess.resample_polyline_to_count(self.s_loop_coords_ordered, self.config.loft_strip_resample_count)
        if resampled_s_target is None or len(resampled_s_target) < 2:
            logger.warning("Resampling of SMPLX target loop for lofting failed.")
            return None
            
        kdt_s_target = cKDTree(resampled_s_target)
        num_b_pts_actual = len(self.b_loop_coords_aligned)
        stitch_strip_vertices_np = np.vstack((self.b_loop_coords_aligned, resampled_s_target))

        for i in range(num_b_pts_actual):
            b_curr_idx = i
            b_next_idx = (i + 1) % num_b_pts_actual
            
            # Query for closest points on the resampled SMPLX target loop
            _, s_match_idx_curr = kdt_s_target.query(self.b_loop_coords_aligned[b_curr_idx], k=1)
            _, s_match_idx_next = kdt_s_target.query(self.b_loop_coords_aligned[b_next_idx], k=1)

            # Vertex indices in stitch_strip_vertices_np
            v0 = b_curr_idx  # Current body point
            v1 = b_next_idx  # Next body point
            v2 = s_match_idx_next + num_b_pts_actual # Matched SMPLX point for next body point
            v3 = s_match_idx_curr + num_b_pts_actual # Matched SMPLX point for current body point
            
            new_stitch_triangles_list.extend([[v0, v1, v2], [v0, v2, v3]])
        
        if not new_stitch_triangles_list:
            logger.warning("No stitch triangles generated for lofting.")
            return None

        stitch_strip_faces_np = np.array(new_stitch_triangles_list, dtype=int)
        if stitch_strip_vertices_np.ndim == 2 and stitch_strip_vertices_np.shape[0] > 0 and stitch_strip_vertices_np.shape[1] == 3 and \
           stitch_strip_faces_np.ndim == 2 and stitch_strip_faces_np.shape[0] > 0 and stitch_strip_faces_np.shape[1] == 3:
            
            strip_mesh_obj = trimesh.Trimesh(vertices=stitch_strip_vertices_np, faces=stitch_strip_faces_np, process=False)
            if MeshCleanProcess._is_mesh_valid_for_concat(strip_mesh_obj, f"RawStitchStripV{self.INTERNAL_VERSION_TRACKER}"):
                # Normal Flipping Logic (Copied)
                if hasattr(strip_mesh_obj, 'face_normals') and strip_mesh_obj.face_normals is not None and len(strip_mesh_obj.face_normals) > 0:
                    strip_centroid = strip_mesh_obj.centroid
                    inward_pointing_normals_count = 0
                    for face_idx in range(len(strip_mesh_obj.faces)):
                        vector_to_strip_centroid = strip_centroid - strip_mesh_obj.triangles_center[face_idx]
                        if np.dot(strip_mesh_obj.face_normals[face_idx], vector_to_strip_centroid) > 1e-6: # dot product positive means normal points somewhat towards centroid
                            inward_pointing_normals_count += 1
                    if inward_pointing_normals_count > len(strip_mesh_obj.faces) / 2:
                        strip_mesh_obj.invert()
                        logger.debug("Loft stitch mesh normals inverted.")
                logger.debug("Loft stitch mesh created successfully.")
                return strip_mesh_obj
        
        logger.warning("Failed to create a valid loft stitch mesh object.")
        return None

    def _attempt_body_driven_loft(self) -> bool:
        """Attempt the 'BODY_DRIVEN_LOFT' stitch method."""
        logger.info("=== STEP 4 & 5: Attempting 'BODY_DRIVEN_LOFT' ===")
        if not self._extract_smplx_face_loop():
            logger.warning("Cannot proceed with loft: SMPLX face loop extraction failed.")
            return False

        b_loop_coords_ordered_pre_align = self._extract_and_order_body_hole_loop()
        if b_loop_coords_ordered_pre_align is None or len(b_loop_coords_ordered_pre_align) < 3:
            logger.warning("Cannot proceed with loft: Body hole loop extraction failed.")
            return False

        if not self._align_body_loop_to_smplx_loop(b_loop_coords_ordered_pre_align):
            logger.warning("Cannot proceed with loft: Loop alignment failed.")
            return False

        strip_mesh_obj = self._create_loft_stitch_mesh()
        if not strip_mesh_obj or not MeshCleanProcess._is_mesh_valid_for_concat(strip_mesh_obj, f"StitchStripV{self.INTERNAL_VERSION_TRACKER}"):
            logger.warning("Cannot proceed with loft: Stitch mesh creation failed or invalid.")
            return False

        # Concatenate components
        valid_cs = [m for m in [self.original_smplx_face_geom_tri, self.body_with_hole_trimesh, strip_mesh_obj] 
                    if MeshCleanProcess._is_mesh_valid_for_concat(m, f"LoftFinalCompV{self.INTERNAL_VERSION_TRACKER}")]
        
        if len(valid_cs) == 3:
            try:
                cand_m = trimesh.util.concatenate(valid_cs)
                if MeshCleanProcess._is_mesh_valid_for_concat(cand_m, f"ConcatLoftResV{self.INTERNAL_VERSION_TRACKER}"):
                    final_m_proc = cand_m.copy()
                    final_m_proc.merge_vertices(merge_tex=False, merge_norm=False) # Original params
                    final_m_proc.remove_unreferenced_vertices()
                    final_m_proc.remove_degenerate_faces()
                    if MeshCleanProcess._is_mesh_valid_for_concat(final_m_proc, f"ProcessedLoftResV{self.INTERNAL_VERSION_TRACKER}"):
                        self.stitched_mesh_intermediate = final_m_proc
                    else:
                        logger.warning("Processed lofted mesh is invalid, using raw concatenated.")
                        self.stitched_mesh_intermediate = cand_m
                    
                    if self.stitched_mesh_intermediate is not None and not self.stitched_mesh_intermediate.is_empty:
                        logger.info(f"Stitch method '{self.config.stitch_method}' applied successfully.")
                        return True
                    else:
                        logger.warning("Lofted concatenation resulted in an empty or None mesh.")
                        self.stitched_mesh_intermediate = None
                        return False
            except Exception as e_f_cat:
                logger.error(f"Exception during final loft concatenation/merge: {e_f_cat}", exc_info=True)
                self.stitched_mesh_intermediate = None
                return False
        else:
            logger.warning("Not all three components (face, body_hole, strip) were valid for loft concatenation.")
            return False
        return False


    def _perform_simple_concatenation(self) -> bool:
        """Perform simple concatenation as a fallback or default."""
        logger.info("=== STEP 5 (Effective): Performing simple concatenation ===")
        if not self.original_smplx_face_geom_tri or not self.body_with_hole_trimesh:
             logger.error("Cannot perform simple concatenation: one or both base meshes are missing.")
             return False

        valid_fb_comps = [m for m in [self.original_smplx_face_geom_tri, self.body_with_hole_trimesh] 
                          if MeshCleanProcess._is_mesh_valid_for_concat(m, f"FallbackCompV{self.INTERNAL_VERSION_TRACKER}")]
        
        if not valid_fb_comps:
            logger.fatal("No valid fallback components for concatenation.")
            return False
            
        try:
            self.stitched_mesh_intermediate = trimesh.util.concatenate(valid_fb_comps)
            if not MeshCleanProcess._is_mesh_valid_for_concat(self.stitched_mesh_intermediate, f"FallbackConcatV{self.INTERNAL_VERSION_TRACKER}"):
                logger.fatal("Fallback concatenation resulted in invalid mesh.")
                self.stitched_mesh_intermediate = None
                return False
            self.stitched_mesh_intermediate.fix_normals() # Original had this here
            logger.info("Simple concatenation successful.")
            return True
        except Exception as e_concat:
            logger.error(f"Exception during simple concatenation: {e_concat}", exc_info=True)
            self.stitched_mesh_intermediate = None
            return False

    def _stitch_components(self) -> bool:
        """Orchestrates the stitching process based on config."""
        if self.config.stitch_method == "body_driven_loft":
            if self._attempt_body_driven_loft():
                return True
            else: # Lofting failed or was not applicable
                logger.info(f"Stitch method '{self.config.stitch_method}' did not produce a result or was not applicable. Defaulting to simple concatenation.")
                return self._perform_simple_concatenation()
        else: # Other stitch methods or 'none'
            logger.info(f"Stitch method is '{self.config.stitch_method}'. Proceeding with simple concatenation.")
            return self._perform_simple_concatenation()

    def _fill_seam_holes_ear_clip(self) -> bool:
        if not self.stitched_mesh_intermediate or \
           not MeshCleanProcess._is_mesh_valid_for_concat(self.stitched_mesh_intermediate, f"MeshBeforeSeamFill_V{self.INTERNAL_VERSION_TRACKER}"):
            logger.warning("Skipping seam hole fill: stitched mesh is invalid or missing.")
            self.final_processed_mesh = self.stitched_mesh_intermediate 
            return True 

        if self.config.stitch_method not in ["trimesh_dynamic_loft", "body_driven_loft"]:
            logger.info(f"Skipping ear-clip seam hole fill as stitch method was '{self.config.stitch_method}'.")
            self.final_processed_mesh = self.stitched_mesh_intermediate
            return True

        logger.info("Attempting EAR-CLIP SEAM HOLE FILL post-loft...")
        mesh_to_fill = self.stitched_mesh_intermediate.copy() 
        if not hasattr(mesh_to_fill, 'face_normals') or mesh_to_fill.face_normals is None or len(mesh_to_fill.face_normals) != len(mesh_to_fill.faces):
            mesh_to_fill.fix_normals()
        
        original_vertices_for_fill = mesh_to_fill.vertices.copy()
        current_faces_list = list(mesh_to_fill.faces)
        added_any_fill_faces_this_pass = False 
        
        s_loop_ref = self.s_loop_coords_ordered 
        b_loop_ref = self.b_loop_coords_aligned 
        
        all_ordered_loops_on_stitched_mesh = MeshCleanProcess.get_all_boundary_loops(mesh_to_fill, min_loop_len=3)
        logger.debug(f"Found {len(all_ordered_loops_on_stitched_mesh)} boundary loops for potential filling.")

        kdt_s_ref, kdt_b_ref = None, None
        if s_loop_ref is not None and len(s_loop_ref) > 0: kdt_s_ref = cKDTree(s_loop_ref)
        if b_loop_ref is not None and len(b_loop_ref) > 0: kdt_b_ref = cKDTree(b_loop_ref)
        
        proximity_to_seam_threshold_fill = 0.025 

        # --- Heuristic for limb openings (e.g., hands, feet) ---
        # Use the bounding box of the *simplified body before hole carving* as a reference
        # This assumes simplified_body_trimesh is available and valid from previous steps
        z_threshold_for_hands_feet = None
        if self.simplified_body_trimesh is not None and MeshCleanProcess._is_mesh_valid_for_concat(self.simplified_body_trimesh, "BodyForLimbZThresh"):
            body_bounds_min, body_bounds_max = self.simplified_body_trimesh.bounds
            body_height = body_bounds_max[2] - body_bounds_min[2]
            if body_height > 1e-6: # Ensure body has some height
                # Consider holes in the lower 30% of the body as potential limb openings
                z_threshold_for_hands_feet = body_bounds_min[2] + body_height * 0.30 
                logger.debug(f"Z-threshold for excluding limb openings during fill: {z_threshold_for_hands_feet:.4f}")
        # --- End limb opening heuristic ---

        for loop_idx, current_hole_v_indices_ordered in enumerate(all_ordered_loops_on_stitched_mesh):
            if not (current_hole_v_indices_ordered is not None and \
                    3 <= len(current_hole_v_indices_ordered) <= self.config.max_seam_hole_fill_vertices):
                continue
            if current_hole_v_indices_ordered.max() >= len(original_vertices_for_fill) or current_hole_v_indices_ordered.min() < 0:
                continue
            
            current_hole_coords_3d = original_vertices_for_fill[current_hole_v_indices_ordered]
            loop_centroid_z = np.mean(current_hole_coords_3d[:, 2])

            # --- Apply limb opening heuristic ---
            if z_threshold_for_hands_feet is not None and loop_centroid_z < z_threshold_for_hands_feet:
                logger.debug(f"Loop {loop_idx} (centroid Z: {loop_centroid_z:.3f}) is below Z-threshold. Likely a limb opening. Skipping fill.")
                continue
            # --- End limb opening heuristic ---
            
            is_seam_hole_for_fill = False # Default to False
            # Proximity check (optional, can be made stricter or conditional)
            if kdt_s_ref is not None and kdt_b_ref is not None:
                d_s, _ = kdt_s_ref.query(current_hole_coords_3d, k=1, distance_upper_bound=proximity_to_seam_threshold_fill * 2)
                avg_d_s = np.mean(d_s[np.isfinite(d_s)]) if np.any(np.isfinite(d_s)) else float('inf')
                d_b, _ = kdt_b_ref.query(current_hole_coords_3d, k=1, distance_upper_bound=proximity_to_seam_threshold_fill * 2)
                avg_d_b = np.mean(d_b[np.isfinite(d_b)]) if np.any(np.isfinite(d_b)) else float('inf')
                if avg_d_s < proximity_to_seam_threshold_fill and avg_d_b < proximity_to_seam_threshold_fill:
                    is_seam_hole_for_fill = True
                    logger.debug(f"Loop {loop_idx} identified as a seam hole by proximity for filling.")
            else:
                # If proximity checks cannot be done (e.g. reference loops missing),
                # we might assume all remaining (non-limb) holes are candidates
                # or add other criteria. For now, if no prox check, don't mark as seam_hole.
                # This means only limb-checked holes would be filled if this branch is taken.
                # To fill all non-limb holes: is_seam_hole_for_fill = True
                logger.debug(f"Loop {loop_idx}: Proximity refs not available, not marking as seam hole by proximity.")


            if is_seam_hole_for_fill: # Only proceed if explicitly marked as seam hole
                # ... (rest of the ear-clipping logic: plane fit, triangulation, normal check, adding faces) ...
                # This part remains the same as the last correct version
                try:
                    if len(current_hole_coords_3d) < 3: continue
                    plane_origin_loop, normal_loop = MeshCleanProcess.get_dominant_plane(current_hole_coords_3d)
                    if np.allclose(normal_loop,[0,0,0]): logger.debug(f"Loop {loop_idx} degenerate normal. Skipping."); continue
                    transform_to_2d = trimesh.geometry.plane_transform(plane_origin_loop, normal_loop); loop_coords_2d_projected = trimesh.transform_points(current_hole_coords_3d, transform_to_2d)[:, :2]
                    if len(np.unique(loop_coords_2d_projected, axis=0)) < 3: logger.debug(f"Loop {loop_idx} degenerate 2D proj. Skipping."); continue
                    path_2d_object = trimesh.path.Path2D(entities=[trimesh.path.entities.Line(np.arange(len(loop_coords_2d_projected)))], vertices=loop_coords_2d_projected)
                    patch_faces_local_idx, patch_vertices_2d = None, None
                    try: patch_faces_local_idx, patch_vertices_2d = trimesh.creation.triangulate_polygon(path_2d_object, triangle_args='p')
                    except Exception:
                        try: patch_faces_local_idx, patch_vertices_2d = trimesh.creation.triangulate_polygon(path_2d_object)
                        except Exception as e_tri: logger.debug(f"Triangulation failed for loop {loop_idx}: {e_tri}"); continue
                    if patch_faces_local_idx is not None and len(patch_faces_local_idx) > 0:
                        if len(patch_vertices_2d) == len(loop_coords_2d_projected) and np.allclose(patch_vertices_2d, loop_coords_2d_projected, atol=1e-5):
                            new_fill_faces_global_candidate = current_hole_v_indices_ordered[patch_faces_local_idx]
                            temp_patch = trimesh.Trimesh(vertices=original_vertices_for_fill, faces=new_fill_faces_global_candidate, process=False); temp_patch.fix_normals()
                            if len(temp_patch.faces) > 0 and len(current_hole_v_indices_ordered) >=2:
                                shared_edge_vidx_tuple = tuple(sorted((current_hole_v_indices_ordered[0], current_hole_v_indices_ordered[1])))
                                try:
                                    if not hasattr(mesh_to_fill, 'edges_unique') or mesh_to_fill.edges_unique is None: _ = mesh_to_fill.edges_unique 
                                    if mesh_to_fill.edges_unique is not None and len(mesh_to_fill.edges_unique) > 0:
                                        edge_id_in_mesh_to_fill = mesh_to_fill.edges_unique.tolist().index(list(shared_edge_vidx_tuple))
                                        adj_faces_indices_in_mesh_to_fill = mesh_to_fill.edge_faces[edge_id_in_mesh_to_fill]
                                        if len(adj_faces_indices_in_mesh_to_fill) > 0: 
                                            valid_adj_face_idx = -1
                                            for f_idx_adj in adj_faces_indices_in_mesh_to_fill:
                                                if f_idx_adj != -1 and f_idx_adj < len(mesh_to_fill.face_normals): valid_adj_face_idx = f_idx_adj; break
                                            if valid_adj_face_idx != -1:
                                                normal_existing_adj_face = mesh_to_fill.face_normals[valid_adj_face_idx]
                                                if hasattr(temp_patch, 'face_normals') and temp_patch.face_normals is not None and len(temp_patch.face_normals) > 0: 
                                                    normal_patch_adj_face = temp_patch.face_normals[0] 
                                                    if np.dot(normal_patch_adj_face, normal_existing_adj_face) < 0.1: new_fill_faces_global_candidate = new_fill_faces_global_candidate[:, ::-1]
                                except (ValueError, IndexError, AttributeError): pass
                            current_faces_list.extend(new_fill_faces_global_candidate); added_any_fill_faces_this_pass = True
                            logger.debug(f"Filled loop {loop_idx} with {len(new_fill_faces_global_candidate)} faces.")
                except Exception as e_ear_clip: 
                    logger.warning(f"Error during ear-clip processing for loop {loop_idx}: {e_ear_clip}", exc_info=True)
        
        if added_any_fill_faces_this_pass:
            updated_mesh_after_fill = trimesh.Trimesh(vertices=original_vertices_for_fill, 
                                                      faces=np.array(current_faces_list, dtype=int), 
                                                      process=False) 
            updated_mesh_after_fill.merge_vertices(merge_tex=False, merge_norm=False) 
            updated_mesh_after_fill.remove_unreferenced_vertices(); updated_mesh_after_fill.remove_degenerate_faces(); 
            updated_mesh_after_fill.fix_normals() 
            if MeshCleanProcess._is_mesh_valid_for_concat(updated_mesh_after_fill, f"MeshAfterEarClipFill_V{self.INTERNAL_VERSION_TRACKER}"):
                self.final_processed_mesh = updated_mesh_after_fill
                logger.info("Ear-clip seam hole filling applied.")
            else:
                logger.warning("Mesh became invalid after ear-clip fill. Reverting to pre-fill mesh.")
                self.final_processed_mesh = self.stitched_mesh_intermediate # Revert
        else:
            logger.info("No seam holes were filled by ear-clipping.")
            self.final_processed_mesh = self.stitched_mesh_intermediate # Pass through
        return True
    def _apply_final_polish(self) -> bool:
        """STEP 5.5: Applying Final Polish using PyMeshLab."""
        if not self.final_processed_mesh or \
           not MeshCleanProcess._is_mesh_valid_for_concat(self.final_processed_mesh, f"MeshBeforeFinalPolish_V{self.INTERNAL_VERSION_TRACKER}"):
            logger.warning("Skipping final polish: mesh is invalid or missing.")
            return True # Not a failure of this step

        logger.info("=== STEP 5.5: Applying Final Polish ===")
        temp_polish_in = self._make_temp_path(f"final_polish_in_v{self.INTERNAL_VERSION_TRACKER}", use_ply=True)
        temp_polish_out = self._make_temp_path(f"final_polish_out_v{self.INTERNAL_VERSION_TRACKER}", use_ply=True)
        polished_mesh_loaded = None
        ms_polish = None
        
        try:
            self.final_processed_mesh.export(temp_polish_in)
            ms_polish = pymeshlab.MeshSet()
            ms_polish.load_new_mesh(temp_polish_in)
            if ms_polish.current_mesh_id() != -1 and ms_polish.current_mesh().vertex_number() > 0:
                ms_polish.meshing_remove_duplicate_vertices()
                try:
                    ms_polish.meshing_repair_non_manifold_edges(method='Split Vertices')
                except pymeshlab.PyMeshLabException:
                    try:
                        ms_polish.meshing_repair_non_manifold_edges(method='Remove Faces')
                    except pymeshlab.PyMeshLabException:
                        logger.info("Polish: Both non-manifold edge repair methods failed.")
                
                is_manifold_for_hole_closing = False
                try:
                    topo_measures = ms_polish.get_topological_measures()
                    nm_edges_metric = topo_measures.get('non_manifold_edges', -1) # As per original
                    if nm_edges_metric == 0 : is_manifold_for_hole_closing = True
                except Exception: 
                    logger.debug("Could not get topological measures for hole closing check during polish.")

                if is_manifold_for_hole_closing:
                    try:
                        ms_polish.meshing_close_holes(maxholesize=self.config.final_polish_max_hole_edges, newfaceselected=False)
                    except pymeshlab.PyMeshLabException as e_close_holes:
                        logger.info(f"Polish: meshing_close_holes failed: {e_close_holes}")
                else:
                    logger.info("Polish: Skipping meshing_close_holes as mesh not determined to be sufficiently manifold.")
                
                ms_polish.meshing_remove_unreferenced_vertices()
                ms_polish.compute_normal_per_face() # As per original
                ms_polish.save_current_mesh(temp_polish_out)
                polished_mesh_loaded = trimesh.load_mesh(temp_polish_out, process=False)
            
            if MeshCleanProcess._is_mesh_valid_for_concat(polished_mesh_loaded, f"PolishedMeshFromPML_V{self.INTERNAL_VERSION_TRACKER}"):
                self.final_processed_mesh = polished_mesh_loaded
                self.final_processed_mesh.fix_normals() # Ensure normals are good after trimesh load
                logger.info("Final polish step applied.")
            else:
                logger.warning("Final polish resulted in invalid/empty mesh. Keeping pre-polish mesh.")
                # self.final_processed_mesh remains as it was before this step
        except Exception as e_polish:
            logger.warning(f"Error during main final polish step: {e_polish}", exc_info=True)
        finally:
            if ms_polish is not None: del ms_polish
        return True

    def _filter_spider_triangles(self) -> bool:
        """STEP 5.75: Filtering Spider-Web Triangles."""
        if not self.final_processed_mesh or \
           not MeshCleanProcess._is_mesh_valid_for_concat(self.final_processed_mesh, f"MeshBeforeSpiderFilter_V{self.INTERNAL_VERSION_TRACKER}"):
            logger.warning("Skipping spider filter: mesh is invalid or missing.")
            return True # Not a failure of this step
        
        if self.config.spider_filter_area_factor is None and self.config.spider_filter_max_edge_len_factor is None:
            logger.info("Skipping spider filter: no criteria defined.")
            return True

        logger.info("=== STEP 5.75: Filtering Spider-Web Triangles ===")
        target_faces_for_spider_filter = None
        if MeshCleanProcess._is_mesh_valid_for_concat(self.original_smplx_face_geom_tri, "OrigFaceForSpiderRefBounds"):
            if len(self.final_processed_mesh.faces) > 0 and len(self.original_smplx_face_geom_tri.vertices) > 0:
                final_mesh_centroids = self.final_processed_mesh.triangles_center
                smplx_bounds_min, smplx_bounds_max = self.original_smplx_face_geom_tri.bounds
                padding = 0.05 
                min_b = smplx_bounds_min - padding; max_b = smplx_bounds_max + padding
                candidate_indices = [idx for idx, centroid in enumerate(final_mesh_centroids) if 
                                     (min_b[0] <= centroid[0] <= max_b[0] and
                                      min_b[1] <= centroid[1] <= max_b[1] and
                                      min_b[2] <= centroid[2] <= max_b[2])]
                if candidate_indices: target_faces_for_spider_filter = np.array(candidate_indices, dtype=int)
                logger.debug(f"Targeting {len(target_faces_for_spider_filter) if target_faces_for_spider_filter is not None else 0} faces near SMPLX bounds for spider filter.")

        effective_max_edge_length = None
        if self.config.spider_filter_max_edge_len_factor is not None and self.config.spider_filter_max_edge_len_factor > 0:
            if MeshCleanProcess._is_mesh_valid_for_concat(self.simplified_body_trimesh, "BodyForSpiderEdgeRef") and \
               self.simplified_body_trimesh and len(self.simplified_body_trimesh.edges_unique_length) > 0: # Check self.simplified_body_trimesh exists
                body_edge_lengths = self.simplified_body_trimesh.edges_unique_length
                if len(body_edge_lengths) > 0:
                    # Original logic was max * (1+factor), seems like it should be a factor of a typical/median edge length.
                    # Reverting to original's interpretation for strict logic preservation for now:
                    effective_max_edge_length = np.max(body_edge_lengths) * (1.0 + self.config.spider_filter_max_edge_len_factor)
                    logger.debug(f"Spider filter effective max edge length: {effective_max_edge_length}")

        ref_for_stats = self.original_smplx_face_geom_tri if MeshCleanProcess._is_mesh_valid_for_concat(self.original_smplx_face_geom_tri, "RefForSpiderAreaStats") else None
        
        filtered_mesh = MeshCleanProcess._filter_large_triangles_from_fill(
            self.final_processed_mesh,
            target_face_indices=target_faces_for_spider_filter,
            max_allowed_edge_length=effective_max_edge_length,
            max_allowed_area_factor=self.config.spider_filter_area_factor,
            reference_mesh_for_stats=ref_for_stats,
            mesh_name_for_debug=f"FinalOutput_SpiderFilter_V{self.INTERNAL_VERSION_TRACKER}"
        )
        if not MeshCleanProcess._is_mesh_valid_for_concat(filtered_mesh, f"MeshAfterSpiderFilter_V{self.INTERNAL_VERSION_TRACKER}"):
            logger.critical("Mesh invalid after spider-web filter. CANNOT PROCEED TO SAVE.")
            return False # This is a hard failure for saving.
        
        self.final_processed_mesh = filtered_mesh
        self.final_processed_mesh.fix_normals() # Ensure normals are good
        logger.info("Spider-web triangle filter applied.")
        return True

    def process(self) -> Optional[trimesh.Trimesh]:
        """
        Executes the full face grafting pipeline.
        Returns the processed Trimesh object if successful, None otherwise.
        """
        logger.info(f"--- Starting Mesh Processing: Grafting SMPLX Face (V{self.INTERNAL_VERSION_TRACKER}) ---")
        pipeline_successful = False
        try:
            if not self._load_and_simplify_meshes(): return None
            if not self._perform_iterative_body_repair(): return None
            if not self._determine_hole_faces_and_create_body_with_hole(): return None
            
            if not self._stitch_components(): # Populates self.stitched_mesh_intermediate
                logger.error("Component stitching failed. Aborting.")
                return None
            
            # From this point, operations are on self.stitched_mesh_intermediate, result in self.final_processed_mesh
            if not self._fill_seam_holes_ear_clip(): # Operates on intermediate, output to final_processed_mesh
                # This step might be skippable or might revert, so not necessarily fatal.
                # It internally sets self.final_processed_mesh
                pass 

            if not self._apply_final_polish(): # Operates on self.final_processed_mesh
                 # This step might also be skippable, using pre-polish mesh
                pass

            if not self._filter_spider_triangles(): # Operates on self.final_processed_mesh
                logger.error("Spider triangle filtering failed critically. Aborting.")
                return None

            # Final check before saving
            if not MeshCleanProcess._is_mesh_valid_for_concat(self.final_processed_mesh, f"Mesh Before Final Save V{self.INTERNAL_VERSION_TRACKER}"):
                logger.critical("Final_processed_mesh is invalid or empty before final save. CANNOT SAVE.")
                return None

            self.final_processed_mesh.export(self.output_path)
            logger.info(f"--- Mesh Grafting V{self.INTERNAL_VERSION_TRACKER} Finished. Output: {self.output_path} ---")
            pipeline_successful = True
            return self.final_processed_mesh

        except Exception as e_main_pipeline:
            logger.error(f"--- Pipeline V{self.INTERNAL_VERSION_TRACKER} Failed (Outer Try-Except Block) --- {e_main_pipeline}", exc_info=True)
            return None
        finally:
            self._cleanup_temp_files()
            if not pipeline_successful:
                logger.error(f"Pipeline V{self.INTERNAL_VERSION_TRACKER} did not complete successfully.")

class MeshCleanProcess:
    def __init__(self, input_path, output_path):
        """
        Initialize the MeshCleanProcess class.

        Parameters:
            input_path (str): Path to the input mesh file.
            output_path (str): Path to save the processed mesh.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.ms = pymeshlab.MeshSet()

    def load_mesh(self):
        """Loads the mesh into the MeshSet."""
        try:
            self.ms.load_new_mesh(self.input_path)
            if self.ms.current_mesh().vertex_number() == 0:
                 print(f"Warning: Loaded mesh from {self.input_path} is empty.")
                 return False
            return True
        except pymeshlab.PyMeshLabException as e:
            print(f"Error loading mesh {self.input_path}: {e}")
            return False


    def clean_mesh(self):
        """Cleans the mesh by removing duplicate vertices, faces, and merging close vertices."""
        if self.ms.current_mesh().vertex_number() == 0: return # Skip if mesh is empty
        try:
            self.ms.meshing_remove_duplicate_faces()
            self.ms.meshing_remove_duplicate_vertices()
            # These repair functions can sometimes fail on complex meshes, wrap in try/except
            try:
                self.ms.meshing_repair_non_manifold_edges()
            except pymeshlab.PyMeshLabException as e:
                print(f"Warning: Could not repair non-manifold edges: {e}")
            try:
                self.ms.meshing_repair_non_manifold_vertices()
            except pymeshlab.PyMeshLabException as e:
                 print(f"Warning: Could not repair non-manifold vertices: {e}")
            self.ms.meshing_remove_unreferenced_vertices()
        except pymeshlab.PyMeshLabException as e:
            print(f"Error during mesh cleaning: {e}")


    def fill_holes(self):
        """
        Fills holes in the mesh.
        """
        if self.ms.current_mesh().vertex_number() == 0: return # Skip if mesh is empty
        try:
            # Consider adding a max hole size parameter if needed
            self.ms.meshing_close_holes()
        except pymeshlab.PyMeshLabException as e:
            print(f"Warning: Could not close holes: {e}")


    def reconstruct_surface(self, method='poisson', **kwargs):
        """
        Reconstructs the surface to make the mesh watertight.

        Parameters:
            method (str): Reconstruction method ('poisson').
            **kwargs: Additional parameters for the reconstruction method.
        """
        if self.ms.current_mesh().vertex_number() == 0: return # Skip if mesh is empty
        if method == 'poisson':
            depth = kwargs.get('depth', 10) # Increased default depth slightly
            # Add other potentially useful poisson params if needed (e.g., samples_per_node)
            try:
                self.ms.generate_surface_reconstruction_screened_poisson(depth=depth, preclean=True)
                # After reconstruction, the original mesh is replaced. Check if the new mesh is valid.
                if self.ms.current_mesh().vertex_number() == 0:
                    print(f"Warning: Poisson reconstruction resulted in an empty mesh.")
                    return False
                return True
            except pymeshlab.PyMeshLabException as e:
                 print(f"Error during Poisson reconstruction: {e}")
                 return False
        else:
            raise ValueError("Unsupported reconstruction method. Use 'poisson'.")

    def check_watertight(self):
        """
        Checks if the mesh is watertight.

        Returns:
            bool: True if the mesh is watertight, False otherwise.
        """
        if self.ms.current_mesh().vertex_number() == 0: return False # Empty mesh is not watertight
        try:
            # geo_metrics = self.ms.get_geometric_measures() # Not directly needed for watertightness check
            metrics = self.ms.get_topological_measures()
            is_wt = (
                metrics.get('boundary_edges', -1) == 0 and
                metrics.get('number_holes', -1) == 0 and
                metrics.get('is_mesh_two_manifold', False)
                # Non-manifold checks can sometimes be overly strict or buggy in libraries
                # metrics.get('non_two_manifold_edges', -1) == 0 and
                # metrics.get('non_two_manifold_vertices', -1) == 0
            )
            if not is_wt:
                print(f"Watertight Check Failed: Boundary Edges={metrics.get('boundary_edges', 'N/A')}, Holes={metrics.get('number_holes', 'N/A')}, IsTwoManifold={metrics.get('is_mesh_two_manifold', 'N/A')}")
            return is_wt
        except pymeshlab.PyMeshLabException as e:
            print(f"Error checking watertightness: {e}")
            return False


    def save_mesh(self):
        """Saves the processed mesh to the output path."""
        if self.ms.current_mesh().vertex_number() == 0:
            print(f"Skipping save for empty mesh to {self.output_path}")
            return False
        try:
            self.ms.save_current_mesh(self.output_path)
            return True
        except pymeshlab.PyMeshLabException as e:
            print(f"Error saving mesh {self.output_path}: {e}")
            return False

    def process(self, reconstruction_method='poisson', **kwargs):
        """
        Full pipeline to make the mesh watertight.

        Parameters:
            reconstruction_method (str): Reconstruction method ('poisson').
            **kwargs: Additional parameters for the reconstruction method (e.g., depth).

        Returns:
            bool: True if the processed mesh is watertight and saved, False otherwise.
        """
        if not self.load_mesh():
             return False # Stop if loading failed or mesh is empty

        self.clean_mesh()
        self.fill_holes() # Try filling holes before reconstruction

        if not self.reconstruct_surface(method=reconstruction_method, **kwargs):
             print("Stopping process because surface reconstruction failed or resulted in empty mesh.")
             return False # Stop if reconstruction failed

        # Clean again after reconstruction as Poisson can introduce small issues
        self.clean_mesh()

        is_watertight = self.check_watertight()
        saved_successfully = self.save_mesh()

        return is_watertight and saved_successfully
    
    @staticmethod
    def get_all_boundary_loops(mesh: trimesh.Trimesh, min_loop_len: int = 3) -> List[np.ndarray]:
        """
        Returns a list of all distinct boundary loops (as arrays of ordered vertex indices).
        Uses manual edge counting and component finding. Orders each loop if possible.
        """
        all_loops_ordered = []
        if not MeshCleanProcess._is_mesh_valid_for_concat(mesh, "InputMeshForAllLoops"):
            return all_loops_ordered

        boundary_edges = MeshCleanProcess._get_boundary_edges_manually_from_faces(mesh)
        if boundary_edges is None or len(boundary_edges) < min_loop_len: # Need at least min_loop_len edges
            return all_loops_ordered

        # Find connected components (vertex sets for each loop)
        # Removed 'count' param for broader trimesh compatibility
        loop_components_vidx = trimesh.graph.connected_components(boundary_edges, min_len=min_loop_len) 

        if not loop_components_vidx or not any(c is not None and len(c) >= min_loop_len for c in loop_components_vidx):
            return all_loops_ordered

        for i, comp_vidx_unordered in enumerate(loop_components_vidx):
            if comp_vidx_unordered is None or len(comp_vidx_unordered) < min_loop_len:
                continue
            
            # Filter boundary_edges to get only edges belonging to this specific component
            comp_set = set(comp_vidx_unordered)
            edges_for_this_comp = [e for e in boundary_edges if e[0] in comp_set and e[1] in comp_set]

            if not edges_for_this_comp or len(edges_for_this_comp) < len(comp_vidx_unordered) -1 : # Heuristic for enough edges
                 print(f"DEBUG get_all_boundary_loops: Comp {i} ({len(comp_vidx_unordered)}V) has too few specific edges ({len(edges_for_this_comp)}E). Skipping ordering for it.")
                 # Could add the unordered component if desired, but usually need ordered for filling.
                 # all_loops_ordered.append(np.array(comp_vidx_unordered, dtype=int)) # Add unordered
                 continue

            ordered_vidx = MeshCleanProcess._order_loop_vertices_from_edges(
                f"Loop_{i}_Order", comp_vidx_unordered, np.array(edges_for_this_comp)
            )
            if ordered_vidx is not None and len(ordered_vidx) >= min_loop_len:
                all_loops_ordered.append(ordered_vidx)
            # else:
                # print(f"DEBUG get_all_boundary_loops: Ordering failed for component {i} ({len(comp_vidx_unordered)}V).")
        
        return all_loops_ordered
                                    
    @staticmethod
    def _is_mesh_valid_for_concat(mesh: Optional[trimesh.Trimesh], mesh_name: str) -> bool:
        """Helper to check if a trimesh object is suitable for concatenation."""
        if mesh is None:
            print(f"CRIT_DEBUG_CONCAT_CHECK: {mesh_name} is None.")
            return False
        if mesh.is_empty:
            print(f"CRIT_DEBUG_CONCAT_CHECK: {mesh_name} is empty (V={len(mesh.vertices)}, F={len(mesh.faces)}).")
            return False
        if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces'):
            print(f"CRIT_DEBUG_CONCAT_CHECK: {mesh_name} is missing vertices or faces attributes.")
            return False
        if mesh.vertices.ndim != 2 or mesh.vertices.shape[1] != 3:
            print(f"CRIT_DEBUG_CONCAT_CHECK: {mesh_name} vertices have wrong shape {mesh.vertices.shape}.")
            return False
        if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
            print(f"CRIT_DEBUG_CONCAT_CHECK: {mesh_name} faces have wrong shape {mesh.faces.shape}.")
            return False
        # print(f"DEBUG_CONCAT_CHECK: {mesh_name} seems valid for concat (V={len(mesh.vertices)}, F={len(mesh.faces)}).")
        return True
                                    
    @staticmethod
    def _get_boundary_edges_manually_from_faces(mesh: trimesh.Trimesh) -> Optional[np.ndarray]:
        if not (mesh and not mesh.is_empty and \
                hasattr(mesh, 'vertices') and mesh.vertices is not None and len(mesh.vertices) > 0 and \
                hasattr(mesh, 'faces') and mesh.faces is not None and len(mesh.faces) > 0 ):
            return None
        try:
            all_face_edges_canonical = []
            for face_idx, face in enumerate(mesh.faces):
                if len(face) != 3: continue
                all_face_edges_canonical.append(tuple(sorted((face[0], face[1]))))
                all_face_edges_canonical.append(tuple(sorted((face[1], face[2]))))
                all_face_edges_canonical.append(tuple(sorted((face[2], face[0]))))
            if not all_face_edges_canonical: return np.array([], dtype=int).reshape(-1,2)
            edge_counts = Counter(all_face_edges_canonical)
            manual_boundary_edges_list = [list(edge_tuple) for edge_tuple, count in edge_counts.items() if count == 1]
            if not manual_boundary_edges_list: return np.array([], dtype=int).reshape(-1,2)
            return np.array(manual_boundary_edges_list, dtype=int)
        except Exception as e:
            print(f"DEBUG: Error in _get_boundary_edges_manually_from_faces: {e}"); traceback.print_exc(); return None

    @staticmethod
    def _order_loop_vertices_from_edges(mesh_name_for_debug: str, loop_vertex_indices_unique: np.ndarray, all_edges_for_this_loop: np.ndarray, min_completeness_for_partial_ok: float = 0.75, min_path_len_for_partial_ok: int = 10) -> Optional[np.ndarray]:
        if loop_vertex_indices_unique is None or len(loop_vertex_indices_unique) < 3 or all_edges_for_this_loop is None or len(all_edges_for_this_loop) < max(0, len(loop_vertex_indices_unique) -1) : return None
        adj = {v_idx: [] for v_idx in loop_vertex_indices_unique}; vertex_degrees_in_loop = {v_idx: 0 for v_idx in loop_vertex_indices_unique}
        for u, v in all_edges_for_this_loop:
            if u in adj and v in adj: adj[u].append(v); adj[v].append(u); vertex_degrees_in_loop[u] += 1; vertex_degrees_in_loop[v] += 1
        if not loop_vertex_indices_unique.size: return None
        start_node_manual = loop_vertex_indices_unique[0]
        deg1_nodes = [v_idx for v_idx in loop_vertex_indices_unique if vertex_degrees_in_loop.get(v_idx,0) == 1]
        if deg1_nodes: start_node_manual = deg1_nodes[0]
        else:
            deg2_nodes = [v_idx for v_idx in loop_vertex_indices_unique if vertex_degrees_in_loop.get(v_idx,0) == 2]
            if deg2_nodes: start_node_manual = deg2_nodes[0]
        ordered_path_manual = [start_node_manual]; visited_edges_in_path = set(); current_node = start_node_manual
        for _ in range(len(all_edges_for_this_loop) + 2):
            found_next = False; neighbors = adj.get(current_node, [])
            for neighbor in neighbors:
                edge_cand = tuple(sorted((current_node, neighbor)))
                if edge_cand not in visited_edges_in_path:
                    ordered_path_manual.append(neighbor); visited_edges_in_path.add(edge_cand); current_node = neighbor; found_next = True; break
            if not found_next: break
        if len(ordered_path_manual) > 1 and ordered_path_manual[0] == ordered_path_manual[-1]: ordered_path_manual = ordered_path_manual[:-1]
        num_unique_in_path = len(np.unique(ordered_path_manual)); num_expected_unique = len(loop_vertex_indices_unique)
        if num_unique_in_path == num_expected_unique and len(ordered_path_manual) == num_expected_unique: return np.array(ordered_path_manual, dtype=int)
        if num_unique_in_path >= num_expected_unique * min_completeness_for_partial_ok and len(ordered_path_manual) >= min_path_len_for_partial_ok: return np.array(ordered_path_manual, dtype=int)
        else: return None

    @staticmethod
    def resample_polyline_to_count(polyline_vertices: np.ndarray, target_count: int) -> Optional[np.ndarray]:
        if polyline_vertices is None or not isinstance(polyline_vertices, np.ndarray) or polyline_vertices.ndim != 2 or polyline_vertices.shape[1] != 3: return None
        if len(polyline_vertices) < 2 : return polyline_vertices
        if target_count < 2: return polyline_vertices[:target_count] if target_count > 0 else np.array([],dtype=polyline_vertices.dtype).reshape(0,3)
        if len(polyline_vertices) == target_count: return polyline_vertices
        distances = np.linalg.norm(np.diff(polyline_vertices, axis=0), axis=1)
        cumulative_distances = np.concatenate(([0], np.cumsum(distances))); total_length = cumulative_distances[-1]
        if total_length < 1e-9:
            if target_count > 0 and len(polyline_vertices) > 0: return np.tile(polyline_vertices[0], (target_count, 1))
            else: return np.array([],dtype=polyline_vertices.dtype).reshape(0,3)
        sampled_arc_lengths = np.linspace(0, total_length, target_count); resampled_points_list = []
        for s_len in sampled_arc_lengths:
            idx = np.searchsorted(cumulative_distances, s_len, side='right') - 1; idx = np.clip(idx, 0, len(polyline_vertices) - 2)
            p0, p1 = polyline_vertices[idx], polyline_vertices[idx+1]; seg_len = cumulative_distances[idx+1] - cumulative_distances[idx]
            t = (s_len - cumulative_distances[idx]) / seg_len if seg_len > 1e-9 else 0.0
            resampled_points_list.append(p0 + np.clip(t, 0, 1) * (p1 - p0))
        return np.array(resampled_points_list) if resampled_points_list else np.array([], dtype=polyline_vertices.dtype).reshape(0,3)

    @staticmethod
    def get_dominant_plane(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if points is None or not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
            origin = np.mean(points, axis=0) if (points is not None and isinstance(points, np.ndarray) and points.ndim==2 and points.shape[1]==3 and len(points) > 0) else np.array([0.0, 0.0, 0.0])
            return np.array([0.0, 0.0, 1.0]), origin
        center = np.mean(points, axis=0); centered_points = points - center
        try:
            _, _, vh = np.linalg.svd(centered_points, full_matrices=False); plane_normal = vh[-1, :]
            return trimesh.util.unitize(plane_normal), center
        except np.linalg.LinAlgError: return np.array([0.0, 0.0, 1.0]), center
        except Exception: return np.array([0.0, 0.0, 1.0]), center

    @staticmethod
    def _get_hole_boundary_edges_from_removed_faces(original_mesh: trimesh.Trimesh, faces_removed_mask: np.ndarray) -> Optional[np.ndarray]:
        if not MeshCleanProcess._is_mesh_valid_for_concat(original_mesh, "OriginalMeshForHoleBoundary") or \
           faces_removed_mask is None or not isinstance(faces_removed_mask, np.ndarray) or faces_removed_mask.ndim != 1 or len(faces_removed_mask) != len(original_mesh.faces):
            return None
        if not np.any(faces_removed_mask) or np.all(faces_removed_mask): return np.array([], dtype=int).reshape(-1, 2)
        try:
            adj_face_pairs = original_mesh.face_adjacency; adj_shared_edges = original_mesh.face_adjacency_edges
            hole_boundary_edges_list = [adj_shared_edges[i] for i in range(len(adj_face_pairs)) if faces_removed_mask[adj_face_pairs[i][0]] != faces_removed_mask[adj_face_pairs[i][1]]]
            if not hole_boundary_edges_list: return np.array([], dtype=int).reshape(-1, 2)
            return np.array(hole_boundary_edges_list, dtype=int)
        except Exception as e:
            print(f"DEBUG: Error in _get_hole_boundary_edges_from_removed_faces: {e}"); traceback.print_exc(); return None

    @staticmethod
    def _iterative_non_manifold_repair_pml(
        input_mesh: trimesh.Trimesh,
        max_iterations: int = 5,
        debug_dir: Optional[str] = None,
        temp_file_prefix: str = "nm_repair"
    ) -> trimesh.Trimesh:
        if not MeshCleanProcess._is_mesh_valid_for_concat(input_mesh, "InputForIterativeRepair"):
            print(f"INFO V_IR: Input mesh for iterative repair ('{temp_file_prefix}') is invalid. Skipping repair.") # V_IR for Verbose Iterative Repair
            return input_mesh

        print(f"--- V_IR: Starting Iterative Non-Manifold Repair (max {max_iterations} iterations) for '{temp_file_prefix}' ---")
        current_mesh = input_mesh.copy()
        local_temp_files = []
        
        def make_local_temp_path(suffix:str, directory:Optional[str])->str:
            _fd, path = tempfile.mkstemp(suffix=suffix + ".ply", dir=directory)
            os.close(_fd); local_temp_files.append(path); return path

        initial_v_count = current_mesh.vertices.shape[0]
        initial_f_count = current_mesh.faces.shape[0]
        print(f"  V_IR: Initial state: Vertices={initial_v_count}, Faces={initial_f_count}")

        prev_v_count_geom_stagnant = -1
        prev_f_count_geom_stagnant = -1
        prev_selected_nm_faces_stagnant = -1
        prev_selected_nm_vertices_stagnant = -1
        stagnation_counter = 0

        for i in range(max_iterations):
            print(f"  V_IR: Repair Iteration {i+1}/{max_iterations}:")
            
            selected_nm_faces_before = -1
            selected_nm_vertices_before = -1
            ms_check_before = None
            temp_check_in_before_path = ""
            try:
                ms_check_before = pymeshlab.MeshSet()
                temp_check_in_before_path = make_local_temp_path(f"{temp_file_prefix}_iter{i}_check_in_before", debug_dir)
                current_mesh.export(temp_check_in_before_path)
                ms_check_before.load_new_mesh(temp_check_in_before_path)
            
                if ms_check_before.current_mesh_id() == -1 or ms_check_before.current_mesh().vertex_number() == 0:
                    print("    V_IR ERROR: Mesh became empty or invalid before repair check. Stopping iteration.")
                    break 
                
                v_count_iter_start = ms_check_before.current_mesh().vertex_number()
                f_count_iter_start = ms_check_before.current_mesh().face_number()

                ms_check_before.apply_filter('compute_selection_by_non_manifold_edges_per_face')
                selected_nm_faces_before = ms_check_before.current_mesh().selected_face_number()
                ms_check_before.load_new_mesh(temp_check_in_before_path) 

                ms_check_before.apply_filter('select_non_manifold_vertices')
                selected_nm_vertices_before = ms_check_before.current_mesh().selected_vertex_number()
                
                print(f"    V_IR: Before repair: V={v_count_iter_start}, F={f_count_iter_start}, NM Edge-Faces={selected_nm_faces_before}, NM Verts={selected_nm_vertices_before}")

                if selected_nm_faces_before == 0 and selected_nm_vertices_before == 0 :
                    print("    V_IR: No non-manifold elements selected. Repair considered complete.")
                    break
            except AttributeError as e_attr: 
                print(f"    V_IR FATAL ERROR: PyMeshLab filter for non-manifold selection not found or misnamed: {e_attr}. Stopping iterative repair.")
                for f_path in local_temp_files:
                    if os.path.exists(f_path):
                        try: os.remove(f_path)
                        except OSError: pass
                return current_mesh 
            except Exception as e_check_before:
                print(f"    V_IR ERROR during pre-repair check: {e_check_before}. Stopping iteration.")
                break
            finally:
                if ms_check_before is not None: del ms_check_before

            ms_repair = None
            repaired_this_iter_flag = False
            repair_method_used = "None" 
            try:
                ms_repair = pymeshlab.MeshSet()
                temp_repair_in = make_local_temp_path(f"{temp_file_prefix}_iter{i}_repair_in", debug_dir)
                current_mesh.export(temp_repair_in)
                ms_repair.load_new_mesh(temp_repair_in)
                
                if ms_repair.current_mesh_id() == -1 or ms_repair.current_mesh().vertex_number() == 0:
                    print("    V_IR ERROR: Mesh became empty or invalid before repair attempt. Stopping iteration.")
                    break

                try:
                    print("    V_IR: Attempting repair with 'Split Vertices'...")
                    ms_repair.meshing_repair_non_manifold_edges(method='Split Vertices')
                    repaired_this_iter_flag = True; repair_method_used = "Split Vertices"
                    print("      V_IR: 'Split Vertices' applied.")
                except pymeshlab.PyMeshLabException:
                    print("      V_IR: 'Split Vertices' failed. Attempting repair with 'Remove Faces'...")
                    try:
                        ms_repair.meshing_repair_non_manifold_edges(method='Remove Faces')
                        repaired_this_iter_flag = True; repair_method_used = "Remove Faces"
                        print("      V_IR: 'Remove Faces' applied.")
                    except pymeshlab.PyMeshLabException as e_repair_fallback:
                        print(f"    V_IR ERROR: Both non-manifold edge repair methods failed: {e_repair_fallback}. Stopping iteration.")
                        break 
                
                if not repaired_this_iter_flag:
                    print("    V_IR: No repair method was successfully applied in this iteration. Stopping.")
                    break

                ms_repair.meshing_remove_unreferenced_vertices()
                temp_repair_out = make_local_temp_path(f"{temp_file_prefix}_iter{i}_repair_out", debug_dir)
                ms_repair.save_current_mesh(temp_repair_out)
                repaired_mesh_trimesh_iter = trimesh.load_mesh(temp_repair_out, process=False)
                
                if MeshCleanProcess._is_mesh_valid_for_concat(repaired_mesh_trimesh_iter, f"RepairedIter{i}"):
                    current_mesh = repaired_mesh_trimesh_iter
                    v_count_after_repair = current_mesh.vertices.shape[0]
                    f_count_after_repair = current_mesh.faces.shape[0]
                    print(f"    V_IR: After repair (method: {repair_method_used}): V={v_count_after_repair}, F={f_count_after_repair}")
                else:
                    print("    V_IR ERROR: Repaired mesh is invalid after loading back. Stopping iteration.")
                    break
            except Exception as e_repair_block:
                 print(f"    V_IR ERROR during repair block: {e_repair_block}. Stopping iteration.")
                 break
            finally:
                if ms_repair is not None: del ms_repair

            selected_nm_faces_after = -1; selected_nm_vertices_after = -1
            ms_check_after = None; temp_check_out_after_path = ""
            current_v_count_geom_after = -1; current_f_count_geom_after = -1
            try:
                ms_check_after = pymeshlab.MeshSet()
                temp_check_out_after_path = make_local_temp_path(f"{temp_file_prefix}_iter{i}_check_out_after", debug_dir)
                current_mesh.export(temp_check_out_after_path)
                ms_check_after.load_new_mesh(temp_check_out_after_path)
                if ms_check_after.current_mesh_id() != -1 and ms_check_after.current_mesh().vertex_number() > 0:
                    current_v_count_geom_after = ms_check_after.current_mesh().vertex_number()
                    current_f_count_geom_after = ms_check_after.current_mesh().face_number()
                    ms_check_after.apply_filter('compute_selection_by_non_manifold_edges_per_face')
                    selected_nm_faces_after = ms_check_after.current_mesh().selected_face_number()
                    ms_check_after.load_new_mesh(temp_check_out_after_path)
                    ms_check_after.apply_filter('select_non_manifold_vertices')
                    selected_nm_vertices_after = ms_check_after.current_mesh().selected_vertex_number()
                    
                    print(f"    V_IR: After repair check: V={current_v_count_geom_after}, F={current_f_count_geom_after}, NM Edge-Faces={selected_nm_faces_after}, NM Verts={selected_nm_vertices_after}")
                    
                    if selected_nm_faces_after == 0 and selected_nm_vertices_after == 0:
                        print(f"    V_IR: Non-manifold elements resolved (by selection check) in iteration {i+1}.")
                        break
                    if current_v_count_geom_after == prev_v_count_geom_stagnant and \
                       current_f_count_geom_after == prev_f_count_geom_stagnant and \
                       selected_nm_faces_after == prev_selected_nm_faces_stagnant and \
                       selected_nm_vertices_after == prev_selected_nm_vertices_stagnant:
                        stagnation_counter += 1
                        if stagnation_counter >= 2: 
                            print(f"    V_IR: Repair process stagnated for {stagnation_counter} iterations (geom and NM selections unchanged). Stopping.")
                            break
                    else: stagnation_counter = 0
                    prev_v_count_geom_stagnant = current_v_count_geom_after
                    prev_f_count_geom_stagnant = current_f_count_geom_after
                    prev_selected_nm_faces_stagnant = selected_nm_faces_after
                    prev_selected_nm_vertices_stagnant = selected_nm_vertices_after
                else: 
                    print("    V_IR ERROR: Mesh became empty or invalid for post-repair check. Stopping iteration.")
                    break
            except AttributeError as e_attr_after: 
                print(f"    V_IR FATAL ERROR during post-repair check: PyMeshLab filter for non-manifold selection not found or misnamed: {e_attr_after}. Stopping iterative repair.")
                for f_path in local_temp_files:
                    if os.path.exists(f_path):
                        try: os.remove(f_path)
                        except OSError: pass
                return current_mesh
            except Exception as e_check_after_block:
                print(f"    V_IR ERROR during post-repair check: {e_check_after_block}. Assuming repair failed to improve.")
                break
            finally:
                if ms_check_after is not None: del ms_check_after
            
            if i == max_iterations - 1:
                print(f"  V_IR: Reached max iterations ({max_iterations}). Final selected NM Edge-Faces: {selected_nm_faces_after}, NM Verts: {selected_nm_vertices_after}")
        
        final_v_count = current_mesh.vertices.shape[0]
        final_f_count = current_mesh.faces.shape[0]
        print(f"--- V_IR: Finished Iterative Non-Manifold Repair for '{temp_file_prefix}' ---")
        print(f"  V_IR: Final state: Vertices={final_v_count} (Initial: {initial_v_count}), Faces={final_f_count} (Initial: {initial_f_count})")

        for f_path in local_temp_files:
            if os.path.exists(f_path):
                try: os.remove(f_path)
                except OSError: pass
        return current_mesh
    
    @staticmethod
    def _filter_large_triangles_from_fill(
        mesh: trimesh.Trimesh,
        target_face_indices: Optional[np.ndarray] = None, # New parameter
        max_allowed_edge_length: Optional[float] = None,
        max_allowed_area_factor: Optional[float] = None,
        reference_mesh_for_stats: Optional[trimesh.Trimesh] = None,
        mesh_name_for_debug: str = "large_tri_filter"
    ) -> trimesh.Trimesh:
        if not MeshCleanProcess._is_mesh_valid_for_concat(mesh, f"InputForLargeTriFilter_{mesh_name_for_debug}"):
            return mesh
        if max_allowed_edge_length is None and max_allowed_area_factor is None:
            return mesh # No criteria to filter by

        # print(f"--- V_FILTER_LARGE_TRI: Filtering large triangles for {mesh_name_for_debug} ---")
        
        # Start with all faces marked to be kept
        faces_to_keep_mask = np.ones(len(mesh.faces), dtype=bool)
        
        # If target_face_indices are provided, we only consider these for potential removal.
        # Other faces are implicitly kept.
        # So, we'll build a "faces_to_potentially_remove_mask" based on target_face_indices.
        candidate_faces_for_removal_mask = np.zeros(len(mesh.faces), dtype=bool)
        if target_face_indices is not None and len(target_face_indices) > 0:
            # Ensure indices are valid
            valid_target_indices = target_face_indices[target_face_indices < len(mesh.faces)]
            candidate_faces_for_removal_mask[valid_target_indices] = True
            # print(f"  V_FILTER_LARGE_TRI: Targeting {np.sum(candidate_faces_for_removal_mask)} faces for potential removal.")
        else:
            # If no target_face_indices, all faces are candidates for removal based on criteria
            candidate_faces_for_removal_mask.fill(True)
            # print(f"  V_FILTER_LARGE_TRI: Targeting ALL faces for potential removal.")

        # Temporary mask for faces that meet removal criteria within the candidates
        faces_meeting_removal_criteria_mask = np.zeros(len(mesh.faces), dtype=bool)

        # 1. Filter by Max Edge Length (Absolute)
        if max_allowed_edge_length is not None and max_allowed_edge_length > 0:
            # print(f"  V_FILTER_LARGE_TRI: Applying max_allowed_edge_length: {max_allowed_edge_length}")
            if np.any(candidate_faces_for_removal_mask): # Only proceed if there are candidates
                face_edge_lengths = np.zeros((len(mesh.faces), 3))
                # Calculate edge lengths only for candidate faces to save computation,
                # but it's simpler to calculate for all and then mask.
                for i, face_verts_indices in enumerate(mesh.faces):
                    if candidate_faces_for_removal_mask[i]: # Only compute for candidates
                        if face_verts_indices.max() < len(mesh.vertices):
                            v0, v1, v2 = mesh.vertices[face_verts_indices]
                            face_edge_lengths[i, 0] = np.linalg.norm(v0 - v1)
                            face_edge_lengths[i, 1] = np.linalg.norm(v1 - v2)
                            face_edge_lengths[i, 2] = np.linalg.norm(v2 - v0)
                        else: face_edge_lengths[i, :] = np.inf 
                    else:
                        face_edge_lengths[i, :] = 0 # Not a candidate, won't be removed by this
                
                max_edge_per_face = np.max(face_edge_lengths, axis=1)
                # Mark for removal if it's a candidate AND its max edge is too long
                marked_by_edge = (max_edge_per_face > max_allowed_edge_length) & candidate_faces_for_removal_mask
                faces_meeting_removal_criteria_mask[marked_by_edge] = True
                # num_removed_by_edge = np.sum(marked_by_edge)
                # if num_removed_by_edge > 0:
                    # print(f"  V_FILTER_LARGE_TRI: Marked {num_removed_by_edge} faces within target for removal by edge length.")

        # 2. Filter by Face Area (Relative to typical surface triangles)
        if max_allowed_area_factor is not None and max_allowed_area_factor > 0:
            # print(f"  V_FILTER_LARGE_TRI: Applying max_allowed_area_factor: {max_allowed_area_factor}")
            if np.any(candidate_faces_for_removal_mask): # Only proceed if there are candidates
                current_mesh_face_areas = mesh.area_faces
                median_area_stat = 0
                if reference_mesh_for_stats is not None and \
                   MeshCleanProcess._is_mesh_valid_for_concat(reference_mesh_for_stats, "RefMeshForAreaStats") and \
                   len(reference_mesh_for_stats.faces)>0:
                    median_area_stat = np.median(reference_mesh_for_stats.area_faces)
                elif len(current_mesh_face_areas[candidate_faces_for_removal_mask & ~faces_meeting_removal_criteria_mask]) > 0 : 
                    # Use stats from candidate faces not already marked for removal by edge length
                    valid_areas_for_stat_calc = current_mesh_face_areas[candidate_faces_for_removal_mask & ~faces_meeting_removal_criteria_mask]
                    median_area_stat = np.median(valid_areas_for_stat_calc)
                
                if median_area_stat > 1e-9: 
                    area_threshold = median_area_stat * max_allowed_area_factor
                    # Mark for removal if it's a candidate AND its area is too large
                    marked_by_area = (current_mesh_face_areas > area_threshold) & candidate_faces_for_removal_mask
                    faces_meeting_removal_criteria_mask[marked_by_area] = True 
                    # num_removed_by_area = np.sum(marked_by_area & ~faces_to_remove_by_edge_length_within_candidates) # Count only newly marked by area
                    # if num_removed_by_area > 0:
                        # print(f"  V_FILTER_LARGE_TRI: Marked {num_removed_by_area} additional faces within target for removal by area.")

        # Now, update the main faces_to_keep_mask
        faces_to_keep_mask[faces_meeting_removal_criteria_mask] = False
        
        num_total_removed = np.sum(~faces_to_keep_mask)
        if num_total_removed > 0:
            # print(f"  V_FILTER_LARGE_TRI: Total faces to remove: {num_total_removed} out of {len(mesh.faces)}")
            if num_total_removed == len(mesh.faces):
                # print(f"  V_FILTER_LARGE_TRI WARN: All faces would be removed. Aborting filtering for {mesh_name_for_debug}.")
                return mesh 

            filtered_mesh = mesh.copy(); filtered_mesh.update_faces(faces_to_keep_mask); filtered_mesh.remove_unreferenced_vertices()
            if MeshCleanProcess._is_mesh_valid_for_concat(filtered_mesh, f"FilteredMesh_{mesh_name_for_debug}"):
                # print(f"  V_FILTER_LARGE_TRI: Mesh after filtering: V={len(filtered_mesh.vertices)}, F={len(filtered_mesh.faces)}")
                return filtered_mesh
            else:
                # print(f"  V_FILTER_LARGE_TRI WARN: Mesh became invalid after filtering. Returning original.")
                return mesh 
        # else: print(f"  V_FILTER_LARGE_TRI: No faces met criteria for large triangle removal on {mesh_name_for_debug}.")
        return mesh

    @staticmethod
    def run_face_grafting_pipeline(
        full_body_mesh_path: str,
        output_path: str,
        smplx_face_mesh_path: str,
        # Pass all the config parameters here or a pre-made config object
        projection_footprint_threshold: float = 0.01,
        footprint_dilation_rings: int = 1,
        body_simplification_target_faces: int = 12000,
        stitch_method: str = "body_driven_loft", 
        smplx_face_neck_loop_strategy: str = "full_face_silhouette",
        alignment_resample_count: int = 1000,
        loft_strip_resample_count: int = 100,
        max_seam_hole_fill_vertices: int = 250,
        final_polish_max_hole_edges: int = 100,
        iterative_repair_s1_iters: int = 5,
        # iterative_repair_s2_iters: int = 5, # Not directly used in this refactor pass
        # iterative_repair_s2_remesh_percent: Optional[float] = None, # Not directly used
        spider_filter_area_factor: Optional[float] = 200.0,
        spider_filter_max_edge_len_factor: Optional[float] = 0.15,
        debug_dir: Optional[str] = None
    ) -> Optional[trimesh.Trimesh]:
        
        config = FaceGraftingConfig(
            projection_footprint_threshold=projection_footprint_threshold,
            footprint_dilation_rings=footprint_dilation_rings,
            body_simplification_target_faces=body_simplification_target_faces,
            stitch_method=stitch_method,
            smplx_face_neck_loop_strategy=smplx_face_neck_loop_strategy,
            alignment_resample_count=alignment_resample_count,
            loft_strip_resample_count=loft_strip_resample_count,
            max_seam_hole_fill_vertices=max_seam_hole_fill_vertices,
            final_polish_max_hole_edges=final_polish_max_hole_edges,
            iterative_repair_s1_iters=iterative_repair_s1_iters,
            # iterative_repair_s2_iters=iterative_repair_s2_iters,
            # iterative_repair_s2_remesh_percent=iterative_repair_s2_remesh_percent,
            spider_filter_area_factor=spider_filter_area_factor,
            spider_filter_max_edge_len_factor=spider_filter_max_edge_len_factor,
            debug_dir=debug_dir
        )
        
        pipeline = FaceGraftingPipeline(
            full_body_mesh_path=full_body_mesh_path,
            smplx_face_mesh_path=smplx_face_mesh_path,
            output_path=output_path,
            config=config
        )
        return pipeline.process()
