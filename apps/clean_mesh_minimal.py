import pymeshlab
import trimesh
import numpy as np
from scipy.spatial import cKDTree  
import os 
# import traceback # No longer used
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
    projection_footprint_threshold: float = 0.01
    footprint_dilation_rings: int = 10
    body_simplification_target_faces: int = 12000
    stitch_method: str = "body_driven_loft"
    smplx_face_neck_loop_strategy: str = "full_face_silhouette"
    alignment_resample_count: int = 1000
    # loft_strip_resample_count: int = 100 # Unused by the current _create_loft_stitch_mesh
    max_seam_hole_fill_vertices: int = 250
    final_polish_max_hole_edges: int = 100
    iterative_repair_s1_iters: int = 5
    # iterative_repair_s2_iters: int = 5 # Unused placeholder
    # iterative_repair_s2_remesh_percent: Optional[float] = None # Unused placeholder
    spider_filter_area_factor: Optional[float] = 300.0
    spider_filter_max_edge_len_factor: Optional[float] = 0.15
    vertex_coordinate_precision_digits: Optional[int] = 5 
    hole_boundary_smoothing_iterations: int = 10
    hole_boundary_smoothing_factor: float = 0.5

class FaceGraftingPipeline:
    """
    Pipeline for grafting an SMPLX face mesh onto a full body mesh.
    """
    INTERNAL_VERSION_TRACKER = "5.80_refactored_pipeline_with_seam_check"

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
                                         version_tracker: str) -> Optional[trimesh.Trimesh]:
        if mesh is None: 
            logger.warning(f"Skipping thorough cleaning for {mesh_name_for_log}: input mesh is None.")
            return None
        
        if not MeshCleanProcess._is_mesh_valid_for_concat(mesh, f"{mesh_name_for_log}_PreClean_V{version_tracker}"):
            logger.warning(f"Skipping thorough cleaning for {mesh_name_for_log}: input mesh is initially invalid.")
            return mesh 

        # logger.info(f"Thoroughly cleaning Trimesh object: {mesh_name_for_log} (Initial V={len(mesh.vertices)}, F={len(mesh.faces)})")
        try:
            mesh.remove_duplicate_faces()
            if mesh.is_empty: 
                logger.warning(f"  {mesh_name_for_log} became empty after remove_duplicate_faces.")
                return None 
            mesh.merge_vertices(merge_tex=False, merge_norm=False)
            if mesh.is_empty:
                logger.warning(f"  {mesh_name_for_log} became empty after merge_vertices.")
                return None
            mesh.remove_unreferenced_vertices()
            mesh.remove_degenerate_faces()
            if mesh.is_empty:
                logger.warning(f"  {mesh_name_for_log} became empty after remove_degenerate_faces.")
                return None
            mesh.fix_normals(multibody=True) 
            if not MeshCleanProcess._is_mesh_valid_for_concat(mesh, f"{mesh_name_for_log}_PostClean_V{version_tracker}"):
                logger.warning(f"Thorough cleaning for {mesh_name_for_log} resulted in an invalid mesh.")
                return None 
            # logger.info(f"Thorough cleaning for {mesh_name_for_log} completed. Final V={len(mesh.vertices)}, F={len(mesh.faces)}")
            return mesh
        except Exception as e:
            logger.error(f"Exception during thorough cleaning of {mesh_name_for_log}: {e}", exc_info=True)
            return None

    def _load_and_simplify_meshes(self) -> bool:
        logger.info("=== STEP 1 & 2: Load meshes, clean face, simplify and clean body ===")
        
        self.original_smplx_face_geom_tri = trimesh.load_mesh(self.smplx_face_mesh_path, process=True)
        if self.original_smplx_face_geom_tri is None or \
           not MeshCleanProcess._is_mesh_valid_for_concat(self.original_smplx_face_geom_tri, "InitialSMPLXFace"):
            logger.critical(f"Failed to load or invalid SMPLX Face mesh: '{self.smplx_face_mesh_path}'. Aborting.")
            return False
        
        cleaned_smplx_face = FaceGraftingPipeline._thoroughly_clean_trimesh_object(
            self.original_smplx_face_geom_tri.copy(), "InitialSMPLXFace", self.INTERNAL_VERSION_TRACKER
        )
        if cleaned_smplx_face is not None: self.original_smplx_face_geom_tri = cleaned_smplx_face
        else: logger.warning("Thorough cleaning of initial SMPLX face failed. Using original loaded version.")

        self.simplified_body_trimesh = trimesh.load_mesh(self.full_body_mesh_path, process=True)
        if self.simplified_body_trimesh is None or \
           not MeshCleanProcess._is_mesh_valid_for_concat(self.simplified_body_trimesh, "InitialFullBody"):
            logger.critical(f"Failed to load or invalid Full Body mesh: '{self.full_body_mesh_path}'. Aborting.")
            return False

        if self.original_smplx_face_geom_tri: self.original_smplx_face_geom_tri.fix_normals()
        if self.simplified_body_trimesh: self.simplified_body_trimesh.fix_normals()

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
                        except (AttributeError, pymeshlab.PyMeshLabException): pass # Ignore if not available or fails
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

        cleaned_simplified_body = FaceGraftingPipeline._thoroughly_clean_trimesh_object(
            self.simplified_body_trimesh.copy(), "SimplifiedBodyFinalClean", self.INTERNAL_VERSION_TRACKER
        )
        if cleaned_simplified_body is not None: self.simplified_body_trimesh = cleaned_simplified_body
        else: logger.warning("Final thorough cleaning of simplified body failed. Using pre-clean version.")

        if not MeshCleanProcess._is_mesh_valid_for_concat(self.simplified_body_trimesh, "FinalSimplifiedBody"):
            logger.critical("Simplified body mesh is invalid after all steps in _load_and_simplify. Aborting.")
            return False
            
        if self.simplified_body_trimesh is not None and not self.simplified_body_trimesh.is_empty:
            try:
                _ = self.simplified_body_trimesh.edges # Ensure graph properties
                if len(self.simplified_body_trimesh.boundary_edges) > 0:
                    logger.warning(f"FINAL SIMPLIFIED BODY MESH has boundary edges. Not watertight.")
            except Exception: pass # Ignore errors in this non-critical check
        
        logger.info(f"Finished loading and simplifying. Final body: V={len(self.simplified_body_trimesh.vertices)}, F={len(self.simplified_body_trimesh.faces)}")
        return True

    def _smooth_main_hole_boundary(self) -> bool:
        if self.config.hole_boundary_smoothing_iterations <= 0:
            return True
        if not self.body_with_hole_trimesh or \
           not MeshCleanProcess._is_mesh_valid_for_concat(self.body_with_hole_trimesh, "BodyForHoleSmooth"):
            logger.warning("Cannot smooth hole boundary: body_with_hole_trimesh is invalid or None.")
            return False

        logger.info(f"Smoothing main hole boundary ({self.config.hole_boundary_smoothing_iterations} iter)...")
        all_loops_vidx = MeshCleanProcess.get_all_boundary_loops(self.body_with_hole_trimesh, min_loop_len=3)
        if not all_loops_vidx: return True
        largest_loop_vidx = max(all_loops_vidx, key=len, default=None)
        if largest_loop_vidx is None or len(largest_loop_vidx) < 3: return True
        
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
        if hasattr(self.body_with_hole_trimesh, '_cache'): self.body_with_hole_trimesh._cache.clear()
        return True

    def _determine_hole_faces_and_create_body_with_hole(self) -> bool:
        if self.simplified_body_trimesh is None or self.original_smplx_face_geom_tri is None: 
            logger.error("Missing meshes for hole creation. Aborting.")
            return False 
        
        logger.info(f"=== STEP 3: Determining Hole Faces & Creating Body With Hole ===")
        if len(self.simplified_body_trimesh.faces) == 0 and self.config.projection_footprint_threshold > 0: 
             logger.warning("Simplified body has no faces; body_with_hole will be empty if removal intended.")
             self.faces_to_remove_mask_on_body = np.array([], dtype=bool)
        else:
            self.faces_to_remove_mask_on_body = np.zeros(len(self.simplified_body_trimesh.faces), dtype=bool)
        
        if self.original_smplx_face_geom_tri.vertices.shape[0] > 0 and \
           hasattr(self.original_smplx_face_geom_tri, 'vertex_normals') and \
           self.original_smplx_face_geom_tri.vertex_normals.shape == self.original_smplx_face_geom_tri.vertices.shape and \
           len(self.simplified_body_trimesh.faces) > 0: 
            try:
                _, d_cp, t_cp = trimesh.proximity.closest_point(self.simplified_body_trimesh, self.original_smplx_face_geom_tri.vertices)
                if d_cp is not None and t_cp is not None:
                    self.faces_to_remove_mask_on_body[np.unique(t_cp[d_cp < self.config.projection_footprint_threshold])] = True
                offset = self.config.projection_footprint_threshold * 0.5
                p_f = self.original_smplx_face_geom_tri.vertices + self.original_smplx_face_geom_tri.vertex_normals * offset
                p_b = self.original_smplx_face_geom_tri.vertices - self.original_smplx_face_geom_tri.vertex_normals * offset
                _, _, t_f = trimesh.proximity.closest_point(self.simplified_body_trimesh, p_f)
                if t_f is not None: self.faces_to_remove_mask_on_body[np.unique(t_f)] = True
                _, dists_b, t_b = trimesh.proximity.closest_point(self.simplified_body_trimesh, p_b)
                if dists_b is not None and t_b is not None:
                    self.faces_to_remove_mask_on_body[np.unique(t_b[dists_b < self.config.projection_footprint_threshold])] = True
            except Exception as e_p: logger.warning(f"Error during robust hole determination: {e_p}")

            if self.config.footprint_dilation_rings > 0 and np.any(self.faces_to_remove_mask_on_body):
                adj = self.simplified_body_trimesh.face_adjacency
                current_wavefront = self.faces_to_remove_mask_on_body.copy()
                for _ in range(self.config.footprint_dilation_rings):
                    wave_indices = np.where(current_wavefront)[0]
                    if not wave_indices.size: break
                    all_neigh = [p[0] if p[1] in wave_indices else p[1] for p in adj if p[0] in wave_indices or p[1] in wave_indices]
                    if not all_neigh: break
                    unique_neigh = np.unique(all_neigh)
                    new_to_add = unique_neigh[~self.faces_to_remove_mask_on_body[unique_neigh]]
                    if not new_to_add.size: break
                    self.faces_to_remove_mask_on_body[new_to_add] = True
                    current_wavefront.fill(False); current_wavefront[new_to_add] = True

        body_state_before_hole_cut = self.simplified_body_trimesh.copy() 
        current_body_for_hole = self.simplified_body_trimesh.copy() 
        if np.any(self.faces_to_remove_mask_on_body):
            if len(self.faces_to_remove_mask_on_body) == len(current_body_for_hole.faces): 
                if np.all(self.faces_to_remove_mask_on_body): 
                    logger.warning("All body faces marked for removal. Body_with_hole will be empty.")
                    current_body_for_hole = trimesh.Trimesh() 
                else:
                    current_body_for_hole.update_faces(~self.faces_to_remove_mask_on_body)
                    current_body_for_hole.remove_unreferenced_vertices()
            else: logger.error("Face removal mask length mismatch. Skipping removal.")
        
        self.body_with_hole_trimesh = current_body_for_hole 
        if self.body_with_hole_trimesh is not None and not self.body_with_hole_trimesh.is_empty and \
           MeshCleanProcess._is_mesh_valid_for_concat(self.body_with_hole_trimesh, "BodyWHolePreSplit"):
            if hasattr(self.body_with_hole_trimesh, 'split'):
                components = self.body_with_hole_trimesh.split(only_watertight=False)
                if components:
                    largest_comp = max(components, key=lambda c: len(c.faces if hasattr(c, 'faces') else []))
                    if MeshCleanProcess._is_mesh_valid_for_concat(largest_comp, "LargestCompBodyWHole"):
                        self.body_with_hole_trimesh = largest_comp 
                    else: logger.warning("Largest component of body-with-hole invalid. Using pre-split.")
        elif self.body_with_hole_trimesh is None: logger.error("body_with_hole_trimesh is None. Unexpected."); return False

        body_state_before_final_clean = self.body_with_hole_trimesh.copy() if self.body_with_hole_trimesh is not None else None
        cleaned_body_with_hole = FaceGraftingPipeline._thoroughly_clean_trimesh_object(
            body_state_before_final_clean, "BodyWithHoleClean", self.INTERNAL_VERSION_TRACKER
        )
        if cleaned_body_with_hole is not None: self.body_with_hole_trimesh = cleaned_body_with_hole
        else: 
            logger.warning("Thorough cleaning of body-with-hole failed. Using pre-clean version or fallback.")
            if body_state_before_final_clean is not None and MeshCleanProcess._is_mesh_valid_for_concat(body_state_before_final_clean, "BodyWHoleFallback1"):
                self.body_with_hole_trimesh = body_state_before_final_clean
            else: 
                self.body_with_hole_trimesh = FaceGraftingPipeline._thoroughly_clean_trimesh_object(
                    body_state_before_hole_cut.copy(), "BodyWHoleFallbackSimplified", self.INTERNAL_VERSION_TRACKER)

        if self.body_with_hole_trimesh is not None and self.config.hole_boundary_smoothing_iterations > 0:
            if not self._smooth_main_hole_boundary(): logger.warning("Hole boundary smoothing failed or skipped.")
            else:
                if not MeshCleanProcess._is_mesh_valid_for_concat(self.body_with_hole_trimesh, "BodyWHolePostSmooth"):
                    logger.warning("Body-with-hole invalid after smoothing. Attempting re-clean.")
                    cleaned_after_smooth = FaceGraftingPipeline._thoroughly_clean_trimesh_object(
                        self.body_with_hole_trimesh.copy(), "BodyWHolePostSmoothReClean", self.INTERNAL_VERSION_TRACKER
                    )
                    if cleaned_after_smooth is not None: self.body_with_hole_trimesh = cleaned_after_smooth
                    else: logger.warning("Re-cleaning after smoothing failed. Proceeding with potentially problematic mesh.")

        if not MeshCleanProcess._is_mesh_valid_for_concat(self.body_with_hole_trimesh, "FinalBodyWHole"):
             logger.critical("Final Body-with-hole mesh is invalid. Aborting.")
             return False
        
        # Removed DEBUG save of body_with_hole_trimesh
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

        # logger.info("Aligning body loop to SMPLX face loop.")
        b_loop_coords_to_align = b_loop_coords_ordered_pre_align.copy()
        s_start_pt = self.s_loop_coords_ordered[0]
        _, closest_idx_on_b = cKDTree(b_loop_coords_to_align).query(s_start_pt, k=1)
        b_rolled = np.roll(b_loop_coords_to_align, -closest_idx_on_b, axis=0)
        
        resample_count = self.config.alignment_resample_count
        if len(self.s_loop_coords_ordered) >= 2 and len(b_rolled) >= 2 and resample_count >= 2:
            s_r = MeshCleanProcess.resample_polyline_to_count(self.s_loop_coords_ordered, resample_count)
            b_rf = MeshCleanProcess.resample_polyline_to_count(b_rolled, resample_count) 
            b_rb = MeshCleanProcess.resample_polyline_to_count(b_rolled[::-1], resample_count) 
            if s_r is not None and b_rf is not None and b_rb is not None and \
               len(s_r) == resample_count and len(b_rf) == resample_count and len(b_rb) == resample_count:
                if np.sum(np.linalg.norm(s_r - b_rb, axis=1)) < np.sum(np.linalg.norm(s_r - b_rf, axis=1)):
                    self.b_loop_coords_aligned = b_rolled[::-1].copy()
                else: self.b_loop_coords_aligned = b_rolled.copy()
            else:
                logger.warning("Resampling for alignment failed; using rolled loop.")
                self.b_loop_coords_aligned = b_rolled.copy()
        else: self.b_loop_coords_aligned = b_rolled.copy()
        return self.b_loop_coords_aligned is not None and len(self.b_loop_coords_aligned) >=3

    def _create_loft_stitch_mesh(self) -> Optional[trimesh.Trimesh]:
        if (self.s_loop_coords_ordered is None or len(self.s_loop_coords_ordered) < 3 or
                self.b_loop_coords_aligned   is None or len(self.b_loop_coords_aligned)   < 3):
            logger.warning("Loft strip: missing or too-short loops for creation.")
            return None
        b_loop, s_loop = self.b_loop_coords_aligned, self.s_loop_coords_ordered
        Nb, Ns = len(b_loop), len(s_loop)
        nearest_s_idx = cKDTree(s_loop).query(b_loop, k=1)[1]
        strip_verts = np.vstack((b_loop, s_loop))
        faces = []
        for i in range(Nb):
            i_next = (i + 1) % Nb
            v0, v1 = i, i_next
            s_curr, s_next = nearest_s_idx[i] + Nb, nearest_s_idx[i_next]  + Nb
            faces.extend([[v0, v1, s_next], [v0, s_next, s_curr]])
        strip = trimesh.Trimesh(vertices=strip_verts, faces=np.asarray(faces, dtype=int), process=True)
        if not MeshCleanProcess._is_mesh_valid_for_concat(strip, "LoftStripExactFace"):
            logger.warning("Loft strip failed validation.")
            return None
        if strip.face_normals is not None and len(strip.face_normals): # Basic orientation check
            if (np.dot(strip.face_normals[0], strip.triangles_center[0] - strip.centroid) > 0):
                strip.invert()
        # logger.info("Loft stitch mesh (exact face verts) built.")
        return strip

    def _extract_main_boundary_loop_from_body_with_hole(self) -> Optional[np.ndarray]:
        if not self.body_with_hole_trimesh or \
           not MeshCleanProcess._is_mesh_valid_for_concat(self.body_with_hole_trimesh, "BodyForLoopExtract"):
            logger.warning("EXTRACT_MAIN_BWH_LOOP: body_with_hole_trimesh is invalid or None.")
            return None
        if self.s_loop_coords_ordered is None or len(self.s_loop_coords_ordered) < 3:
            logger.warning("EXTRACT_MAIN_BWH_LOOP: s_loop (SMPLX face) missing. Fallback to largest loop on body_with_hole.")
            all_loops_vidx = MeshCleanProcess.get_all_boundary_loops(self.body_with_hole_trimesh, min_loop_len=3)
            if not all_loops_vidx: return None
            largest_loop_vidx = max(all_loops_vidx, key=len, default=None)
            if largest_loop_vidx is not None and len(largest_loop_vidx) >=3 and largest_loop_vidx.max() < len(self.body_with_hole_trimesh.vertices):
                return self.body_with_hole_trimesh.vertices[largest_loop_vidx]
            return None

        all_loops_vidx = MeshCleanProcess.get_all_boundary_loops(self.body_with_hole_trimesh, min_loop_len=3)
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
        logger.info("=== STEP 4 & 5: Attempting 'BODY_DRIVEN_LOFT' ===")
        if not self._extract_smplx_face_loop(): logger.warning("Loft: SMPLX face loop extraction failed."); return False
        b_loop_pre_align = self._extract_main_boundary_loop_from_body_with_hole()
        if b_loop_pre_align is None or len(b_loop_pre_align) < 3: logger.warning("Loft: Body hole loop extraction failed."); return False
        if not self._align_body_loop_to_smplx_loop(b_loop_pre_align): logger.warning("Loft: Loop alignment failed."); return False
        strip_mesh_obj = self._create_loft_stitch_mesh()
        if not strip_mesh_obj or not MeshCleanProcess._is_mesh_valid_for_concat(strip_mesh_obj, "StitchStrip"):
            logger.warning("Loft: Stitch mesh creation failed or invalid."); return False

        face_concat, body_concat = self.original_smplx_face_geom_tri, self.body_with_hole_trimesh
        if self.config.vertex_coordinate_precision_digits is not None:
            digits = self.config.vertex_coordinate_precision_digits
            if face_concat is not None:
                temp_f = face_concat.copy(); temp_f.vertices = np.round(temp_f.vertices, digits)
                if MeshCleanProcess._is_mesh_valid_for_concat(temp_f, "RoundedFace"): face_concat = temp_f
                else: logger.warning("Rounding SMPLX face vertices made it invalid.")
            if body_concat is not None:
                temp_b = body_concat.copy(); temp_b.vertices = np.round(temp_b.vertices, digits)
                if MeshCleanProcess._is_mesh_valid_for_concat(temp_b, "RoundedBodyHole"): body_concat = temp_b
                else: logger.warning("Rounding body_with_hole vertices made it invalid.")
        
        if not face_concat or not body_concat: logger.error("Loft: SMPLX face or body_with_hole missing for concat."); return False
        valid_cs = [m for m in [face_concat, body_concat, strip_mesh_obj] if MeshCleanProcess._is_mesh_valid_for_concat(m, "LoftFinalComp")]
        
        if len(valid_cs) == 3:
            try:
                cand_m = trimesh.util.concatenate(valid_cs)
                trimesh.constants.tol.merge = 1e-5 # Fine-tune merge tolerance
                cand_m.merge_vertices(merge_tex=False, merge_norm=False)

                if not MeshCleanProcess._is_mesh_valid_for_concat(cand_m, "ConcatLoftRaw"):
                    logger.warning("Concatenated loft mesh (raw) is invalid."); self.stitched_mesh_intermediate = None; return False
                
                final_m_proc = None 
                temp_pml_in = self._make_temp_path("merge_pml_in", use_ply=True)
                temp_pml_out = self._make_temp_path("merge_pml_out", use_ply=True)
                ms_merge = None
                try:
                    if cand_m.export(temp_pml_in):
                        ms_merge = pymeshlab.MeshSet(); ms_merge.load_new_mesh(temp_pml_in)
                        if ms_merge.current_mesh_id() != -1 and ms_merge.current_mesh().vertex_number() > 0:
                            # logger.info(f"PML merge input: V={ms_merge.current_mesh().vertex_number()}, F={ms_merge.current_mesh().face_number()}")
                            ms_merge.meshing_merge_close_vertices(threshold=pymeshlab.AbsoluteValue(1e-5))
                            ms_merge.meshing_remove_unreferenced_vertices()
                            if ms_merge.current_mesh().vertex_number() > 0:
                                try: ms_merge.meshing_close_holes(maxholesize=100, newfaceselected=False)
                                except Exception: pass # Ignore PML close holes error
                            ms_merge.save_current_mesh(temp_pml_out)
                            reloaded_pml = trimesh.load_mesh(temp_pml_out, process=True)
                            if MeshCleanProcess._is_mesh_valid_for_concat(reloaded_pml, "LoadedPMLMerge"):
                                reloaded_pml.fill_holes() # Trimesh fill holes
                                if MeshCleanProcess._is_mesh_valid_for_concat(reloaded_pml, "PMLPlusTrimeshFill"):
                                    final_m_proc = reloaded_pml.copy()
                                    # logger.info(f"PML merge path succeeded. V={len(final_m_proc.vertices)}, F={len(final_m_proc.faces)}")
                except Exception as e_pml_merge_block: logger.warning(f"PML merge block exception: {e_pml_merge_block}")
                finally:
                    if ms_merge is not None: del ms_merge
                
                if final_m_proc is None: 
                    logger.info("PML merge path failed or skipped. Using Trimesh merge on concatenated mesh.")
                    final_m_proc = cand_m.copy() 
                    final_m_proc.merge_vertices(merge_tex=False, merge_norm=False) # Redundant if already merged above, but safe.
                        
                final_m_proc.remove_unreferenced_vertices(); final_m_proc.remove_degenerate_faces()

                if MeshCleanProcess._is_mesh_valid_for_concat(final_m_proc, "ProcessedLoft"):
                    self.stitched_mesh_intermediate = final_m_proc
                    if self.stitched_mesh_intermediate is not None and MeshCleanProcess._is_mesh_valid_for_concat(self.stitched_mesh_intermediate, "IntermediateForSeam"):
                        self._verify_seam_closure_on_mesh(self.stitched_mesh_intermediate, "IntermediateStitchedMesh")
                else:
                    logger.warning("Processed lofted mesh invalid. Setting stitched_mesh_intermediate to initial concat if valid.")
                    self.stitched_mesh_intermediate = cand_m if MeshCleanProcess._is_mesh_valid_for_concat(cand_m, "CandMFallback") else None
                
                if self.stitched_mesh_intermediate is not None and not self.stitched_mesh_intermediate.is_empty:
                    logger.info(f"Stitch method '{self.config.stitch_method}' successful.")
                    return True
                logger.warning("Lofted concatenation resulted in empty/None mesh after processing."); return False
            except Exception as e_f_cat:
                logger.error(f"Exception in final loft concat/merge: {e_f_cat}", exc_info=True)
                self.stitched_mesh_intermediate = None; return False
        else:
            logger.warning(f"Not all 3 components valid for loft concat. Valid: {len(valid_cs)}"); return False

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
        
    def _perform_simple_concatenation(self) -> bool:
        logger.info("=== STEP 5 (Fallback): Performing simple concatenation ===")
        if not self.original_smplx_face_geom_tri or not self.body_with_hole_trimesh:
             logger.error("Simple concat: one or both base meshes missing."); return False
        valid_comps = [m for m in [self.original_smplx_face_geom_tri, self.body_with_hole_trimesh] 
                          if MeshCleanProcess._is_mesh_valid_for_concat(m, "FallbackComp")]
        if not valid_comps: logger.critical("No valid components for simple concat."); return False
        try:
            self.stitched_mesh_intermediate = trimesh.util.concatenate(valid_comps)
            self.stitched_mesh_intermediate.merge_vertices(merge_tex=False, merge_norm=False) # Important for simple concat too
            if not MeshCleanProcess._is_mesh_valid_for_concat(self.stitched_mesh_intermediate, "FallbackConcatResult"):
                logger.critical("Simple concatenation resulted in invalid mesh."); return False
            self.stitched_mesh_intermediate.fix_normals() 
            logger.info("Simple concatenation successful.")
            return True
        except Exception as e_concat:
            logger.error(f"Exception during simple concatenation: {e_concat}", exc_info=True)
            return False

    def _stitch_components(self) -> bool:
        if self.config.stitch_method == "body_driven_loft":
            if self._attempt_body_driven_loft(): return True
            logger.info("Body_driven_loft failed or not applicable. Falling back to simple concatenation.")
            return self._perform_simple_concatenation()
        # logger.info(f"Stitch method is '{self.config.stitch_method}'. Proceeding with simple concatenation.")
        return self._perform_simple_concatenation()

    def _fill_seam_holes_ear_clip(self) -> bool:
        from scipy.spatial import cKDTree 
        if not self.stitched_mesh_intermediate or \
           not MeshCleanProcess._is_mesh_valid_for_concat(self.stitched_mesh_intermediate, "MeshBeforeSeamFill"):
            logger.warning("Skipping seam hole fill: stitched mesh invalid/missing.")
            self.final_processed_mesh = self.stitched_mesh_intermediate; return True 
        if self.config.stitch_method not in ["trimesh_dynamic_loft", "body_driven_loft"]: # trimesh_dynamic_loft is not implemented here
            self.final_processed_mesh = self.stitched_mesh_intermediate; return True

        logger.info("Attempting EAR-CLIP SEAM HOLE FILL post-loft...")
        mesh_to_fill = self.stitched_mesh_intermediate.copy() 
        try: # Pre-computation
            if hasattr(mesh_to_fill, 'fix_normals'): mesh_to_fill.fix_normals(multibody=True)
            if hasattr(mesh_to_fill, 'edges_unique') and hasattr(mesh_to_fill, 'face_adjacency_edges') and hasattr(mesh_to_fill, 'edge_faces'):
                _ = mesh_to_fill.edges_unique; _ = mesh_to_fill.face_adjacency_edges; _ = mesh_to_fill.edge_faces
                mesh_to_fill._edges_unique_sorted_lookup = {tuple(sorted(edge)): idx for idx, edge in enumerate(mesh_to_fill.edges_unique)}
        except Exception: logger.warning("Could not fully pre-cache graph properties for ear-clip fill.")

        orig_verts = mesh_to_fill.vertices.copy()
        curr_faces_list = list(mesh_to_fill.faces)
        added_fill_faces = False 
        s_loop_ref, b_loop_ref = self.s_loop_coords_ordered, self.b_loop_coords_aligned
        all_loops = MeshCleanProcess.get_all_boundary_loops(mesh_to_fill, min_loop_len=3)
        kdt_s, kdt_b = (cKDTree(s_loop_ref) if s_loop_ref is not None else None), (cKDTree(b_loop_ref) if b_loop_ref is not None else None)
        prox_thresh = 0.025 
        z_thresh_limb = None
        if self.simplified_body_trimesh and MeshCleanProcess._is_mesh_valid_for_concat(self.simplified_body_trimesh, "BodyLimbZ"):
            b_min, b_max = self.simplified_body_trimesh.bounds; height = b_max[2] - b_min[2]
            if height > 1e-6: z_thresh_limb = b_min[2] + height * 0.30 

        for loop_vidx in all_loops:
            if not (3 <= len(loop_vidx) <= self.config.max_seam_hole_fill_vertices and loop_vidx.max() < len(orig_verts)): continue
            loop_coords = orig_verts[loop_vidx]
            if z_thresh_limb is not None and np.mean(loop_coords[:, 2]) < z_thresh_limb: continue
            is_seam = False 
            if kdt_s and kdt_b:
                d_s, _ = kdt_s.query(loop_coords, k=1, distance_upper_bound=prox_thresh*2)
                d_b, _ = kdt_b.query(loop_coords, k=1, distance_upper_bound=prox_thresh*2)
                if np.mean(d_s[np.isfinite(d_s)]) < prox_thresh and np.mean(d_b[np.isfinite(d_b)]) < prox_thresh: is_seam = True
            
            if is_seam:
                new_faces_cand = None
                try:
                    if len(loop_coords) < 3: continue
                    plane_orig, normal = MeshCleanProcess.get_dominant_plane(loop_coords)
                    if np.allclose(normal, [0,0,0]): continue
                    xform_2d = trimesh.geometry.plane_transform(plane_orig, normal)
                    loop_2d = trimesh.transform_points(loop_coords, xform_2d)[:, :2]
                    if len(np.unique(loop_2d, axis=0)) < 3: continue
                    from shapely.geometry import Polygon
                    patch_v_2d, patch_f_loc = None, None; tri_ok = False
                    try:
                        poly2d = Polygon(loop_2d) 
                        patch_v_2d, patch_f_loc = trimesh.creation.triangulate_polygon(poly2d, triangle_args="p")
                        tri_ok = (patch_f_loc is not None and len(patch_f_loc) > 0)
                    except Exception: # Catch all for PSLG, then try ear-cut
                        path_2d_obj = trimesh.path.Path2D(entities=[trimesh.path.entities.Line(np.arange(len(loop_2d)))], vertices=loop_2d)
                        if hasattr(path_2d_obj, 'polygons_full') and path_2d_obj.polygons_full:
                           poly_earcut = path_2d_obj.polygons_full[0] 
                           patch_v_2d, patch_f_loc = trimesh.creation.triangulate_polygon(poly_earcut)
                           tri_ok = (patch_f_loc is not None and len(patch_f_loc) > 0)
                    if not tri_ok: continue
                    idx_map = cKDTree(loop_2d).query(patch_v_2d, k=1)[1]
                    new_faces_cand = loop_vidx[idx_map[patch_f_loc]]

                    if new_faces_cand is not None and len(new_faces_cand) > 0:
                        patch_mesh = trimesh.Trimesh(vertices=orig_verts, faces=new_faces_cand, process=False); patch_mesh.fix_normals()
                        if len(patch_mesh.faces) > 0 and hasattr(patch_mesh, 'face_normals') and patch_mesh.face_normals is not None:
                            avg_patch_n = trimesh.util.unitize(np.mean(patch_mesh.face_normals, axis=0))
                            adj_normals = []
                            if hasattr(mesh_to_fill, '_edges_unique_sorted_lookup'):
                                for i in range(len(loop_vidx)):
                                    e_tuple = tuple(sorted((loop_vidx[i], loop_vidx[(i+1)%len(loop_vidx)])))
                                    e_idx = mesh_to_fill._edges_unique_sorted_lookup.get(e_tuple)
                                    if e_idx is not None and hasattr(mesh_to_fill, 'edge_faces') and hasattr(mesh_to_fill, 'face_normals') and \
                                       e_idx < len(mesh_to_fill.edge_faces):
                                        for f_idx in mesh_to_fill.edge_faces[e_idx]:
                                            if f_idx != -1 and f_idx < len(mesh_to_fill.face_normals): adj_normals.append(mesh_to_fill.face_normals[f_idx])
                            if adj_normals:
                                target_n = trimesh.util.unitize(np.mean(np.array(adj_normals), axis=0))
                                if np.dot(avg_patch_n, target_n) < 0.1: 
                                    new_faces_cand = new_faces_cand[:, ::-1] # Flip
                                    logger.info(f"Loop {loop_idx}: Flipped patch orientation for seam fill.")
                    if new_faces_cand is not None: curr_faces_list.extend(new_faces_cand); added_fill_faces = True
                except Exception as e_outer: logger.warning(f"Error ear-clipping loop {loop_idx}: {e_outer}", exc_info=False) # exc_info False for brevity
        
        if added_fill_faces:
            updated_mesh = trimesh.Trimesh(vertices=orig_verts, faces=np.array(curr_faces_list,dtype=int), process=True) 
            if MeshCleanProcess._is_mesh_valid_for_concat(updated_mesh, "MeshAfterEarClip"):
                self.final_processed_mesh = updated_mesh; logger.info("Ear-clip seam hole filling applied.")
            else:
                logger.warning("Mesh invalid after ear-clip. Reverting."); self.final_processed_mesh = self.stitched_mesh_intermediate 
        else: self.final_processed_mesh = self.stitched_mesh_intermediate 
        return True
        
    def _apply_final_polish(self) -> bool:
        if not self.final_processed_mesh or \
           not MeshCleanProcess._is_mesh_valid_for_concat(self.final_processed_mesh, "MeshBeforeFinalPolish"):
            logger.warning("Skipping final polish: mesh invalid/missing."); return True 

        logger.info("=== STEP 5.5: Applying Final Polish ===")
        temp_in = self._make_temp_path("polish_in", use_ply=True)
        temp_out = self._make_temp_path("polish_out", use_ply=True)
        polished_loaded = None; ms_polish = None
        try:
            self.final_processed_mesh.export(temp_in)
            ms_polish = pymeshlab.MeshSet(); ms_polish.load_new_mesh(temp_in)
            if ms_polish.current_mesh_id() != -1 and ms_polish.current_mesh().vertex_number() > 0:
                ms_polish.meshing_remove_duplicate_vertices()
                try: ms_polish.meshing_repair_non_manifold_edges(method='Split Vertices')
                except Exception: 
                    try: ms_polish.meshing_repair_non_manifold_edges(method='Remove Faces')
                    except Exception: pass
                is_mani = False
                try:
                    if ms_polish.get_topological_measures().get('non_manifold_edges', -1) == 0: is_mani = True
                except Exception: pass
                if is_mani:
                    try: ms_polish.meshing_close_holes(maxholesize=self.config.final_polish_max_hole_edges)
                    except Exception as e_ch: logger.info(f"Polish: PML close_holes failed: {e_ch}")
                ms_polish.meshing_remove_unreferenced_vertices(); ms_polish.compute_normal_per_face() 
                ms_polish.save_current_mesh(temp_out)
                polished_loaded = trimesh.load_mesh(temp_out, process=True)
            
            if MeshCleanProcess._is_mesh_valid_for_concat(polished_loaded, "PolishedMeshPML"):
                self.final_processed_mesh = polished_loaded; self.final_processed_mesh.fix_normals() 
                logger.info("Final polish applied.")
            else: logger.warning("Final polish resulted in invalid mesh. Keeping pre-polish.")
        except Exception as e_polish: logger.warning(f"Error in final polish: {e_polish}", exc_info=True)
        finally:
            if ms_polish is not None: del ms_polish
        return True

    def _filter_spider_triangles(self) -> bool:
        if not self.final_processed_mesh or \
           not MeshCleanProcess._is_mesh_valid_for_concat(self.final_processed_mesh, "MeshBeforeSpiderFilter"):
            logger.warning("Skipping spider filter: mesh invalid/missing."); return True 
        if self.config.spider_filter_area_factor is None and self.config.spider_filter_max_edge_len_factor is None:
            return True
        logger.info("=== STEP 5.75: Filtering Spider-Web Triangles ===")
        target_faces = None
        if MeshCleanProcess._is_mesh_valid_for_concat(self.original_smplx_face_geom_tri, "SpiderRefBounds") and \
           self.final_processed_mesh.faces.shape[0] > 0 and self.original_smplx_face_geom_tri.vertices.shape[0] > 0: 
            centroids = self.final_processed_mesh.triangles_center
            s_min, s_max = self.original_smplx_face_geom_tri.bounds; pad = 0.05 
            min_b, max_b = s_min - pad, s_max + pad
            ids = [i for i, c in enumerate(centroids) if (min_b[0] <= c[0] <= max_b[0] and min_b[1] <= c[1] <= max_b[1] and min_b[2] <= c[2] <= max_b[2])]
            if ids: target_faces = np.array(ids, dtype=int)
        max_edge_len = None
        if self.config.spider_filter_max_edge_len_factor is not None and self.config.spider_filter_max_edge_len_factor > 0:
            if self.simplified_body_trimesh and MeshCleanProcess._is_mesh_valid_for_concat(self.simplified_body_trimesh, "SpiderEdgeRef") and \
               len(self.simplified_body_trimesh.edges_unique_length) > 0:
                max_edge_len = np.max(self.simplified_body_trimesh.edges_unique_length) * (1.0 + self.config.spider_filter_max_edge_len_factor)
        ref_stats = self.original_smplx_face_geom_tri if MeshCleanProcess._is_mesh_valid_for_concat(self.original_smplx_face_geom_tri, "SpiderAreaRef") else None
        filtered = MeshCleanProcess._filter_large_triangles_from_fill(
            self.final_processed_mesh, target_face_indices=target_faces, max_allowed_edge_length=max_edge_len,
            max_allowed_area_factor=self.config.spider_filter_area_factor, reference_mesh_for_stats=ref_stats
        )
        if not MeshCleanProcess._is_mesh_valid_for_concat(filtered, "MeshAfterSpiderFilter"):
            logger.critical("Mesh invalid after spider filter. Aborting save."); return False 
        self.final_processed_mesh = filtered; self.final_processed_mesh.fix_normals()
        # logger.info("Spider-web triangle filter applied.")
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
        gap_thresh = getattr(self.config, 'seam_gap_verification_threshold', 1e-4) 
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

    def process(self) -> Optional[trimesh.Trimesh]:
        logger.info(f"--- Starting Face Grafting Pipeline (V{self.INTERNAL_VERSION_TRACKER}) ---")
        pipeline_ok = False
        try:
            if not self._load_and_simplify_meshes(): return None
            if not self._perform_iterative_body_repair(): return None
            if not self._determine_hole_faces_and_create_body_with_hole(): return None
            if not self._stitch_components(): logger.error("Component stitching failed."); return None
            self._fill_seam_holes_ear_clip() # Updates self.final_processed_mesh
            self._apply_final_polish()      # Updates self.final_processed_mesh

            # _filter_spider_triangles() is commented out by user
            # if not self._filter_spider_triangles(): logger.error("Spider filtering failed critically."); return None

            if not MeshCleanProcess._is_mesh_valid_for_concat(self.final_processed_mesh, "MeshBeforeFinalSave"):
                logger.critical("Final mesh is invalid or empty before save. CANNOT SAVE."); return None
            self.final_processed_mesh.export(self.output_path)
            logger.info(f"--- Pipeline Finished. Output: {self.output_path} ---")
            self._verify_seam_closure()
            logger.info("--- Performing Final Watertightness Check ---")
            if self.final_processed_mesh and not self.final_processed_mesh.is_empty:
                temp_final_check = self._make_temp_path("final_watertight_check", use_ply=True)
                if self.final_processed_mesh.export(temp_final_check):
                    checker = MeshCleanProcess(temp_final_check, "dummy.ply")
                    if checker.load_mesh(): logger.info(f"Final Watertightness: {checker.check_watertight()}")
                    else: logger.warning("Could not load final for watertight check.")
                    del checker
            pipeline_ok = True
            return self.final_processed_mesh
        except Exception as e: 
            logger.error(f"--- Pipeline V{self.INTERNAL_VERSION_TRACKER} Failed: {e}", exc_info=True); return None
        finally:
            self._cleanup_temp_files()
            if not pipeline_ok: logger.error(f"Pipeline V{self.INTERNAL_VERSION_TRACKER} did not complete successfully.")

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
            cleaned.remove_degenerate_faces()
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

    def check_watertight(self):
        if self.ms.current_mesh().vertex_number() == 0: return False 
        try:
            metrics = self.ms.get_topological_measures()
            is_wt = (metrics.get('boundary_edges',-1)==0 and metrics.get('number_holes',-1)==0 and metrics.get('is_mesh_two_manifold',False))
            if not is_wt: logger.info(f"Watertight Check (PML): BE={metrics.get('boundary_edges')}, Holes={metrics.get('number_holes')}, IsTwoManifold={metrics.get('is_mesh_two_manifold')}")
            return is_wt
        except pymeshlab.PyMeshLabException as e: logger.error(f"Error checking watertight (PML): {e}"); return False

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
    def get_all_boundary_loops(mesh: trimesh.Trimesh, min_loop_len: int = 3) -> List[np.ndarray]:
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
    def _order_loop_vertices_from_edges(mesh_name_for_debug: str, loop_vidx_unique: np.ndarray, all_edges_loop: np.ndarray, 
                                        min_comp_ok: float = 0.75, min_path_len_ok: int = 10) -> Optional[np.ndarray]:
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
        if unique_in_path >= expected_unique*min_comp_ok and len(path) >= min_path_len_ok: return np.array(path,dtype=int) # Partial ok
        return None

    @staticmethod
    def resample_polyline_to_count(polyline: np.ndarray, count: int) -> Optional[np.ndarray]:
        if polyline is None or polyline.ndim!=2 or polyline.shape[1]!=3: return None
        if len(polyline)<2: return polyline
        if count < 2: return polyline[:count] if count > 0 else np.array([],dtype=polyline.dtype).reshape(0,3)
        if len(polyline)==count: return polyline
        dists = np.linalg.norm(np.diff(polyline,axis=0),axis=1)
        cum_dists = np.concatenate(([0], np.cumsum(dists))); total_len = cum_dists[-1]
        if total_len < 1e-9: return np.tile(polyline[0],(count,1)) if count > 0 and len(polyline)>0 else np.array([],dtype=polyline.dtype).reshape(0,3)
        sampled_lens = np.linspace(0,total_len,count); resampled = []
        for s_len in sampled_lens:
            idx = np.searchsorted(cum_dists, s_len, side='right')-1; idx=np.clip(idx,0,len(polyline)-2)
            p0,p1=polyline[idx],polyline[idx+1]; seg_len=cum_dists[idx+1]-cum_dists[idx]
            t = (s_len-cum_dists[idx])/seg_len if seg_len > 1e-9 else 0.0
            resampled.append(p0+np.clip(t,0,1)*(p1-p0))
        return np.array(resampled) if resampled else np.array([],dtype=polyline.dtype).reshape(0,3)

    @staticmethod
    def get_dominant_plane(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if points is None or points.ndim != 2 or points.shape[1]!=3 or len(points)<3:
            return np.array([0.,0.,1.]), (np.mean(points,axis=0) if (points is not None and points.ndim==2 and points.shape[1]==3 and len(points)>0) else np.array([0.,0.,0.]))
        center = np.mean(points,axis=0); centered = points-center
        try: _,_,vh = np.linalg.svd(centered,full_matrices=False); return trimesh.util.unitize(vh[-1,:]), center
        except Exception: return np.array([0.,0.,1.]), center

    @staticmethod
    def _get_hole_boundary_edges_from_removed_faces(orig_mesh: trimesh.Trimesh, faces_removed: np.ndarray) -> Optional[np.ndarray]:
        if not MeshCleanProcess._is_mesh_valid_for_concat(orig_mesh, "OrigMeshHoleBoundary") or \
           faces_removed is None or faces_removed.ndim!=1 or len(faces_removed)!=len(orig_mesh.faces): return None
        if not np.any(faces_removed) or np.all(faces_removed): return np.array([],dtype=int).reshape(-1,2)
        try:
            adj_pairs, adj_edges = orig_mesh.face_adjacency, orig_mesh.face_adjacency_edges
            hole_edges = [adj_edges[i] for i in range(len(adj_pairs)) if faces_removed[adj_pairs[i][0]] != faces_removed[adj_pairs[i][1]]]
            return np.array(hole_edges,dtype=int) if hole_edges else np.array([],dtype=int).reshape(-1,2)
        except Exception as e: logger.error(f"Error in _get_hole_boundary_edges: {e}", exc_info=True); return None

    @staticmethod
    def _filter_large_triangles_from_fill(
        mesh: trimesh.Trimesh, target_face_indices: Optional[np.ndarray] = None, 
        max_allowed_edge_length: Optional[float] = None, max_allowed_area_factor: Optional[float] = None,
        reference_mesh_for_stats: Optional[trimesh.Trimesh] = None, mesh_name_for_debug: str = "large_tri" # name not used if logs removed
    ) -> trimesh.Trimesh:
        if not MeshCleanProcess._is_mesh_valid_for_concat(mesh, f"InputLargeTriFilter"): return mesh
        if max_allowed_edge_length is None and max_allowed_area_factor is None: return mesh 
        keep_mask = np.ones(len(mesh.faces),dtype=bool)
        cand_mask = np.zeros(len(mesh.faces),dtype=bool)
        if target_face_indices is not None and len(target_face_indices)>0:
            valid_targets = target_face_indices[target_face_indices < len(mesh.faces)]
            cand_mask[valid_targets]=True
        else: cand_mask.fill(True)
        remove_criteria_mask = np.zeros(len(mesh.faces),dtype=bool)
        if max_allowed_edge_length is not None and max_allowed_edge_length > 0 and np.any(cand_mask): 
            face_edges_len = np.zeros((len(mesh.faces),3))
            for i,f_verts in enumerate(mesh.faces):
                if cand_mask[i]: 
                    if f_verts.max()<len(mesh.vertices): v0,v1,v2=mesh.vertices[f_verts]; face_edges_len[i,:]=[np.linalg.norm(v0-v1),np.linalg.norm(v1-v2),np.linalg.norm(v2-v0)]
                    else: face_edges_len[i,:]=np.inf 
            remove_criteria_mask[(np.max(face_edges_len,axis=1) > max_allowed_edge_length) & cand_mask] = True
        if max_allowed_area_factor is not None and max_allowed_area_factor > 0 and np.any(cand_mask): 
            areas = mesh.area_faces; median_area = 0
            if reference_mesh_for_stats and MeshCleanProcess._is_mesh_valid_for_concat(reference_mesh_for_stats,"RefStatsArea") and len(reference_mesh_for_stats.faces)>0:
                median_area = np.median(reference_mesh_for_stats.area_faces)
            elif len(areas[cand_mask & ~remove_criteria_mask])>0: median_area = np.median(areas[cand_mask & ~remove_criteria_mask])
            if median_area > 1e-9: remove_criteria_mask[(areas > median_area*max_allowed_area_factor) & cand_mask] = True 
        keep_mask[remove_criteria_mask]=False
        if np.sum(~keep_mask) > 0:
            if np.all(~keep_mask): logger.warning("All faces removed by large tri filter. Aborting filter."); return mesh 
            filtered = mesh.copy(); filtered.update_faces(keep_mask); filtered.remove_unreferenced_vertices()
            if MeshCleanProcess._is_mesh_valid_for_concat(filtered, f"FilteredMeshLargeTri"): return filtered
            logger.warning("Mesh invalid after large tri filter. Returning original."); return mesh 
        return mesh
    
    @staticmethod
    def run_face_grafting_pipeline(
        full_body_mesh_path: str, output_path: str, smplx_face_mesh_path: str,
        projection_footprint_threshold: float = 0.01, footprint_dilation_rings: int = 3,
        body_simplification_target_faces: int = 12000, stitch_method: str = "body_driven_loft", 
        smplx_face_neck_loop_strategy: str = "full_face_silhouette", alignment_resample_count: int = 1000,
        # loft_strip_resample_count removed as unused by current loft
        max_seam_hole_fill_vertices: int = 250, final_polish_max_hole_edges: int = 100,
        iterative_repair_s1_iters: int = 5, spider_filter_area_factor: Optional[float] = 300.0,
        spider_filter_max_edge_len_factor: Optional[float] = 0.20,
        hole_boundary_smoothing_iterations=10, hole_boundary_smoothing_factor=0.5
        # debug_dir removed
    ) -> Optional[trimesh.Trimesh]:
        config = FaceGraftingConfig(
            projection_footprint_threshold=projection_footprint_threshold,
            footprint_dilation_rings=footprint_dilation_rings,
            body_simplification_target_faces=body_simplification_target_faces,
            stitch_method=stitch_method,
            smplx_face_neck_loop_strategy=smplx_face_neck_loop_strategy,
            alignment_resample_count=alignment_resample_count,
            # loft_strip_resample_count=loft_strip_resample_count, # Unused
            max_seam_hole_fill_vertices=max_seam_hole_fill_vertices,
            final_polish_max_hole_edges=final_polish_max_hole_edges,
            iterative_repair_s1_iters=iterative_repair_s1_iters,
            spider_filter_area_factor=spider_filter_area_factor,
            spider_filter_max_edge_len_factor=spider_filter_max_edge_len_factor,
            vertex_coordinate_precision_digits=6, 
            hole_boundary_smoothing_iterations=hole_boundary_smoothing_iterations,
            hole_boundary_smoothing_factor=hole_boundary_smoothing_factor
        )
        pipeline = FaceGraftingPipeline(full_body_mesh_path,smplx_face_mesh_path,output_path,config)
        return pipeline.process()
        
def main():
    base_input_dir = "/home/ubuntu/projects/induxr/econ_s/ECON/results/Fulden/IFN+_face_thresh_0.01/econ/obj"
    body_mesh_filename = "fulden_tpose_f1_0_full_wt.obj" 
    face_mesh_filename = "fulden_tpose_f1_0_face.obj"
    full_body_mesh_path = os.path.join(base_input_dir, body_mesh_filename)
    smplx_face_mesh_path = os.path.join(base_input_dir, face_mesh_filename)
    output_dir = '/home/ubuntu/projects/induxr/econ_s/ECON/results/standalone_face_graft_test_results'
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"grafted_{os.path.splitext(body_mesh_filename)[0]}_with_{os.path.splitext(face_mesh_filename)[0]}.obj"
    output_path = os.path.join(output_dir, output_filename)

    script_logger = logging.getLogger("standalone_graft_test")
    if not logging.getLogger().hasHandlers(): # Ensure basicConfig is set if no handlers exist
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    script_logger.setLevel(logging.INFO) # Explicitly set level for this logger

    script_logger.info(f"--- Standalone Face Grafting Test ---")
    script_logger.info(f"Body Mesh: {full_body_mesh_path}")
    script_logger.info(f"Face Mesh: {smplx_face_mesh_path}")
    script_logger.info(f"Output: {output_path}")

    if not os.path.exists(full_body_mesh_path):
        script_logger.error(f"CRITICAL: Body mesh not found: '{full_body_mesh_path}'. Update filename/path."); return
    if not os.path.exists(smplx_face_mesh_path):
        script_logger.error(f"CRITICAL: Face mesh not found: '{smplx_face_mesh_path}'. Update filename/path."); return
    script_logger.info("Input files found. Starting pipeline...")
    try:
        result_mesh = MeshCleanProcess.run_face_grafting_pipeline(
            full_body_mesh_path=full_body_mesh_path, smplx_face_mesh_path=smplx_face_mesh_path, output_path=output_path,
            hole_boundary_smoothing_iterations=10, hole_boundary_smoothing_factor=0.5    
        )
    except Exception as e:
        script_logger.error(f"Unhandled pipeline exception: {e}", exc_info=True); result_mesh = None
    if result_mesh:
        script_logger.info(f"--- Pipeline successful. Result V={len(result_mesh.vertices)}, F={len(result_mesh.faces)}. Saved to: {output_path} ---")
    else:
        script_logger.error(f"--- Pipeline failed or no result. Output might not be saved to {output_path}. Check logs. ---")

if __name__ == "__main__":
    main()