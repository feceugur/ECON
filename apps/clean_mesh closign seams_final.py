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

# Setup basic logging for the module if not already configured
# In a real application, the calling code would configure logging.
# For this standalone refactor, this ensures logs are visible.
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # Debug level set by basicConfig or calling application


@dataclass
class FaceGraftingConfig:
    """Configuration for the FaceGraftingPipeline."""
    projection_footprint_threshold: float = 0.01
    footprint_dilation_rings: int = 10
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
    spider_filter_area_factor: Optional[float] = 300.0
    spider_filter_max_edge_len_factor: Optional[float] = 0.15
    vertex_coordinate_precision_digits: Optional[int] = 5 # Digits for rounding when merging, e.g., 6 for 1e-6 precision
    # debug_dir: Optional[str] = None # Removed debug_dir

class FaceGraftingPipeline:
    """
    Pipeline for grafting an SMPLX face mesh onto a full body mesh.
    Refactored from MeshCleanProcess.process_mesh_graft_smplx_face_v5.
    """
    INTERNAL_VERSION_TRACKER = "5.80_refactored_pipeline_with_seam_check" # Version updated

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
        self.ordered_s_vidx_loop: Optional[np.ndarray] = None # Vertex indices on original_smplx_face_geom_tri
        self.s_loop_coords_ordered: Optional[np.ndarray] = None # 3D coordinates of smplx face loop
        self.b_loop_coords_aligned: Optional[np.ndarray] = None # 3D coordinates of aligned body hole loop

        # if self.config.debug_dir: # Removed debug_dir logic
        #     os.makedirs(self.config.debug_dir, exist_ok=True)

    def _make_temp_path(self, suffix_label: str, use_ply: bool = False) -> str:
        """Helper to create and track temporary file paths."""
        actual_suffix = suffix_label if suffix_label.startswith('_') else '_' + suffix_label
        file_ext = ".ply" if use_ply else ".obj"
        # Ensure tempfile has a unique prefix related to the pipeline instance if needed, or rely on mkstemp uniqueness
        fd, path = tempfile.mkstemp(suffix=actual_suffix + file_ext, dir=None, prefix="graft_") # dir=None for default temp
        os.close(fd)
        self.temp_files_to_clean.append(path)
        return path

    def _cleanup_temp_files(self):
        """Removes all temporary files created during the process."""
        # logger.debug(f"Cleaning up {len(self.temp_files_to_clean)} temporary files.")
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
        """
        Applies a sequence of cleaning operations to a Trimesh object.
        Returns the cleaned mesh or None if input is None or becomes invalid.
        Operates on the passed mesh object (intended to be a copy).
        """
        if mesh is None: 
            logger.warning(f"Skipping thorough cleaning for {mesh_name_for_log}: input mesh is None.")
            return None
        
        if not MeshCleanProcess._is_mesh_valid_for_concat(mesh, f"{mesh_name_for_log}_PreClean_V{version_tracker}"):
            logger.warning(f"Skipping thorough cleaning for {mesh_name_for_log}: input mesh is initially invalid based on _is_mesh_valid_for_concat check.")
            return mesh 

        logger.info(f"Thoroughly cleaning Trimesh object: {mesh_name_for_log} (Initial V={len(mesh.vertices)}, F={len(mesh.faces)})")
        
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
            
            logger.info(f"Thorough cleaning for {mesh_name_for_log} completed successfully. Final V={len(mesh.vertices)}, F={len(mesh.faces)}")
            return mesh
        except Exception as e:
            logger.error(f"Exception during thorough cleaning of {mesh_name_for_log}: {e}", exc_info=True)
            return None

    def _load_and_simplify_meshes(self) -> bool:
        """STEP 1 & 2: Load meshes, clean face, simplify and clean body."""
        logger.info("=== STEP 1 & 2: Load meshes, clean face, simplify and clean body ===")
        
        self.original_smplx_face_geom_tri = trimesh.load_mesh(self.smplx_face_mesh_path, process=True)
        if self.original_smplx_face_geom_tri is None:
            logger.critical(f"Failed to load SMPLX Face mesh from '{self.smplx_face_mesh_path}'. Aborting.")
            return False
        if not MeshCleanProcess._is_mesh_valid_for_concat(self.original_smplx_face_geom_tri, f"InitialSMPLXFaceLoadCheck_V{self.INTERNAL_VERSION_TRACKER}"):
            logger.critical(f"Initial SMPLX Face mesh ('{self.smplx_face_mesh_path}') is invalid after loading. Aborting.")
            return False
        logger.info("Applying thorough cleaning to initial SMPLX face mesh...")
        smplx_face_copy_for_cleaning = self.original_smplx_face_geom_tri.copy()
        cleaned_smplx_face = FaceGraftingPipeline._thoroughly_clean_trimesh_object(
            smplx_face_copy_for_cleaning, "InitialSMPLXFace", self.INTERNAL_VERSION_TRACKER
        )
        if cleaned_smplx_face is not None:
            self.original_smplx_face_geom_tri = cleaned_smplx_face
            logger.info("Thorough cleaning of initial SMPLX face successful.")
        else:
            logger.warning("Thorough cleaning of initial SMPLX face failed. Using original loaded version.")
            if not MeshCleanProcess._is_mesh_valid_for_concat(self.original_smplx_face_geom_tri, f"FallbackUncleanedSMPLXFace_V{self.INTERNAL_VERSION_TRACKER}"):
                 logger.critical(f"Original SMPLX face mesh is also invalid after failed cleaning attempt. Aborting.")
                 return False

        self.simplified_body_trimesh = trimesh.load_mesh(self.full_body_mesh_path, process=True)
        if self.simplified_body_trimesh is None:
            logger.critical(f"Failed to load Full Body mesh from '{self.full_body_mesh_path}'. Aborting.")
            return False
        if not MeshCleanProcess._is_mesh_valid_for_concat(self.simplified_body_trimesh, f"InitialFullBodyLoadCheck_V{self.INTERNAL_VERSION_TRACKER}"):
            logger.critical(f"Initial Full Body mesh ('{self.full_body_mesh_path}') is invalid after loading. Aborting.")
            return False

        if self.original_smplx_face_geom_tri: self.original_smplx_face_geom_tri.fix_normals()
        if self.simplified_body_trimesh: self.simplified_body_trimesh.fix_normals()

        if self.simplified_body_trimesh.faces.shape[0] > self.config.body_simplification_target_faces:
            logger.info(f"Simplifying body mesh from {self.simplified_body_trimesh.faces.shape[0]} to target {self.config.body_simplification_target_faces} faces, focusing on manifold output.")
            body_state_before_pml_simplification = self.simplified_body_trimesh.copy() 
            
            # Assume _make_temp_path is defined in the full class
            temp_in = self._make_temp_path(f"b_v{self.INTERNAL_VERSION_TRACKER}_s_in", use_ply=True)
            temp_out = self._make_temp_path(f"b_v{self.INTERNAL_VERSION_TRACKER}_s_out_manifold", use_ply=True)
            loaded_s_from_pml = None
            ms_s = None
            try:
                if self.simplified_body_trimesh.export(temp_in): 
                    ms_s = pymeshlab.MeshSet()
                    ms_s.load_new_mesh(temp_in)
                    if ms_s.current_mesh_id() != -1 and ms_s.current_mesh().face_number() > 0:
                        logger.info("    PML: Applying Quadric Edge Collapse Decimation...")
                        ms_s.meshing_decimation_quadric_edge_collapse(
                            targetfacenum=self.config.body_simplification_target_faces, 
                            preservenormal=True, preservetopology=True, optimalplacement=True,
                            planarquadric=True, qualitythr=0.7 
                        )
                        logger.info(f"    PML: Decimation complete. V={ms_s.current_mesh().vertex_number()}, F={ms_s.current_mesh().face_number()}")

                        logger.info("    PML: Attempting to ensure/repair manifoldness post-decimation...")
                        try:
                            logger.info("    PML: Attempting non-manifold edge repair by removing faces.")
                            ms_s.meshing_repair_non_manifold_edges(method='Remove Faces')
                            logger.info(f"    PML: After repair non-manifold edges by removing faces. V={ms_s.current_mesh().vertex_number()}, F={ms_s.current_mesh().face_number()}")
                            
                            logger.info("    PML: Attempting non-manifold edge repair by splitting vertices.")
                            ms_s.meshing_repair_non_manifold_edges(method='Split Vertices') 
                            logger.info(f"    PML: After repair non-manifold edges by splitting vertices. V={ms_s.current_mesh().vertex_number()}, F={ms_s.current_mesh().face_number()}")

                            try:
                                logger.info("    PML: Attempting non-manifold vertex repair by splitting.")
                                ms_s.meshing_repair_non_manifold_vertices_by_splitting(threshold=pymeshlab.Percentage(0.01))
                                logger.info(f"    PML: After repair non-manifold vertices by splitting. V={ms_s.current_mesh().vertex_number()}, F={ms_s.current_mesh().face_number()}")
                            except AttributeError:
                                logger.warning("    PML: 'meshing_repair_non_manifold_vertices_by_splitting' not available or misnamed for this PyMeshLab version.")
                            except pymeshlab.PyMeshLabException as e_vsplit_pml:
                                logger.warning(f"    PML: Error during 'meshing_repair_non_manifold_vertices_by_splitting': {e_vsplit_pml}")

                            ms_s.meshing_remove_unreferenced_vertices()
                            logger.info(f"    PML: After remove unreferenced. V={ms_s.current_mesh().vertex_number()}, F={ms_s.current_mesh().face_number()}")

                            try:
                                measures = ms_s.get_topological_measures()
                                nm_edges = measures.get('non_manifold_edges', -1)
                                nm_verts = measures.get('non_manifold_vertices', -1) 
                                is_two_manifold_pml = measures.get('is_mesh_two_manifold', False)
                                logger.info(f"    PML: Post-repair topological measures: NM Edges={nm_edges}, NM Verts={nm_verts}, IsTwoManifold={is_two_manifold_pml}")
                                if not is_two_manifold_pml and nm_edges == 0: 
                                    logger.info("    PML: Mesh has no non-manifold edges but not strictly 2-manifold per PML.")
                                elif not is_two_manifold_pml:
                                     logger.warning("    PML: Mesh still reported as NOT strictly 2-manifold by PyMeshLab after repair attempts.")
                            except Exception as e_topo:
                                logger.warning(f"    PML: Could not get topological measures: {e_topo}")

                        except pymeshlab.PyMeshLabException as e_pml_manifold_repair:
                            logger.warning(f"    PML: PyMeshLabException during PyMeshLab manifold repair phase: {e_pml_manifold_repair}")
                        except Exception as e_generic_manifold_repair:
                             logger.warning(f"    PML: Generic exception during PyMeshLab manifold repair phase: {e_generic_manifold_repair}")
                        
                        ms_s.save_current_mesh(temp_out)
                        loaded_s_from_pml = trimesh.load_mesh(temp_out, process=True)
                    else:
                        logger.warning("Mesh in PyMeshLab for simplification was empty or invalid before decimation.")
                
                if loaded_s_from_pml is not None and MeshCleanProcess._is_mesh_valid_for_concat(loaded_s_from_pml, f"SimpPML_V{self.INTERNAL_VERSION_TRACKER}"):
                    if loaded_s_from_pml.is_watertight:
                        logger.info("Simplified mesh from PML is watertight (Trimesh check).")
                    else:
                        num_boundary_edges_pml_load = -1
                        try:
                            # Ensure graph properties are computed for loaded_s_from_pml
                            _ = loaded_s_from_pml.edges
                            num_boundary_edges_pml_load = len(loaded_s_from_pml.boundary_edges)
                        except AttributeError: # Should not happen if .edges worked
                             logger.warning(f"AttributeError for boundary_edges on PML loaded mesh, even after .edges access.")
                        except Exception: pass # Catch other potential errors during len()
                        logger.warning(f"Simplified mesh from PML Trimesh check: is_watertight={loaded_s_from_pml.is_watertight} (Boundary edges: {num_boundary_edges_pml_load if num_boundary_edges_pml_load !=-1 else 'N/A'})")
                    
                    self.simplified_body_trimesh = loaded_s_from_pml
                    logger.info(f"Body simplification via PyMeshLab successful. New face count: {self.simplified_body_trimesh.faces.shape[0]}")
                else:
                    logger.warning("Body simplification (PML) path failed, resulted in invalid/None mesh. Reverting to pre-PML-simplification body.")
                    self.simplified_body_trimesh = body_state_before_pml_simplification
            except Exception as e_simp_block: 
                logger.warning(f"Exception during body simplification (PML) block: {e_simp_block}. Reverting to pre-PML-simplification body.", exc_info=True) 
                self.simplified_body_trimesh = body_state_before_pml_simplification 
            finally:
                if ms_s is not None: del ms_s
        
        if self.simplified_body_trimesh is None:
            logger.critical("self.simplified_body_trimesh is None before final cleaning stage in _load_and_simplify_meshes. Aborting.")
            return False

        logger.info("Applying final thorough Trimesh cleaning to simplified body mesh (post-PML or if PML skipped)...")
        body_copy_for_final_clean = self.simplified_body_trimesh.copy() 
        
        cleaned_simplified_body = FaceGraftingPipeline._thoroughly_clean_trimesh_object(
            body_copy_for_final_clean, 
            "SimplifiedBodyPostPMLAndManifoldAttempt",
            self.INTERNAL_VERSION_TRACKER
        )
        if cleaned_simplified_body is not None: 
            self.simplified_body_trimesh = cleaned_simplified_body
            logger.info("Final thorough Trimesh cleaning of simplified body successful.")
        else:
            logger.warning("Final thorough Trimesh cleaning of simplified body failed or made it invalid. Using version before this Trimesh cleaning attempt.")
            if not MeshCleanProcess._is_mesh_valid_for_concat(self.simplified_body_trimesh, "SimplifiedBody_PreFinalThoroughClean_Fallback"):
                logger.error("State of simplified_body_trimesh before final thorough clean was ALREADY invalid. This shouldn't happen. Aborting.")
                return False

        if not MeshCleanProcess._is_mesh_valid_for_concat(self.simplified_body_trimesh, f"FinalSimplifiedBodyCheck_V{self.INTERNAL_VERSION_TRACKER}"):
            logger.critical("Simplified body mesh is invalid after all simplification and cleaning attempts in _load_and_simplify_meshes. Aborting.")
            return False
            
        # has_boundary_edges = True # This variable is not used after this block, can be removed if not needed elsewhere
        if self.simplified_body_trimesh is not None and not self.simplified_body_trimesh.is_empty:
            try:
                # Explicitly access 'edges' to encourage computation of graph properties
                # This should make 'boundary_edges' available.
                _ = self.simplified_body_trimesh.edges 

                num_boundary_edges = len(self.simplified_body_trimesh.boundary_edges)
                if num_boundary_edges == 0:
                    # has_boundary_edges = False # Not used
                    logger.info("FINAL SIMPLIFIED BODY MESH has no boundary edges (is_watertight=True via Trimesh check). This is good.")
                else:
                    logger.warning(f"FINAL SIMPLIFIED BODY MESH has {num_boundary_edges} boundary edges (Trimesh check). This implies it's not watertight and may have manifold issues impacting hole cutting/stitching.")
            except AttributeError as e_attr: # More specific catch for the original error
                logger.warning(f"Could not reliably determine boundary edges for final simplified body due to AttributeError: {e_attr}. This might indicate an issue with the Trimesh object state or version.")
            except Exception as e_be_check: # Catch other potential errors
                logger.warning(f"Could not reliably determine boundary edges for final simplified body: {e_be_check}")
        else:
            logger.warning("Final simplified body mesh is None or empty when checking boundary edges at end of _load_and_simplify_meshes.")
            
        logger.info(f"Finished _load_and_simplify_meshes. Final simplified_body: V={len(self.simplified_body_trimesh.vertices) if self.simplified_body_trimesh else 'N/A'}, F={len(self.simplified_body_trimesh.faces) if self.simplified_body_trimesh else 'N/A'}")
        return True

    def _determine_hole_faces_and_create_body_with_hole(self) -> bool:
        """STEP 3: Determining Hole Faces and Creating Body with Hole."""
        if self.simplified_body_trimesh is None or self.original_smplx_face_geom_tri is None: 
            logger.error("Missing simplified_body_trimesh or original_smplx_face_geom_tri for hole creation. Aborting step.")
            return False 
        
        logger.info(f"=== STEP 3: Determining Hole Faces (Input simplified_body V={len(self.simplified_body_trimesh.vertices)}, F={len(self.simplified_body_trimesh.faces)}) ===")
        
        if len(self.simplified_body_trimesh.faces) == 0 and np.any(self.config.projection_footprint_threshold > 0): 
             logger.warning("Simplified body has no faces. Cannot determine hole faces. Body with hole will be empty if removal was intended.")
             self.faces_to_remove_mask_on_body = np.array([], dtype=bool)
        else:
            self.faces_to_remove_mask_on_body = np.zeros(len(self.simplified_body_trimesh.faces), dtype=bool)
        
        if self.original_smplx_face_geom_tri.vertices.shape[0] > 0 and \
           hasattr(self.original_smplx_face_geom_tri, 'vertex_normals') and \
           self.original_smplx_face_geom_tri.vertex_normals.shape == self.original_smplx_face_geom_tri.vertices.shape and \
           len(self.simplified_body_trimesh.faces) > 0: 
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
                # logger.debug(f"Initial faces to remove from projection: {np.sum(self.faces_to_remove_mask_on_body)}")
            except Exception as e_p:
                logger.warning(f"Error during robust hole determination: {e_p}")

            if self.config.footprint_dilation_rings > 0 and np.any(self.faces_to_remove_mask_on_body):
                # logger.debug(f"Dilating footprint by {self.config.footprint_dilation_rings} rings.")
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
                # logger.debug(f"Faces to remove after dilation: {np.sum(self.faces_to_remove_mask_on_body)}")

        body_state_before_hole_cut = self.simplified_body_trimesh.copy() 
        
        current_body_for_hole = self.simplified_body_trimesh.copy() 

        if np.any(self.faces_to_remove_mask_on_body):
            if len(self.faces_to_remove_mask_on_body) == len(current_body_for_hole.faces): 
                faces_to_keep_mask = ~self.faces_to_remove_mask_on_body
                if np.any(faces_to_keep_mask):
                    current_body_for_hole.update_faces(faces_to_keep_mask)
                    current_body_for_hole.remove_unreferenced_vertices()
                else: 
                    logger.warning("All faces on body mesh were marked for removal. Body with hole will be empty.")
                    current_body_for_hole = trimesh.Trimesh() 
            else:
                logger.error(f"Mask length ({len(self.faces_to_remove_mask_on_body)}) for face removal mismatches current body face count ({len(current_body_for_hole.faces)}). Skipping face removal.")
        else:
            logger.info("No faces marked for removal to create hole. Body-with-hole will be same as simplified body (before splitting components).")

        self.body_with_hole_trimesh = current_body_for_hole 
        
        if self.body_with_hole_trimesh is not None and not self.body_with_hole_trimesh.is_empty and \
           MeshCleanProcess._is_mesh_valid_for_concat(self.body_with_hole_trimesh, f"TempBodyWHolePreSplit_V{self.INTERNAL_VERSION_TRACKER}"):
            
            if hasattr(self.body_with_hole_trimesh, 'split') and callable(self.body_with_hole_trimesh.split):
                components = self.body_with_hole_trimesh.split(only_watertight=False)
                if components:
                    largest_comp = max(components, key=lambda c: len(c.faces if hasattr(c, 'faces') and c.faces is not None else []))
                    if MeshCleanProcess._is_mesh_valid_for_concat(largest_comp, f"LargestCompBodyWHole_V{self.INTERNAL_VERSION_TRACKER}"):
                        self.body_with_hole_trimesh = largest_comp 
                        # logger.debug(f"Kept largest component of body with hole. V={len(self.body_with_hole_trimesh.vertices)}, F={len(self.body_with_hole_trimesh.faces)}")
                    else:
                        logger.warning("Largest component of body-with-hole was invalid. Using pre-split version.")
                else: 
                    logger.info("Split operation on body-with-hole yielded no components. Using pre-split version.") # Changed from debug
        elif self.body_with_hole_trimesh is not None and self.body_with_hole_trimesh.is_empty:
             logger.warning("Body-with-hole is empty before component splitting. This likely means all faces were removed or input was empty.")
        elif self.body_with_hole_trimesh is None: 
             logger.error("self.body_with_hole_trimesh is None before component splitting. This is unexpected.")
             return False

        body_state_before_final_clean = self.body_with_hole_trimesh.copy() if self.body_with_hole_trimesh is not None else None

        logger.info("Applying thorough cleaning to body-with-hole mesh (after cut and component selection)...")
        cleaned_body_with_hole = FaceGraftingPipeline._thoroughly_clean_trimesh_object(
            body_state_before_final_clean, 
            "BodyWithHoleAfterCutAndSplit",
            self.INTERNAL_VERSION_TRACKER
        )
        if cleaned_body_with_hole is not None: 
            self.body_with_hole_trimesh = cleaned_body_with_hole
            logger.info("Thorough cleaning of body-with-hole successful.")
        else: 
            logger.warning("Thorough cleaning of body-with-hole failed or input was None/invalidated. Attempting to use version before this Trimesh cleaning.")
            if body_state_before_final_clean is not None and MeshCleanProcess._is_mesh_valid_for_concat(body_state_before_final_clean, "BodyWHolePreFinalCleanFallback"):
                self.body_with_hole_trimesh = body_state_before_final_clean
            else: 
                logger.warning("State before final cleaning of body-with-hole was also invalid/None. Falling back to a cleaned version of the input simplified_body_trimesh (no hole).")
                self.body_with_hole_trimesh = FaceGraftingPipeline._thoroughly_clean_trimesh_object(
                                                    body_state_before_hole_cut.copy(), 
                                                    "SimplifiedBodyAsFallbackForBodyWithHole",
                                                    self.INTERNAL_VERSION_TRACKER)

        if not MeshCleanProcess._is_mesh_valid_for_concat(self.body_with_hole_trimesh, f"FinalBodyWHole_V{self.INTERNAL_VERSION_TRACKER}"):
             logger.critical("Body-with-hole mesh (self.body_with_hole_trimesh) is invalid after all processing in _determine_hole_faces_and_create_body_with_hole. Aborting.")
             return False
        
        logger.info(f"Finished _determine_hole_faces_and_create_body_with_hole. Final body_with_hole_trimesh: V={len(self.body_with_hole_trimesh.vertices)}, F={len(self.body_with_hole_trimesh.faces)}")
        return True

    def _extract_smplx_face_loop(self) -> bool:
        if not self.original_smplx_face_geom_tri: return False
        logger.info("Extracting SMPLX face loop.") # Changed from debug
        
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
                logger.info(f"Extracted SMPLX face loop with {len(self.s_loop_coords_ordered)} vertices.") # Changed from debug
                return True
            else:
                logger.warning("SMPLX face loop vertex indices out of bounds.")
                self.ordered_s_vidx_loop = None 
                self.s_loop_coords_ordered = None
        
        logger.warning("Failed to extract a valid SMPLX face loop.")
        return False

    def _extract_and_order_body_hole_loop(self) -> Optional[np.ndarray]:
        if self.simplified_body_trimesh is None or self.faces_to_remove_mask_on_body is None or \
        self.s_loop_coords_ordered is None:
            logger.warning("EXTRACT_BODY_LOOP: Prerequisites missing (simplified_body_trimesh, faces_to_remove_mask_on_body, or s_loop_coords_ordered).")
            return None
        
        logger.info(f"EXTRACT_BODY_LOOP: Starting. Simplified body V={len(self.simplified_body_trimesh.vertices)}, F={len(self.simplified_body_trimesh.faces)}. Faces to remove sum={np.sum(self.faces_to_remove_mask_on_body)}") # Changed from debug

        body_hole_defining_edges = MeshCleanProcess._get_hole_boundary_edges_from_removed_faces(
            self.simplified_body_trimesh, self.faces_to_remove_mask_on_body
        )
        
        if body_hole_defining_edges is None:
            logger.warning("EXTRACT_BODY_LOOP: _get_hole_boundary_edges_from_removed_faces returned None.")
            return None
        
        # logger.debug(f"EXTRACT_BODY_LOOP: Found {len(body_hole_defining_edges)} body_hole_defining_edges.")
        if len(body_hole_defining_edges) < 3:
            logger.warning(f"EXTRACT_BODY_LOOP: Not enough body_hole_defining_edges found ({len(body_hole_defining_edges)}). Cannot form a loop.")
            return None

        unord_b_comp_selected_vidx = np.array([], dtype=int)
        hole_boundary_components_vidx_list = trimesh.graph.connected_components(body_hole_defining_edges, min_len=3)
        
        if not hole_boundary_components_vidx_list:
            logger.warning("EXTRACT_BODY_LOOP: No connected components found from body_hole_defining_edges.")
            return None
            
        # logger.debug(f"EXTRACT_BODY_LOOP: Found {len(hole_boundary_components_vidx_list)} raw hole boundary components.")
        hole_comps_filtered = [comp for comp in hole_boundary_components_vidx_list if comp is not None and len(comp) >= 3]
        # logger.debug(f"EXTRACT_BODY_LOOP: Found {len(hole_comps_filtered)} filtered hole boundary components (len >= 3).")

        if not hole_comps_filtered:
            logger.warning("EXTRACT_BODY_LOOP: No valid hole components after filtering (len >= 3).")
            return None
            
        best_loop_prox_vidx = None
        best_score_prox = np.inf
        candidate_loops_info = [] 

        for comp_idx, current_comp_vidx in enumerate(hole_comps_filtered):
            if current_comp_vidx.max() >= len(self.simplified_body_trimesh.vertices) or current_comp_vidx.min() < 0:
                # logger.debug(f"EXTRACT_BODY_LOOP: Comp {comp_idx} vertex indices out of bounds. Skipping.")
                continue
            loop_coords_candidate = self.simplified_body_trimesh.vertices[current_comp_vidx]
            if len(loop_coords_candidate) < 3:
                # logger.debug(f"EXTRACT_BODY_LOOP: Comp {comp_idx} has < 3 coords after indexing. Skipping.")
                continue
            
            candidate_loops_info.append({'id': comp_idx, 'len': len(current_comp_vidx)})
            try:
                tree_candidate = cKDTree(loop_coords_candidate)
                dists_to_candidate, _ = tree_candidate.query(self.s_loop_coords_ordered, k=1)
                current_score_prox = np.mean(dists_to_candidate) if len(dists_to_candidate) > 0 else float('inf')
                candidate_loops_info[-1]['prox_score'] = current_score_prox
                # logger.debug(f"EXTRACT_BODY_LOOP: Comp {comp_idx} (len {len(current_comp_vidx)}) prox_score: {current_score_prox:.6f}")
                if current_score_prox < best_score_prox:
                    best_score_prox = current_score_prox
                    best_loop_prox_vidx = current_comp_vidx
            except Exception as e_prox:
                logger.warning(f"EXTRACT_BODY_LOOP: Exception during proximity scoring for comp {comp_idx}: {e_prox}")
                candidate_loops_info[-1]['prox_score'] = float('inf')
                continue
        
        if best_loop_prox_vidx is not None:
            unord_b_comp_selected_vidx = best_loop_prox_vidx
            logger.info(f"EXTRACT_BODY_LOOP: Selected best loop by proximity (score: {best_score_prox:.6f}, len: {len(unord_b_comp_selected_vidx)}).")
        elif hole_comps_filtered: 
            logger.warning("EXTRACT_BODY_LOOP: Proximity selection failed to find a best loop. Falling back to largest component.")
            sorted_candidates = sorted(candidate_loops_info, key=lambda x: (-x['len'], x.get('prox_score', float('inf'))))
            if sorted_candidates:
                best_candidate_info = sorted_candidates[0]
                unord_b_comp_selected_vidx = hole_comps_filtered[best_candidate_info['id']] 
                logger.info(f"EXTRACT_BODY_LOOP: Fallback selected loop by size/score: ID {best_candidate_info['id']}, len {best_candidate_info['len']}, prox {best_candidate_info.get('prox_score', 'N/A')}")
            else:
                logger.error("EXTRACT_BODY_LOOP: Proximity selection failed AND no candidates for fallback. This is unexpected.")
                return None 
        else: 
            logger.error("EXTRACT_BODY_LOOP: No filtered components available for selection. This state should not be reached.")
            return None
        
        if len(unord_b_comp_selected_vidx) < 3 : 
            logger.warning(f"EXTRACT_BODY_LOOP: Selected component has < 3 vertices ({len(unord_b_comp_selected_vidx)}). Cannot order.")
            return None

        ordered_b_vidx_footprint = None
        b_comp_set = set(unord_b_comp_selected_vidx)
        if body_hole_defining_edges is not None and len(body_hole_defining_edges) > 0:
            b_edges_for_selected_comp = [e for e in body_hole_defining_edges if e[0] in b_comp_set and e[1] in b_comp_set]
            if len(b_edges_for_selected_comp) >= len(unord_b_comp_selected_vidx) -1 : 
                # logger.debug(f"EXTRACT_BODY_LOOP: Ordering {len(unord_b_comp_selected_vidx)} verts with {len(b_edges_for_selected_comp)} edges for selected component.")
                ordered_b_vidx_footprint = MeshCleanProcess._order_loop_vertices_from_edges(
                    f"BodyHoleFootprint_Order_V{self.INTERNAL_VERSION_TRACKER}", 
                    unord_b_comp_selected_vidx, 
                    np.array(b_edges_for_selected_comp)
                )
            else:
                logger.warning(f"EXTRACT_BODY_LOOP: Not enough specific edges ({len(b_edges_for_selected_comp)}) for selected component ({len(unord_b_comp_selected_vidx)} verts) to order.")
        
        if ordered_b_vidx_footprint is not None and len(ordered_b_vidx_footprint) >= 3:
            if ordered_b_vidx_footprint.max() < len(self.simplified_body_trimesh.vertices) and ordered_b_vidx_footprint.min() >= 0:
                logger.info(f"EXTRACT_BODY_LOOP: Successfully extracted and ordered body hole loop with {len(ordered_b_vidx_footprint)} vertices.")
                return self.simplified_body_trimesh.vertices[ordered_b_vidx_footprint]
            else:
                logger.warning("EXTRACT_BODY_LOOP: Ordered body hole loop vertex indices out of bounds.")
        else:
            logger.warning("EXTRACT_BODY_LOOP: Failed to order the selected body hole component or it was too short after ordering.")
        
        logger.warning("EXTRACT_BODY_LOOP: Failed to extract a valid body hole loop (reached end of function).")
        return None

    def _align_body_loop_to_smplx_loop(self, b_loop_coords_ordered_pre_align: np.ndarray) -> bool:
        if self.s_loop_coords_ordered is None or len(self.s_loop_coords_ordered) < 2 or \
           b_loop_coords_ordered_pre_align is None or len(b_loop_coords_ordered_pre_align) < 2: 
            logger.warning("Not enough points in loops for alignment or loops missing.")
            return False

        logger.info("Aligning body loop to SMPLX face loop.") # Changed from debug
        b_loop_coords_to_align = b_loop_coords_ordered_pre_align.copy()
        
        s_start_pt = self.s_loop_coords_ordered[0]
        kdt_b_align = cKDTree(b_loop_coords_to_align)
        _, closest_idx_on_b_to_s_start = kdt_b_align.query(s_start_pt, k=1)
        
        b_loop_coords_rolled = np.roll(b_loop_coords_to_align, -closest_idx_on_b_to_s_start, axis=0)
        
        resample_count = self.config.alignment_resample_count
        if len(self.s_loop_coords_ordered) >= 2 and len(b_loop_coords_rolled) >= 2 and resample_count >= 2:
            s_r_a = MeshCleanProcess.resample_polyline_to_count(self.s_loop_coords_ordered, resample_count)
            b_r_f = MeshCleanProcess.resample_polyline_to_count(b_loop_coords_rolled, resample_count) 
            b_r_b = MeshCleanProcess.resample_polyline_to_count(b_loop_coords_rolled[::-1], resample_count) 

            if s_r_a is not None and b_r_f is not None and b_r_b is not None and \
               len(s_r_a) == resample_count and len(b_r_f) == resample_count and len(b_r_b) == resample_count:
                dist_fwd = np.sum(np.linalg.norm(s_r_a - b_r_f, axis=1))
                dist_bwd = np.sum(np.linalg.norm(s_r_a - b_r_b, axis=1))
                if dist_bwd < dist_fwd:
                    self.b_loop_coords_aligned = b_loop_coords_rolled[::-1].copy()
                    # logger.debug("Aligned body loop (reversed orientation found better).")
                else:
                    self.b_loop_coords_aligned = b_loop_coords_rolled.copy()
                    # logger.debug("Aligned body loop (forward orientation).")
            else:
                logger.warning("Resampling for alignment failed, using rolled loop.")
                self.b_loop_coords_aligned = b_loop_coords_rolled.copy()
        else:
            # logger.debug("Using rolled loop directly (not enough points or resample count too low for full alignment).")
            self.b_loop_coords_aligned = b_loop_coords_rolled.copy()
            
        return self.b_loop_coords_aligned is not None and len(self.b_loop_coords_aligned) >=3

    def _create_loft_stitch_mesh(self) -> Optional[trimesh.Trimesh]:
        if self.s_loop_coords_ordered is None or len(self.s_loop_coords_ordered) < 2 or \
           self.b_loop_coords_aligned is None or len(self.b_loop_coords_aligned) < 2:
            logger.warning("Not enough data for loft stitch mesh creation or loops missing.")
            return None

        logger.info("Creating loft stitch mesh.") # Changed from debug
        
        s_loop_coords_for_stitch = self.s_loop_coords_ordered.copy()
        b_loop_coords_for_stitch = self.b_loop_coords_aligned.copy()

        if self.config.vertex_coordinate_precision_digits is not None:
            digits = self.config.vertex_coordinate_precision_digits
            # logger.debug(f"Rounding s_loop_coords and b_loop_coords to {digits} decimal places for stitch mesh.")
            s_loop_coords_for_stitch = np.round(s_loop_coords_for_stitch, decimals=digits)
            b_loop_coords_for_stitch = np.round(b_loop_coords_for_stitch, decimals=digits)

        new_stitch_triangles_list = []
        
        resampled_s_target = MeshCleanProcess.resample_polyline_to_count(s_loop_coords_for_stitch, self.config.loft_strip_resample_count)
        if resampled_s_target is None or len(resampled_s_target) < 2:
            logger.warning("Resampling of SMPLX target loop (potentially rounded) for lofting failed.")
            return None
            
        kdt_s_target = cKDTree(resampled_s_target)
        num_b_pts_actual = len(b_loop_coords_for_stitch) 
        
        stitch_strip_vertices_np = np.vstack((b_loop_coords_for_stitch, resampled_s_target))

        for i in range(num_b_pts_actual):
            b_curr_idx = i
            b_next_idx = (i + 1) % num_b_pts_actual
            
            _, s_match_idx_curr = kdt_s_target.query(b_loop_coords_for_stitch[b_curr_idx], k=1)
            _, s_match_idx_next = kdt_s_target.query(b_loop_coords_for_stitch[b_next_idx], k=1)

            v0 = b_curr_idx
            v1 = b_next_idx
            v2 = s_match_idx_next + num_b_pts_actual 
            v3 = s_match_idx_curr + num_b_pts_actual 
            
            new_stitch_triangles_list.extend([[v0, v1, v2], [v0, v2, v3]])
        
        if not new_stitch_triangles_list:
            logger.warning("No stitch triangles generated for lofting.")
            return None

        stitch_strip_faces_np = np.array(new_stitch_triangles_list, dtype=int)
        if stitch_strip_vertices_np.ndim == 2 and stitch_strip_vertices_np.shape[0] > 0 and stitch_strip_vertices_np.shape[1] == 3 and \
           stitch_strip_faces_np.ndim == 2 and stitch_strip_faces_np.shape[0] > 0 and stitch_strip_faces_np.shape[1] == 3:
            
            strip_mesh_obj = trimesh.Trimesh(vertices=stitch_strip_vertices_np, faces=stitch_strip_faces_np, process=True)
            if MeshCleanProcess._is_mesh_valid_for_concat(strip_mesh_obj, f"RawStitchStripV{self.INTERNAL_VERSION_TRACKER}"):
                if hasattr(strip_mesh_obj, 'face_normals') and strip_mesh_obj.face_normals is not None and len(strip_mesh_obj.face_normals) > 0:
                    strip_centroid = strip_mesh_obj.centroid
                    inward_pointing_normals_count = 0
                    for face_idx in range(len(strip_mesh_obj.faces)):
                        vector_to_strip_centroid = strip_centroid - strip_mesh_obj.triangles_center[face_idx]
                        if np.dot(strip_mesh_obj.face_normals[face_idx], vector_to_strip_centroid) > 1e-6:
                            inward_pointing_normals_count += 1
                    if inward_pointing_normals_count > len(strip_mesh_obj.faces) / 2:
                        strip_mesh_obj.invert()
                        # logger.debug("Loft stitch mesh normals inverted.")
                logger.info("Loft stitch mesh created successfully.") # Changed from debug
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

        smplx_face_for_concat = self.original_smplx_face_geom_tri 
        body_with_hole_for_concat = self.body_with_hole_trimesh 

        if self.config.vertex_coordinate_precision_digits is not None:
            digits = self.config.vertex_coordinate_precision_digits
            # logger.debug(f"Attempting to round SMPLX face and Body-with-hole vertices to {digits} decimal places before concatenation.")
            if smplx_face_for_concat is not None:
                temp_smplx_face_rounded = smplx_face_for_concat.copy()
                temp_smplx_face_rounded.vertices = np.round(temp_smplx_face_rounded.vertices, decimals=digits)
                if MeshCleanProcess._is_mesh_valid_for_concat(temp_smplx_face_rounded, "RoundedSMPLXFace"):
                    smplx_face_for_concat = temp_smplx_face_rounded 
                    # logger.debug("Using rounded SMPLX face for concatenation.")
                else:
                    logger.warning("Rounding SMPLX face vertices resulted in invalid mesh, using original non-rounded for concat.")
                    smplx_face_for_concat = self.original_smplx_face_geom_tri 
            if body_with_hole_for_concat is not None:
                temp_body_with_hole_rounded = body_with_hole_for_concat.copy()
                temp_body_with_hole_rounded.vertices = np.round(temp_body_with_hole_rounded.vertices, decimals=digits)
                if MeshCleanProcess._is_mesh_valid_for_concat(temp_body_with_hole_rounded, "RoundedBodyWithHole"):
                    body_with_hole_for_concat = temp_body_with_hole_rounded 
                    # logger.debug("Using rounded Body-with-hole for concatenation.")
                else:
                    logger.warning("Rounding body_with_hole vertices resulted in invalid mesh, using original non-rounded for concat.")
                    body_with_hole_for_concat = self.body_with_hole_trimesh 
        
        # Removed DEBUG_VERT_CHECK block

        if not smplx_face_for_concat or not body_with_hole_for_concat:
             logger.error("Loft concatenation: SMPLX face or body_with_hole mesh is missing before final component check.")
             return False

        valid_cs = [m for m in [smplx_face_for_concat, body_with_hole_for_concat, strip_mesh_obj] 
                    if MeshCleanProcess._is_mesh_valid_for_concat(m, f"LoftFinalComp_V{self.INTERNAL_VERSION_TRACKER}")]
        
        if len(valid_cs) == 3:
            try:
                cand_m = trimesh.util.concatenate(valid_cs)
                
                if MeshCleanProcess._is_mesh_valid_for_concat(cand_m, f"ConcatLoftRes_V{self.INTERNAL_VERSION_TRACKER}"):
                    final_m_proc = None 

                    logger.info("Attempting vertex merging using PyMeshLab, then Trimesh process=True and fill_holes.")
                    temp_merge_pml_in = self._make_temp_path(f"merge_pml_in_v{self.INTERNAL_VERSION_TRACKER}", use_ply=True)
                    temp_merge_pml_out = self._make_temp_path(f"merge_pml_out_v{self.INTERNAL_VERSION_TRACKER}", use_ply=True)
                    
                    merged_via_pml_trimesh = None
                    ms_merge = None
                    initial_cand_m_v_count = cand_m.vertices.shape[0]
                    initial_cand_m_f_count = cand_m.faces.shape[0]
                    logger.info(f"cand_m (input to PML merge): V={initial_cand_m_v_count}, F={initial_cand_m_f_count}")
                    
                    # Removed debug save of cand_m_PRE_MERGE

                    try:
                        export_success = cand_m.export(temp_merge_pml_in)
                        if not export_success:
                            logger.error(f"Failed to export cand_m to {temp_merge_pml_in} for PyMeshLab merge.")
                            raise RuntimeError(f"Export failed for PyMeshLab merge input {temp_merge_pml_in}")

                        ms_merge = pymeshlab.MeshSet()
                        ms_merge.load_new_mesh(temp_merge_pml_in)
                        if ms_merge.current_mesh_id() == -1 or ms_merge.current_mesh().vertex_number() == 0:
                            logger.warning("Mesh became empty or invalid after loading into PyMeshLab for merge. Skipping PML merge operation.")
                        else:
                            pml_v_before_merge = ms_merge.current_mesh().vertex_number()
                            pml_f_before_merge = ms_merge.current_mesh().face_number()
                            logger.info(f"PyMeshLab mesh stats BEFORE merge_close_vertices: V={pml_v_before_merge}, F={pml_f_before_merge}")
                            
                            merge_threshold = 1e-5 
                            logger.info(f"PyMeshLab meshing_merge_close_vertices with threshold: {merge_threshold}")
                            ms_merge.meshing_merge_close_vertices(threshold=pymeshlab.AbsoluteValue(merge_threshold))
                            
                            pml_v_after_merge = ms_merge.current_mesh().vertex_number()
                            pml_f_after_merge = ms_merge.current_mesh().face_number()
                            logger.info(f"PyMeshLab mesh stats AFTER merge_close_vertices: V={pml_v_after_merge}, F={pml_f_after_merge}")
                            if pml_v_after_merge < pml_v_before_merge:
                                logger.info(f"PyMeshLab merge_close_vertices REDUCED vertex count by {pml_v_before_merge - pml_v_after_merge}.")
                            else:
                                logger.info("PyMeshLab merge_close_vertices did NOT reduce vertex count.")
                            
                            pml_v_after_merge = ms_merge.current_mesh().vertex_number()
                            logger.info(f"PyMeshLab mesh stats AFTER merge_close_vertices: V={pml_v_after_merge}...")
                            ms_merge.meshing_remove_unreferenced_vertices()
                            v_after_pml_remove_unref = ms_merge.current_mesh().vertex_number()
                            logger.info(f"PyMeshLab mesh stats AFTER remove_unreferenced_vertices: V={v_after_pml_remove_unref}...")

                            if v_after_pml_remove_unref > 0 : 
                                try:
                                    logger.info("Attempting PyMeshLab meshing_close_holes on PML-merged mesh.")
                                    max_hole_edges_pml = 100 
                                    ms_merge.meshing_close_holes(maxholesize=max_hole_edges_pml, newfaceselected=False)
                                    logger.info(f"PyMeshLab meshing_close_holes completed. Mesh stats: V={ms_merge.current_mesh().vertex_number()}, F={ms_merge.current_mesh().face_number()}")
                                except pymeshlab.PyMeshLabException as e_pml_close:
                                    logger.warning(f"PyMeshLab meshing_close_holes failed: {e_pml_close}")
                                except Exception as e_pml_close_generic:
                                    logger.warning(f"Generic exception in PyMeshLab meshing_close_holes: {e_pml_close_generic}")
                            else:
                                logger.warning("Skipping PyMeshLab meshing_close_holes as mesh is empty after vertex merge/remove_unref.")

                            ms_merge.save_current_mesh(temp_merge_pml_out)
                            
                            logger.info(f"Reloading mesh from PyMeshLab output ({temp_merge_pml_out}) with process=True.")
                            merged_via_pml_trimesh = trimesh.load_mesh(temp_merge_pml_out, process=True) 
                            
                            if not MeshCleanProcess._is_mesh_valid_for_concat(merged_via_pml_trimesh, "LoadedFromPML_ProcessTrue"):
                                logger.warning("PyMeshLab merge result became invalid/None when reloaded (process=True). Will attempt Trimesh merge on original concatenated.")
                                merged_via_pml_trimesh = None 
                            else:
                                logger.info(f"Successfully reloaded mesh after PyMeshLab merge (process=True). Stats: V={merged_via_pml_trimesh.vertices.shape[0]}, F={merged_via_pml_trimesh.faces.shape[0]}")
                                
                                logger.info("Attempting trimesh.Trimesh.fill_holes() on PML-merged & Trimesh-processed result.")
                                merged_via_pml_trimesh.fill_holes() 
                                logger.info(f"Mesh stats after trimesh.Trimesh.fill_holes(): V={merged_via_pml_trimesh.vertices.shape[0]}, F={merged_via_pml_trimesh.faces.shape[0]}")
                                
                                if not MeshCleanProcess._is_mesh_valid_for_concat(merged_via_pml_trimesh, "AfterTrimeshFillHolesOnPML"):
                                    logger.warning("Mesh became invalid after trimesh.Trimesh.fill_holes(). PML path considered failed.")
                                    merged_via_pml_trimesh = None 
                                else:
                                    logger.info("trimesh.Trimesh.fill_holes() completed on PML+processed mesh.")
                                    final_m_proc = merged_via_pml_trimesh.copy() 
                                
                                # Removed debug save of DEBUG_B_PML_MERGED

                    except pymeshlab.PyMeshLabException as e_pml_filter:
                        logger.warning(f"PyMeshLabException during PyMeshLab processing: {e_pml_filter}. Will attempt Trimesh merge.")
                    except Exception as e_pml_merge_block:
                        logger.warning(f"Generic exception during PyMeshLab processing block: {e_pml_merge_block}. Will attempt Trimesh merge.")
                    finally:
                        if ms_merge is not None: del ms_merge
                    
                    if final_m_proc is None: 
                        logger.info("PML merge path did not produce a valid final_m_proc. Falling back to Trimesh vertex merging on original concatenated mesh.")
                        final_m_proc = cand_m.copy() 

                        logger.info(f"Trimesh merge tolerance (tol.merge) being used: {trimesh.constants.tol.merge}")
                        final_m_proc.merge_vertices(merge_tex=False, merge_norm=False)
                        
                        # Removed debug save of DEBUG_D_TRIMESH_FALLBACK

                    final_m_proc.remove_unreferenced_vertices() 
                    final_m_proc.remove_degenerate_faces()

                    if MeshCleanProcess._is_mesh_valid_for_concat(final_m_proc, f"ProcessedLoftRes_V{self.INTERNAL_VERSION_TRACKER}"):
                        self.stitched_mesh_intermediate = final_m_proc
                        # if self.config.debug_dir and self.stitched_mesh_intermediate is not None and \ # Removed debug_dir
                        if self.stitched_mesh_intermediate is not None and \
                           MeshCleanProcess._is_mesh_valid_for_concat(self.stitched_mesh_intermediate, "IntermediateForSeamCheck"):
                            logger.info("--- Running Seam Verification on Intermediate Stitched Mesh (after selected merge and final Trimesh cleaning) ---")
                            self._verify_seam_closure_on_mesh(self.stitched_mesh_intermediate, "IntermediateStitchedMeshAfterFullProcess")
                    else:
                        logger.warning("Processed lofted mesh (final_m_proc) is invalid. Setting stitched_mesh_intermediate to original concatenated (cand_m).")
                        self.stitched_mesh_intermediate = cand_m if MeshCleanProcess._is_mesh_valid_for_concat(cand_m, "CandMFallback") else None
                    
                    if self.stitched_mesh_intermediate is not None and not self.stitched_mesh_intermediate.is_empty:
                        logger.info(f"Stitch method '{self.config.stitch_method}' applied successfully.")
                        return True
                    else:
                        logger.warning("Lofted concatenation resulted in an empty or None mesh after processing.")
                        self.stitched_mesh_intermediate = None
                        return False
                else: 
                    logger.warning("Concatenated mesh (cand_m) is invalid before merge_vertices. Cannot proceed with merge.")
                    self.stitched_mesh_intermediate = None
                    return False
            except Exception as e_f_cat:
                logger.error(f"Exception during final loft concatenation/merge block: {e_f_cat}", exc_info=True)
                self.stitched_mesh_intermediate = None
                return False
        else:
            logger.warning(f"Not all three components (face, body_hole, strip) were valid for loft concatenation. Valid count: {len(valid_cs)}")
            if not MeshCleanProcess._is_mesh_valid_for_concat(smplx_face_for_concat, "smplx_face_for_concat_final_check"):
                logger.warning("  smplx_face_for_concat was invalid or None.")
            if not MeshCleanProcess._is_mesh_valid_for_concat(body_with_hole_for_concat, "body_with_hole_for_concat_final_check"):
                logger.warning("  body_with_hole_for_concat was invalid or None.")
            if not MeshCleanProcess._is_mesh_valid_for_concat(strip_mesh_obj, "strip_mesh_obj_final_check"):
                logger.warning("  strip_mesh_obj was invalid or None.")
            return False
        # return False # This line was unreachable, removed.
            
    def _verify_seam_closure_on_mesh(self, mesh_to_check: trimesh.Trimesh, mesh_name_for_log: str) -> bool:
        """
        Verifies seam closure on a given mesh against the pipeline's
        s_loop_coords_ordered and b_loop_coords_aligned.
        """
        logger.info(f"--- Verifying Seam Closure on: {mesh_name_for_log} (V_Check: {self.INTERNAL_VERSION_TRACKER}) ---")
        
        if not MeshCleanProcess._is_mesh_valid_for_concat(mesh_to_check, f"MeshForSeamCheck_{mesh_name_for_log}"):
            logger.warning(f"Seam check for {mesh_name_for_log}: Mesh is invalid or None. Skipping.")
            return True 

        if self.s_loop_coords_ordered is None or len(self.s_loop_coords_ordered) < 3:
            logger.info(f"Seam check for {mesh_name_for_log}: SMPLX face loop (s_loop_coords_ordered) is not available or too short. Skipping check.")
            return True 

        if self.b_loop_coords_aligned is None or len(self.b_loop_coords_aligned) < 3:
            logger.info(f"Seam check for {mesh_name_for_log}: Aligned body hole loop (b_loop_coords_aligned) is not available or too short. Skipping check.")
            return True 

        try:
            original_seam_coords = np.vstack((self.s_loop_coords_ordered, self.b_loop_coords_aligned))
            if len(original_seam_coords) == 0:
                logger.info(f"Seam check for {mesh_name_for_log}: Combined original seam coordinates are empty. Skipping check.")
                return True
            original_seam_kdtree = cKDTree(original_seam_coords)
        except Exception as e_kdtree:
            logger.warning(f"Seam check for {mesh_name_for_log}: Failed to build KDTree from original seam coordinates: {e_kdtree}. Skipping check.")
            return True

        boundary_edges_final = MeshCleanProcess._get_boundary_edges_manually_from_faces(mesh_to_check)
        
        if boundary_edges_final is None or len(boundary_edges_final) == 0:
            logger.info(f"Seam check for {mesh_name_for_log}: Mesh has no boundary edges. Seam is considered closed.")
            return True 

        gap_check_threshold = getattr(self.config, 'seam_gap_verification_threshold', 1e-4) 
        
        gap_edges_found_indices: List[Tuple[int,int]] = [] 
        
        for v_idx_a, v_idx_b in boundary_edges_final:
            if v_idx_a >= len(mesh_to_check.vertices) or \
               v_idx_b >= len(mesh_to_check.vertices):
                # logger.debug(f"Skipping boundary edge ({v_idx_a}, {v_idx_b}) in {mesh_name_for_log} due to out-of-bounds vertex index (mesh_verts_len={len(mesh_to_check.vertices)}).")
                continue

            coord_a = mesh_to_check.vertices[v_idx_a]
            coord_b = mesh_to_check.vertices[v_idx_b]

            try:
                dist_a, _ = original_seam_kdtree.query(coord_a, k=1, distance_upper_bound=gap_check_threshold * 1.1) 
                dist_b, _ = original_seam_kdtree.query(coord_b, k=1, distance_upper_bound=gap_check_threshold * 1.1)
            except Exception as e_query: 
                logger.warning(f"Seam check for {mesh_name_for_log}: KDTree query failed for edge ({v_idx_a}, {v_idx_b}): {e_query}. Skipping edge.")
                continue

            if dist_a < gap_check_threshold and dist_b < gap_check_threshold:
                gap_edges_found_indices.append((v_idx_a, v_idx_b))
        
        if not gap_edges_found_indices:
            logger.info(f"Seam check for {mesh_name_for_log}: No boundary edges found with both vertices within {gap_check_threshold:.1e} of the original seam. Seam appears effectively closed.")
            return True 
        else:
            logger.warning(f"Seam check for {mesh_name_for_log}: Found {len(gap_edges_found_indices)} boundary edges where both vertices are close (within {gap_check_threshold:.1e}) to the original seam. This may indicate tiny gaps.")
            # for i, edge_indices_tuple in enumerate(gap_edges_found_indices[:min(5, len(gap_edges_found_indices))]):
            #      logger.debug(f"  {mesh_name_for_log} - Potential gap edge {i+1}: Vertices {edge_indices_tuple}")
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
            logger.critical("No valid fallback components for concatenation.") # Changed from fatal
            return False
            
        try:
            self.stitched_mesh_intermediate = trimesh.util.concatenate(valid_fb_comps)
            if not MeshCleanProcess._is_mesh_valid_for_concat(self.stitched_mesh_intermediate, f"FallbackConcatV{self.INTERNAL_VERSION_TRACKER}"):
                logger.critical("Fallback concatenation resulted in invalid mesh.") # Changed from fatal
                self.stitched_mesh_intermediate = None
                return False
            self.stitched_mesh_intermediate.fix_normals() 
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
            else: 
                logger.info(f"Stitch method '{self.config.stitch_method}' did not produce a result or was not applicable. Defaulting to simple concatenation.")
                return self._perform_simple_concatenation()
        else: 
            logger.info(f"Stitch method is '{self.config.stitch_method}'. Proceeding with simple concatenation.")
            return self._perform_simple_concatenation()

    def _fill_seam_holes_ear_clip(self) -> bool:
        from scipy.spatial import cKDTree 

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
        
        try:
            _ = mesh_to_fill.face_adjacency 
            # logger.debug("Successfully pre-cached graph properties for mesh_to_fill.")
        except Exception as e_cache_graph:
            logger.warning(f"Could not pre-cache graph properties for mesh_to_fill: {e_cache_graph}. Normal alignment might fail.")

        original_vertices_for_fill = mesh_to_fill.vertices.copy()
        current_faces_list = list(mesh_to_fill.faces)
        added_any_fill_faces_this_pass = False 
        
        s_loop_ref = self.s_loop_coords_ordered 
        b_loop_ref = self.b_loop_coords_aligned 
        
        all_ordered_loops_on_stitched_mesh = MeshCleanProcess.get_all_boundary_loops(mesh_to_fill, min_loop_len=3)
        # logger.debug(f"Found {len(all_ordered_loops_on_stitched_mesh)} boundary loops for potential filling.")

        kdt_s_ref, kdt_b_ref = None, None
        if s_loop_ref is not None and len(s_loop_ref) > 0: kdt_s_ref = cKDTree(s_loop_ref)
        if b_loop_ref is not None and len(b_loop_ref) > 0: kdt_b_ref = cKDTree(b_loop_ref)
        
        proximity_to_seam_threshold_fill = 0.025 

        z_threshold_for_hands_feet = None
        if self.simplified_body_trimesh is not None and MeshCleanProcess._is_mesh_valid_for_concat(self.simplified_body_trimesh, "BodyForLimbZThresh"):
            body_bounds_min, body_bounds_max = self.simplified_body_trimesh.bounds
            body_height = body_bounds_max[2] - body_bounds_min[2]
            if body_height > 1e-6: 
                z_threshold_for_hands_feet = body_bounds_min[2] + body_height * 0.30 
                # logger.debug(f"Z-threshold for excluding limb openings during fill: {z_threshold_for_hands_feet:.4f}")

        for loop_idx, current_hole_v_indices_ordered in enumerate(all_ordered_loops_on_stitched_mesh):
            # logger.debug(f"Processing Loop {loop_idx} with {len(current_hole_v_indices_ordered) if current_hole_v_indices_ordered is not None else 'N/A'} vertices.")
            if not (current_hole_v_indices_ordered is not None and \
                    3 <= len(current_hole_v_indices_ordered) <= self.config.max_seam_hole_fill_vertices):
                continue
            if current_hole_v_indices_ordered.max() >= len(original_vertices_for_fill) or current_hole_v_indices_ordered.min() < 0:
                logger.warning(f"Loop {loop_idx} vertex indices out of bounds. Max index: {current_hole_v_indices_ordered.max()}, Vertices available: {len(original_vertices_for_fill)}. Skipping.")
                continue
            
            current_hole_coords_3d = original_vertices_for_fill[current_hole_v_indices_ordered]
            loop_centroid_z = np.mean(current_hole_coords_3d[:, 2])

            if z_threshold_for_hands_feet is not None and loop_centroid_z < z_threshold_for_hands_feet:
                # logger.debug(f"Loop {loop_idx} (centroid Z: {loop_centroid_z:.3f}) is below Z-threshold. Likely a limb opening. Skipping fill.")
                continue
            
            is_seam_hole_for_fill = False 
            if kdt_s_ref is not None and kdt_b_ref is not None:
                d_s, _ = kdt_s_ref.query(current_hole_coords_3d, k=1, distance_upper_bound=proximity_to_seam_threshold_fill * 2)
                avg_d_s = np.mean(d_s[np.isfinite(d_s)]) if np.any(np.isfinite(d_s)) else float('inf')
                d_b, _ = kdt_b_ref.query(current_hole_coords_3d, k=1, distance_upper_bound=proximity_to_seam_threshold_fill * 2)
                avg_d_b = np.mean(d_b[np.isfinite(d_b)]) if np.any(np.isfinite(d_b)) else float('inf')
                if avg_d_s < proximity_to_seam_threshold_fill and avg_d_b < proximity_to_seam_threshold_fill:
                    is_seam_hole_for_fill = True
                    # logger.debug(f"Loop {loop_idx} identified as a seam hole by proximity for filling.")
            # else:
                # logger.debug(f"Loop {loop_idx}: Proximity refs (s_loop_ref or b_loop_ref) not available, cannot mark as seam hole by proximity.")

            if is_seam_hole_for_fill:
                try:
                    if len(current_hole_coords_3d) < 3: 
                        # logger.debug(f"Loop {loop_idx} has < 3 coordinates. Skipping.")
                        continue
                    
                    plane_origin_loop, normal_loop = MeshCleanProcess.get_dominant_plane(current_hole_coords_3d)
                    if np.allclose(normal_loop,[0,0,0]): 
                        # logger.debug(f"Loop {loop_idx} degenerate normal. Skipping.")
                        continue
                    
                    transform_to_2d = trimesh.geometry.plane_transform(plane_origin_loop, normal_loop)
                    loop_coords_2d_projected = trimesh.transform_points(current_hole_coords_3d, transform_to_2d)[:, :2]
                    
                    if len(np.unique(loop_coords_2d_projected, axis=0)) < 3: 
                        # logger.debug(f"Loop {loop_idx} degenerate 2D projection (less than 3 unique points). Skipping.")
                        continue
                                        
                    path_2d_entities = [trimesh.path.entities.Line(np.arange(len(loop_coords_2d_projected)))]
                    path_2d_object = trimesh.path.Path2D(
                        entities=path_2d_entities,
                        vertices=loop_coords_2d_projected
                    )

                    from shapely.geometry import Polygon 
                    
                    patch_vertices_2d, patch_faces_local_idx = None, None
                    triangulation_successful = False

                    # logger.debug(f"Loop {loop_idx}: Attempting triangulation with triangle_args='p'.")
                    try:
                        poly2d = Polygon(loop_coords_2d_projected) 
                        patch_vertices_2d, patch_faces_local_idx = trimesh.creation.triangulate_polygon(
                            poly2d, triangle_args="p" 
                        )
                        triangulation_successful = (patch_faces_local_idx is not None and len(patch_faces_local_idx) > 0)
                        # logger.debug(f"Loop {loop_idx}: PSLG triangulation (Shapely+Triangle) {'succeeded' if triangulation_successful else 'returned no faces'}.")
                    except ImportError as e_shapely: 
                        logger.warning(f"Loop {loop_idx}: Shapely not available ({e_shapely}), cannot use Shapely polygon for PSLG. Will attempt Path2D fallback if possible.")
                        try:
                             patch_vertices_2d, patch_faces_local_idx = trimesh.creation.triangulate_polygon(
                                path_2d_object, triangle_args="p" 
                            )
                             triangulation_successful = (patch_faces_local_idx is not None and len(patch_faces_local_idx) > 0)
                             # logger.debug(f"Loop {loop_idx}: PSLG triangulation (Path2D+Triangle) {'succeeded' if triangulation_successful else 'returned no faces'}.")
                        except Exception as e_tri_a_fallback:
                             logger.warning(f"Loop {loop_idx}: PSLG triangulation (Path2D+Triangle) failed: {e_tri_a_fallback}")
                    except AttributeError as e_attr_path2d_tri: 
                        logger.warning(f"Loop {loop_idx}: AttributeError during PSLG triangulation with Path2D ('{e_attr_path2d_tri}'). This can happen with older Trimesh and 'triangle_args'.")
                    except Exception as e_tri_a:
                        logger.warning(f"Loop {loop_idx}: PSLG triangulation attempt failed: {e_tri_a}")

                    if not triangulation_successful:
                        # logger.debug(f"Loop {loop_idx}: Attempting fallback triangulation (ear-cut with Path2D).")
                        try:
                            if hasattr(path_2d_object, 'polygons_full') and path_2d_object.polygons_full:
                                poly2d_for_earcut = path_2d_object.polygons_full[0] 
                                patch_vertices_2d, patch_faces_local_idx = trimesh.creation.triangulate_polygon(poly2d_for_earcut)
                                triangulation_successful = (patch_faces_local_idx is not None and len(patch_faces_local_idx) > 0)
                                # logger.debug(f"Loop {loop_idx}: ear-cut fallback {'succeeded' if triangulation_successful else 'returned no faces'}.")
                            else:
                                logger.warning(f"Loop {loop_idx}: path_2d_object.polygons_full not available for ear-cut fallback. Skipping triangulation.")
                                triangulation_successful = False
                        except AttributeError as e_attr_path2d_earcut: 
                             logger.warning(f"Loop {loop_idx}: AttributeError during ear-cut fallback ('{e_attr_path2d_earcut}'). This can happen with older Trimesh. Skipping triangulation for this loop.")
                             triangulation_successful = False
                        except Exception as e_tri_b:
                            logger.warning(f"Loop {loop_idx}: ear-cut fallback failed: {e_tri_b}")
                            triangulation_successful = False

                    if not triangulation_successful:
                        # logger.debug(f"Loop {loop_idx}: All triangulation attempts failed. Skipping hole.")
                        continue
                    
                    try:
                        from scipy.spatial import cKDTree           
                        idx_map = cKDTree(loop_coords_2d_projected).query(
                            patch_vertices_2d, k=1
                        )[1]
                    except ImportError:                            
                        idx_map = np.array([
                            np.argmin(np.linalg.norm(loop_coords_2d_projected - v, axis=1))
                            for v in patch_vertices_2d
                        ])
                    new_fill_faces_global_candidate = current_hole_v_indices_ordered[idx_map[patch_faces_local_idx]]

                    temp_patch = trimesh.Trimesh(
                        vertices=original_vertices_for_fill,
                        faces=new_fill_faces_global_candidate,
                        process=True 
                    )

                    try:
                        if len(temp_patch.faces) > 0 and len(current_hole_v_indices_ordered) >= 2:
                            shared_edge_tuple = tuple(
                                sorted((
                                    current_hole_v_indices_ordered[0],
                                    current_hole_v_indices_ordered[1]
                                ))
                            )
                            
                            if hasattr(mesh_to_fill, 'edge_faces') and mesh_to_fill.edge_faces is not None and \
                               hasattr(mesh_to_fill, 'edges_unique') and mesh_to_fill.edges_unique is not None and \
                               len(mesh_to_fill.edges_unique) > 0: 

                                sorted_edges_unique_cached = np.sort(mesh_to_fill.edges_unique, axis=1)
                                sorted_shared_edge_arr = np.sort(np.array(shared_edge_tuple))

                                edge_exists_mask = np.all(sorted_edges_unique_cached == sorted_shared_edge_arr, axis=1)
                                
                                if np.any(edge_exists_mask):
                                    edge_id = np.where(edge_exists_mask)[0][0]
                                    
                                    if edge_id < len(mesh_to_fill.edge_faces):
                                        adj_faces_indices = mesh_to_fill.edge_faces[edge_id]
                                        valid_adj_face_idx = -1
                                        for f_idx_adj in adj_faces_indices:
                                            if f_idx_adj != -1 and hasattr(mesh_to_fill, 'face_normals') and \
                                               mesh_to_fill.face_normals is not None and \
                                               f_idx_adj < len(mesh_to_fill.face_normals):
                                                valid_adj_face_idx = f_idx_adj
                                                break
                                        
                                        if valid_adj_face_idx != -1 and \
                                           hasattr(temp_patch, 'face_normals') and \
                                           temp_patch.face_normals is not None and \
                                           len(temp_patch.face_normals) > 0: 
                                            normal_existing_adj_face = mesh_to_fill.face_normals[valid_adj_face_idx]
                                            normal_patch_adj_face = temp_patch.face_normals[0] 
                                            if np.dot(normal_patch_adj_face, normal_existing_adj_face) < 0.1: 
                                                new_fill_faces_global_candidate = new_fill_faces_global_candidate[:, ::-1]
                                                # logger.debug(f"Loop {loop_idx}: flipped patch orientation based on adjacent face normal.")
                                        elif len(temp_patch.face_normals) == 0:
                                            logger.warning(f"Loop {loop_idx}: temp_patch has no face normals for orientation check.")
                                        # else:
                                             # logger.debug(f"Loop {loop_idx}: No valid adjacent face found or temp_patch normals missing for orientation check.")
                                    else:
                                        logger.warning(f"Loop {loop_idx}: edge_id {edge_id} out of bounds for mesh_to_fill.edge_faces (len {len(mesh_to_fill.edge_faces)}).")
                                # else:
                                    # logger.debug(f"Loop {loop_idx}: Shared edge {shared_edge_tuple} not found in mesh_to_fill.edges_unique for normal alignment.")
                            else:
                                logger.warning(f"Loop {loop_idx}: mesh_to_fill.edge_faces or .edges_unique not populated or empty after cache attempt. Skipping normal alignment.")
                    except Exception as e_norm:
                        logger.warning(f"Loop {loop_idx}: normal alignment processing failed: {e_norm}", exc_info=True)

                    current_faces_list.extend(new_fill_faces_global_candidate)
                    added_any_fill_faces_this_pass = True
                    # logger.debug(
                    #     f"Filled loop {loop_idx} with {len(new_fill_faces_global_candidate)} faces."
                    # )
                except Exception as e_ear_clip_outer: 
                    logger.warning(f"Error during ear-clip processing for loop {loop_idx}: {e_ear_clip_outer}", exc_info=True)
        
        if added_any_fill_faces_this_pass:
            updated_mesh_after_fill = trimesh.Trimesh(vertices=original_vertices_for_fill, 
                                                      faces=np.array(current_faces_list, dtype=int), 
                                                      process=True) 

            if MeshCleanProcess._is_mesh_valid_for_concat(updated_mesh_after_fill, f"MeshAfterEarClipFill_V{self.INTERNAL_VERSION_TRACKER}"):
                self.final_processed_mesh = updated_mesh_after_fill
                logger.info("Ear-clip seam hole filling applied.")
            else:
                logger.warning("Mesh became invalid after ear-clip fill. Reverting to pre-fill mesh.")
                self.final_processed_mesh = self.stitched_mesh_intermediate 
        else:
            logger.info("No seam holes were filled by ear-clipping (either no suitable holes or triangulation failed).")
            self.final_processed_mesh = self.stitched_mesh_intermediate 
        return True
        
    def _apply_final_polish(self) -> bool:
        """STEP 5.5: Applying Final Polish using PyMeshLab."""
        if not self.final_processed_mesh or \
           not MeshCleanProcess._is_mesh_valid_for_concat(self.final_processed_mesh, f"MeshBeforeFinalPolish_V{self.INTERNAL_VERSION_TRACKER}"):
            logger.warning("Skipping final polish: mesh is invalid or missing.")
            return True 

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
                    nm_edges_metric = topo_measures.get('non_manifold_edges', -1) 
                    if nm_edges_metric == 0 : is_manifold_for_hole_closing = True
                except Exception: 
                    logger.info("Could not get topological measures for hole closing check during polish.") # Changed from debug

                if is_manifold_for_hole_closing:
                    try:
                        ms_polish.meshing_close_holes(maxholesize=self.config.final_polish_max_hole_edges, newfaceselected=False)
                    except pymeshlab.PyMeshLabException as e_close_holes:
                        logger.info(f"Polish: meshing_close_holes failed: {e_close_holes}")
                else:
                    logger.info("Polish: Skipping meshing_close_holes as mesh not determined to be sufficiently manifold.")
                
                ms_polish.meshing_remove_unreferenced_vertices()
                ms_polish.compute_normal_per_face() 
                ms_polish.save_current_mesh(temp_polish_out)
                polished_mesh_loaded = trimesh.load_mesh(temp_polish_out, process=True)
            
            if MeshCleanProcess._is_mesh_valid_for_concat(polished_mesh_loaded, f"PolishedMeshFromPML_V{self.INTERNAL_VERSION_TRACKER}"):
                self.final_processed_mesh = polished_mesh_loaded
                self.final_processed_mesh.fix_normals() 
                logger.info("Final polish step applied.")
            else:
                logger.warning("Final polish resulted in invalid/empty mesh. Keeping pre-polish mesh.")
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
            return True 
        
        if self.config.spider_filter_area_factor is None and self.config.spider_filter_max_edge_len_factor is None:
            logger.info("Skipping spider filter: no criteria defined.")
            return True

        logger.info("=== STEP 5.75: Filtering Spider-Web Triangles ===")
        target_faces_for_spider_filter = None
        if MeshCleanProcess._is_mesh_valid_for_concat(self.original_smplx_face_geom_tri, "OrigFaceForSpiderRefBounds"):
            if self.final_processed_mesh.faces.shape[0] > 0 and self.original_smplx_face_geom_tri.vertices.shape[0] > 0: 
                final_mesh_centroids = self.final_processed_mesh.triangles_center
                smplx_bounds_min, smplx_bounds_max = self.original_smplx_face_geom_tri.bounds
                padding = 0.05 
                min_b = smplx_bounds_min - padding; max_b = smplx_bounds_max + padding
                candidate_indices = [idx for idx, centroid in enumerate(final_mesh_centroids) if 
                                     (min_b[0] <= centroid[0] <= max_b[0] and
                                      min_b[1] <= centroid[1] <= max_b[1] and
                                      min_b[2] <= centroid[2] <= max_b[2])]
                if candidate_indices: target_faces_for_spider_filter = np.array(candidate_indices, dtype=int)
                # logger.debug(f"Targeting {len(target_faces_for_spider_filter) if target_faces_for_spider_filter is not None else 0} faces near SMPLX bounds for spider filter.")

        effective_max_edge_length = None
        if self.config.spider_filter_max_edge_len_factor is not None and self.config.spider_filter_max_edge_len_factor > 0:
            if self.simplified_body_trimesh and MeshCleanProcess._is_mesh_valid_for_concat(self.simplified_body_trimesh, "BodyForSpiderEdgeRef") and \
               len(self.simplified_body_trimesh.edges_unique_length) > 0:
                body_edge_lengths = self.simplified_body_trimesh.edges_unique_length
                if len(body_edge_lengths) > 0:
                    effective_max_edge_length = np.max(body_edge_lengths) * (1.0 + self.config.spider_filter_max_edge_len_factor)
                    # logger.debug(f"Spider filter effective max edge length: {effective_max_edge_length}")

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
            return False 
        
        self.final_processed_mesh = filtered_mesh
        self.final_processed_mesh.fix_normals()
        logger.info("Spider-web triangle filter applied.")
        return True

    def _verify_seam_closure(self) -> bool:
        """
        Verifies if there are any residual boundary edges along the original seam
        in the final processed mesh, indicating a potential gap.
        This is a diagnostic check and does not alter the mesh.
        Returns True if seam appears closed or check cannot be performed, False if potential gaps are found.
        """
        logger.info(f"=== STEP 6 (Verify): Verifying Seam Closure (V{self.INTERNAL_VERSION_TRACKER}) ===")
        if not MeshCleanProcess._is_mesh_valid_for_concat(self.final_processed_mesh, f"FinalMeshForSeamCheck_V{self.INTERNAL_VERSION_TRACKER}"):
            logger.warning("Seam closure verification: Final processed mesh is invalid or missing. Skipping check.")
            return True 

        if self.s_loop_coords_ordered is None or len(self.s_loop_coords_ordered) < 3:
            logger.info("Seam closure verification: SMPLX face loop (s_loop_coords_ordered) is not available or too short. Skipping check.")
            return True 

        if self.b_loop_coords_aligned is None or len(self.b_loop_coords_aligned) < 3:
            logger.info("Seam closure verification: Aligned body hole loop (b_loop_coords_aligned) is not available or too short. Skipping check.")
            return True 

        try:
            original_seam_coords = np.vstack((self.s_loop_coords_ordered, self.b_loop_coords_aligned))
            if len(original_seam_coords) == 0:
                logger.info("Seam closure verification: Combined original seam coordinates are empty. Skipping check.")
                return True
                
            original_seam_kdtree = cKDTree(original_seam_coords)
        except Exception as e_kdtree:
            logger.warning(f"Seam closure verification: Failed to build KDTree from original seam coordinates: {e_kdtree}. Skipping check.")
            return True

        boundary_edges_final = MeshCleanProcess._get_boundary_edges_manually_from_faces(self.final_processed_mesh)
        
        if boundary_edges_final is None or len(boundary_edges_final) == 0:
            logger.info("Seam closure verification: Final mesh has no boundary edges. Seam is considered closed.")
            return True

        gap_check_threshold = getattr(self.config, 'seam_gap_verification_threshold', 1e-4) 
        
        gap_edges_found = []
        for v_idx_a, v_idx_b in boundary_edges_final:
            if v_idx_a >= len(self.final_processed_mesh.vertices) or \
               v_idx_b >= len(self.final_processed_mesh.vertices):
                # logger.debug(f"Skipping boundary edge ({v_idx_a}, {v_idx_b}) due to out-of-bounds vertex index for final mesh vertices (len={len(self.final_processed_mesh.vertices)}).")
                continue

            coord_a = self.final_processed_mesh.vertices[v_idx_a]
            coord_b = self.final_processed_mesh.vertices[v_idx_b]

            try:
                dist_a, _ = original_seam_kdtree.query(coord_a, k=1, distance_upper_bound=gap_check_threshold * 1.1) 
                dist_b, _ = original_seam_kdtree.query(coord_b, k=1, distance_upper_bound=gap_check_threshold * 1.1)
            except Exception as e_query: 
                logger.warning(f"Seam closure verification: KDTree query failed for edge ({v_idx_a}, {v_idx_b}): {e_query}. Skipping edge.")
                continue

            if dist_a < gap_check_threshold and dist_b < gap_check_threshold:
                gap_edges_found.append(((v_idx_a, v_idx_b), (coord_a, coord_b)))
        
        if not gap_edges_found:
            logger.info(f"Seam closure verification: No boundary edges found with both vertices within {gap_check_threshold:.1e} of the original seam. Seam appears effectively closed.")
            return True 
        else:
            logger.warning(f"Seam closure verification: Found {len(gap_edges_found)} boundary edges where both vertices are close (within {gap_check_threshold:.1e}) to the original seam. This may indicate tiny gaps.")
            # for i, (edge_indices, edge_coords) in enumerate(gap_edges_found[:min(5, len(gap_edges_found))]): 
            #     logger.debug(f"  Potential gap edge {i+1}: Vertices {edge_indices}, Coords approx [{edge_coords[0][0]:.4f},{edge_coords[0][1]:.4f},{edge_coords[0][2]:.4f}] to [{edge_coords[1][0]:.4f},{edge_coords[1][1]:.4f},{edge_coords[1][2]:.4f}]")
            return False 

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
            
            if not self._stitch_components(): 
                logger.error("Component stitching failed. Aborting.")
                return None
            
            if not self._fill_seam_holes_ear_clip(): 
                pass 

            if not self._apply_final_polish():
                pass

            # if not self._filter_spider_triangles(): 
            #     logger.error("Spider triangle filtering failed critically. Aborting.")
            #     return None

            if not MeshCleanProcess._is_mesh_valid_for_concat(self.final_processed_mesh, f"Mesh Before Final Save V{self.INTERNAL_VERSION_TRACKER}"):
                logger.critical("Final_processed_mesh is invalid or empty before final save. CANNOT SAVE.")
                return None

            self.final_processed_mesh.export(self.output_path)
            logger.info(f"--- Mesh Grafting V{self.INTERNAL_VERSION_TRACKER} Finished. Output: {self.output_path} ---")
            
            self._verify_seam_closure()

            # Perform final watertightness check
            logger.info("--- Performing Final Watertightness Check on Output Mesh ---")
            if self.final_processed_mesh and not self.final_processed_mesh.is_empty:
                temp_final_mesh_path = self._make_temp_path("final_check_watertight", use_ply=True)
                export_success = False
                try:
                    export_success = self.final_processed_mesh.export(temp_final_mesh_path)
                except Exception as e_export_final:
                    logger.error(f"Failed to export final mesh for watertightness check: {e_export_final}")

                if export_success:
                    # Use a dummy output path for MeshCleanProcess as it's not saving here
                    # Need to ensure MeshCleanProcess is defined or imported correctly
                    checker_mcp = MeshCleanProcess(input_path=temp_final_mesh_path, output_path="dummy_output_for_check.ply")
                    if checker_mcp.load_mesh():
                        is_final_watertight = checker_mcp.check_watertight()
                        # The check_watertight method in MeshCleanProcess prints its own details.
                        logger.info(f"Final mesh watertightness (PyMeshLab check result): {is_final_watertight}")
                    else:
                        logger.warning("Could not load final mesh into PyMeshLab for watertightness check.")
                    del checker_mcp 
                # Temp file will be cleaned by _cleanup_temp_files in finally block
            else:
                logger.warning("Final processed mesh is None or empty, skipping final watertightness check.")

            pipeline_successful = True
            return self.final_processed_mesh

        except Exception as e_main_pipeline: 
            logger.error(f"--- Pipeline V{self.INTERNAL_VERSION_TRACKER} Failed (Outer Try-Except Block) --- {e_main_pipeline}", exc_info=True)
            return None
        finally:
            self._cleanup_temp_files()
            if not pipeline_successful:
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
        debug_dir: Optional[str] = None, # Kept for signature, but will be None
        temp_file_prefix: str = "nm_repair_aggressive"
    ) -> Optional[trimesh.Trimesh]:

        if not MeshCleanProcess._is_mesh_valid_for_concat(input_mesh, f"InputForAggressiveRepair_{temp_file_prefix}"):
            logger.warning(f"V_IR_AGG: Input mesh for aggressive repair ('{temp_file_prefix}') is invalid or None. Skipping.")
            return input_mesh

        logger.info(f"--- V_IR_AGG: Starting AGGRESSIVE Iterative Non-Manifold Repair for '{temp_file_prefix}' ---")
        
        current_mesh_trimesh = input_mesh.copy()
        
        local_temp_files: List[str] = []
        def make_local_temp_path(suffix_label: str) -> str:
            actual_suffix = suffix_label if suffix_label.startswith('_') else '_' + suffix_label
            fd, path = tempfile.mkstemp(suffix=actual_suffix + ".ply", dir=debug_dir, prefix=f"{temp_file_prefix}_") # debug_dir will be None
            os.close(fd)
            local_temp_files.append(path)
            return path

        initial_v_count = current_mesh_trimesh.vertices.shape[0]
        initial_f_count = current_mesh_trimesh.faces.shape[0]

        for main_iter in range(max_main_iterations):

            if not MeshCleanProcess._is_mesh_valid_for_concat(current_mesh_trimesh, f"PreMainIter{main_iter}"):
                logger.error(f"  V_IR_AGG: Mesh became invalid before main iter {main_iter + 1}. Aborting repair.")
                break 

            temp_main_in_path = make_local_temp_path(f"main_iter{main_iter}_in")
            current_mesh_trimesh.export(temp_main_in_path)

            ms = pymeshlab.MeshSet()
            try:
                ms.load_new_mesh(temp_main_in_path)
                if ms.current_mesh_id() == -1 or ms.current_mesh().vertex_number() == 0:
                    logger.error("  V_IR_AGG: Mesh empty or invalid after loading into PyMeshLab. Aborting this iteration.")
                    break 

                v_before_iter = ms.current_mesh().vertex_number()
                f_before_iter = ms.current_mesh().face_number()

                for edge_iter in range(max_edge_repair_iters):
                    # logger.debug(f"      V_IR_AGG: Edge Repair Sub-Iter {edge_iter + 1}/{max_edge_repair_iters}")
                    nm_edges_selection_before = -1
                    try:
                        ms.set_selection_none() 
                        ms.apply_filter('compute_selection_by_non_manifold_edges_per_face') 
                        nm_edges_selection_before = ms.current_mesh().selected_face_number()
                        if nm_edges_selection_before == 0:
                            # logger.debug("        V_IR_AGG: No non-manifold edge faces selected. Skipping edge repair sub-iter.")
                            break 
                        # logger.debug(f"        V_IR_AGG: Non-manifold edge faces selected: {nm_edges_selection_before}")
                        ms.meshing_repair_non_manifold_edges(method='Split Vertices')
                    except pymeshlab.PyMeshLabException as e_edge_split:
                        # logger.debug(f"        V_IR_AGG: Edge repair by split failed ({e_edge_split}). Trying remove faces.")
                        try: 
                            ms.set_selection_none() 
                            ms.apply_filter('compute_selection_by_non_manifold_edges_per_face')
                            if ms.current_mesh().selected_face_number() > 0:
                                ms.meshing_repair_non_manifold_edges(method='Remove Faces') 
                        except pymeshlab.PyMeshLabException as e_edge_remove:
                            logger.warning(f"        V_IR_AGG: Edge repair by remove faces also failed: {e_edge_remove}")
                            break 
                
                try:
                    ms.meshing_remove_duplicate_faces()
                    ms.meshing_remove_duplicate_vertices() 
                    ms.meshing_remove_unreferenced_vertices()
                except pymeshlab.PyMeshLabException as e_pml_clean:
                    logger.warning(f"      V_IR_AGG: Error during basic PML cleaning: {e_pml_clean}")

                if min_component_faces_to_keep > 0:
                    try:
                        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=min_component_faces_to_keep, removeunref=True)
                    except pymeshlab.PyMeshLabException as e_iso:
                        logger.warning(f"      V_IR_AGG: Failed to remove isolated pieces by face count: {e_iso}")

                is_manifold_for_pml_hole_fill = False
                try:
                    topo_measures = ms.get_topological_measures()
                    nm_edges_metric = topo_measures.get('non_manifold_edges', -1)
                    if nm_edges_metric == 0:
                        is_manifold_for_pml_hole_fill = True
                except Exception as e_topo: 
                    logger.warning(f"      V_IR_AGG: Could not get topological measures for PML hole closing check: {e_topo}") # Changed from debug

                if is_manifold_for_pml_hole_fill:
                    for hf_iter in range(max_hole_fill_iters):
                        boundary_edges_before_fill = -1 
                        try:
                            current_topo_measures_for_hole = ms.get_topological_measures()
                            boundary_edges_before_fill = current_topo_measures_for_hole.get('boundary_edges', -1)
                        except Exception:
                            logger.warning("      V_IR_AGG: Failed to get fresh topo measures before hole fill attempt.")

                        if boundary_edges_before_fill == 0:
                            break
                        try:
                            ms.meshing_close_holes(maxholesize=pml_hole_fill_max_edges, newfaceselected=False)
                            boundary_edges_after_fill = -1
                            try:
                                current_topo_measures_after_hole = ms.get_topological_measures()
                                boundary_edges_after_fill = current_topo_measures_after_hole.get('boundary_edges', -1)
                            except Exception:
                                logger.warning("      V_IR_AGG: Failed to get fresh topo measures after hole fill attempt.")

                            if boundary_edges_after_fill == boundary_edges_before_fill and boundary_edges_before_fill !=0 and boundary_edges_before_fill != -1 :
                                break
                            if boundary_edges_after_fill == 0:
                                break
                        except pymeshlab.PyMeshLabException as e_pml_hf:
                            logger.warning(f"      V_IR_AGG: PyMeshLab meshing_close_holes failed: {e_pml_hf}")
                            break 

                v_after_iter = ms.current_mesh().vertex_number()
                f_after_iter = ms.current_mesh().face_number()

                temp_main_out_path = make_local_temp_path(f"main_iter{main_iter}_out")
                ms.save_current_mesh(temp_main_out_path)
                reloaded_mesh = trimesh.load_mesh(temp_main_out_path, process=False)

                if MeshCleanProcess._is_mesh_valid_for_concat(reloaded_mesh, f"ReloadedPMLIter{main_iter}"):
                    current_mesh_trimesh = reloaded_mesh
                else:
                    logger.error(f"  V_IR_AGG: Mesh became invalid after saving from PyMeshLab and reloading in main iter {main_iter + 1}. Using state before this PML processing.")
                    break
            
            except Exception as e_main_iter_processing:
                logger.error(f"  V_IR_AGG: Exception during PML processing in main iteration {main_iter + 1}: {e_main_iter_processing}", exc_info=True)
                break
            finally:
                if 'ms' in locals() and ms is not None:
                    del ms
        
        if merge_vertices_at_end and current_mesh_trimesh is not None: 
            mesh_for_final_clean = current_mesh_trimesh.copy() 
            original_v = len(mesh_for_final_clean.vertices)
            original_f = len(mesh_for_final_clean.faces)

            mesh_for_final_clean.merge_vertices(merge_tex=False, merge_norm=False)
            mesh_for_final_clean.remove_unreferenced_vertices()
            mesh_for_final_clean.remove_degenerate_faces()
            mesh_for_final_clean.fix_normals()

            if MeshCleanProcess._is_mesh_valid_for_concat(mesh_for_final_clean, "AfterFinalTrimeshClean"):
                current_mesh_trimesh = mesh_for_final_clean
                logger.info(f"  V_IR_AGG: Final Trimesh clean applied. V={len(current_mesh_trimesh.vertices)} (was {original_v}), F={len(current_mesh_trimesh.faces)} (was {original_f})")
            else:
                logger.warning(f"  V_IR_AGG: Final Trimesh clean resulted in invalid mesh. Using state before this final clean (V={original_v}, F={original_f}).")

        logger.info(f"--- V_IR_AGG: Finished AGGRESSIVE Iterative Non-Manifold Repair for '{temp_file_prefix}' ---")
        final_v_count = current_mesh_trimesh.vertices.shape[0] if current_mesh_trimesh else -1
        final_f_count = current_mesh_trimesh.faces.shape[0] if current_mesh_trimesh else -1
        logger.info(f"  V_IR_AGG: Final state: Vertices={final_v_count} (Initial: {initial_v_count}), Faces={final_f_count} (Initial: {initial_f_count})")

        for f_path in local_temp_files:
            if os.path.exists(f_path):
                try: os.remove(f_path)
                except OSError as e_rm: logger.warning(f"Could not remove temp file {f_path}: {e_rm}")

        if not MeshCleanProcess._is_mesh_valid_for_concat(current_mesh_trimesh, f"FinalAggressiveRepair_{temp_file_prefix}"):
            logger.error(f"  V_IR_AGG: Mesh is invalid at the end of aggressive repair. Returning None.")
            return None 
                
        return current_mesh_trimesh

    def _perform_iterative_body_repair(self) -> bool:
        """STEP 2.5: Iterative Non-Manifold Repair on Simplified Body."""
        if self.simplified_body_trimesh is None: 
            logger.error("Cannot perform iterative body repair: simplified_body_trimesh is None.")
            return False
        
        if self.config.iterative_repair_s1_iters > 0:
            logger.info("=== STEP 2.5: AGGRESSIVE Iterative Non-Manifold Repair on Simplified Body ===")
            
            body_before_aggressive_repair = self.simplified_body_trimesh.copy()

            repaired_body = FaceGraftingPipeline._iterative_non_manifold_repair_pml_aggressive(
                self.simplified_body_trimesh, 
                pml_hole_fill_max_edges=self.config.final_polish_max_hole_edges, 
                max_main_iterations=self.config.iterative_repair_s1_iters,
                max_edge_repair_iters=3, 
                max_hole_fill_iters=2,
                min_component_faces_to_keep=100, 
                merge_vertices_at_end=True,
                debug_dir=None, # Pass None as debug_dir is removed from config
                temp_file_prefix=f"body_v{self.INTERNAL_VERSION_TRACKER}_s1_agg_repair"
            )
            
            if repaired_body is not None and MeshCleanProcess._is_mesh_valid_for_concat(repaired_body, f"BodyAfterAggressiveRepairS1_V{self.INTERNAL_VERSION_TRACKER}"):
                self.simplified_body_trimesh = repaired_body
                logger.info("Aggressive iterative body repair (S1) completed.")
            else:
                logger.warning("Aggressive iterative body repair (S1) failed or resulted in invalid/None mesh. Reverting to state before this repair.")
                self.simplified_body_trimesh = body_before_aggressive_repair
                if not MeshCleanProcess._is_mesh_valid_for_concat(self.simplified_body_trimesh, "BodyAfterFailedAggRepair_FallbackCheck"):
                    logger.error("Body mesh is also invalid after reverting from failed aggressive repair. Critical error.")
                    return False
        else:
            logger.info("Skipping aggressive iterative body repair (S1) as iterations are 0.")
        return True

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
                 logger.warning(f"Loaded mesh from {self.input_path} is empty.")
                 return False
            return True
        except pymeshlab.PyMeshLabException as e:
            logger.error(f"Error loading mesh {self.input_path}: {e}")
            return False


    def clean_mesh(self):
        """Cleans the mesh by removing duplicate vertices, faces, and merging close vertices."""
        if self.ms.current_mesh().vertex_number() == 0: return 
        try:
            self.ms.meshing_remove_duplicate_faces()
            self.ms.meshing_remove_duplicate_vertices()
            try:
                self.ms.meshing_repair_non_manifold_edges()
            except pymeshlab.PyMeshLabException as e:
                logger.warning(f"Could not repair non-manifold edges: {e}")
            try:
                self.ms.meshing_repair_non_manifold_vertices()
            except pymeshlab.PyMeshLabException as e:
                 logger.warning(f"Could not repair non-manifold vertices: {e}")
            self.ms.meshing_remove_unreferenced_vertices()
        except pymeshlab.PyMeshLabException as e:
            logger.error(f"Error during mesh cleaning: {e}")


    def fill_holes(self):
        """
        Fills holes in the mesh.
        """
        if self.ms.current_mesh().vertex_number() == 0: return 
        try:
            self.ms.meshing_close_holes()
        except pymeshlab.PyMeshLabException as e:
            logger.warning(f"Could not close holes: {e}")


    def reconstruct_surface(self, method='poisson', **kwargs):
        """
        Reconstructs the surface to make the mesh watertight.

        Parameters:
            method (str): Reconstruction method ('poisson').
            **kwargs: Additional parameters for the reconstruction method.
        """
        if self.ms.current_mesh().vertex_number() == 0: return 
        if method == 'poisson':
            depth = kwargs.get('depth', 10) 
            try:
                self.ms.generate_surface_reconstruction_screened_poisson(depth=depth, preclean=True)
                if self.ms.current_mesh().vertex_number() == 0:
                    logger.warning(f"Poisson reconstruction resulted in an empty mesh.")
                    return False
                return True
            except pymeshlab.PyMeshLabException as e:
                 logger.error(f"Error during Poisson reconstruction: {e}")
                 return False
        else:
            raise ValueError("Unsupported reconstruction method. Use 'poisson'.")

    def check_watertight(self):
        """
        Checks if the mesh is watertight.

        Returns:
            bool: True if the mesh is watertight, False otherwise.
        """
        if self.ms.current_mesh().vertex_number() == 0: return False 
        try:
            metrics = self.ms.get_topological_measures()
            is_wt = (
                metrics.get('boundary_edges', -1) == 0 and
                metrics.get('number_holes', -1) == 0 and
                metrics.get('is_mesh_two_manifold', False)
            )
            if not is_wt:
                logger.info(f"Watertight Check Failed: Boundary Edges={metrics.get('boundary_edges', 'N/A')}, Holes={metrics.get('number_holes', 'N/A')}, IsTwoManifold={metrics.get('is_mesh_two_manifold', 'N/A')}")
            return is_wt
        except pymeshlab.PyMeshLabException as e:
            logger.error(f"Error checking watertightness: {e}")
            return False


    def save_mesh(self):
        """Saves the processed mesh to the output path."""
        if self.ms.current_mesh().vertex_number() == 0:
            logger.warning(f"Skipping save for empty mesh to {self.output_path}")
            return False
        try:
            self.ms.save_current_mesh(self.output_path)
            return True
        except pymeshlab.PyMeshLabException as e:
            logger.error(f"Error saving mesh {self.output_path}: {e}")
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
             return False 

        self.clean_mesh()
        self.fill_holes() 

        if not self.reconstruct_surface(method=reconstruction_method, **kwargs):
             logger.warning("Stopping process because surface reconstruction failed or resulted in empty mesh.")
             return False 

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
        if boundary_edges is None or len(boundary_edges) < min_loop_len: 
            return all_loops_ordered

        loop_components_vidx = trimesh.graph.connected_components(boundary_edges, min_len=min_loop_len) 

        if not loop_components_vidx or not any(c is not None and len(c) >= min_loop_len for c in loop_components_vidx):
            return all_loops_ordered

        for i, comp_vidx_unordered in enumerate(loop_components_vidx):
            if comp_vidx_unordered is None or len(comp_vidx_unordered) < min_loop_len:
                continue
            
            comp_set = set(comp_vidx_unordered)
            edges_for_this_comp = [e for e in boundary_edges if e[0] in comp_set and e[1] in comp_set]

            if not edges_for_this_comp or len(edges_for_this_comp) < len(comp_vidx_unordered) -1 : 
                 continue

            ordered_vidx = MeshCleanProcess._order_loop_vertices_from_edges(
                f"Loop_{i}_Order", comp_vidx_unordered, np.array(edges_for_this_comp)
            )
            if ordered_vidx is not None and len(ordered_vidx) >= min_loop_len:
                all_loops_ordered.append(ordered_vidx)
        
        return all_loops_ordered
                                    
    @staticmethod
    def _is_mesh_valid_for_concat(mesh: Optional[trimesh.Trimesh], mesh_name: str) -> bool:
        """Helper to check if a trimesh object is suitable for concatenation."""
        if mesh is None:
            # logger.warning(f"CONCAT_CHECK: {mesh_name} is None.") # Changed from print
            return False
        if mesh.is_empty:
            # logger.warning(f"CONCAT_CHECK: {mesh_name} is empty (V={len(mesh.vertices)}, F={len(mesh.faces)}).") # Changed from print
            return False
        if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces'):
            # logger.warning(f"CONCAT_CHECK: {mesh_name} is missing vertices or faces attributes.") # Changed from print
            return False
        if mesh.vertices.ndim != 2 or mesh.vertices.shape[1] != 3:
            # logger.warning(f"CONCAT_CHECK: {mesh_name} vertices have wrong shape {mesh.vertices.shape}.") # Changed from print
            return False
        if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
            # logger.warning(f"CONCAT_CHECK: {mesh_name} faces have wrong shape {mesh.faces.shape}.") # Changed from print
            return False
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
            logger.error(f"Error in _get_boundary_edges_manually_from_faces: {e}", exc_info=True); return None

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
            logger.error(f"Error in _get_hole_boundary_edges_from_removed_faces: {e}", exc_info=True); return None

    @staticmethod
    def _filter_large_triangles_from_fill(
        mesh: trimesh.Trimesh,
        target_face_indices: Optional[np.ndarray] = None, 
        max_allowed_edge_length: Optional[float] = None,
        max_allowed_area_factor: Optional[float] = None,
        reference_mesh_for_stats: Optional[trimesh.Trimesh] = None,
        mesh_name_for_debug: str = "large_tri_filter"
    ) -> trimesh.Trimesh:
        if not MeshCleanProcess._is_mesh_valid_for_concat(mesh, f"InputForLargeTriFilter_{mesh_name_for_debug}"):
            return mesh
        if max_allowed_edge_length is None and max_allowed_area_factor is None:
            return mesh 

        faces_to_keep_mask = np.ones(len(mesh.faces), dtype=bool)
        
        candidate_faces_for_removal_mask = np.zeros(len(mesh.faces), dtype=bool)
        if target_face_indices is not None and len(target_face_indices) > 0:
            valid_target_indices = target_face_indices[target_face_indices < len(mesh.faces)]
            candidate_faces_for_removal_mask[valid_target_indices] = True
        else:
            candidate_faces_for_removal_mask.fill(True)

        faces_meeting_removal_criteria_mask = np.zeros(len(mesh.faces), dtype=bool)

        if max_allowed_edge_length is not None and max_allowed_edge_length > 0:
            if np.any(candidate_faces_for_removal_mask): 
                face_edge_lengths = np.zeros((len(mesh.faces), 3))
                for i, face_verts_indices in enumerate(mesh.faces):
                    if candidate_faces_for_removal_mask[i]: 
                        if face_verts_indices.max() < len(mesh.vertices):
                            v0, v1, v2 = mesh.vertices[face_verts_indices]
                            face_edge_lengths[i, 0] = np.linalg.norm(v0 - v1)
                            face_edge_lengths[i, 1] = np.linalg.norm(v1 - v2)
                            face_edge_lengths[i, 2] = np.linalg.norm(v2 - v0)
                        else: face_edge_lengths[i, :] = np.inf 
                    else:
                        face_edge_lengths[i, :] = 0 
                
                max_edge_per_face = np.max(face_edge_lengths, axis=1)
                marked_by_edge = (max_edge_per_face > max_allowed_edge_length) & candidate_faces_for_removal_mask
                faces_meeting_removal_criteria_mask[marked_by_edge] = True

        if max_allowed_area_factor is not None and max_allowed_area_factor > 0:
            if np.any(candidate_faces_for_removal_mask): 
                current_mesh_face_areas = mesh.area_faces
                median_area_stat = 0
                if reference_mesh_for_stats is not None and \
                   MeshCleanProcess._is_mesh_valid_for_concat(reference_mesh_for_stats, "RefMeshForAreaStats") and \
                   len(reference_mesh_for_stats.faces)>0:
                    median_area_stat = np.median(reference_mesh_for_stats.area_faces)
                elif len(current_mesh_face_areas[candidate_faces_for_removal_mask & ~faces_meeting_removal_criteria_mask]) > 0 : 
                    valid_areas_for_stat_calc = current_mesh_face_areas[candidate_faces_for_removal_mask & ~faces_meeting_removal_criteria_mask]
                    median_area_stat = np.median(valid_areas_for_stat_calc)
                
                if median_area_stat > 1e-9: 
                    area_threshold = median_area_stat * max_allowed_area_factor
                    marked_by_area = (current_mesh_face_areas > area_threshold) & candidate_faces_for_removal_mask
                    faces_meeting_removal_criteria_mask[marked_by_area] = True 

        faces_to_keep_mask[faces_meeting_removal_criteria_mask] = False
        
        num_total_removed = np.sum(~faces_to_keep_mask)
        if num_total_removed > 0:
            if num_total_removed == len(mesh.faces):
                logger.warning(f"All faces would be removed by large triangle filter for {mesh_name_for_debug}. Aborting filtering.")
                return mesh 

            filtered_mesh = mesh.copy(); filtered_mesh.update_faces(faces_to_keep_mask); filtered_mesh.remove_unreferenced_vertices()
            if MeshCleanProcess._is_mesh_valid_for_concat(filtered_mesh, f"FilteredMesh_{mesh_name_for_debug}"):
                return filtered_mesh
            else:
                logger.warning(f"Mesh became invalid after large triangle filtering for {mesh_name_for_debug}. Returning original.")
                return mesh 
        return mesh
    
    @staticmethod
    def run_face_grafting_pipeline(
        full_body_mesh_path: str,
        output_path: str,
        smplx_face_mesh_path: str,
        # Pass all the config parameters here or a pre-made config object
        projection_footprint_threshold: float = 0.01,
        footprint_dilation_rings: int = 3,
        body_simplification_target_faces: int = 12000,
        stitch_method: str = "body_driven_loft", 
        smplx_face_neck_loop_strategy: str = "full_face_silhouette",
        alignment_resample_count: int = 1000,
        loft_strip_resample_count: int = 50,
        max_seam_hole_fill_vertices: int = 250,
        final_polish_max_hole_edges: int = 100,
        iterative_repair_s1_iters: int = 5,
        # iterative_repair_s2_iters: int = 5, # Not directly used in this refactor pass
        # iterative_repair_s2_remesh_percent: Optional[float] = None, # Not directly used
        spider_filter_area_factor: Optional[float] = 300.0,
        spider_filter_max_edge_len_factor: Optional[float] = 0.20,
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
            vertex_coordinate_precision_digits=6, 
        )
        
        pipeline = FaceGraftingPipeline(
            full_body_mesh_path=full_body_mesh_path,
            smplx_face_mesh_path=smplx_face_mesh_path,
            output_path=output_path,
            config=config
        )
        return pipeline.process()
