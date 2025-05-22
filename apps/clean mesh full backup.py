import pymeshlab
import trimesh
import numpy as np
from scipy.spatial import cKDTree, ConvexHull # Added ConvexHull
import os # <-- Import os for path manipulation
import open3d as o3d # If not already there
import traceback
import tempfile 
from typing import List, Union, Optional, Tuple # Add Tuple
import trimesh.path # For trimesh.path.polygons
import trimesh.util as util # <<<<<<< ADD THIS LINE or ensure trimesh.util is accessible
from collections import Counter, defaultdict
from matplotlib.path import Path as MplPath # For robust point-in-polygon


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
    def process_mesh_graft_smplx_face(
        full_body_mesh_path: str,
        output_path: str,
        smplx_face_mesh_path: str,
        face_scale_for_hole_carving: float = 1.10,
        body_simplification_target_faces: int = 12000,
        final_stitch_poisson_depth: int = 10,
        debug_dir: str = None
    ) -> Optional[trimesh.Trimesh]:
        """
        Grafts the SMPLX face onto a processed body mesh.
        1. Loads SMPLX face (pristine).
        2. Loads and simplifies the full body mesh.
        3. Carves a hole in the simplified body mesh based on the SMPLX face's footprint.
        4. Concatenates the pristine face and the body-with-hole.
        5. Uses Poisson reconstruction to stitch them.
        """
        print(f"--- Starting Mesh Processing: Grafting SMPLX Face (Pristine Face Method) ---")
        print(f"SMPLX Face Path: {smplx_face_mesh_path}")
        print(f"Full Body Input Path: {full_body_mesh_path}")
        print(f"Output Path: {output_path}")

        temp_files_to_clean = []

        def make_temp_path(suffix, directory):
            _fd, path = tempfile.mkstemp(suffix=suffix, dir=directory)
            os.close(_fd)
            temp_files_to_clean.append(path)
            return path

        try:
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)
                print(f"Debug outputs will be saved in: {debug_dir}")

            # === STEP 1: Load SMPLX Face Mesh ===
            print("\n=== STEP 1: Loading SMPLX Face Mesh ===")
            smplx_face_trimesh = trimesh.load_mesh(smplx_face_mesh_path, process=False)
            if smplx_face_trimesh.is_empty:
                print(f"Fatal Error: SMPLX face mesh at {smplx_face_mesh_path} is empty.")
                return None
            smplx_face_trimesh.fix_normals() # Ensure normals are consistent
            print(f"Loaded SMPLX face: V={smplx_face_trimesh.vertices.shape[0]}, F={smplx_face_trimesh.faces.shape[0]}")
            if debug_dir: smplx_face_trimesh.export(os.path.join(debug_dir, "graft_01_smplx_face_loaded.obj"))

            # === STEP 2: Load and Simplify Full Body Mesh ===
            print("\n=== STEP 2: Loading and Simplifying Full Body Mesh ===")
            full_body_trimesh = trimesh.load_mesh(full_body_mesh_path, process=False)
            if full_body_trimesh.is_empty:
                print(f"Fatal Error: Full body mesh at {full_body_mesh_path} is empty.")
                return None
            
            simplified_body_trimesh = full_body_trimesh # Fallback
            if full_body_trimesh.faces.shape[0] > body_simplification_target_faces:
                print(f"Simplifying body from {full_body_trimesh.faces.shape[0]} to {body_simplification_target_faces} faces.")
                temp_full_body_in_pml = make_temp_path("_full_body_pml.ply", debug_dir)
                temp_simplified_body_out_pml = make_temp_path("_simp_body_pml.ply", debug_dir)
                try:
                    full_body_trimesh.export(temp_full_body_in_pml)
                    ms_simplify = pymeshlab.MeshSet()
                    ms_simplify.load_new_mesh(temp_full_body_in_pml)
                    if ms_simplify.current_mesh().face_number() > 0:
                        ms_simplify.meshing_decimation_quadric_edge_collapse(
                            targetfacenum=int(body_simplification_target_faces),
                            preservenormal=True,
                            preservetopology=False # Allow topo changes for aggressive simplification
                        )
                        ms_simplify.save_current_mesh(temp_simplified_body_out_pml)
                        simplified_body_trimesh_loaded = trimesh.load_mesh(temp_simplified_body_out_pml, process=False)
                        if not simplified_body_trimesh_loaded.is_empty:
                            simplified_body_trimesh = simplified_body_trimesh_loaded
                        else:
                            print("Warning: Body simplification in PyMeshLab resulted in an empty mesh. Using unsimplified.")
                    else:
                        print("Warning: Full body mesh was empty after loading into PyMeshLab for simplification.")
                except Exception as e_simplify:
                    print(f"Warning: Body simplification failed: {e_simplify}. Using unsimplified body.")
            else:
                print("Body mesh already meets target face count or is smaller. No simplification needed.")
            
            simplified_body_trimesh.fix_normals()
            print(f"Simplified body mesh: V={simplified_body_trimesh.vertices.shape[0]}, F={simplified_body_trimesh.faces.shape[0]}")
            if debug_dir: simplified_body_trimesh.export(os.path.join(debug_dir, "graft_02_simplified_body.obj"))

            # === STEP 3: Carve Hole in Simplified Body Mesh ===
            print("\n=== STEP 3: Carving Hole in Simplified Body Mesh ===")
            # Use the bounding box of the SMPLX face, slightly scaled, to define the cut region.
            face_bounds_min, face_bounds_max = smplx_face_trimesh.bounds
            face_center = smplx_face_trimesh.centroid
            
            # Scale bounds from the center
            scaled_min_bounds = face_center - (face_center - face_bounds_min) * face_scale_for_hole_carving
            scaled_max_bounds = face_center + (face_bounds_max - face_center) * face_scale_for_hole_carving

            body_vertices = simplified_body_trimesh.vertices
            # Identify vertices within the scaled bounding box
            vertices_in_bbox_mask = (
                (body_vertices[:, 0] >= scaled_min_bounds[0]) & (body_vertices[:, 0] <= scaled_max_bounds[0]) &
                (body_vertices[:, 1] >= scaled_min_bounds[1]) & (body_vertices[:, 1] <= scaled_max_bounds[1]) &
                (body_vertices[:, 2] >= scaled_min_bounds[2]) & (body_vertices[:, 2] <= scaled_max_bounds[2])
            )
            
            # We want to REMOVE faces if ALL their vertices are inside this "carve" zone
            # faces_to_remove_mask = vertices_in_bbox_mask[simplified_body_trimesh.faces].all(axis=1) # Option 1: remove if all in
            # Or, remove faces if ANY of their vertices are inside the "carve" zone (more aggressive cut)
            faces_to_remove_mask = vertices_in_bbox_mask[simplified_body_trimesh.faces].any(axis=1) # Option 2: remove if any in

            faces_to_keep_mask = ~faces_to_remove_mask

            body_with_hole_trimesh = simplified_body_trimesh # Fallback
            if not np.any(faces_to_keep_mask) and np.any(faces_to_remove_mask): # All faces would be removed
                print("Warning: Carving hole would remove all body faces. This is likely an issue with scale or alignment.")
                print("Proceeding with the full simplified body for now, but stitching may be poor.")
            elif np.any(faces_to_remove_mask): # Only proceed if there's something to remove
                # Create a submesh with the faces to keep
                # Using trimesh.Trimesh constructor is safer for submeshing by face mask
                body_with_hole_trimesh = trimesh.Trimesh(
                    vertices=simplified_body_trimesh.vertices,
                    faces=simplified_body_trimesh.faces[faces_to_keep_mask]
                )
                body_with_hole_trimesh.remove_unreferenced_vertices() # Clean up
                body_with_hole_trimesh.fill_holes() # Attempt to make the new boundary cleaner
                print(f"Hole carved in body. Kept {np.sum(faces_to_keep_mask)} faces.")
            else:
                print("No faces identified for removal based on hole carving criteria. Using full simplified body.")

            body_with_hole_trimesh.fix_normals()
            print(f"Body with hole: V={body_with_hole_trimesh.vertices.shape[0]}, F={body_with_hole_trimesh.faces.shape[0]}")
            if debug_dir: body_with_hole_trimesh.export(os.path.join(debug_dir, "graft_03_body_with_hole.obj"))

            # === STEP 4: Concatenate SMPLX Face and Body-with-Hole ===
            print("\n=== STEP 4: Concatenating Meshes ===")
            meshes_to_combine = []
            if not smplx_face_trimesh.is_empty: meshes_to_combine.append(smplx_face_trimesh)
            if not body_with_hole_trimesh.is_empty: meshes_to_combine.append(body_with_hole_trimesh)

            if not meshes_to_combine:
                print("Fatal Error: Both face and body-with-hole are empty before concatenation.")
                return None
            
            concatenated_mesh = trimesh.util.concatenate(meshes_to_combine)
            if concatenated_mesh.is_empty:
                print("Fatal Error: Concatenated mesh is empty.")
                return None
            
            # It's crucial that normals are somewhat consistent for Poisson
            concatenated_mesh.fix_normals()
            # Merging vertices here can be problematic if there's a slight intentional gap for Poisson.
            # If they are perfectly aligned, merge_vertices is okay.
            # concatenated_mesh.merge_vertices() # Optional: merge coincident vertices
            print(f"Concatenated for Poisson: V={concatenated_mesh.vertices.shape[0]}, F={concatenated_mesh.faces.shape[0]}")
            if debug_dir: concatenated_mesh.export(os.path.join(debug_dir, "graft_04_concatenated_for_poisson.obj"))

            # === STEP 5: Final Stitching with Poisson Reconstruction ===
            print("\n=== STEP 5: Stitching with Screened Poisson Reconstruction ===")
            final_stitched_trimesh = None # Fallback
            temp_concat_for_pml = make_temp_path("_concat_pml.ply", debug_dir)
            temp_stitched_out_pml = make_temp_path("_stitched_pml.ply", debug_dir)
            try:
                concatenated_mesh.export(temp_concat_for_pml)
                ms_stitch = pymeshlab.MeshSet()
                ms_stitch.load_new_mesh(temp_concat_for_pml)

                if ms_stitch.current_mesh().face_number() > 0:
                    print(f"Applying Screened Poisson with depth: {final_stitch_poisson_depth}")
                    # Screened Poisson often works best if input has normals
                    ms_stitch.compute_normals_for_point_sets() # If it's treated as points
                    
                    ms_stitch.generate_surface_reconstruction_screened_poisson(
                        depth=int(final_stitch_poisson_depth),
                        preclean=True # Clean artifacts from concatenation before reconstruction
                        # Higher depth = more detail/closer to surface, but can also create more islands.
                        # Lower depth = smoother, better for bridging, but might lose SMPLX face detail.
                        # Consider 'samplespernode' (e.g., 1.5 or 2.0) and 'pointweight' if needing finer control
                        # For preserving the face: higher pointweight might help, but needs testing.
                    )
                    
                    # Post-Poisson cleanup is vital
                    ms_stitch.meshing_remove_connected_component_by_diameter(mincomponentdiag=pymeshlab.Percentage(70)) # Keep largest
                    ms_stitch.meshing_close_holes() # Try to ensure final watertightness
                    ms_stitch.meshing_remove_unreferenced_vertices()

                    if ms_stitch.current_mesh().face_number() == 0:
                        print("Warning: Poisson stitching resulted in an empty mesh.")
                    else:
                        ms_stitch.save_current_mesh(temp_stitched_out_pml)
                        final_stitched_trimesh = trimesh.load_mesh(temp_stitched_out_pml, process=False)
                        if final_stitched_trimesh.is_empty:
                            print("Warning: Loading stitched mesh from PyMeshLab resulted in an empty mesh.")
                            final_stitched_trimesh = None # Reset
                else:
                    print("Warning: Concatenated mesh was empty after loading into PyMeshLab for stitching.")
            except Exception as e_poisson:
                print(f"Error during Poisson stitching: {e_poisson}")
                traceback.print_exc()
            
            if final_stitched_trimesh is None:
                print("Poisson stitching failed or resulted in empty mesh. Falling back to pre-stitch concatenated mesh (will have seams).")
                final_stitched_trimesh = concatenated_mesh # Use the non-stitched version

            final_stitched_trimesh.fix_normals()
            print(f"Final stitched mesh: V={final_stitched_trimesh.vertices.shape[0]}, F={final_stitched_trimesh.faces.shape[0]}")
            if debug_dir: final_stitched_trimesh.export(os.path.join(debug_dir, "graft_05_final_stitched_mesh.obj"))

            # === STEP 6: Save Output ===
            final_stitched_trimesh.export(output_path)
            print(f"--- Mesh Grafting (Pristine Face Method) Finished. Output: {output_path} ---")
            return final_stitched_trimesh

        except Exception as e_main:
            print(f"--- Mesh Grafting (Pristine Face Method) Failed (Main Try-Except) ---")
            print(f"Error details: {e_main}")
            traceback.print_exc()
            return None
        finally:
            print("Cleaning up temporary files for grafting method...")
            for temp_path in temp_files_to_clean:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError as e_rem:
                        print(f"Warning: Could not remove temp file {temp_path}: {e_rem}")


    @staticmethod
    def process_mesh_graft_smplx_face_v2(
        full_body_mesh_path: str,
        output_path: str,
        smplx_face_mesh_path: str,
        smplx_body_for_neck_ref_path: Optional[str] = None, # Not actively used in this version for carving
        face_overlap_threshold: float = 0.01, # Critical for "Face Overlap" carving
        body_simplification_target_faces: int = 15000,
        stitch_method: str = "none", # "none", "gentle_poisson"
        gentle_poisson_depth: int = 8,
        debug_dir: str = None
    ) -> Optional[trimesh.Trimesh]:
        print(f"--- Starting Mesh Processing: Grafting SMPLX Face (V2 - Face Overlap Carving) ---")
        print(f"Full Body Input Path: {full_body_mesh_path}")
        print(f"SMPLX Face Path: {smplx_face_mesh_path}")
        print(f"Output Path: {output_path}")
        print(f"Stitch method: {stitch_method}")
        if stitch_method == "gentle_poisson":
            print(f"Gentle Poisson Depth: {gentle_poisson_depth}")
        print(f"Face Overlap Threshold for Carving: {face_overlap_threshold}") # Changed print statement
        print(f"Body Simplification Target Faces: {body_simplification_target_faces}")

        temp_files_to_clean = []

        def make_temp_path(suffix, directory):
            actual_suffix = suffix if suffix.startswith('_') else '_' + suffix
            # Use .obj for debug Trimesh exports if PyMeshLab is not involved yet, .ply if it is
            # For general temp, .ply is safer for PyMeshLab.
            _fd, path = tempfile.mkstemp(suffix=actual_suffix + ".ply", dir=directory)
            os.close(_fd)
            temp_files_to_clean.append(path)
            return path
        
        try:
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)
                print(f"Debug outputs will be saved in: {debug_dir}")

            # === STEP 1: Load Pristine SMPLX Face Mesh (_face.obj - the "mask") ===
            print("\n=== STEP 1: Loading SMPLX Face ('Mask') Mesh ===")
            original_smplx_face_geom = trimesh.load_mesh(smplx_face_mesh_path, process=False)
            if original_smplx_face_geom.is_empty:
                print(f"Fatal: SMPLX face mesh at {smplx_face_mesh_path} is empty."); return None
            if debug_dir:
                original_smplx_face_geom.export(os.path.join(debug_dir, "graftv2_OL_01_smplx_face_mask_loaded.obj"))

            # === STEP 2: Load and Simplify Full Body Mesh ===
            print("\n=== STEP 2: Loading and Simplifying Body Mesh ===")
            full_body_input_trimesh = trimesh.load_mesh(full_body_mesh_path, process=False)
            if full_body_input_trimesh.is_empty:
                print(f"Fatal: Body mesh at {full_body_mesh_path} is empty."); return None
            
            simplified_body_trimesh = full_body_input_trimesh 
            if full_body_input_trimesh.faces.shape[0] > body_simplification_target_faces:
                print(f"Simplifying body from {full_body_input_trimesh.faces.shape[0]} to target ~{body_simplification_target_faces} faces.")
                temp_body_in_pml = make_temp_path("body_simp_in", debug_dir)
                temp_body_out_pml = make_temp_path("body_simp_out", debug_dir)
                try:
                    full_body_input_trimesh.export(temp_body_in_pml)
                    ms_s = pymeshlab.MeshSet()
                    ms_s.load_new_mesh(temp_body_in_pml)
                    if ms_s.current_mesh().face_number() > 0:
                        ms_s.meshing_decimation_quadric_edge_collapse(
                            targetfacenum=int(body_simplification_target_faces), 
                            preservenormal=True, preservetopology=False )
                        ms_s.save_current_mesh(temp_body_out_pml)
                        simplified_body_trimesh_loaded = trimesh.load_mesh(temp_body_out_pml, process=False)
                        if not simplified_body_trimesh_loaded.is_empty:
                            simplified_body_trimesh = simplified_body_trimesh_loaded
                        else: print("Warning: Body simp in PML resulted in empty mesh.")
                    else: print("Warning: Body mesh empty in PML for simp.")
                except Exception as e_bsimp: print(f"Warning: Body simp failed: {e_bsimp}.")
            else: print("Body mesh already meets target. No simp performed.")
            
            simplified_body_trimesh.fix_normals() 
            if debug_dir:
                simplified_body_trimesh.export(os.path.join(debug_dir, "graftv2_OL_02_simplified_body.obj"))

            # === STEP 3: Carve Hole in Body based on SMPLX Face Surface Overlap ===
            print("\n=== STEP 3: Carving Hole in Body using Face Surface Overlap ===")
            body_with_hole_trimesh = simplified_body_trimesh # Fallback

            if original_smplx_face_geom.is_empty:
                print("Warning: SMPLX face is empty. Cannot use it for hole carving by overlap.")
            else:
                print(f"Calculating closest points from body (V={len(simplified_body_trimesh.vertices)}) to SMPLX face surface (V={len(original_smplx_face_geom.vertices)}). This may take a moment...")
                try:
                    closest_points_on_smplx_face_surf, distances_body_to_smplx_face_surf, _ = \
                        trimesh.proximity.closest_point(
                            original_smplx_face_geom, simplified_body_trimesh.vertices
                        )
                    print(f"Closest points calculation done. Using face_overlap_threshold: {face_overlap_threshold}")

                    # Remove body vertices that are very close (i.e., "covered by") the SMPLX face surface.
                    body_vertices_to_remove_mask = distances_body_to_smplx_face_surf < face_overlap_threshold

                    if not np.any(body_vertices_to_remove_mask): # Check if any vertices are marked for removal
                        print("Warning: No body vertices found close enough to SMPLX face surface to be removed.")
                        print("Consider checking alignment of face and body, or increasing 'face_overlap_threshold'.")
                    else:
                        print(f"Marking {np.sum(body_vertices_to_remove_mask)} body vertices for potential removal.")
                    
                    # Remove faces if ANY of their vertices are marked for removal
                    body_faces_to_remove_mask = body_vertices_to_remove_mask[simplified_body_trimesh.faces].any(axis=1)
                
                    if np.any(body_faces_to_remove_mask):
                        body_faces_to_keep_mask = ~body_faces_to_remove_mask
                        print(f"Identified {np.sum(body_faces_to_remove_mask)} body faces for removal based on overlap.")
                        body_with_hole_trimesh = trimesh.Trimesh(
                            vertices=simplified_body_trimesh.vertices, 
                            faces=simplified_body_trimesh.faces[body_faces_to_keep_mask]
                        )
                        body_with_hole_trimesh.remove_unreferenced_vertices()
                        body_with_hole_trimesh.fix_normals() # Fix normals after creating the hole
                    else:
                        print("Warning: No body faces ultimately marked for removal based on overlap. Body mesh remains unholed.")
                except Exception as e_prox:
                    print(f"Error during proximity query for hole carving: {e_prox}")
                    traceback.print_exc()
                    print("Skipping hole carving due to error.")

            if debug_dir:
                body_with_hole_trimesh.export(os.path.join(debug_dir, "graftv2_OL_03_body_with_overlap_hole.obj"))

            # === STEP 4: Combine Face ("Mask") and Body-with-Hole (Geometry Only) ===
            # (This step is identical to previous version)
            print("\n=== STEP 4: Combining Meshes (Geometry Preservation Focus) ===")
            meshes_to_concat = []
            if original_smplx_face_geom and not original_smplx_face_geom.is_empty:
                meshes_to_concat.append(original_smplx_face_geom) 
            if body_with_hole_trimesh and not body_with_hole_trimesh.is_empty:
                 meshes_to_concat.append(body_with_hole_trimesh)
            if not meshes_to_concat:
                print("Fatal Error: No valid meshes to concatenate."); return None
            final_mesh_concatenated_geom_only = trimesh.util.concatenate(meshes_to_concat)
            if final_mesh_concatenated_geom_only.is_empty:
                print("Fatal Error: Concatenation resulted in empty mesh."); return None
            if debug_dir:
                 final_mesh_concatenated_geom_only.export(os.path.join(debug_dir, "graftv2_OL_04a_concatenated_geom_only.obj"))
            final_mesh_output = final_mesh_concatenated_geom_only.copy()
            final_mesh_output.fix_normals() 
            if debug_dir:
                final_mesh_output.export(os.path.join(debug_dir, "graftv2_OL_04b_concatenated_with_fixed_normals.obj"))

            # === STEP 5: Stitching (Conditional) ===
            # This logic now applies to a mesh where the face is a mask, and the body has a hole under it.
            # Poisson will try to close ALL open boundaries: neck, eyes, mouth of the mask, and the hole boundary.
            if stitch_method == "gentle_poisson":
                print("\n=== STEP 5: Stitching with Gentle Poisson (will affect all open boundaries) ===")
                # (Poisson logic remains the same as previous version)
                temp_for_gentle_poisson = make_temp_path("gentle_poisson_in_ol", debug_dir) # OL for OverLap
                temp_after_gentle_poisson = make_temp_path("gentle_poisson_out_ol", debug_dir)
                final_mesh_output.export(temp_for_gentle_poisson)
                
                ms_gp = pymeshlab.MeshSet()
                ms_gp.load_new_mesh(temp_for_gentle_poisson)
                if ms_gp.current_mesh().face_number() > 0:
                    try:
                        print(f"Applying Screened Poisson with depth: {gentle_poisson_depth}")
                        ms_gp.generate_surface_reconstruction_screened_poisson(
                            depth=int(gentle_poisson_depth), preclean=True )
                        ms_gp.meshing_remove_connected_component_by_diameter(mincomponentdiag=pymeshlab.Percentage(70))
                        ms_gp.meshing_close_holes() 
                        ms_gp.meshing_remove_unreferenced_vertices()
                        if ms_gp.current_mesh().face_number() > 0:
                            ms_gp.save_current_mesh(temp_after_gentle_poisson)
                            final_mesh_loaded_from_poisson = trimesh.load_mesh(temp_after_gentle_poisson, process=False)
                            if not final_mesh_loaded_from_poisson.is_empty:
                                final_mesh_output = final_mesh_loaded_from_poisson 
                                print("Gentle Poisson stitching applied.")
                            else: print("Warning: Gentle Poisson resulted in empty mesh (loaded back).")
                        else: print("Warning: Gentle Poisson resulted in empty mesh (in PML).")
                    except Exception as e_gp: print(f"Error during gentle Poisson: {e_gp}."); traceback.print_exc()
                else: print("Mesh empty before gentle Poisson.")
                if debug_dir:
                    final_mesh_output.export(os.path.join(debug_dir, "graftv2_OL_05_after_gentle_poisson_block.obj"))
            elif stitch_method == "none":
                print("\n=== STEP 5: No Stitching Applied (stitch_method='none') ===")
                pass # final_mesh_output is already set to concatenated with fixed normals
            else:
                print(f"Warning: Unknown stitch_method '{stitch_method}'.")

            final_mesh_output.fix_normals() # Final fix before saving

            # === STEP 6: Final Output ===
            print(f"\n=== STEP 6: Saving Final Output Mesh ===")
            print(f"Final mesh V={len(final_mesh_output.vertices)}, F={len(final_mesh_output.faces)}")
            final_mesh_output.export(output_path)
            print(f"--- Mesh Grafting V2 (Face Overlap Carving) Finished. Output: {output_path} ---")
            return final_mesh_output

        except Exception as e_main_v2:
            print(f"--- Mesh Grafting V2 (Face Overlap Carving) Failed ---"); print(f"Error: {e_main_v2}"); traceback.print_exc(); return None
        finally:
            print("Cleaning up temporary files for grafting method v2 (OL)...")
            # (temp file cleanup logic as before)
            for temp_path in temp_files_to_clean:
                if temp_path and os.path.exists(temp_path): 
                    try: os.remove(temp_path)
                    except OSError as e_rem: print(f"Warning: Could not remove temp file {temp_path}: {e_rem}")

    @staticmethod
    def get_boundary_edges_manual(mesh: trimesh.Trimesh) -> Optional[np.ndarray]:
        # print("DEBUG: Entered get_boundary_edges_manual (v3 - understanding group_rows with require_count=1)")
        if not (mesh and not mesh.is_empty and \
                hasattr(mesh, 'vertices') and mesh.vertices is not None and len(mesh.vertices) > 0 and \
                hasattr(mesh, 'faces') and mesh.faces is not None and len(mesh.faces) > 0 and \
                hasattr(mesh, 'edges_sorted')):
            print("DEBUG: get_boundary_edges_manual - Mesh is invalid or missing attributes.")
            return None
        
        try:
            sorted_edges = mesh.edges_sorted # This is an (M, 2) array of unique edges, sorted internally
            
            if sorted_edges is None or len(sorted_edges) == 0:
                print("DEBUG: get_boundary_edges_manual - mesh.edges_sorted is empty or None.")
                return np.array([], dtype=int).reshape(-1,2)

            # print(f"DEBUG: get_boundary_edges_manual - Processing {len(sorted_edges)} edges from mesh.edges_sorted.")

            # To find boundary edges, we need to know how many faces each edge belongs to.
            # mesh.edges_sorted gives *unique* edges. We need to count occurrences based on faces.

            # ----- THIS IS THE CORRECT MANUAL METHOD USING FACE EDGES AND COUNTER -----
            # (The one from my very last response was correct for manual counting if group_rows was an issue)
            all_face_edges_canonical = []
            for face in mesh.faces:
                all_face_edges_canonical.append(tuple(sorted((face[0], face[1]))))
                all_face_edges_canonical.append(tuple(sorted((face[1], face[2]))))
                all_face_edges_canonical.append(tuple(sorted((face[2], face[0]))))
            
            if not all_face_edges_canonical:
                print("DEBUG: get_boundary_edges_manual - No edges derived from faces.")
                return np.array([], dtype=int).reshape(-1,2)

            edge_counts = Counter(all_face_edges_canonical)
            
            manual_boundary_edges_list = []
            for edge_tuple, count in edge_counts.items():
                if count == 1: # Boundary edges are shared by only one face
                    manual_boundary_edges_list.append(list(edge_tuple))
            
            if not manual_boundary_edges_list:
                # print("DEBUG: get_boundary_edges_manual - No boundary edges found (all edges shared by 2 faces or counts not 1).")
                return np.array([], dtype=int).reshape(-1,2)
                
            manual_boundary_edges = np.array(manual_boundary_edges_list, dtype=int)
            # print(f"DEBUG: get_boundary_edges_manual - Found {len(manual_boundary_edges)} boundary edges manually via face edge counting.")
            return manual_boundary_edges
            # ----- END OF CORRECT MANUAL METHOD -----

        except Exception as e:
            print(f"DEBUG: Error in get_boundary_edges_manual: {e}")
            traceback.print_exc()
            return None
        
    @staticmethod
    def _get_boundary_edges_manually_from_faces(mesh: trimesh.Trimesh) -> Optional[np.ndarray]:
        # --- This is the CORRECTED version using collections.Counter on all face edges ---
        # print("DEBUG: Entered _get_boundary_edges_manually_from_faces")
        if not (mesh and not mesh.is_empty and \
                hasattr(mesh, 'vertices') and mesh.vertices is not None and len(mesh.vertices) > 0 and \
                hasattr(mesh, 'faces') and mesh.faces is not None and len(mesh.faces) > 0 ):
            print("DEBUG: _get_boundary_edges_manually_from_faces - Mesh is invalid or missing attributes.")
            return None 
        
        try:
            all_face_edges_canonical = []
            for face_idx, face in enumerate(mesh.faces):
                if len(face) != 3: continue # Skip non-triangular if any
                all_face_edges_canonical.append(tuple(sorted((face[0], face[1]))))
                all_face_edges_canonical.append(tuple(sorted((face[1], face[2]))))
                all_face_edges_canonical.append(tuple(sorted((face[2], face[0]))))
            
            if not all_face_edges_canonical:
                print("DEBUG: _get_boundary_edges_manually_from_faces - No edges derived from faces.")
                return np.array([], dtype=int).reshape(-1,2)

            edge_counts = Counter(all_face_edges_canonical)
            manual_boundary_edges_list = []
            for edge_tuple, count in edge_counts.items():
                if count == 1: 
                    manual_boundary_edges_list.append(list(edge_tuple))
            
            if not manual_boundary_edges_list:
                return np.array([], dtype=int).reshape(-1,2)
            return np.array(manual_boundary_edges_list, dtype=int)
        except Exception as e:
            print(f"DEBUG: Error in _get_boundary_edges_manually_from_faces: {e}"); traceback.print_exc(); return None

    @staticmethod
    def get_outermost_boundary_loop_vertices(mesh: trimesh.Trimesh, strategy="longest_loop", mesh_name_for_debug="mesh") -> np.ndarray:
        print(f"DEBUG {mesh_name_for_debug}: Entered get_outermost_boundary_loop_vertices")
        if not (mesh and not mesh.is_empty and hasattr(mesh,'vertices') and mesh.vertices is not None and \
                len(mesh.vertices)>0 and hasattr(mesh,'faces') and mesh.faces is not None and len(mesh.faces)>0):
            print(f"DEBUG {mesh_name_for_debug}: Input mesh empty/invalid.")
            return np.array([],dtype=int)
        
        boundary_edges = None
        try:
            if hasattr(mesh, 'edges_unique_boundary'): boundary_edges = mesh.edges_unique_boundary
            else: boundary_edges = MeshCleanProcess._get_boundary_edges_manually_from_faces(mesh)
            if boundary_edges is None or len(boundary_edges)==0: print(f"DEBUG {mesh_name_for_debug}: No boundary edges found."); return np.array([],dtype=int)
            
            all_loop_vidx_sets = trimesh.graph.connected_components(boundary_edges, min_len=3)
            candidate_loops_unique_vidx_sets = []
            if all_loop_vidx_sets and any(c is not None and len(c)>0 for c in all_loop_vidx_sets):
                for loop_v_comp in all_loop_vidx_sets:
                    if loop_v_comp is not None and len(loop_v_comp)>2:
                        candidate_loops_unique_vidx_sets.append(loop_v_comp.astype(int))
            if not candidate_loops_unique_vidx_sets: print(f"DEBUG {mesh_name_for_debug}: No valid candidate loops from components."); return np.array([],dtype=int)
            
            sel_loop_unord_vidx = np.array([],dtype=int)
            # ... (Strategy selection for sel_loop_unord_vidx as before) ...
            if strategy=="longest_loop": sel_loop_unord_vidx=max(candidate_loops_unique_vidx_sets,key=len,default=np.array([],dtype=int))
            elif strategy=="lowest_z_loop": # ... (lowest_z_loop logic)
                def get_avg_z(v_idx):
                    if len(v_idx)==0: return float('inf')
                    valid_v=v_idx[v_idx<len(mesh.vertices)]; 
                    if len(valid_v)==0:return float('inf')
                    return np.mean(mesh.vertices[valid_v,2])
                valid_c=[l for l in candidate_loops_unique_vidx_sets if len(l)>0]
                if valid_c:sel_loop_unord_vidx=min(valid_c,key=get_avg_z,default=np.array([],dtype=int))
            else: sel_loop_unord_vidx=max(candidate_loops_unique_vidx_sets,key=len,default=np.array([],dtype=int))


            if len(sel_loop_unord_vidx)<3: print(f"DEBUG {mesh_name_for_debug}: Selected loop too short ({len(sel_loop_unord_vidx)}V)."); return np.array([],dtype=int)
            
            print(f"DEBUG {mesh_name_for_debug}: Selected UNORDERED loop has {len(sel_loop_unord_vidx)} vertices. Attempting to order...")
            
            # --- ORDER THE SELECTED LOOP ---
            # Filter edges to only those connecting vertices within the selected loop component
            sel_loop_set = set(sel_loop_unord_vidx)
            edges_for_selected_loop_list = [edge for edge in boundary_edges if edge[0] in sel_loop_set and edge[1] in sel_loop_set]

            if not edges_for_selected_loop_list or len(edges_for_selected_loop_list) < len(sel_loop_unord_vidx) - 1:
                print(f"DEBUG {mesh_name_for_debug}: Not enough edges ({len(edges_for_selected_loop_list)}) for selected loop vertices ({len(sel_loop_unord_vidx)}). Returning UNORDERED.")
                return sel_loop_unord_vidx 

            edges_for_selected_loop = np.array(edges_for_selected_loop_list, dtype=int)
            
            ordered_vidx_path_final = np.array([], dtype=int)
            # Attempt 1: trimesh.graph.traverse_edges (if available)
            if hasattr(trimesh.graph, 'traverse_edges'):
                # print(f"DEBUG {mesh_name_for_debug}: Attempting trimesh.graph.traverse_edges...")
                try:
                    start_node_for_traverse = sel_loop_unord_vidx[0] # Start from any vertex in the component
                    path_from_traverse = trimesh.graph.traverse_edges(edges_for_selected_loop, start_node_for_traverse)
                    
                    # Validate traversal result
                    if path_from_traverse is not None and len(path_from_traverse) >= 3 and \
                       len(np.unique(path_from_traverse)) >= len(sel_loop_unord_vidx) * 0.8: # Contains most unique verts
                        # print(f"DEBUG {mesh_name_for_debug}: traverse_edges path length: {len(path_from_traverse)}.")
                        if len(path_from_traverse) > 1 and path_from_traverse[0] == path_from_traverse[-1]: # Is it a closed path?
                            ordered_vidx_path_final = np.array(path_from_traverse[:-1], dtype=int) # Make open for resampling
                        else:
                            ordered_vidx_path_final = np.array(path_from_traverse, dtype=int)
                    else:
                        print(f"DEBUG {mesh_name_for_debug}: traverse_edges result unsatisfactory. Will try manual ordering.")
                except AttributeError: # traverse_edges doesn't exist
                    print(f"DEBUG {mesh_name_for_debug}: 'trimesh.graph.traverse_edges' not found (AttributeError). Will use manual ordering.")
                except Exception as e_traverse_init:
                    print(f"DEBUG {mesh_name_for_debug}: Exception during trimesh.graph.traverse_edges: {e_traverse_init}. Will use manual ordering.")
            else:
                 print(f"DEBUG {mesh_name_for_debug}: 'trimesh.graph.traverse_edges' not available. Will use manual ordering.")

            # Attempt 2: Manual Ordering (if traverse_edges failed or produced a bad result)
            if len(ordered_vidx_path_final) < 3: # If traverse_edges didn't yield a good path
                print(f"DEBUG {mesh_name_for_debug}: Attempting MANUAL ordering...")
                # Build adjacency list for ONLY the vertices and edges in the selected loop
                adj = {v: [] for v in sel_loop_unord_vidx}
                for u_node, v_node in edges_for_selected_loop:
                    adj[u_node].append(v_node)
                    adj[v_node].append(u_node)

                start_node = sel_loop_unord_vidx[0]
                curr = start_node
                # prev = -1 # Using None is safer for checks
                prev = None
                manual_path = [curr]
                
                # Max iterations: number of vertices in the loop. A simple cycle visits each vertex once.
                for _iter_count in range(len(sel_loop_unord_vidx)): 
                    if len(manual_path) == len(sel_loop_unord_vidx): # Found all vertices for an open path
                        # Check if it can close to start_node (excluding direct backtrack)
                        if start_node in adj.get(curr,[]) and start_node != prev:
                             # print(f"DEBUG {mesh_name_for_debug}: Manual path seems complete and can close.")
                            pass # Path is complete and open
                        break 

                    found_next = False
                    # Prioritize neighbors not already in the path (excluding prev for next step choice)
                    # but current `prev` is for the *previous edge*. We need to ensure we don't add `curr`'s direct predecessor.
                    
                    # Get neighbors of `curr`, excluding `prev`
                    possible_next_nodes = [n for n in adj.get(curr, []) if n != prev]
                    
                    # Prefer unvisited nodes among these
                    actual_next_node = None
                    for next_node_candidate in possible_next_nodes:
                        if next_node_candidate not in manual_path:
                            actual_next_node = next_node_candidate
                            break
                    
                    if actual_next_node is None and possible_next_nodes: # All neighbors (excluding prev) are already in path
                        # This can happen if we are about to close the loop OR if path is stuck
                        if possible_next_nodes[0] == start_node and len(manual_path) == len(sel_loop_unord_vidx) -1 : # Trying to close to start
                            actual_next_node = possible_next_nodes[0]
                        # else:
                            # print(f"DEBUG {mesh_name_for_debug}: Manual ordering: all valid next steps for {curr} are already in path {manual_path}. Path might be stuck or short.")
                            # break # Path is stuck before visiting all nodes
                    
                    if actual_next_node is not None:
                        manual_path.append(actual_next_node)
                        prev = curr
                        curr = actual_next_node
                        found_next = True
                    else: # No valid next node found
                        # print(f"DEBUG {mesh_name_for_debug}: Manual ordering dead end at vertex {curr} after path {manual_path}.")
                        break
                
                # Validate the manually constructed path
                if len(manual_path) >= 3 and len(np.unique(manual_path)) >= len(sel_loop_unord_vidx) * 0.8:
                    # print(f"DEBUG {mesh_name_for_debug}: Manual ordering produced path of length {len(manual_path)}.")
                    ordered_vidx_path_final = np.array(manual_path, dtype=int)
                else:
                    print(f"DEBUG {mesh_name_for_debug}: Manual ordering failed or path too short (len {len(manual_path)}).")
                    # ordered_vidx_path_final remains as set by traverse_edges (or empty)

            # Final Decision
            if len(ordered_vidx_path_final) >= 3 :
                # print(f"DEBUG {mesh_name_for_debug}: Successfully obtained ORDERED loop of {len(ordered_vidx_path_final)} vertices.")
                return ordered_vidx_path_final # This is an OPEN path
            else:
                print(f"DEBUG {mesh_name_for_debug}: All ordering attempts failed. Returning UNORDERED loop (len {len(sel_loop_unord_vidx)}).")
                return sel_loop_unord_vidx 

        except Exception as e: 
            print(f"Crit Error get_outermost ({mesh_name_for_debug}): {e}"); traceback.print_exc(); return np.array([], dtype=int)
        
    @staticmethod
    def resample_polyline_to_count(polyline_vertices: np.ndarray, target_count: int) -> Optional[np.ndarray]:
        # --- Assumed to be correctly implemented from before ---
        if polyline_vertices is None or len(polyline_vertices) < 2: return polyline_vertices
        if target_count < 2: return polyline_vertices[:target_count] if target_count > 0 else np.array([])
        if len(polyline_vertices) == target_count: return polyline_vertices

        distances = np.linalg.norm(np.diff(polyline_vertices, axis=0), axis=1)
        cumulative_distances = np.concatenate(([0], np.cumsum(distances)))
        total_length = cumulative_distances[-1]

        if total_length == 0: return np.tile(polyline_vertices[0], (target_count, 1))

        sampled_arc_lengths = np.linspace(0, total_length, target_count)
        resampled_points = []
        for s_len in sampled_arc_lengths:
            idx = np.searchsorted(cumulative_distances, s_len, side='right') -1
            idx = np.clip(idx, 0, len(polyline_vertices) - 2)
            p0, p1 = polyline_vertices[idx], polyline_vertices[idx+1]
            seg_len = cumulative_distances[idx+1] - cumulative_distances[idx]
            t = (s_len - cumulative_distances[idx]) / seg_len if seg_len > 1e-9 else 0.0 # Avoid div by zero
            resampled_points.append(p0 + np.clip(t,0,1) * (p1 - p0))
        return np.array(resampled_points) if resampled_points else None
                            
    @staticmethod
    def process_mesh_graft_smplx_face_v3( # Internally, this is becoming v3.4
        full_body_mesh_path: str,
        output_path: str,
        smplx_face_mesh_path: str,
        outer_boundary_strategy: str = "longest_loop",
        cut_expansion_offset: float = 0.01, # Used as radius in contains_points
        body_simplification_target_faces: int = 15000,
        stitch_method: str = "none",
        gentle_poisson_depth: int = 8,
        projection_plane_normal_config: Optional[Union[str, list, tuple, np.ndarray]] = None,
        front_face_dot_product_positive: bool = True,
        # New parameter to define the "forward" direction for orienting the projection plane normal
        expected_forward_vector: Optional[np.ndarray] = None, # e.g., np.array([0,1,0]) if +Y is forward
        debug_dir: str = None
    ) -> Optional[trimesh.Trimesh]:
        # Using v3.4 for internal debug print clarity
        print(f"--- Starting Mesh Processing: Grafting SMPLX Face (V3.4 - Projection & Artifact Refinement) ---")
        # ... (Initial prints, temp file setup) ...
        temp_files_to_clean = [] # Initialize the list first

        # Define make_temp_path as a proper nested function
        def make_temp_path(suffix: str, directory: Optional[str]) -> str:
            actual_suffix = suffix if suffix.startswith('_') else '_' + suffix
            _fd, path = tempfile.mkstemp(suffix=actual_suffix + ".ply", dir=directory) 
            os.close(_fd)
            temp_files_to_clean.append(path)
            return path
        
        from trimesh import util

        try:
            if debug_dir: os.makedirs(debug_dir, exist_ok=True)

            # === STEP 1 & 2: Load Meshes & Simplify Body ===
            # ... (No changes from v3.3, just update debug filenames if desired to v3.4) ...
            print("\n=== STEP 1: Loading SMPLX Face ('Mask') Mesh ===")
            original_smplx_face_geom = trimesh.load_mesh(smplx_face_mesh_path, process=False)
            if original_smplx_face_geom.is_empty: print(f"Fatal: SMPLX face empty."); return None
            if debug_dir: original_smplx_face_geom.export(os.path.join(debug_dir, "graftv34_01_smplx_mask_loaded.obj"))

            print("\n=== STEP 2: Loading and Simplifying Body Mesh ===")
            full_body_input_trimesh = trimesh.load_mesh(full_body_mesh_path, process=False)
            if full_body_input_trimesh.is_empty: print(f"Fatal: Body mesh empty."); return None
            simplified_body_trimesh = full_body_input_trimesh 
            if full_body_input_trimesh.faces.shape[0] > body_simplification_target_faces:
                print(f"Simplifying body to target {body_simplification_target_faces} faces.")
                # ... (Simplification logic using PyMeshLab as before) ...
                temp_body_in_pml = make_temp_path("body_v34_simp_in", debug_dir); temp_body_out_pml = make_temp_path("body_v34_simp_out", debug_dir)
                try:
                    full_body_input_trimesh.export(temp_body_in_pml)
                    ms_s = pymeshlab.MeshSet(); ms_s.load_new_mesh(temp_body_in_pml)
                    if ms_s.current_mesh().face_number() > 0:
                        ms_s.meshing_decimation_quadric_edge_collapse(targetfacenum=body_simplification_target_faces, preservenormal=True)
                        ms_s.save_current_mesh(temp_body_out_pml)
                        simplified_body_trimesh_loaded = trimesh.load_mesh(temp_body_out_pml, process=False)
                        if not simplified_body_trimesh_loaded.is_empty: simplified_body_trimesh = simplified_body_trimesh_loaded
                except Exception as e_bsimp_v34: print(f"Warning: Body simp v3.4 failed {e_bsimp_v34}")
            if debug_dir: simplified_body_trimesh.export(os.path.join(debug_dir, "graftv34_02_simplified_body.obj"))


            # === STEP 3: Define Hole in Body based on SMPLX Mask's "Outer Boundary" ===
            print(f"\n=== STEP 3: Defining Hole based on Mask's Outer Boundary ===")
            smplx_mask_outer_boundary_vidx = MeshCleanProcess.get_outermost_boundary_loop_vertices(
                original_smplx_face_geom, strategy=outer_boundary_strategy )
            body_with_hole_trimesh = simplified_body_trimesh.copy() 

            if len(smplx_mask_outer_boundary_vidx) < 3:
                print("Warning: Could not identify suitable mask outer boundary. Skipping hole carving.")
            else:
                smplx_mask_outer_boundary_verts_3d = np.array([])
                if smplx_mask_outer_boundary_vidx.max() < len(original_smplx_face_geom.vertices) and smplx_mask_outer_boundary_vidx.min() >=0:
                    smplx_mask_outer_boundary_verts_3d = original_smplx_face_geom.vertices[smplx_mask_outer_boundary_vidx]
                # ... (Debug export for 03a with enhanced checks from v3.3 - assumed present) ...
                if debug_dir:
                    if smplx_mask_outer_boundary_verts_3d.shape[0] > 0 and smplx_mask_outer_boundary_verts_3d.ndim == 2 and smplx_mask_outer_boundary_verts_3d.shape[1] == 3:
                        trimesh.points.PointCloud(smplx_mask_outer_boundary_verts_3d).export(os.path.join(debug_dir, f"graftv34_03a_smplx_mask_outer_boundary_pts.obj"))
                
                if smplx_mask_outer_boundary_verts_3d.shape[0] < 3:
                    print("Warning: smplx_mask_outer_boundary_verts_3d has < 3 points. Skipping hole carving.")
                else:
                    kdtree_body_verts = cKDTree(simplified_body_trimesh.vertices)
                    _, closest_body_vidx_to_smplx_boundary = kdtree_body_verts.query(smplx_mask_outer_boundary_verts_3d, workers=-1)
                    mapped_body_boundary_vidx = np.unique(closest_body_vidx_to_smplx_boundary)

                    if len(mapped_body_boundary_vidx) < 3:
                        print("Warning: Mapped body boundary < 3 unique Vidx. Skipping hole carving.")
                    else:
                        mapped_body_boundary_verts_3d = simplified_body_trimesh.vertices[mapped_body_boundary_vidx]
                        # ... (Debug export for 03b with enhanced checks - assumed present) ...
                        if debug_dir:
                            if mapped_body_boundary_verts_3d.shape[0] > 0 and mapped_body_boundary_verts_3d.ndim == 2 and mapped_body_boundary_verts_3d.shape[1] == 3:
                                trimesh.points.PointCloud(mapped_body_boundary_verts_3d).export(os.path.join(debug_dir, "graftv34_03b_mapped_body_boundary_pts.obj"))

                        # Determine projection plane normal
                        plane_normal_to_use, plane_origin_to_use = np.array([0,0,1.0]), np.array([0,0,0.0])
                        # ... (Projection plane determination logic as in v3.3 - assumed present) ...
                        if projection_plane_normal_config is None: 
                            plane_normal_to_use, plane_origin_to_use = MeshCleanProcess.get_dominant_plane(mapped_body_boundary_verts_3d)
                            # ** NEW: Ensure consistent orientation for auto-detected normal **
                            if expected_forward_vector is not None:
                                if np.dot(plane_normal_to_use, expected_forward_vector) < 0:
                                    plane_normal_to_use = -plane_normal_to_use # Flip it
                                    print(f"DEBUG Step3: Auto-detected plane normal was flipped to align with expected_forward_vector. New normal: {plane_normal_to_use}")
                        elif isinstance(projection_plane_normal_config, str): # ... (handle "XY", "XZ", "YZ")
                            plane_origin_to_use = np.mean(mapped_body_boundary_verts_3d, axis=0)
                            if projection_plane_normal_config.upper() == 'XY': plane_normal_to_use = np.array([0,0,1.0])
                            elif projection_plane_normal_config.upper() == 'XZ': plane_normal_to_use = np.array([0,1.0,0])
                            elif projection_plane_normal_config.upper() == 'YZ': plane_normal_to_use = np.array([1.0,0,0])
                            else: plane_normal_to_use = np.array([0,0,1.0])
                        elif isinstance(projection_plane_normal_config, (list, tuple, np.ndarray)): # ... (handle custom normal array)
                             try:
                                plane_normal_to_use_arr = np.array(projection_plane_normal_config, dtype=float)
                                if plane_normal_to_use_arr.shape == (3,):
                                    plane_normal_to_use = util.unitize(plane_normal_to_use_arr); plane_origin_to_use = np.mean(mapped_body_boundary_verts_3d, axis=0)
                                else: plane_normal_to_use, plane_origin_to_use = MeshCleanProcess.get_dominant_plane(mapped_body_boundary_verts_3d)
                             except ValueError: plane_normal_to_use, plane_origin_to_use = MeshCleanProcess.get_dominant_plane(mapped_body_boundary_verts_3d)
                        else: plane_normal_to_use, plane_origin_to_use = MeshCleanProcess.get_dominant_plane(mapped_body_boundary_verts_3d)

                        # print(f"DEBUG Step3: Using final projection normal: {plane_normal_to_use}, origin: {plane_origin_to_use}")
                        
                        # Transform points to 2D plane (same as v3.3)
                        transform_to_2d_plane = trimesh.geometry.plane_transform(plane_origin_to_use, plane_normal_to_use)
                        transform_from_2d_plane = np.linalg.inv(transform_to_2d_plane)
                        mapped_body_boundary_verts_proj_3d = trimesh.transform_points(mapped_body_boundary_verts_3d, transform_to_2d_plane)
                        body_face_centroids_3d = simplified_body_trimesh.triangles_center
                        body_face_centroids_proj_3d = trimesh.transform_points(body_face_centroids_3d, transform_to_2d_plane)
                        mapped_body_boundary_pts_2d = mapped_body_boundary_verts_proj_3d[:, :2]
                        body_face_centroids_2d = body_face_centroids_proj_3d[:, :2]

                        if len(np.unique(mapped_body_boundary_pts_2d, axis=0)) < 3:
                            print("Warning: Not enough unique 2D points for Convex Hull after projection.")
                        else:
                            try:
                                hull = ConvexHull(mapped_body_boundary_pts_2d)
                                cutting_polygon_2d_convex_hull = mapped_body_boundary_pts_2d[hull.vertices]
                                
                                # ** NEW: Attempt to use the original mapped boundary points if convex hull significantly reduces vertex count **
                                # This might help with concavities better than just the hull.
                                # The polygon must be simple (not self-intersecting) for MplPath.
                                # Ordering them is the key. `trimesh.path.simplify.simplify_bidirectional` can help order a 2D path.
                                # For now, we'll use a simple heuristic: if the hull has far fewer points than the mapped boundary,
                                # it might be oversimplifying.
                                if len(cutting_polygon_2d_convex_hull) < len(mapped_body_boundary_pts_2d) * 0.5 and len(mapped_body_boundary_pts_2d) > 10 : # Heuristic
                                    print("DEBUG Step3: Convex hull significantly simplified the boundary. Attempting to use more original points.")
                                    # Try to order the original mapped_body_boundary_pts_2d
                                    # This is complex. For now, still defaulting to convex hull for MplPath simplicity.
                                    # A robust ordering algorithm would be needed here.
                                    # If shapely is available: from shapely.geometry import MultiPoint, Polygon; ordered_pts = np.array(MultiPoint(mapped_body_boundary_pts_2d).convex_hull.exterior.coords)
                                    # For now, we stick to ConvexHull for the cutting_polygon_2d_for_test for simplicity.
                                    print("DEBUG Step3: Sticking to Convex Hull for MplPath. For more detail, advanced path ordering needed.")
                                
                                cutting_polygon_2d_for_test = cutting_polygon_2d_convex_hull
                                # (Offsetting is still an issue without trimesh.path.polygons, radius in contains_points is the workaround)

                                if debug_dir and len(cutting_polygon_2d_for_test) > 0:
                                    # ... (Debug export 03c as PointCloud with enhanced checks from v3.3 - assumed present) ...
                                    cutting_polygon_3d_vis = np.hstack((cutting_polygon_2d_for_test, np.full((len(cutting_polygon_2d_for_test), 1), np.mean(mapped_body_boundary_verts_proj_3d[:,2])) ))
                                    cutting_polygon_3d_vis = trimesh.transform_points(cutting_polygon_3d_vis, transform_from_2d_plane)
                                    if cutting_polygon_3d_vis.ndim == 2 and cutting_polygon_3d_vis.shape[1] == 3:
                                         trimesh.points.PointCloud(cutting_polygon_3d_vis).export(os.path.join(debug_dir, "graftv34_03c_cutting_polygon_3d_pts.obj"))
                                
                                if len(cutting_polygon_2d_for_test) > 2 and len(body_face_centroids_2d) > 0:
                                    mpl_path_obj = MplPath(cutting_polygon_2d_for_test) # Vertices of polygon
                                    centroids_inside_2d_polygon_mask = mpl_path_obj.contains_points(
                                        body_face_centroids_2d, 
                                        radius=cut_expansion_offset # Positive makes it "larger", negative "smaller"
                                    )
                                    
                                    body_face_normals_3d = simplified_body_trimesh.face_normals
                                    dot_products = np.einsum('ij,j->i', body_face_normals_3d, plane_normal_to_use)
                                    
                                    if front_face_dot_product_positive:
                                        front_facing_mask = dot_products > 0.0 
                                    else:
                                        front_facing_mask = dot_products < 0.0 
                                    
                                    faces_to_remove_combined_mask = centroids_inside_2d_polygon_mask & front_facing_mask
                                    num_faces_to_remove = np.sum(faces_to_remove_combined_mask)

                                    if num_faces_to_remove > 0:
                                        print(f"Identified {num_faces_to_remove} body faces (inside polygon & 'front-facing') for removal.")
                                        faces_to_keep_mask = ~faces_to_remove_combined_mask
                                        body_with_hole_trimesh = trimesh.Trimesh(
                                            vertices=simplified_body_trimesh.vertices,
                                            faces=simplified_body_trimesh.faces[faces_to_keep_mask] )
                                        body_with_hole_trimesh.remove_unreferenced_vertices()
                                        body_with_hole_trimesh.fix_normals()
                                    else:
                                        print("Warning: No body faces matched all removal criteria.")
                                # ... (else for invalid cutting polygon/centroids)
                            except Exception as e_hull_poly:
                                print(f"Error during Convex Hull or point-in-polygon (Matplotlib): {e_hull_poly}")
                                traceback.print_exc()
            if debug_dir: body_with_hole_trimesh.export(os.path.join(debug_dir, "graftv34_03d_body_with_final_hole.obj"))

            # === STEP 4, 5, 6: Concatenate, Stitch (Optional), Save ===
            # ... (Identical to v3.3, use "graftv34_" for debug filenames) ...
            print("\n=== STEP 4: Combining Meshes ===") # ... (Concatenation logic as before)
            meshes_to_concat = [];
            if original_smplx_face_geom and not original_smplx_face_geom.is_empty: meshes_to_concat.append(original_smplx_face_geom) 
            if body_with_hole_trimesh and not body_with_hole_trimesh.is_empty: meshes_to_concat.append(body_with_hole_trimesh)
            if not meshes_to_concat: print("Fatal: No meshes to concat v3.4."); return None
            final_mesh_concatenated_geom_only = trimesh.util.concatenate(meshes_to_concat)
            if final_mesh_concatenated_geom_only.is_empty: print("Fatal: Concat empty v3.4."); return None
            if debug_dir: final_mesh_concatenated_geom_only.export(os.path.join(debug_dir, "graftv34_04a_concatenated_geom_only.obj"))
            final_mesh_output = final_mesh_concatenated_geom_only.copy(); final_mesh_output.fix_normals() 
            if debug_dir: final_mesh_output.export(os.path.join(debug_dir, "graftv34_04b_concatenated_with_fixed_normals.obj"))

            if stitch_method == "gentle_poisson": # ... (Poisson logic as before)
                print("\n=== STEP 5: Stitching with Gentle Poisson ===")
                temp_for_gp_v34 = make_temp_path("gp_in_v34", debug_dir); temp_after_gp_v34 = make_temp_path("gp_out_v34", debug_dir)
                final_mesh_output.export(temp_for_gp_v34)
                ms_gp_v34 = pymeshlab.MeshSet(); ms_gp_v34.load_new_mesh(temp_for_gp_v34)
                if ms_gp_v34.current_mesh().face_number() > 0:
                    try:
                        ms_gp_v34.generate_surface_reconstruction_screened_poisson(depth=int(gentle_poisson_depth), preclean=True)
                        ms_gp_v34.meshing_remove_connected_component_by_diameter(mincomponentdiag=pymeshlab.Percentage(70))
                        ms_gp_v34.meshing_close_holes(); ms_gp_v34.meshing_remove_unreferenced_vertices()
                        if ms_gp_v34.current_mesh().face_number() > 0:
                            ms_gp_v34.save_current_mesh(temp_after_gp_v34)
                            loaded_gp_v34 = trimesh.load_mesh(temp_after_gp_v34, process=False)
                            if not loaded_gp_v34.is_empty: final_mesh_output = loaded_gp_v34
                    except Exception as e_gp_v34: print(f"Error GP v3.4: {e_gp_v34}"); traceback.print_exc()
            elif stitch_method == "none": print("\n=== STEP 5: No Stitching (v3.4) ===")
            else: print(f"Warning: Unknown stitch method v3.4: {stitch_method}")

            final_mesh_output.fix_normals()
            print(f"\n=== STEP 6: Saving Final Output Mesh (v3.4) ===")
            final_mesh_output.export(output_path)
            print(f"--- Mesh Grafting V3.4 Finished. Output: {output_path} ---"); return final_mesh_output

        except Exception as e_main_v34:
            print(f"--- Mesh Grafting V3.4 Failed ---"); print(f"Error: {e_main_v34}"); traceback.print_exc(); return None
        finally: # ... (cleanup logic) ...
            print("Cleaning up temporary files for grafting method v3.4...")
            for temp_path in temp_files_to_clean:
                if temp_path and os.path.exists(temp_path): 
                    try: os.remove(temp_path)
                    except OSError as e_rem: print(f"Warning: Could not remove temp file {temp_path}: {e_rem}")
    @staticmethod
    def process_mesh_graft_smplx_face_v4( # New version name
        full_body_mesh_path: str,
        output_path: str,
        smplx_face_mesh_path: str, # This is your "mask"
        # Threshold for how close a body point must be to the projection of SMPL face to be removed
        projection_footprint_threshold: float = 0.04, # e.g., 1cm
        # How many layers of adjacent faces to also remove around the initial footprint
        footprint_dilation_rings: int = 3, 
        body_simplification_target_faces: int = 15000,
        stitch_method: str = "none", 
        gentle_poisson_depth: int = 8,
        debug_dir: str = None
    ) -> Optional[trimesh.Trimesh]:
        print(f"--- Starting Mesh Processing: Grafting SMPLX Face (V4 - Footprint Projection) ---")
        print(f"Projection Footprint Threshold: {projection_footprint_threshold}")
        print(f"Footprint Dilation Rings: {footprint_dilation_rings}")
        # ... (Other initial prints)

        temp_files_to_clean = []
        def make_temp_path(suffix, directory): # ... (same helper as before) ...
            actual_suffix = suffix if suffix.startswith('_') else '_' + suffix
            _fd, path = tempfile.mkstemp(suffix=actual_suffix + ".ply", dir=directory)
            os.close(_fd); temp_files_to_clean.append(path); return path
        
        from trimesh import util 

        try:
            if debug_dir: os.makedirs(debug_dir, exist_ok=True)

            # === STEP 1: Load SMPLX Face ('Mask') Mesh ===
            print("\n=== STEP 1: Loading SMPLX Face ('Mask') Mesh ===")
            original_smplx_face_geom = trimesh.load_mesh(smplx_face_mesh_path, process=False)
            if original_smplx_face_geom.is_empty: print(f"Fatal: SMPLX face empty."); return None
            if not (hasattr(original_smplx_face_geom, 'vertices') and hasattr(original_smplx_face_geom, 'faces')):
                print(f"Fatal: Loaded SMPLX face mesh is not a valid Trimesh object with vertices/faces."); return None
            if debug_dir: original_smplx_face_geom.export(os.path.join(debug_dir, "graftv4_01_smplx_mask_loaded.obj"))

            # === STEP 2: Load and Simplify Full Body Mesh ===
            print("\n=== STEP 2: Loading and Simplifying Body Mesh ===")
            full_body_input_trimesh = trimesh.load_mesh(full_body_mesh_path, process=False)
            if full_body_input_trimesh.is_empty: print(f"Fatal: Body mesh empty."); return None
            if not (hasattr(full_body_input_trimesh, 'vertices') and hasattr(full_body_input_trimesh, 'faces')):
                print(f"Fatal: Loaded body mesh is not a valid Trimesh object with vertices/faces."); return None
            
            simplified_body_trimesh = full_body_input_trimesh 
            if full_body_input_trimesh.faces.shape[0] > body_simplification_target_faces:
                print(f"Simplifying body to target {body_simplification_target_faces} faces.")
                # ... (Simplification logic using PyMeshLab as before) ...
                temp_body_in_pml = make_temp_path("body_v4_simp_in", debug_dir); temp_body_out_pml = make_temp_path("body_v4_simp_out", debug_dir)
                try:
                    full_body_input_trimesh.export(temp_body_in_pml)
                    ms_s = pymeshlab.MeshSet(); ms_s.load_new_mesh(temp_body_in_pml)
                    if ms_s.current_mesh().face_number() > 0:
                        ms_s.meshing_decimation_quadric_edge_collapse(targetfacenum=body_simplification_target_faces, preservenormal=True)
                        ms_s.save_current_mesh(temp_body_out_pml)
                        simplified_body_trimesh_loaded = trimesh.load_mesh(temp_body_out_pml, process=False)
                        if not simplified_body_trimesh_loaded.is_empty: simplified_body_trimesh = simplified_body_trimesh_loaded
                    else: print("Warning: Body mesh empty in PML for simp v4.")
                except Exception as e_bsimp_v4: print(f"Warning: Body simp v4 failed {e_bsimp_v4}")
            if debug_dir: simplified_body_trimesh.export(os.path.join(debug_dir, "graftv4_02_simplified_body.obj"))


            # === STEP 3: Carve Hole in Body based on SMPLX Face Footprint Projection ===
            print("\n=== STEP 3: Carving Hole in Body using SMPLX Face Footprint Projection ===")
            body_with_hole_trimesh = simplified_body_trimesh.copy() # Start with a copy

            if original_smplx_face_geom.vertices.shape[0] == 0:
                print("Warning: SMPLX face has no vertices. Cannot project footprint.")
            else:
                print(f"Finding closest points on body surface from {original_smplx_face_geom.vertices.shape[0]} SMPLX face vertices...")
                try:
                    # For each vertex in `original_smplx_face_geom`, find its closest point
                    # on the surface of `simplified_body_trimesh`.
                    closest_points_on_body_surf, distances_to_body_surf, body_triangle_ids_hit = \
                        trimesh.proximity.closest_point(
                            simplified_body_trimesh, # Mesh to query against
                            original_smplx_face_geom.vertices # Points to query from
                        )
                    
                    # Select body triangles that are "under" the SMPL face
                    # These are directly hit OR very close to an SMPL face vertex
                    # We use `body_triangle_ids_hit` for directly hit faces.
                    # And `distances_to_body_surf` to potentially expand this.

                    # Initial set of faces to remove are those directly identified by `closest_point`
                    # for SMPL face vertices that are very close to the body surface.
                    initially_hit_face_indices = body_triangle_ids_hit[distances_to_body_surf < projection_footprint_threshold]
                    
                    faces_to_remove_mask = np.zeros(len(simplified_body_trimesh.faces), dtype=bool)
                    if len(initially_hit_face_indices) > 0:
                        unique_hit_face_indices = np.unique(initially_hit_face_indices)
                        print(f"DEBUG: Initially hit {len(unique_hit_face_indices)} unique body faces.")
                        faces_to_remove_mask[unique_hit_face_indices] = True

                        # Optional: Dilate the selection by N rings
                        if footprint_dilation_rings > 0:
                            print(f"DEBUG: Dilating footprint by {footprint_dilation_rings} face rings...")
                            # Build face adjacency graph for the body mesh
                            adjacency = simplified_body_trimesh.face_adjacency
                            current_selection = unique_hit_face_indices
                            for _ in range(footprint_dilation_rings):
                                # Find neighbors of the currently selected faces
                                new_neighbors = []
                                for face_idx in current_selection:
                                    # Neighbors of face_idx via shared edges
                                    adj_faces = simplified_body_trimesh.face_adjacency_edges[simplified_body_trimesh.face_adjacency_convex[face_idx]]
                                    # This gives edges. We need faces connected to these edges.
                                    # Simpler: use face_adjacency array which lists pairs of adjacent faces
                                    # Find rows in adjacency where one face is face_idx
                                    rows_with_face = np.any(adjacency == face_idx, axis=1)
                                    neighboring_pairs = adjacency[rows_with_face]
                                    # Get the other face from each pair
                                    for pair in neighboring_pairs:
                                        new_neighbors.append(pair[0] if pair[1] == face_idx else pair[1])
                                
                                if not new_neighbors: break # No more neighbors to add
                                
                                new_neighbors_unique = np.unique(new_neighbors)
                                # Add only those not already in faces_to_remove_mask
                                newly_added_mask = ~faces_to_remove_mask[new_neighbors_unique]
                                faces_to_remove_mask[new_neighbors_unique[newly_added_mask]] = True
                                current_selection = new_neighbors_unique[newly_added_mask] # Dilate from newly added
                                if not np.any(newly_added_mask): break # No new faces added in this ring
                            print(f"DEBUG: After dilation, {np.sum(faces_to_remove_mask)} faces marked for removal.")

                    if np.any(faces_to_remove_mask):
                        faces_to_keep_mask = ~faces_to_remove_mask
                        body_with_hole_trimesh = trimesh.Trimesh(
                            vertices=simplified_body_trimesh.vertices,
                            faces=simplified_body_trimesh.faces[faces_to_keep_mask]
                        )
                        body_with_hole_trimesh.remove_unreferenced_vertices()
                        body_with_hole_trimesh.fix_normals()
                    else:
                        print("Warning: No body faces identified for removal based on SMPL face footprint projection.")

                except Exception as e_prox_v4:
                    print(f"Error during proximity query for footprint projection: {e_prox_v4}")
                    traceback.print_exc()
                    print("Skipping hole carving due to error.")

            if debug_dir:
                body_with_hole_trimesh.export(os.path.join(debug_dir, "graftv4_03_body_with_footprint_hole.obj"))

            # === STEP 4: Combine Face ("Mask") and Body-with-Hole ===
            print("\n=== STEP 4: Combining Meshes ===")
            meshes_to_concat = []
            if original_smplx_face_geom and not original_smplx_face_geom.is_empty:
                meshes_to_concat.append(original_smplx_face_geom) 
            if body_with_hole_trimesh and not body_with_hole_trimesh.is_empty:
                 meshes_to_concat.append(body_with_hole_trimesh)
            if not meshes_to_concat: print("Fatal Error: No meshes to concat v4."); return None
            final_mesh_concatenated_geom_only = trimesh.util.concatenate(meshes_to_concat)
            if final_mesh_concatenated_geom_only.is_empty: print("Fatal Error: Concat empty v4."); return None
            if debug_dir: final_mesh_concatenated_geom_only.export(os.path.join(debug_dir, "graftv4_04a_concatenated_geom_only.obj"))
            final_mesh_output = final_mesh_concatenated_geom_only.copy(); final_mesh_output.fix_normals() 
            if debug_dir: final_mesh_output.export(os.path.join(debug_dir, "graftv4_04b_concatenated_with_fixed_normals.obj"))

            # === STEP 5: Stitching (Conditional) ===
            if stitch_method == "gentle_poisson":
                print("\n=== STEP 5: Stitching with Gentle Poisson ===")
                # ... (Poisson logic as before, update temp file names if desired for v4) ...
                temp_for_gp_v4 = make_temp_path("gp_in_v4", debug_dir); temp_after_gp_v4 = make_temp_path("gp_out_v4", debug_dir)
                final_mesh_output.export(temp_for_gp_v4)
                ms_gp_v4 = pymeshlab.MeshSet(); ms_gp_v4.load_new_mesh(temp_for_gp_v4)
                if ms_gp_v4.current_mesh().face_number() > 0:
                    try:
                        ms_gp_v4.generate_surface_reconstruction_screened_poisson(depth=int(gentle_poisson_depth), preclean=True)
                        ms_gp_v4.meshing_remove_connected_component_by_diameter(mincomponentdiag=pymeshlab.Percentage(70))
                        ms_gp_v4.meshing_close_holes(); ms_gp_v4.meshing_remove_unreferenced_vertices()
                        if ms_gp_v4.current_mesh().face_number() > 0:
                            ms_gp_v4.save_current_mesh(temp_after_gp_v4)
                            loaded_gp_v4 = trimesh.load_mesh(temp_after_gp_v4, process=False)
                            if not loaded_gp_v4.is_empty: final_mesh_output = loaded_gp_v4
                        else: print("Warn: GP v4 mesh empty in PML.")
                    except Exception as e_gp_v4: print(f"Error GP v4: {e_gp_v4}"); traceback.print_exc()
                else: print("Mesh empty before GP v4.")
                if debug_dir: final_mesh_output.export(os.path.join(debug_dir, "graftv4_05_after_gentle_poisson.obj"))

            elif stitch_method == "none": print("\n=== STEP 5: No Stitching (v4) ===")
            else: print(f"Warning: Unknown stitch method v4: {stitch_method}")

            final_mesh_output.fix_normals()
            print(f"\n=== STEP 6: Saving Final Output Mesh (v4) ===")
            final_mesh_output.export(output_path)
            print(f"--- Mesh Grafting V4 (Footprint Projection) Finished. Output: {output_path} ---"); return final_mesh_output
        except Exception as e_main_v4:
            print(f"--- Mesh Grafting V4 Failed ---"); print(f"Error: {e_main_v4}"); traceback.print_exc(); return None
        finally: # ... (cleanup logic) ...
            print("Cleaning up temporary files for grafting method v4...")
            for temp_path in temp_files_to_clean:
                if temp_path and os.path.exists(temp_path): 
                    try: os.remove(temp_path)
                    except OSError as e_rem: print(f"Warning: Could not remove temp file {temp_path}: {e_rem}")
        
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
    def _create_sequential_edges(num_vertices: int, closed_loop: bool = False) -> np.ndarray:
        """Helper to create edges for a sequence of vertices."""
        if num_vertices < 2:
            return np.array([], dtype=int).reshape(-1, 2)
        edges = [[i, i + 1] for i in range(num_vertices - 1)]
        if closed_loop and num_vertices > 1: # Ensure we can close it
            edges.append([num_vertices - 1, 0])
        return np.array(edges, dtype=int)

    @staticmethod
    def _get_hole_boundary_edges_from_removed_faces(
        original_mesh: trimesh.Trimesh, 
        faces_removed_mask: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Identifies edges that form the boundary of a region defined by removed faces.
        An edge is part of this boundary if one of its adjacent faces was removed and the other was kept.
        """
        # print("DEBUG: Entered _get_hole_boundary_edges_from_removed_faces")
        if not MeshCleanProcess._is_mesh_valid_for_concat(original_mesh, "OriginalMeshForHoleBoundary") or \
           faces_removed_mask is None or len(faces_removed_mask) != len(original_mesh.faces):
            print("DEBUG: _get_hole_boundary_edges - Invalid input mesh or mask.")
            return None

        if not np.any(faces_removed_mask) or np.all(faces_removed_mask):
            print("DEBUG: _get_hole_boundary_edges - All faces removed or no faces removed. No hole boundary defined this way.")
            return np.array([], dtype=int).reshape(-1, 2)

        try:
            # face_adjacency: (n, 2) array of face indices for adjacent faces
            # face_adjacency_edges: (n, 2) array of vertex indices for the edge shared by adjacent faces
            adj_face_pairs = original_mesh.face_adjacency
            adj_shared_edges = original_mesh.face_adjacency_edges
            
            hole_boundary_edges_list = []
            for i in range(len(adj_face_pairs)):
                f1_idx, f2_idx = adj_face_pairs[i]
                # Check if one face is in the removed set and the other is not
                f1_removed = faces_removed_mask[f1_idx]
                f2_removed = faces_removed_mask[f2_idx]
                
                if f1_removed != f2_removed: # XOR condition: one is true, other is false
                    hole_boundary_edges_list.append(adj_shared_edges[i])
            
            if not hole_boundary_edges_list:
                print("DEBUG: _get_hole_boundary_edges - No edges found on boundary of removed/kept regions.")
                return np.array([], dtype=int).reshape(-1, 2)

            # print(f"DEBUG: _get_hole_boundary_edges - Found {len(hole_boundary_edges_list)} raw hole boundary edges.")
            return np.array(hole_boundary_edges_list, dtype=int)
            
        except Exception as e:
            print(f"DEBUG: Error in _get_hole_boundary_edges_from_removed_faces: {e}")
            traceback.print_exc()
            return None
                
    @staticmethod
    def _extrude_loop_and_create_flange(
        source_mesh_vertices: np.ndarray, # All vertices of the source mesh
        ordered_loop_vidx: np.ndarray,    # Ordered indices of the loop on source_mesh
        extrusion_distance: float,
        mesh_name_for_debug: str = "flange",
        # Strategy for extrusion normals:
        # "vertex_normals": use existing vertex normals from source_mesh at loop_vidx (needs source_mesh_vertex_normals)
        # "loop_plane_offset": offset along normal of plane fitted to loop_vidx
        # "radial_out_from_loop_center": extrude outwards from loop centroid (needs careful orientation)
        normal_strategy: str = "vertex_normals", 
        source_mesh_vertex_normals: Optional[np.ndarray] = None, # Required if strategy is "vertex_normals"
        invert_normal_direction: bool = False # To flip extrusion direction
    ) -> Optional[Tuple[trimesh.Trimesh, np.ndarray]]: # Use typing.Tuple
        """
        Extrudes an ordered boundary loop to create a flange (a strip of faces).
        Returns:
            - flange_strip_mesh: A Trimesh object of the extruded strip ONLY. Its vertices are [original_loop_pts, new_extruded_pts].
            - new_outer_loop_coords_ordered: (N,3) array of the 3D coordinates of the new outer loop of the flange.
        Returns None if extrusion fails.
        """
        # print(f"DEBUG _extrude_flange ({mesh_name_for_debug}): Strategy '{normal_strategy}', dist {extrusion_distance}, invert {invert_normal_direction}")
        if ordered_loop_vidx is None or len(ordered_loop_vidx) < 3:
            print(f"DEBUG _extrude_flange ({mesh_name_for_debug}): Input loop too short.")
            return None

        original_loop_coords = source_mesh_vertices[ordered_loop_vidx]
        num_loop_pts = len(original_loop_coords)
        extrusion_vectors = np.zeros_like(original_loop_coords)

        if normal_strategy == "vertex_normals":
            if source_mesh_vertex_normals is None or len(source_mesh_vertex_normals) != len(source_mesh_vertices):
                print(f"DEBUG _extrude_flange ({mesh_name_for_debug}): Vertex normals required but not provided or mismatched. Cannot extrude.")
                return None
            loop_normals = source_mesh_vertex_normals[ordered_loop_vidx]
            extrusion_vectors = trimesh.util.unitize(loop_normals)
        
        elif normal_strategy == "loop_plane_offset":
            if num_loop_pts < 3: print(f"DEBUG _extrude_flange ({mesh_name_for_debug}): Not enough points for plane fit."); return None
            # Fit a plane to the loop points
            # C, N = trimesh.points.plane_fit(original_loop_coords) # This is for point clouds
            # For a loop, it's better to find an average normal from edges if planar
            # For simplicity, use PCA for dominant plane if complex
            plane_normal, _ = MeshCleanProcess.get_dominant_plane(original_loop_coords)
            # Ensure consistent orientation for all points on the loop
            extrusion_vectors = np.tile(plane_normal, (num_loop_pts, 1))
            # Check orientation: does it point "away" from loop center in its plane?
            # This check might be complex if the loop isn't planar or convex.
            # For now, relies on invert_normal_direction for control.

        elif normal_strategy == "radial_out_from_loop_center":
            if num_loop_pts < 3: print(f"DEBUG _extrude_flange ({mesh_name_for_debug}): Not enough points for radial extrusion."); return None
            loop_center = np.mean(original_loop_coords, axis=0)
            extrusion_vectors = trimesh.util.unitize(original_loop_coords - loop_center)
        
        else:
            print(f"DEBUG _extrude_flange ({mesh_name_for_debug}): Unknown normal_strategy '{normal_strategy}'.")
            return None

        if invert_normal_direction:
            extrusion_vectors *= -1.0

        new_extruded_loop_coords_ordered = original_loop_coords + extrusion_vectors * extrusion_distance

        # Create the flange strip mesh
        flange_strip_vertices = np.vstack((original_loop_coords, new_extruded_loop_coords_ordered))
        flange_strip_faces = []
        for i in range(num_loop_pts):
            v0_orig = i                       # Index in the first half of strip_vertices (original loop)
            v1_orig = (i + 1) % num_loop_pts  # Next vertex in original loop

            v0_new = i + num_loop_pts         # Corresponding index in the second half (newly extruded loop)
            v1_new = ((i + 1) % num_loop_pts) + num_loop_pts # Next vertex in newly extruded loop
            
            # Create two triangles for the quad: (v0_orig, v1_orig, v1_new, v0_new)
            # Winding order matters for outward facing normals of the strip
            flange_strip_faces.append([v0_orig, v1_new, v1_orig]) # Tri 1: Uses v0_orig, v1_orig, v1_new
            flange_strip_faces.append([v0_orig, v0_new, v1_new]) # Tri 2: Uses v0_orig, v0_new, v1_new
                                                              # Note: was [v0_orig, v1_new, v0_new] before correction test for winding.
                                                              # standard quad strip might be [i, (i+1)%N, N + (i+1)%N] and [i, N+(i+1)%N, N+i]

        if not flange_strip_faces or flange_strip_vertices.shape[0] == 0: 
             print(f"DEBUG _extrude_flange ({mesh_name_for_debug}): Failed to create strip verts/faces.")
             return None
        
        flange_mesh = trimesh.Trimesh(vertices=flange_strip_vertices, faces=np.array(flange_strip_faces), process=False)
        if not MeshCleanProcess._is_mesh_valid_for_concat(flange_mesh, f"{mesh_name_for_debug}_FlangeStrip"):
            return None
        
        # print(f"DEBUG _extrude_flange ({mesh_name_for_debug}): Flange created V={len(flange_mesh.vertices)}, F={len(flange_mesh.faces)}")
        return flange_mesh, new_extruded_loop_coords_ordered
    
    @staticmethod
    def _order_loop_vertices_from_edges( # Primary ordering attempt
        mesh_name_for_debug: str, 
        loop_vertex_indices_unique: np.ndarray, 
        all_edges_for_this_loop: np.ndarray,
        min_completeness_for_partial_ok: float = 0.75, # Allow if 75% of component verts are in path
        min_path_len_for_partial_ok: int = 10         # And path is at least this long
    ) -> Optional[np.ndarray]: 
        # print(f"DEBUG _order_loop ({mesh_name_for_debug}): Input unique vidx count: {len(loop_vertex_indices_unique)}, edges count: {len(all_edges_for_this_loop)}")
        if loop_vertex_indices_unique is None or len(loop_vertex_indices_unique) < 3 or \
           all_edges_for_this_loop is None or len(all_edges_for_this_loop) < max(0, len(loop_vertex_indices_unique) -1) :
            # print(f"DEBUG _order_loop ({mesh_name_for_debug}): Invalid input for ordering.")
            return None 
        
        # --- PURELY MANUAL Ordering ---
        adj = {v_idx: [] for v_idx in loop_vertex_indices_unique}
        vertex_degrees_in_loop = {v_idx: 0 for v_idx in loop_vertex_indices_unique}
        for u, v in all_edges_for_this_loop:
            if u in adj and v in adj: 
                adj[u].append(v); adj[v].append(u)
                vertex_degrees_in_loop[u] += 1; vertex_degrees_in_loop[v] += 1
        
        non_deg_2_count = sum(1 for v_idx in loop_vertex_indices_unique if vertex_degrees_in_loop.get(v_idx, 0) != 2)
        if non_deg_2_count > 2 and len(loop_vertex_indices_unique) > 10: # Heuristic: many branches
            print(f"DEBUG _order_loop ({mesh_name_for_debug}): Manual ordering - Component topology complex ({non_deg_2_count} V deg!=2). Path may be partial.")

        if not loop_vertex_indices_unique.size: return None

        start_node_manual = loop_vertex_indices_unique[0] 
        for v_idx in loop_vertex_indices_unique: 
            if vertex_degrees_in_loop.get(v_idx,0) == 1: start_node_manual = v_idx; break
        if vertex_degrees_in_loop.get(start_node_manual,0) > 2: 
             for v_idx in loop_vertex_indices_unique:
                 if vertex_degrees_in_loop.get(v_idx,0) == 2: start_node_manual = v_idx; break
        
        ordered_path_manual = [start_node_manual]
        visited_edges_in_path = set() 
        current_node = start_node_manual
        for _ in range(len(all_edges_for_this_loop) + 2): 
            found_next = False; neighbors = adj.get(current_node, [])
            # np.random.shuffle(neighbors) # Optional tie-breaking for path choice
            for neighbor in neighbors:
                edge_cand = tuple(sorted((current_node, neighbor)))
                if edge_cand not in visited_edges_in_path:
                    ordered_path_manual.append(neighbor); visited_edges_in_path.add(edge_cand)
                    current_node = neighbor; found_next = True; break 
            if not found_next: break 
        
        if len(ordered_path_manual) > 1 and ordered_path_manual[0] == ordered_path_manual[-1]:
            ordered_path_manual = ordered_path_manual[:-1] 

        num_unique_in_path = len(np.unique(ordered_path_manual))
        num_expected_unique = len(loop_vertex_indices_unique)

        # Check for perfect ordering first
        if num_unique_in_path == num_expected_unique and len(ordered_path_manual) == num_expected_unique:
            # print(f"DEBUG _order_loop ({mesh_name_for_debug}): Manual ordering successful and COMPLETE. Path length: {len(ordered_path_manual)}")
            return np.array(ordered_path_manual, dtype=int)
        
        # If not perfect, check if it's "good enough" for a partial attempt
        if num_unique_in_path >= num_expected_unique * min_completeness_for_partial_ok and \
           len(ordered_path_manual) >= min_path_len_for_partial_ok:
            print(f"DEBUG _order_loop ({mesh_name_for_debug}): Manual ordering PARTIALLY successful. Path: {len(ordered_path_manual)}V ({num_unique_in_path} uniq) vs Comp: {num_expected_unique}V. USING PARTIAL PATH.")
            # Ensure returned path has unique vertices in sequence found, even if it's not all of them.
            # This might not be strictly necessary if `resample_polyline_to_count` can handle repeats,
            # but cleaner to return a path of unique (in sequence) vertices.
            # For simplicity, just return the path found. Resampling might smooth over minor issues.
            return np.array(ordered_path_manual, dtype=int)
        else: # Ordering is considered failed
            print(f"DEBUG _order_loop ({mesh_name_for_debug}): Manual ordering result INCOMPLETE and NOT substantial enough.")
            print(f"  Path found: {len(ordered_path_manual)} verts, {num_unique_in_path} unique.")
            print(f"  Expected component: {num_expected_unique} unique verts.")
            print(f"  Edges in loop: {len(all_edges_for_this_loop)}. Visited edges: {len(visited_edges_in_path)}")
            print(f"  STRICTLY FAILING ORDERING for {mesh_name_for_debug}.")
            return None 
                    
    @staticmethod
    def get_dominant_plane(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimates the dominant plane from a set of 3D points using PCA.
        Returns:
            tuple: (plane_normal: np.ndarray, plane_origin: np.ndarray)
                   Returns default XY plane if not enough points or SVD fails.
        """
        if points is None or len(points) < 3:
            origin = np.mean(points, axis=0) if (points is not None and len(points) > 0) else np.array([0.0, 0.0, 0.0])
            # print(f"DEBUG get_dominant_plane: Not enough points ({len(points) if points is not None else 'None'}). Returning default Z normal and origin {origin}.")
            return np.array([0.0, 0.0, 1.0]), origin

        center = np.mean(points, axis=0)
        centered_points = points - center
        
        try:
            # We only need vh (or VT from SVD: U S V^T = X)
            # The last row of vh corresponds to the smallest singular value / principal component
            _, _, vh = np.linalg.svd(centered_points, full_matrices=False) 
            plane_normal = vh[-1, :]
            # Ensure plane_normal is a unit vector
            return trimesh.util.unitize(plane_normal), center # Use original center as origin
        except np.linalg.LinAlgError:
            print("DEBUG: get_dominant_plane - SVD did not converge. Returning default XY plane normal with input points centroid.")
            return np.array([0.0, 0.0, 1.0]), center
        except Exception as e_svd:
            print(f"DEBUG: get_dominant_plane - Unexpected error during SVD: {e_svd}. Defaulting.")
            return np.array([0.0, 0.0, 1.0]), center
        
    @staticmethod
    def _order_loop_by_graph_walk(
        mesh_name_for_debug: str,
        loop_vertex_indices_unique: np.ndarray, # Should be the component vertices
        all_edges_for_this_loop: np.ndarray # Edges *only* for this component
    ) -> Optional[np.ndarray]:
        print(f"DEBUG _order_loop_by_graph_walk ({mesh_name_for_debug}): Attempting graph walk.")
        if len(loop_vertex_indices_unique) < 3 or len(all_edges_for_this_loop) < 2:
            return None

        adj_map = defaultdict(list)
        vertex_degrees = defaultdict(int)
        for u, v in all_edges_for_this_loop:
            adj_map[u].append(v)
            adj_map[v].append(u)
            vertex_degrees[u] += 1
            vertex_degrees[v] += 1

        # Try to find a start vertex with degree 2 (ideal for simple loop)
        start_node = -1
        for v_idx in loop_vertex_indices_unique: # Iterate in component order for some consistency
            if vertex_degrees[v_idx] == 2:
                start_node = v_idx
                break
        
        if start_node == -1: # If no degree 2 nodes, try degree 1 (open path end)
            for v_idx in loop_vertex_indices_unique:
                if vertex_degrees[v_idx] == 1:
                    start_node = v_idx
                    break
        
        if start_node == -1: # Still no good start node, pick first from component (less ideal)
            print(f"DEBUG _order_loop_by_graph_walk ({mesh_name_for_debug}): No degree 1 or 2 start node found. Using arbitrary start.")
            start_node = loop_vertex_indices_unique[0] if len(loop_vertex_indices_unique) > 0 else -1
            if start_node == -1 : return None


        ordered_path = [start_node]
        prev_node, current_node = -1, start_node # Use -1 as a sentinel for previous initially

        for _ in range(len(loop_vertex_indices_unique)): # Max iterations to visit all unique nodes once
            neighbors = adj_map.get(current_node, [])
            if not neighbors: # Dead end
                # print(f"DEBUG _order_loop_by_graph_walk ({mesh_name_for_debug}): Stuck at {current_node}, no neighbors.")
                break
            
            if len(neighbors) == 1 and prev_node != -1: # Must be the end of an open path or a spur
                next_node = neighbors[0]
                if next_node == prev_node and len(ordered_path) > 1: # Trying to go back immediately
                     print(f"DEBUG _order_loop_by_graph_walk ({mesh_name_for_debug}): Stuck at {current_node}, only neighbor is previous {prev_node}.")
                     break 
                # If it's a different node, proceed (this handles start of path correctly too where prev_node is -1)
            elif len(neighbors) == 2: # Ideal case for a simple loop interior
                next_node = neighbors[0] if neighbors[0] != prev_node else neighbors[1]
            else: # Degree > 2 (branch) or Degree 1 at start when prev_node = -1
                  # For branching, this simple walk will pick one path.
                  # If it's degree 1 at start: pick the only neighbor
                next_node = -1
                for n_candidate in neighbors:
                    if n_candidate != prev_node:
                        next_node = n_candidate
                        break
                if next_node == -1: # Only neighbor was previous, or no valid next neighbor
                    print(f"DEBUG _order_loop_by_graph_walk ({mesh_name_for_debug}): Stuck at branch node {current_node} or end of path.")
                    break
            
            if next_node == start_node and len(ordered_path) > 1: # Closed the loop
                # print(f"DEBUG _order_loop_by_graph_walk ({mesh_name_for_debug}): Graph walk closed loop.")
                break
            if next_node in ordered_path: # Visited this node again before closing loop (not start node) - complex
                print(f"DEBUG _order_loop_by_graph_walk ({mesh_name_for_debug}): Graph walk revisited node {next_node} prematurely. Loop likely not simple.")
                # We could either stop or continue to see how much we get.
                # For now, let's stop to prefer simpler paths from this algorithm.
                return None # Indicate failure for non-simple loops detected this way

            ordered_path.append(next_node)
            prev_node, current_node = current_node, next_node
        
        # Check if all unique component vertices were visited and form a simple path
        if len(np.unique(ordered_path)) == len(loop_vertex_indices_unique) and \
           len(ordered_path) == len(loop_vertex_indices_unique) :
            print(f"DEBUG _order_loop_by_graph_walk ({mesh_name_for_debug}): Graph walk successful, path length {len(ordered_path)}.")
            return np.array(ordered_path, dtype=int)
        else:
            print(f"DEBUG _order_loop_by_graph_walk ({mesh_name_for_debug}): Graph walk result INCOMPLETE/NOT_SIMPLE. Path:{len(ordered_path)}V,{len(np.unique(ordered_path))}uniq vs Comp:{len(loop_vertex_indices_unique)}uniq. ORDERING FAILED.")
            return None

    @staticmethod
    def _order_loop_by_planar_angle_sort(
        mesh_name_for_debug: str,
        mesh_vertices: np.ndarray, # Full mesh vertices
        loop_vertex_indices_unique: np.ndarray # Unique, unordered Vidx of the target loop
    ) -> Optional[np.ndarray]:
        # print(f"DEBUG _order_loop_by_planar_angle_sort ({mesh_name_for_debug}): Attempting angle sort.")
        if loop_vertex_indices_unique is None or len(loop_vertex_indices_unique) < 3:
            # print(f"DEBUG AngleSort ({mesh_name_for_debug}): Too few unique vertices ({len(loop_vertex_indices_unique) if loop_vertex_indices_unique is not None else 'None'}).")
            return None

        loop_coords_3d = mesh_vertices[loop_vertex_indices_unique]
        if len(loop_coords_3d) < 3: # Double check after indexing
            print(f"DEBUG AngleSort ({mesh_name_for_debug}): Too few 3D coordinates ({len(loop_coords_3d)}).")
            return None
            
        # 1. Fit best plane to the 3D loop coordinates
        plane_origin, plane_normal = MeshCleanProcess.get_dominant_plane(loop_coords_3d)
        if np.allclose(plane_normal, [0,0,0]) or np.any(np.isnan(plane_normal)) or np.any(np.isnan(plane_origin)):
             print(f"WARN _order_loop_by_planar_angle_sort ({mesh_name_for_debug}): Plane fit resulted in zero or NaN normal/origin. Cannot project reliably.")
             return None

        # 2. Create transformation to align this plane with XY plane at Z=0
        try:
            # transform that moves input plane to XY plane
            transform_to_2d_plane = trimesh.geometry.plane_transform(plane_origin, plane_normal)
        except Exception as e_plane_xform: # Catch errors in plane_transform (e.g. if normal is zero)
            print(f"DEBUG AngleSort ({mesh_name_for_debug}): Error creating plane transform: {e_plane_xform}. Using default XY projection.")
            # Fallback to simple XY projection if plane_transform fails
            plane_normal = np.array([0.0, 0.0, 1.0])
            plane_origin = np.mean(loop_coords_3d, axis=0)
            transform_to_2d_plane = trimesh.geometry.plane_transform(plane_origin, plane_normal)


        # 3. Transform 3D loop coordinates and take their XY components for 2D
        loop_coords_transformed_to_plane = trimesh.transform_points(loop_coords_3d, transform_to_2d_plane)
        projected_2d_coords = loop_coords_transformed_to_plane[:, :2] # These are now 2D coordinates on the plane

        if projected_2d_coords is None or len(projected_2d_coords) < 3 or projected_2d_coords.shape[1]!=2:
            print(f"DEBUG AngleSort ({mesh_name_for_debug}): Projection to 2D resulted in invalid points (Shape: {projected_2d_coords.shape if projected_2d_coords is not None else 'None'}).")
            return None
            
        # 4. Compute centroid & angles in 2D
        centroid_2d = np.mean(projected_2d_coords, axis=0)
        delta_coords = projected_2d_coords - centroid_2d
        angles = np.arctan2(delta_coords[:,1], delta_coords[:,0]) # Y then X for standard angle
        
        # 5. Sort original loop_vertex_indices_unique by these angles
        order_indices_for_unique_loop = np.argsort(angles)
        ordered_vidx_by_angle = loop_vertex_indices_unique[order_indices_for_unique_loop]
        
        # print(f"DEBUG _order_loop_by_planar_angle_sort ({mesh_name_for_debug}): Angle sort successful. Path length {len(ordered_vidx_by_angle)}.")
        return ordered_vidx_by_angle

    @staticmethod
    def _is_mesh_valid_for_concat(mesh: Optional[trimesh.Trimesh], mesh_name: str) -> bool:
        if mesh is None: return False
        if mesh.is_empty: return False
        if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces'): return False
        if not isinstance(mesh.vertices, np.ndarray) or mesh.vertices.ndim != 2 or mesh.vertices.shape[1] != 3: return False
        if not isinstance(mesh.faces, np.ndarray) or mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3: return False
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
    def get_outermost_boundary_loop_vertices(mesh: trimesh.Trimesh, strategy="longest_loop", mesh_name_for_debug="mesh") -> np.ndarray:
        if not (mesh and not mesh.is_empty and hasattr(mesh,'vertices') and mesh.vertices is not None and \
                len(mesh.vertices)>0 and hasattr(mesh,'faces') and mesh.faces is not None and len(mesh.faces)>0):
            return np.array([],dtype=int)
        boundary_edges = None
        try:
            boundary_edges = MeshCleanProcess._get_boundary_edges_manually_from_faces(mesh)
            if boundary_edges is None or len(boundary_edges)==0:
                if hasattr(mesh, 'edges_unique_boundary'): boundary_edges = mesh.edges_unique_boundary
                if boundary_edges is None or len(boundary_edges)==0: return np.array([],dtype=int)
            all_loop_vidx_sets = trimesh.graph.connected_components(boundary_edges, min_len=3)
            candidate_loops_unique_vidx_sets = [comp.astype(int) for comp in all_loop_vidx_sets if comp is not None and len(comp)>2]
            if not candidate_loops_unique_vidx_sets: return np.array([],dtype=int)
            sel_loop_unord_vidx = np.array([],dtype=int)
            if strategy=="longest_loop":
                sel_loop_unord_vidx=max(candidate_loops_unique_vidx_sets,key=len,default=np.array([],dtype=int))
            elif strategy=="lowest_z_loop":
                def get_avg_z(v_idx_array):
                    if v_idx_array is None or len(v_idx_array)==0: return float('inf')
                    valid_indices = v_idx_array[(v_idx_array >= 0) & (v_idx_array < len(mesh.vertices))]
                    if len(valid_indices)==0: return float('inf')
                    return np.mean(mesh.vertices[valid_indices,2])
                valid_candidates_for_min = [loop_set for loop_set in candidate_loops_unique_vidx_sets if loop_set is not None and len(loop_set) > 0]
                if valid_candidates_for_min: sel_loop_unord_vidx=min(valid_candidates_for_min,key=get_avg_z,default=np.array([],dtype=int))
                else: sel_loop_unord_vidx=max(candidate_loops_unique_vidx_sets,key=len,default=np.array([],dtype=int))
            else: sel_loop_unord_vidx=max(candidate_loops_unique_vidx_sets,key=len,default=np.array([],dtype=int))
            if len(sel_loop_unord_vidx)<3: return np.array([],dtype=int)
            sel_loop_set = set(sel_loop_unord_vidx)
            edges_for_selected_loop_list = [edge for edge in boundary_edges if edge[0] in sel_loop_set and edge[1] in sel_loop_set]
            if not edges_for_selected_loop_list or len(edges_for_selected_loop_list) < max(0, len(sel_loop_unord_vidx) - 1): return sel_loop_unord_vidx
            edges_for_selected_loop = np.array(edges_for_selected_loop_list, dtype=int)
            ordered_vidx_path_final = MeshCleanProcess._order_loop_vertices_from_edges(
                f"{mesh_name_for_debug}_PrimaryOrder", sel_loop_unord_vidx, edges_for_selected_loop)
            if ordered_vidx_path_final is None or len(ordered_vidx_path_final) < 3:
                if hasattr(trimesh.graph, 'traverse_edges'):
                    try:
                        start_node_for_traverse = sel_loop_unord_vidx[0]
                        path_from_traverse = trimesh.graph.traverse_edges(edges_for_selected_loop, start_node_for_traverse)
                        if path_from_traverse is not None and len(path_from_traverse) >= 3 and len(np.unique(path_from_traverse)) >= len(sel_loop_unord_vidx) * 0.8:
                            ordered_vidx_path_final = np.array(path_from_traverse, dtype=int)
                            if len(ordered_vidx_path_final) > 1 and ordered_vidx_path_final[0] == ordered_vidx_path_final[-1]: ordered_vidx_path_final = ordered_vidx_path_final[:-1]
                    except Exception: pass
            if ordered_vidx_path_final is None or len(ordered_vidx_path_final) < 3:
                ordered_vidx_path_final = MeshCleanProcess._order_loop_by_graph_walk(
                    f"{mesh_name_for_debug}_GraphWalkOrder", sel_loop_unord_vidx, edges_for_selected_loop)
            if ordered_vidx_path_final is None or len(ordered_vidx_path_final) < 3:
                ordered_vidx_path_final = MeshCleanProcess._order_loop_by_planar_angle_sort(
                    f"{mesh_name_for_debug}_AngleSortOrder", mesh.vertices, sel_loop_unord_vidx)
            if ordered_vidx_path_final is not None and len(ordered_vidx_path_final) >= 3: return ordered_vidx_path_final
            else: return sel_loop_unord_vidx
        except Exception as e:
            print(f"Crit Error get_outermost ({mesh_name_for_debug}): {e}"); traceback.print_exc(); return np.array([], dtype=int)

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
    def _order_loop_by_graph_walk(mesh_name_for_debug: str, loop_vertex_indices_unique: np.ndarray, all_edges_for_this_loop: np.ndarray) -> Optional[np.ndarray]:
        if loop_vertex_indices_unique is None or len(loop_vertex_indices_unique) < 3 or all_edges_for_this_loop is None or len(all_edges_for_this_loop) < 2 : return None
        adj_map = defaultdict(list); vertex_degrees = defaultdict(int)
        for u, v in all_edges_for_this_loop: adj_map[u].append(v); adj_map[v].append(u); vertex_degrees[u] += 1; vertex_degrees[v] += 1
        start_node = -1
        deg2_nodes = [v_idx for v_idx in loop_vertex_indices_unique if vertex_degrees.get(v_idx, 0) == 2]
        if deg2_nodes: start_node = deg2_nodes[0]
        else:
            deg1_nodes = [v_idx for v_idx in loop_vertex_indices_unique if vertex_degrees.get(v_idx, 0) == 1]
            if deg1_nodes: start_node = deg1_nodes[0]
            elif len(loop_vertex_indices_unique) > 0 : start_node = loop_vertex_indices_unique[0]
            else: return None
        ordered_path = [start_node]; prev_node, current_node = -1, start_node
        for _ in range(len(loop_vertex_indices_unique)):
            neighbors = adj_map.get(current_node, []); next_node_candidate = -1
            if not neighbors: break
            for n_cand in neighbors:
                if n_cand == prev_node: continue
                if n_cand == start_node and len(ordered_path) > 1: next_node_candidate = n_cand; break
                if n_cand not in ordered_path: next_node_candidate = n_cand; break
            if next_node_candidate == -1: break
            if next_node_candidate == start_node and len(ordered_path) > 1: break
            if next_node_candidate in ordered_path: return None
            ordered_path.append(next_node_candidate); prev_node, current_node = current_node, next_node_candidate
        if len(ordered_path) > 1 and ordered_path[0] == ordered_path[-1]: ordered_path = ordered_path[:-1]
        if len(np.unique(ordered_path)) == len(loop_vertex_indices_unique) and len(ordered_path) == len(loop_vertex_indices_unique) : return np.array(ordered_path, dtype=int)
        else: return None

    @staticmethod
    def _order_loop_by_planar_angle_sort(mesh_name_for_debug: str, mesh_vertices: np.ndarray, loop_vertex_indices_unique: np.ndarray) -> Optional[np.ndarray]:
        if loop_vertex_indices_unique is None or len(loop_vertex_indices_unique) < 3: return None
        if loop_vertex_indices_unique.max() >= len(mesh_vertices) or loop_vertex_indices_unique.min() < 0: return None
        loop_coords_3d = mesh_vertices[loop_vertex_indices_unique]
        if len(loop_coords_3d) < 3: return None
        plane_origin, plane_normal = MeshCleanProcess.get_dominant_plane(loop_coords_3d)
        if np.allclose(plane_normal, [0,0,0]) or np.any(np.isnan(plane_normal)) or np.any(np.isnan(plane_origin)): return None
        try: transform_to_2d_plane = trimesh.geometry.plane_transform(plane_origin, plane_normal)
        except Exception: plane_normal = np.array([0.0, 0.0, 1.0]); plane_origin = np.mean(loop_coords_3d, axis=0); transform_to_2d_plane = trimesh.geometry.plane_transform(plane_origin, plane_normal)
        loop_coords_transformed_to_plane = trimesh.transform_points(loop_coords_3d, transform_to_2d_plane)
        projected_2d_coords = loop_coords_transformed_to_plane[:, :2]
        if projected_2d_coords is None or len(projected_2d_coords) < 3 or projected_2d_coords.shape[1]!=2: return None
        centroid_2d = np.mean(projected_2d_coords, axis=0); delta_coords = projected_2d_coords - centroid_2d
        angles = np.arctan2(delta_coords[:,1], delta_coords[:,0])
        order_indices_for_unique_loop = np.argsort(angles)
        ordered_vidx_by_angle = loop_vertex_indices_unique[order_indices_for_unique_loop]
        return ordered_vidx_by_angle

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
    def _create_sequential_edges(num_vertices: int, closed_loop: bool = False) -> np.ndarray:
        if num_vertices < 2: return np.array([], dtype=int).reshape(-1, 2)
        edges = [[i, i + 1] for i in range(num_vertices - 1)]
        if closed_loop and num_vertices > 1: edges.append([num_vertices - 1, 0])
        return np.array(edges, dtype=int)

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
    def _iterative_non_manifold_repair_pml(
        input_mesh: trimesh.Trimesh,
        max_iterations_stage1: int = 3, 
        max_iterations_stage2: int = 2, 
        stage2_remeshing_length_percentage: Optional[float] = None,
        debug_dir: Optional[str] = None,
        temp_file_prefix: str = "nm_repair"
    ) -> trimesh.Trimesh:
        if not MeshCleanProcess._is_mesh_valid_for_concat(input_mesh, "InputForIterativeRepair"):
            print(f"INFO V_IR: Input mesh for iterative repair ('{temp_file_prefix}') is invalid. Skipping repair.")
            return input_mesh

        print(f"--- V_IR: Starting Iterative Non-Manifold Repair for '{temp_file_prefix}' ---")
        current_mesh = input_mesh.copy()
        local_temp_files = []
        
        def make_local_temp_path(suffix:str, directory:Optional[str])->str:
            _fd, path = tempfile.mkstemp(suffix=suffix + ".ply", dir=directory)
            os.close(_fd); local_temp_files.append(path); return path

        initial_v_count = current_mesh.vertices.shape[0]
        initial_f_count = current_mesh.faces.shape[0]
        print(f"  V_IR: Initial state: Vertices={initial_v_count}, Faces={initial_f_count}")

        # ----- STAGE 1: Standard Repairs -----
        print(f"  V_IR: --- Entering Stage 1 Repairs (max {max_iterations_stage1} iterations) ---")
        prev_v_count_stagnant_s1 = -1; prev_f_count_stagnant_s1 = -1
        prev_nm_faces_stagnant_s1 = -1; prev_nm_verts_stagnant_s1 = -1
        stagnation_count_s1 = 0
        stage1_resolved_nm = False

        for i_s1 in range(max_iterations_stage1):
            print(f"  V_IR: Stage 1 - Iteration {i_s1+1}/{max_iterations_stage1}:")
            selected_nm_faces_before_iter, selected_nm_vertices_before_iter = -1, -1
            ms_check_before = None; temp_check_in_before_path = ""
            v_count_iter_start, f_count_iter_start = -1,-1
            try:
                ms_check_before = pymeshlab.MeshSet()
                temp_check_in_before_path = make_local_temp_path(f"{temp_file_prefix}_s1_iter{i_s1}_check_in", debug_dir)
                current_mesh.export(temp_check_in_before_path)
                ms_check_before.load_new_mesh(temp_check_in_before_path)
                if ms_check_before.current_mesh_id() == -1 or ms_check_before.current_mesh().vertex_number() == 0: break
                v_count_iter_start = ms_check_before.current_mesh().vertex_number(); f_count_iter_start = ms_check_before.current_mesh().face_number()
                ms_check_before.apply_filter('compute_selection_by_non_manifold_edges_per_face')
                selected_nm_faces_before_iter = ms_check_before.current_mesh().selected_face_number()
                ms_check_before.load_new_mesh(temp_check_in_before_path)
                ms_check_before.apply_filter('select_non_manifold_vertices')
                selected_nm_vertices_before_iter = ms_check_before.current_mesh().selected_vertex_number()
                print(f"    V_IR: S1 Before repair: V={v_count_iter_start}, F={f_count_iter_start}, NM Edge-Faces={selected_nm_faces_before_iter}, NM Verts={selected_nm_vertices_before_iter}")
                if selected_nm_faces_before_iter == 0 and selected_nm_vertices_before_iter == 0:
                    print("    V_IR: S1 No non-manifold elements detected. Stage 1 complete."); stage1_resolved_nm = True; break
            except AttributeError as e_attr: print(f"    V_IR FATAL ERROR: PML filter for NM selection misnamed: {e_attr}. Stopping iter repair."); return current_mesh
            except Exception as e_check_s1_before: print(f"    V_IR ERROR during S1 pre-repair check: {e_check_s1_before}"); break
            finally:
                if ms_check_before is not None: del ms_check_before

            ms_repair = None; repair_applied_in_pml_s1 = False; 
            v_start_repair_pml = -1; f_start_repair_pml = -1 
            try:
                ms_repair = pymeshlab.MeshSet()
                temp_repair_in = make_local_temp_path(f"{temp_file_prefix}_s1_iter{i_s1}_repair_in", debug_dir)
                current_mesh.export(temp_repair_in); ms_repair.load_new_mesh(temp_repair_in)
                if ms_repair.current_mesh_id() == -1 or ms_repair.current_mesh().vertex_number() == 0: break
                
                v_start_repair_pml = ms_repair.current_mesh().vertex_number(); f_start_repair_pml = ms_repair.current_mesh().face_number()
                edge_repair_attempted_s1 = False; vertex_repair_attempted_s1 = False
                
                if selected_nm_faces_before_iter > 0 or selected_nm_vertices_before_iter > 0: 
                    try: 
                        print("    V_IR: S1 Attempting edge repair with 'Split Vertices'...")
                        ms_repair.meshing_repair_non_manifold_edges(method='Split Vertices'); edge_repair_attempted_s1 = True
                        print("      V_IR: S1 'Split Vertices' on edges applied.")
                    except pymeshlab.PyMeshLabException:
                        print("      V_IR: S1 'Split Vertices' on edges failed. Trying 'Remove Faces'...")
                        try: 
                            ms_repair.meshing_repair_non_manifold_edges(method='Remove Faces'); edge_repair_attempted_s1 = True
                            print("      V_IR: S1 'Remove Faces' on edges applied.")
                        except pymeshlab.PyMeshLabException: print("    V_IR: S1 Both edge repair methods failed.")
                
                if edge_repair_attempted_s1 and (ms_repair.current_mesh().vertex_number() != v_start_repair_pml or ms_repair.current_mesh().face_number() != f_start_repair_pml):
                    current_mesh_temp_path_for_v_repair = make_local_temp_path(f"{temp_file_prefix}_s1_iter{i_s1}_temp_for_v_repair", debug_dir)
                    ms_repair.save_current_mesh(current_mesh_temp_path_for_v_repair)
                    ms_repair.load_new_mesh(current_mesh_temp_path_for_v_repair)
                    v_start_repair_pml = ms_repair.current_mesh().vertex_number(); f_start_repair_pml = ms_repair.current_mesh().face_number() 

                if selected_nm_vertices_before_iter > 0: 
                    print("    V_IR: S1 Attempting non-manifold vertex repair...")
                    try: 
                        ms_repair.meshing_repair_non_manifold_vertices(); vertex_repair_attempted_s1 = True
                        print("      V_IR: S1 Non-manifold vertex repair applied.")
                    except pymeshlab.PyMeshLabException: print("    V_IR: S1 Vertex repair failed.")
                
                ms_repair.meshing_remove_unreferenced_vertices()

                if ms_repair.current_mesh().vertex_number() != v_start_repair_pml or \
                   ms_repair.current_mesh().face_number() != f_start_repair_pml or \
                   edge_repair_attempted_s1 or vertex_repair_attempted_s1 : 
                    repair_applied_in_pml_s1 = True

                if not repair_applied_in_pml_s1 and not (selected_nm_faces_before_iter == 0 and selected_nm_vertices_before_iter == 0) :
                     pass # No geometry change, but issues might persist. Stagnation check will handle.

                temp_repair_out = make_local_temp_path(f"{temp_file_prefix}_s1_iter{i_s1}_repair_out", debug_dir)
                ms_repair.save_current_mesh(temp_repair_out)
                repaired_mesh_trimesh_iter = trimesh.load_mesh(temp_repair_out, process=False)
                if MeshCleanProcess._is_mesh_valid_for_concat(repaired_mesh_trimesh_iter, f"S1RepairedIter{i_s1}"): 
                    current_mesh = repaired_mesh_trimesh_iter
                    print(f"    V_IR: S1 After repair ops: V={current_mesh.vertices.shape[0]}, F={current_mesh.faces.shape[0]}")
                else: print("    V_IR ERROR: S1 Repaired mesh invalid after loading."); break
            except Exception as e_repair_s1: print(f"    V_IR ERROR during S1 repair block: {e_repair_s1}"); break
            finally:
                if ms_repair is not None: del ms_repair

            selected_nm_faces_after_iter, selected_nm_vertices_after_iter = -1, -1
            ms_check_after = None; temp_check_out_after_path = ""
            v_geom_after_iter, f_geom_after_iter = -1, -1
            try:
                ms_check_after = pymeshlab.MeshSet()
                temp_check_out_after_path = make_local_temp_path(f"{temp_file_prefix}_s1_iter{i_s1}_check_out_after", debug_dir)
                current_mesh.export(temp_check_out_after_path); ms_check_after.load_new_mesh(temp_check_out_after_path)
                if ms_check_after.current_mesh_id() != -1 and ms_check_after.current_mesh().vertex_number() > 0:
                    v_geom_after_iter = ms_check_after.current_mesh().vertex_number(); f_geom_after_iter = ms_check_after.current_mesh().face_number()
                    ms_check_after.apply_filter('compute_selection_by_non_manifold_edges_per_face')
                    selected_nm_faces_after_iter = ms_check_after.current_mesh().selected_face_number()
                    ms_check_after.load_new_mesh(temp_check_out_after_path)
                    ms_check_after.apply_filter('select_non_manifold_vertices')
                    selected_nm_vertices_after_iter = ms_check_after.current_mesh().selected_vertex_number()
                    print(f"    V_IR: S1 After repair check: V={v_geom_after_iter}, F={f_geom_after_iter}, NM Edge-Faces={selected_nm_faces_after_iter}, NM Verts={selected_nm_vertices_after_iter}")
                    if selected_nm_faces_after_iter == 0 and selected_nm_vertices_after_iter == 0:
                        print(f"    V_IR: S1 Non-manifold elements resolved."); stage1_resolved_nm = True; break
                    if v_geom_after_iter == prev_v_count_stagnant_s1 and f_geom_after_iter == prev_f_count_stagnant_s1 and \
                       selected_nm_faces_after_iter == prev_nm_faces_stagnant_s1 and selected_nm_vertices_after_iter == prev_nm_verts_stagnant_s1:
                        stagnation_count_s1 += 1
                        if stagnation_count_s1 >= 2: print(f"    V_IR: S1 Repair stagnated for {stagnation_count_s1} iters. NM issues persist."); break
                    else: stagnation_count_s1 = 0
                    prev_v_count_stagnant_s1 = v_geom_after_iter; prev_f_count_stagnant_s1 = f_geom_after_iter
                    prev_nm_faces_stagnant_s1 = selected_nm_faces_after_iter; prev_nm_verts_stagnant_s1 = selected_nm_vertices_after_iter
                else: print("    V_IR ERROR: S1 Mesh empty/invalid for post-repair check."); break
            except Exception as e_check_s1_after: print(f"    V_IR ERROR during S1 post-repair check: {e_check_s1_after}"); break
            finally:
                if ms_check_after is not None: del ms_check_after
            if i_s1 == max_iterations_stage1 - 1: print(f"  V_IR: S1 Reached max iterations.")
        
        if not stage1_resolved_nm:
            print(f"  V_IR: --- Stage 1 did not resolve all NM issues. Entering Stage 2 (max {max_iterations_stage2} iterations) ---")
            prev_v_count_stagnant_s2 = -1; prev_f_count_stagnant_s2 = -1
            prev_nm_faces_stagnant_s2 = -1; prev_nm_verts_stagnant_s2 = -1
            stagnation_count_s2 = 0
            for i_s2 in range(max_iterations_stage2):
                print(f"  V_IR: Stage 2 - Iteration {i_s2+1}/{max_iterations_stage2}:")
                selected_nm_faces_s2_before, selected_nm_vertices_s2_before = -1, -1
                ms_check_s2_before = None; temp_check_s2_in_before_path = ""
                v_s2_start, f_s2_start = -1,-1
                try:
                    ms_check_s2_before = pymeshlab.MeshSet()
                    temp_check_s2_in_before_path = make_local_temp_path(f"{temp_file_prefix}_s2_iter{i_s2}_check_in", debug_dir)
                    current_mesh.export(temp_check_s2_in_before_path)
                    ms_check_s2_before.load_new_mesh(temp_check_s2_in_before_path)
                    if ms_check_s2_before.current_mesh_id() == -1 or ms_check_s2_before.current_mesh().vertex_number() == 0: break
                    v_s2_start = ms_check_s2_before.current_mesh().vertex_number(); f_s2_start = ms_check_s2_before.current_mesh().face_number()
                    ms_check_s2_before.apply_filter('compute_selection_by_non_manifold_edges_per_face')
                    selected_nm_faces_s2_before = ms_check_s2_before.current_mesh().selected_face_number()
                    ms_check_s2_before.load_new_mesh(temp_check_s2_in_before_path)
                    ms_check_s2_before.apply_filter('select_non_manifold_vertices')
                    selected_nm_vertices_s2_before = ms_check_s2_before.current_mesh().selected_vertex_number()
                    print(f"    V_IR: S2 Before repair: V={v_s2_start}, F={f_s2_start}, NM Edge-Faces={selected_nm_faces_s2_before}, NM Verts={selected_nm_vertices_s2_before}")
                    if selected_nm_faces_s2_before == 0 and selected_nm_vertices_s2_before == 0:
                        print("    V_IR: S2 No non-manifold elements. Stage 2 complete."); stage1_resolved_nm = True; break
                except Exception as e_check_s2_before: print(f"    V_IR ERROR during S2 pre-repair check: {e_check_s2_before}"); break
                finally:
                    if ms_check_s2_before is not None: del ms_check_s2_before
                if stage1_resolved_nm: break

                ms_repair_s2 = None
                try:
                    ms_repair_s2 = pymeshlab.MeshSet()
                    temp_repair_s2_in = make_local_temp_path(f"{temp_file_prefix}_s2_iter{i_s2}_repair_in", debug_dir)
                    current_mesh.export(temp_repair_s2_in); ms_repair_s2.load_new_mesh(temp_repair_s2_in)
                    if ms_repair_s2.current_mesh_id() == -1 or ms_repair_s2.current_mesh().vertex_number() == 0: break
                    print("    V_IR: S2 Attempting aggressive edge repair ('Remove Faces')...")
                    try: ms_repair_s2.meshing_repair_non_manifold_edges(method='Remove Faces')
                    except pymeshlab.PyMeshLabException as e_s2_edge: print(f"    V_IR: S2 'Remove Faces' on edges failed: {e_s2_edge}")
                    print("    V_IR: S2 Attempting non-manifold vertex repair...")
                    try: ms_repair_s2.meshing_repair_non_manifold_vertices()
                    except pymeshlab.PyMeshLabException as e_s2_vert: print(f"    V_IR: S2 Vertex repair failed: {e_s2_vert}")
                    if stage2_remeshing_length_percentage is not None and stage2_remeshing_length_percentage > 0:
                        print(f"    V_IR: S2 Attempting Isotropic Remeshing with target length {stage2_remeshing_length_percentage}% of bbox diag...")
                        try:
                            ms_repair_s2.meshing_isotropic_explicit_remeshing(iterations=3, targetlen=pymeshlab.Percentage(stage2_remeshing_length_percentage))
                            print("      V_IR: S2 Isotropic Remeshing applied.")
                        except pymeshlab.PyMeshLabException as e_remesh: print(f"      V_IR: S2 Isotropic Remeshing failed: {e_remesh}")
                    ms_repair_s2.meshing_remove_unreferenced_vertices()
                    temp_repair_s2_out = make_local_temp_path(f"{temp_file_prefix}_s2_iter{i_s2}_repair_out", debug_dir)
                    ms_repair_s2.save_current_mesh(temp_repair_s2_out)
                    repaired_mesh_trimesh_s2_iter = trimesh.load_mesh(temp_repair_s2_out, process=False)
                    if MeshCleanProcess._is_mesh_valid_for_concat(repaired_mesh_trimesh_s2_iter, f"S2RepairedIter{i_s2}"):
                        current_mesh = repaired_mesh_trimesh_s2_iter
                        print(f"    V_IR: S2 After repair ops: V={current_mesh.vertices.shape[0]}, F={current_mesh.faces.shape[0]}")
                    else: print("    V_IR ERROR: S2 Repaired mesh invalid after loading."); break
                except Exception as e_repair_s2: print(f"    V_IR ERROR during S2 repair block: {e_repair_s2}"); break
                finally:
                    if ms_repair_s2 is not None: del ms_repair_s2

                selected_nm_faces_s2_after, selected_nm_vertices_s2_after = -1,-1
                ms_check_s2_after = None; temp_check_s2_out_after_path = ""
                v_s2_geom_after, f_s2_geom_after = -1,-1
                try:
                    ms_check_s2_after = pymeshlab.MeshSet()
                    temp_check_s2_out_after_path = make_local_temp_path(f"{temp_file_prefix}_s2_iter{i_s2}_check_out_after", debug_dir)
                    current_mesh.export(temp_check_s2_out_after_path); ms_check_s2_after.load_new_mesh(temp_check_s2_out_after_path)
                    if ms_check_s2_after.current_mesh_id() != -1 and ms_check_s2_after.current_mesh().vertex_number() > 0:
                        v_s2_geom_after = ms_check_s2_after.current_mesh().vertex_number(); f_s2_geom_after = ms_check_s2_after.current_mesh().face_number()
                        ms_check_s2_after.apply_filter('compute_selection_by_non_manifold_edges_per_face')
                        selected_nm_faces_s2_after = ms_check_s2_after.current_mesh().selected_face_number()
                        ms_check_s2_after.load_new_mesh(temp_check_s2_out_after_path)
                        ms_check_s2_after.apply_filter('select_non_manifold_vertices')
                        selected_nm_vertices_s2_after = ms_check_s2_after.current_mesh().selected_vertex_number()
                        print(f"    V_IR: S2 After repair check: V={v_s2_geom_after}, F={f_s2_geom_after}, NM Edge-Faces={selected_nm_faces_s2_after}, NM Verts={selected_nm_vertices_s2_after}")
                        if selected_nm_faces_s2_after == 0 and selected_nm_vertices_s2_after == 0:
                            print(f"    V_IR: S2 Non-manifold elements resolved."); stage1_resolved_nm = True; break
                        if v_s2_geom_after == prev_v_count_stagnant_s2 and f_s2_geom_after == prev_f_count_stagnant_s2 and \
                           selected_nm_faces_s2_after == prev_nm_faces_stagnant_s2 and selected_nm_vertices_s2_after == prev_nm_verts_stagnant_s2:
                            stagnation_count_s2 +=1
                            if stagnation_count_s2 >=2: print(f"    V_IR: S2 Repair stagnated for {stagnation_count_s2} iters. NM issues persist."); break
                        else: stagnation_count_s2 = 0
                        prev_v_count_stagnant_s2 = v_s2_geom_after; prev_f_count_stagnant_s2 = f_s2_geom_after
                        prev_nm_faces_stagnant_s2 = selected_nm_faces_s2_after; prev_nm_verts_stagnant_s2 = selected_nm_vertices_s2_after
                    else: print("    V_IR ERROR: S2 Mesh empty/invalid for post-repair check."); break
                except Exception as e_check_s2_after: print(f"    V_IR ERROR during S2 post-repair check: {e_check_s2_after}"); break
                finally:
                    if ms_check_s2_after is not None: del ms_check_s2_after
                if i_s2 == max_iterations_stage2 -1 : print(f"  V_IR: S2 Reached max iterations.")
        
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
    def process_mesh_graft_smplx_face_v5(
        full_body_mesh_path: str,
        output_path: str,
        smplx_face_mesh_path: str,
        projection_footprint_threshold: float = 0.01,
        footprint_dilation_rings: int = 1,
        body_simplification_target_faces: int = 12000,
        stitch_method: str = "body_driven_loft", # Now primarily "body_driven_loft" or implies "none"
        smplx_face_neck_loop_strategy: str = "full_face_silhouette",
        alignment_resample_count: int = 1000,
        loft_strip_resample_count: int = 100,
        max_seam_hole_fill_vertices: int = 250,
        final_polish_max_hole_edges: int = 100,
        iterative_repair_s1_iters: int = 5,
        iterative_repair_s2_iters: int = 5,
        iterative_repair_s2_remesh_percent: Optional[float] = None,
        spider_filter_area_factor: Optional[float] = 200.0,
        spider_filter_max_edge_len_factor: Optional[float] = 0.15,
        debug_dir: Optional[str] = None # Still used for tempfile.mkstemp directory
    ) -> Optional[trimesh.Trimesh]:
        INTERNAL_VERSION_TRACKER = "5.79_cleaned" # Updated version for clarity
        print(f"--- Starting Mesh Processing: Grafting SMPLX Face (V{INTERNAL_VERSION_TRACKER}) ---")
        temp_files_to_clean = []

        def make_temp_path(suffix:str, directory:Optional[str], use_ply: bool = False)->str:
            actual_suffix = suffix if suffix.startswith('_') else '_' + suffix
            file_ext = ".ply" if use_ply else ".obj"
            _fd, path = tempfile.mkstemp(suffix=actual_suffix + file_ext, dir=directory)
            os.close(_fd)
            temp_files_to_clean.append(path)
            return path

        try:
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)

            print("=== STEP 1 & 2: Load meshes and simplify body ===")
            original_smplx_face_geom_tri = trimesh.load_mesh(smplx_face_mesh_path, process=False)
            if not MeshCleanProcess._is_mesh_valid_for_concat(original_smplx_face_geom_tri, f"InitialSMPLXFace_V{INTERNAL_VERSION_TRACKER}"):
                print(f"CRITICAL V{INTERNAL_VERSION_TRACKER}: Initial SMPLX Face mesh is invalid. Aborting.")
                return None
            simplified_body_trimesh = trimesh.load_mesh(full_body_mesh_path, process=False)
            if not MeshCleanProcess._is_mesh_valid_for_concat(simplified_body_trimesh, f"InitialFullBody_V{INTERNAL_VERSION_TRACKER}"):
                print(f"CRITICAL V{INTERNAL_VERSION_TRACKER}: Initial Full Body mesh is invalid. Aborting.")
                return None
            original_smplx_face_geom_tri.fix_normals(); simplified_body_trimesh.fix_normals()

            if simplified_body_trimesh.faces.shape[0] > body_simplification_target_faces:
                body_to_simplify = simplified_body_trimesh
                temp_in = make_temp_path(f"b_v{INTERNAL_VERSION_TRACKER}_s_in", debug_dir, use_ply=True)
                temp_out = make_temp_path(f"b_v{INTERNAL_VERSION_TRACKER}_s_out", debug_dir, use_ply=True)
                loaded_s = None
                try:
                    if body_to_simplify.export(temp_in):
                        ms_s = pymeshlab.MeshSet(); ms_s.load_new_mesh(temp_in)
                        if ms_s.current_mesh().face_number() > 0:
                            ms_s.meshing_decimation_quadric_edge_collapse(targetfacenum=body_simplification_target_faces, preservenormal=True)
                            ms_s.save_current_mesh(temp_out)
                            loaded_s = trimesh.load_mesh(temp_out, process=False)
                    if MeshCleanProcess._is_mesh_valid_for_concat(loaded_s, f"SimpPML_V{INTERNAL_VERSION_TRACKER}"):
                        simplified_body_trimesh = loaded_s
                        simplified_body_trimesh.fix_normals()
                    else:
                        print(f"WARN V{INTERNAL_VERSION_TRACKER}: Body simplification failed or resulted in invalid mesh. Using original.")
                except Exception as e_simp:
                    print(f"WARN V{INTERNAL_VERSION_TRACKER}: Exception during body simplification: {e_simp}. Using original.")
            if not MeshCleanProcess._is_mesh_valid_for_concat(simplified_body_trimesh, f"FinalSimpBody_V{INTERNAL_VERSION_TRACKER}"):
                print(f"CRITICAL V{INTERNAL_VERSION_TRACKER}: Simplified body mesh is invalid. Aborting.")
                return None

            if iterative_repair_s1_iters > 0 or iterative_repair_s2_iters > 0:
                print("=== STEP 2.5: Iterative Non-Manifold Repair on Simplified Body ===")
                simplified_body_trimesh = MeshCleanProcess._iterative_non_manifold_repair_pml(
                    simplified_body_trimesh,
                    max_iterations_stage1=iterative_repair_s1_iters,
                    max_iterations_stage2=iterative_repair_s2_iters,
                    stage2_remeshing_length_percentage=iterative_repair_s2_remesh_percent,
                    debug_dir=debug_dir, # For _iterative_non_manifold_repair_pml's own temp files
                    temp_file_prefix=f"body_v{INTERNAL_VERSION_TRACKER}"
                )
                if not MeshCleanProcess._is_mesh_valid_for_concat(simplified_body_trimesh, f"BodyAfterIterativeRepair_V{INTERNAL_VERSION_TRACKER}"):
                    print(f"CRITICAL V{INTERNAL_VERSION_TRACKER}: Body mesh invalid after iterative repair. Aborting."); return None

            print("=== STEP 3: Determining Hole Faces and Creating Body with Hole ===")
            faces_to_remove_mask = np.zeros(len(simplified_body_trimesh.faces), dtype=bool)
            if original_smplx_face_geom_tri.vertices.shape[0] > 0 and hasattr(original_smplx_face_geom_tri, 'vertex_normals') and \
               original_smplx_face_geom_tri.vertex_normals.shape == original_smplx_face_geom_tri.vertices.shape:
                try:
                    _, d_cp, t_cp = trimesh.proximity.closest_point(simplified_body_trimesh, original_smplx_face_geom_tri.vertices)
                    if d_cp is not None and t_cp is not None and len(d_cp) == len(t_cp):
                        h_cp = t_cp[d_cp < projection_footprint_threshold]
                        if len(h_cp) > 0: faces_to_remove_mask[np.unique(h_cp)] = True
                    
                    offset = projection_footprint_threshold * 0.5
                    p_f = original_smplx_face_geom_tri.vertices + original_smplx_face_geom_tri.vertex_normals * offset
                    p_b = original_smplx_face_geom_tri.vertices - original_smplx_face_geom_tri.vertex_normals * offset
                    
                    _, _, t_f = trimesh.proximity.closest_point(simplified_body_trimesh, p_f)
                    if t_f is not None and len(t_f) > 0: faces_to_remove_mask[np.unique(t_f)] = True
                    
                    _, actual_dists_behind, t_b = trimesh.proximity.closest_point(simplified_body_trimesh, p_b)
                    if actual_dists_behind is not None and t_b is not None and len(actual_dists_behind) == len(t_b):
                        h_b = t_b[actual_dists_behind < projection_footprint_threshold]
                        if len(h_b) > 0: faces_to_remove_mask[np.unique(h_b)] = True
                except Exception as e_p:
                    print(f"WARN V{INTERNAL_VERSION_TRACKER}: Error during robust hole determination: {e_p}")

                if footprint_dilation_rings > 0 and np.any(faces_to_remove_mask):
                    adj = simplified_body_trimesh.face_adjacency
                    current_wavefront_mask = faces_to_remove_mask.copy()
                    for _ in range(footprint_dilation_rings):
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
                        truly_new_face_indices_to_add = unique_neighbors_this_ring[~faces_to_remove_mask[unique_neighbors_this_ring]]
                        if not truly_new_face_indices_to_add.size: break
                        faces_to_remove_mask[truly_new_face_indices_to_add] = True
                        current_wavefront_mask = np.zeros_like(faces_to_remove_mask)
                        current_wavefront_mask[truly_new_face_indices_to_add] = True
            
            temp_body_with_hole = simplified_body_trimesh.copy()
            if np.any(faces_to_remove_mask):
                faces_to_keep_mask = ~faces_to_remove_mask
                if np.any(faces_to_keep_mask): # Ensure we don't remove all faces
                    temp_body_with_hole = trimesh.Trimesh(vertices=simplified_body_trimesh.vertices, faces=simplified_body_trimesh.faces[faces_to_keep_mask])
                    temp_body_with_hole.remove_unreferenced_vertices()
                    temp_body_with_hole.fix_normals()
                # If all faces were to be removed, temp_body_with_hole remains a copy of simplified_body_trimesh
            
            body_with_hole_trimesh = temp_body_with_hole
            if MeshCleanProcess._is_mesh_valid_for_concat(temp_body_with_hole, f"TempBodyWHolePreSplit_V{INTERNAL_VERSION_TRACKER}"):
                if hasattr(temp_body_with_hole, 'split') and callable(temp_body_with_hole.split):
                    components = temp_body_with_hole.split(only_watertight=False)
                    if components:
                        largest_comp = max(components, key=lambda c: len(c.faces if hasattr(c, 'faces') and c.faces is not None else []))
                        if MeshCleanProcess._is_mesh_valid_for_concat(largest_comp, f"LargestCompBodyWHole_V{INTERNAL_VERSION_TRACKER}"):
                            body_with_hole_trimesh = largest_comp
            
            if not MeshCleanProcess._is_mesh_valid_for_concat(body_with_hole_trimesh, f"FinalBodyWHole_V{INTERNAL_VERSION_TRACKER}"):
                print(f"WARN V{INTERNAL_VERSION_TRACKER}: Body with hole is invalid. Attempting to use simplified body directly for concatenation.")
                body_with_hole_trimesh = simplified_body_trimesh.copy() # Fallback

            final_mesh_output = None
            if stitch_method == "body_driven_loft":
                print(f"=== STEP 4 & 5: Attempting 'BODY_DRIVEN_LOFT' ===")
                s_loop_coords_ordered, b_loop_coords_aligned = None, None
                can_proceed_to_stitch_stages = False
                ordered_s_vidx = None

                if smplx_face_neck_loop_strategy == "full_face_silhouette":
                    face_boundary_edges = MeshCleanProcess._get_boundary_edges_manually_from_faces(original_smplx_face_geom_tri)
                    if face_boundary_edges is not None and len(face_boundary_edges) >= 3:
                        all_face_boundary_components_vidx = trimesh.graph.connected_components(face_boundary_edges, min_len=3)
                        valid_face_components = [comp for comp in all_face_boundary_components_vidx if comp is not None and len(comp) >= 3]
                        if valid_face_components:
                            s_silhouette_comp_vidx_unord = max(valid_face_components, key=len, default=np.array([], dtype=int))
                            if len(s_silhouette_comp_vidx_unord) >= 3:
                                silhouette_set = set(s_silhouette_comp_vidx_unord)
                                edges_for_silhouette_comp = [e for e in face_boundary_edges if e[0] in silhouette_set and e[1] in silhouette_set]
                                if edges_for_silhouette_comp:
                                    ordered_s_vidx = MeshCleanProcess._order_loop_vertices_from_edges(
                                        f"SMPLX_Full_Silhouette_Order_V{INTERNAL_VERSION_TRACKER}", s_silhouette_comp_vidx_unord, np.array(edges_for_silhouette_comp))

                if ordered_s_vidx is not None and len(ordered_s_vidx) >= 3:
                    if ordered_s_vidx.max() < len(original_smplx_face_geom_tri.vertices) and ordered_s_vidx.min() >= 0:
                        s_loop_coords_ordered = original_smplx_face_geom_tri.vertices[ordered_s_vidx]
                    else: ordered_s_vidx = None # Invalidate if indices out of bounds
                
                if s_loop_coords_ordered is not None and len(s_loop_coords_ordered) >= 3:
                    body_hole_defining_edges = MeshCleanProcess._get_hole_boundary_edges_from_removed_faces(simplified_body_trimesh, faces_to_remove_mask)
                    unord_b_comp_selected_vidx = np.array([], dtype=int)
                    if body_hole_defining_edges is not None and len(body_hole_defining_edges) >= 3:
                        hole_boundary_components_vidx_list = trimesh.graph.connected_components(body_hole_defining_edges, min_len=3)
                        hole_comps_filtered = [comp for comp in hole_boundary_components_vidx_list if comp is not None and len(comp) >= 3]
                        if hole_comps_filtered:
                            best_loop_prox_vidx = None; best_score_prox = np.inf
                            for comp_idx, current_comp_vidx in enumerate(hole_comps_filtered):
                                if current_comp_vidx.max() >= len(simplified_body_trimesh.vertices) or current_comp_vidx.min() < 0: continue
                                loop_coords_candidate = simplified_body_trimesh.vertices[current_comp_vidx]
                                if len(loop_coords_candidate) < 3: continue
                                try:
                                    tree_candidate = cKDTree(loop_coords_candidate)
                                    dists_to_candidate, _ = tree_candidate.query(s_loop_coords_ordered, k=1)
                                    current_score_prox = np.mean(dists_to_candidate)
                                    if current_score_prox < best_score_prox: best_score_prox = current_score_prox; best_loop_prox_vidx = current_comp_vidx
                                except Exception: continue
                            if best_loop_prox_vidx is not None: unord_b_comp_selected_vidx = best_loop_prox_vidx
                            elif hole_comps_filtered: unord_b_comp_selected_vidx = max(hole_comps_filtered, key=len, default=np.array([], dtype=int))
                    
                    ordered_b_vidx_footprint = None
                    if len(unord_b_comp_selected_vidx) >= 3:
                        b_comp_set = set(unord_b_comp_selected_vidx)
                        if body_hole_defining_edges is not None and len(body_hole_defining_edges) > 0:
                            b_edges_for_selected_comp = [e for e in body_hole_defining_edges if e[0] in b_comp_set and e[1] in b_comp_set]
                            if b_edges_for_selected_comp:
                                ordered_b_vidx_footprint = MeshCleanProcess._order_loop_vertices_from_edges(
                                    f"BodyHoleFootprint_Order_V{INTERNAL_VERSION_TRACKER}", unord_b_comp_selected_vidx, np.array(b_edges_for_selected_comp))
                    
                    if ordered_b_vidx_footprint is not None and len(ordered_b_vidx_footprint) >= 3:
                        if ordered_b_vidx_footprint.max() < len(simplified_body_trimesh.vertices) and ordered_b_vidx_footprint.min() >= 0:
                            b_loop_coords_ordered_pre_align = simplified_body_trimesh.vertices[ordered_b_vidx_footprint]
                            b_loop_coords_to_align = b_loop_coords_ordered_pre_align.copy()
                            if len(s_loop_coords_ordered) > 0 and len(b_loop_coords_to_align) > 0:
                                s_start = s_loop_coords_ordered[0]; kdt_b_align = cKDTree(b_loop_coords_to_align)
                                _, closest_idx_on_b_to_s_start = kdt_b_align.query(s_start, k=1)
                                b_loop_coords_rolled = np.roll(b_loop_coords_to_align, -closest_idx_on_b_to_s_start, axis=0)
                                if len(s_loop_coords_ordered) >= 2 and len(b_loop_coords_rolled) >= 2 and alignment_resample_count >= 2:
                                    s_r_a = MeshCleanProcess.resample_polyline_to_count(s_loop_coords_ordered, alignment_resample_count)
                                    b_r_f = MeshCleanProcess.resample_polyline_to_count(b_loop_coords_rolled, alignment_resample_count)
                                    b_r_b = MeshCleanProcess.resample_polyline_to_count(b_loop_coords_rolled[::-1], alignment_resample_count)
                                    if s_r_a is not None and b_r_f is not None and b_r_b is not None and \
                                       len(s_r_a) == alignment_resample_count and len(b_r_f) == alignment_resample_count and len(b_r_b) == alignment_resample_count:
                                        dist_fwd = np.sum(np.linalg.norm(s_r_a - b_r_f, axis=1))
                                        dist_bwd = np.sum(np.linalg.norm(s_r_a - b_r_b, axis=1))
                                        if dist_bwd < dist_fwd: b_loop_coords_aligned = b_loop_coords_rolled[::-1].copy()
                                        else: b_loop_coords_aligned = b_loop_coords_rolled.copy()
                                    else: b_loop_coords_aligned = b_loop_coords_rolled.copy() # Fallback if resampling fails
                                else: b_loop_coords_aligned = b_loop_coords_rolled.copy()
                            else: b_loop_coords_aligned = b_loop_coords_to_align.copy()
                
                if b_loop_coords_aligned is not None and len(b_loop_coords_aligned) >= 3 and \
                   s_loop_coords_ordered is not None and len(s_loop_coords_ordered) >= 3:
                    can_proceed_to_stitch_stages = True

                if can_proceed_to_stitch_stages:
                    new_stitch_triangles_list = []
                    stitch_strip_vertices_np = None
                    
                    # Body Driven Loft Logic
                    resampled_s_target = MeshCleanProcess.resample_polyline_to_count(s_loop_coords_ordered, loft_strip_resample_count)
                    if resampled_s_target is None or len(resampled_s_target) < 2:
                        can_proceed_to_stitch_stages = False
                    else:
                        kdt_s_target = cKDTree(resampled_s_target)
                        num_b_pts_actual = len(b_loop_coords_aligned)
                        stitch_strip_vertices_np = np.vstack((b_loop_coords_aligned, resampled_s_target))
                        for i in range(num_b_pts_actual):
                            b_curr_idx = i
                            b_next_idx = (i + 1) % num_b_pts_actual
                            _, s_match_idx_curr = kdt_s_target.query(b_loop_coords_aligned[b_curr_idx], k=1)
                            _, s_match_idx_next = kdt_s_target.query(b_loop_coords_aligned[b_next_idx], k=1)
                            v0 = b_curr_idx
                            v1 = b_next_idx
                            v2 = s_match_idx_next + num_b_pts_actual
                            v3 = s_match_idx_curr + num_b_pts_actual
                            new_stitch_triangles_list.extend([[v0, v1, v2], [v0, v2, v3]])
                    
                    if can_proceed_to_stitch_stages and new_stitch_triangles_list and stitch_strip_vertices_np is not None:
                        stitch_strip_faces_np = np.array(new_stitch_triangles_list, dtype=int)
                        if stitch_strip_vertices_np.ndim == 2 and stitch_strip_vertices_np.shape[0] > 0 and stitch_strip_vertices_np.shape[1] == 3 and \
                           stitch_strip_faces_np.ndim == 2 and stitch_strip_faces_np.shape[0] > 0 and stitch_strip_faces_np.shape[1] == 3:
                            strip_mesh_obj = trimesh.Trimesh(vertices=stitch_strip_vertices_np, faces=stitch_strip_faces_np, process=False)
                            if MeshCleanProcess._is_mesh_valid_for_concat(strip_mesh_obj, f"RawStitchStripV{INTERNAL_VERSION_TRACKER}"):
                                if hasattr(strip_mesh_obj, 'face_normals') and strip_mesh_obj.face_normals is not None and len(strip_mesh_obj.face_normals) > 0:
                                    strip_centroid = strip_mesh_obj.centroid
                                    inward_pointing_normals_count = 0
                                    for face_idx in range(len(strip_mesh_obj.faces)):
                                        vector_to_strip_centroid = strip_centroid - strip_mesh_obj.triangles_center[face_idx]
                                        if np.dot(strip_mesh_obj.face_normals[face_idx], vector_to_strip_centroid) > 1e-6:
                                            inward_pointing_normals_count += 1
                                    if inward_pointing_normals_count > len(strip_mesh_obj.faces) / 2:
                                        strip_mesh_obj.invert()
                            
                            if MeshCleanProcess._is_mesh_valid_for_concat(strip_mesh_obj, f"StitchStripV{INTERNAL_VERSION_TRACKER}"):
                                valid_cs = [m for m in [original_smplx_face_geom_tri, body_with_hole_trimesh, strip_mesh_obj] if MeshCleanProcess._is_mesh_valid_for_concat(m, f"LoftFinalCompV{INTERNAL_VERSION_TRACKER}")]
                                if len(valid_cs) == 3:
                                    try:
                                        cand_m = trimesh.util.concatenate(valid_cs)
                                        if MeshCleanProcess._is_mesh_valid_for_concat(cand_m, f"ConcatLoftResV{INTERNAL_VERSION_TRACKER}"):
                                            final_m_proc = cand_m.copy()
                                            final_m_proc.merge_vertices(merge_tex=False, merge_norm=False)
                                            final_m_proc.remove_unreferenced_vertices()
                                            final_m_proc.remove_degenerate_faces()
                                            if MeshCleanProcess._is_mesh_valid_for_concat(final_m_proc, f"ProcessedLoftResV{INTERNAL_VERSION_TRACKER}"):
                                                final_mesh_output = final_m_proc
                                            else:
                                                final_mesh_output = cand_m # Use concatenated if processed is invalid
                                            if final_mesh_output is not None and not final_mesh_output.is_empty:
                                                print(f"INFO V{INTERNAL_VERSION_TRACKER}: Stitch method '{stitch_method}' applied successfully.")
                                            else: final_mesh_output = None # Mark as failed
                                    except Exception as e_f_cat:
                                        print(f"CRITICAL_EXCEPTION V{INTERNAL_VERSION_TRACKER} FinalConcatMerge: {e_f_cat}"); traceback.print_exc()
                                        final_mesh_output = None
            
            if final_mesh_output is None:
                if stitch_method == "body_driven_loft": # Only print if lofting was attempted
                    print(f"INFO V{INTERNAL_VERSION_TRACKER}: Stitch method '{stitch_method}' did not produce a result or was not applicable. Defaulting to simple concatenation.")
                else:
                    print(f"INFO V{INTERNAL_VERSION_TRACKER}: Stitch method '{stitch_method}' not 'body_driven_loft'. Proceeding with simple concatenation.")

                valid_fb_comps = [m for m in [original_smplx_face_geom_tri, body_with_hole_trimesh] if MeshCleanProcess._is_mesh_valid_for_concat(m, f"FallbackCompV{INTERNAL_VERSION_TRACKER}")]
                if not valid_fb_comps:
                    print(f"FATAL V{INTERNAL_VERSION_TRACKER}: No valid fallback components for concatenation."); return None
                final_mesh_output = trimesh.util.concatenate(valid_fb_comps)
                if not MeshCleanProcess._is_mesh_valid_for_concat(final_mesh_output, f"FallbackConcatV{INTERNAL_VERSION_TRACKER}"):
                    print(f"FATAL V{INTERNAL_VERSION_TRACKER}: Fallback concatenation resulted in invalid mesh."); return None
                final_mesh_output.fix_normals()
                print(f"=== STEP 5 (Effective): Stitch method resulted in 'none' (simple concatenation) ===")

            if final_mesh_output is not None and MeshCleanProcess._is_mesh_valid_for_concat(final_mesh_output, f"MeshBeforeSeamFill_V{INTERNAL_VERSION_TRACKER}") and stitch_method == "body_driven_loft":
                print(f"INFO V{INTERNAL_VERSION_TRACKER}: Attempting EAR-CLIP SEAM HOLE FILL post-loft...")
                mesh_to_fill = final_mesh_output.copy()
                if not hasattr(mesh_to_fill, 'face_normals') or mesh_to_fill.face_normals is None or len(mesh_to_fill.face_normals) != len(mesh_to_fill.faces):
                    mesh_to_fill.fix_normals()
                original_vertices_for_fill = mesh_to_fill.vertices.copy(); current_faces_list = list(mesh_to_fill.faces); added_any_fill_faces_this_pass = False
                s_loop_ref_coords_for_prox_check = s_loop_coords_ordered # From lofting stage
                b_loop_ref_coords_for_prox_check = b_loop_coords_aligned # From lofting stage
                
                all_ordered_loops_on_stitched_mesh = MeshCleanProcess.get_all_boundary_loops(mesh_to_fill, min_loop_len=3)
                kdt_s_ref, kdt_b_ref = None, None
                if s_loop_ref_coords_for_prox_check is not None and len(s_loop_ref_coords_for_prox_check) > 0: kdt_s_ref = cKDTree(s_loop_ref_coords_for_prox_check)
                if b_loop_ref_coords_for_prox_check is not None and len(b_loop_ref_coords_for_prox_check) > 0: kdt_b_ref = cKDTree(b_loop_ref_coords_for_prox_check)
                
                proximity_to_seam_threshold_fill = 0.025 # Threshold for identifying seam holes
                for loop_idx, current_hole_v_indices_ordered in enumerate(all_ordered_loops_on_stitched_mesh):
                    if not (current_hole_v_indices_ordered is not None and 3 <= len(current_hole_v_indices_ordered) <= max_seam_hole_fill_vertices): continue
                    if current_hole_v_indices_ordered.max() >= len(original_vertices_for_fill) or current_hole_v_indices_ordered.min() < 0: continue
                    
                    current_hole_coords_3d = original_vertices_for_fill[current_hole_v_indices_ordered]; is_seam_hole_for_fill = False
                    if kdt_s_ref is not None and kdt_b_ref is not None:
                        d_s, _ = kdt_s_ref.query(current_hole_coords_3d, k=1, distance_upper_bound=proximity_to_seam_threshold_fill * 2)
                        avg_d_s = np.mean(d_s[np.isfinite(d_s)]) if np.any(np.isfinite(d_s)) else float('inf')
                        d_b, _ = kdt_b_ref.query(current_hole_coords_3d, k=1, distance_upper_bound=proximity_to_seam_threshold_fill * 2)
                        avg_d_b = np.mean(d_b[np.isfinite(d_b)]) if np.any(np.isfinite(d_b)) else float('inf')
                        if avg_d_s < proximity_to_seam_threshold_fill and avg_d_b < proximity_to_seam_threshold_fill:
                            is_seam_hole_for_fill = True
                    
                    if is_seam_hole_for_fill:
                        try:
                            if len(current_hole_coords_3d) < 3: continue
                            plane_origin_loop, normal_loop = MeshCleanProcess.get_dominant_plane(current_hole_coords_3d)
                            if np.allclose(normal_loop, [0,0,0]): continue
                            transform_to_2d = trimesh.geometry.plane_transform(plane_origin_loop, normal_loop)
                            loop_coords_2d_projected = trimesh.transform_points(current_hole_coords_3d, transform_to_2d)[:, :2]
                            if len(np.unique(loop_coords_2d_projected, axis=0)) < 3: continue # Degenerate 2D polygon
                            
                            path_2d_object = trimesh.path.Path2D(entities=[trimesh.path.entities.Line(np.arange(len(loop_coords_2d_projected)))], vertices=loop_coords_2d_projected)
                            patch_faces_local_idx, patch_vertices_2d = None, None
                            try: patch_faces_local_idx, patch_vertices_2d = trimesh.creation.triangulate_polygon(path_2d_object, triangle_args='p') # Try with 'p' for constrained Delaunay
                            except Exception:
                                try: patch_faces_local_idx, patch_vertices_2d = trimesh.creation.triangulate_polygon(path_2d_object) # Fallback
                                except Exception: continue # Triangulation failed
                            
                            if patch_faces_local_idx is not None and len(patch_faces_local_idx) > 0:
                                if len(patch_vertices_2d) == len(loop_coords_2d_projected) and np.allclose(patch_vertices_2d, loop_coords_2d_projected, atol=1e-5):
                                    new_fill_faces_global_candidate = current_hole_v_indices_ordered[patch_faces_local_idx]
                                    # Attempt to align normals of the patch with adjacent mesh
                                    temp_patch = trimesh.Trimesh(vertices=original_vertices_for_fill, faces=new_fill_faces_global_candidate, process=False); temp_patch.fix_normals()
                                    if len(temp_patch.faces) > 0 and len(current_hole_v_indices_ordered) >= 2:
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
                                                            if np.dot(normal_patch_adj_face, normal_existing_adj_face) < 0.1: # If normals are opposing
                                                                new_fill_faces_global_candidate = new_fill_faces_global_candidate[:, ::-1] # Flip patch normals
                                        except (ValueError, IndexError, AttributeError): pass # Non-critical if normal alignment fails for an edge
                                    current_faces_list.extend(new_fill_faces_global_candidate); added_any_fill_faces_this_pass = True
                        except Exception as e_ear_clip:
                            print(f"WARN V{INTERNAL_VERSION_TRACKER}: Error during ear-clip processing for loop {loop_idx}: {e_ear_clip}"); traceback.print_exc()
                
                if added_any_fill_faces_this_pass:
                    updated_mesh_after_fill = trimesh.Trimesh(vertices=original_vertices_for_fill, faces=np.array(current_faces_list, dtype=int), process=False)
                    updated_mesh_after_fill.merge_vertices(merge_tex=False, merge_norm=False); updated_mesh_after_fill.remove_unreferenced_vertices(); updated_mesh_after_fill.remove_degenerate_faces(); updated_mesh_after_fill.fix_normals()
                    if MeshCleanProcess._is_mesh_valid_for_concat(updated_mesh_after_fill, f"MeshAfterEarClipFill_V{INTERNAL_VERSION_TRACKER}"):
                        final_mesh_output = updated_mesh_after_fill
                    else:
                        print(f"WARN V{INTERNAL_VERSION_TRACKER}: Mesh became invalid after ear-clip fill. Reverting to pre-fill mesh.")


            if MeshCleanProcess._is_mesh_valid_for_concat(final_mesh_output, f"MeshBeforeFinalPolish_V{INTERNAL_VERSION_TRACKER}"):
                print(f"=== STEP 5.5: Applying Final Polish ===")
                temp_polish_in = make_temp_path(f"final_polish_in_v{INTERNAL_VERSION_TRACKER}", debug_dir, use_ply=True)
                temp_polish_out = make_temp_path(f"final_polish_out_v{INTERNAL_VERSION_TRACKER}", debug_dir, use_ply=True)
                polished_mesh_loaded = None; ms_polish = None
                try:
                    final_mesh_output.export(temp_polish_in)
                    ms_polish = pymeshlab.MeshSet(); ms_polish.load_new_mesh(temp_polish_in)
                    if ms_polish.current_mesh_id() != -1 and ms_polish.current_mesh().vertex_number() > 0:
                        ms_polish.meshing_remove_duplicate_vertices()
                        try: ms_polish.meshing_repair_non_manifold_edges(method='Split Vertices')
                        except pymeshlab.PyMeshLabException:
                            try: ms_polish.meshing_repair_non_manifold_edges(method='Remove Faces')
                            except pymeshlab.PyMeshLabException: print(f"INFO V{INTERNAL_VERSION_TRACKER}: Polish: Both non-manifold edge repair methods failed.")
                        
                        is_manifold_for_hole_closing = False # Check manifoldness before closing holes
                        try:
                            topo_measures = ms_polish.get_topological_measures()
                            nm_edges_metric = topo_measures.get('non_manifold_edges', -1)
                            if nm_edges_metric == 0: is_manifold_for_hole_closing = True
                            # Add more robust check if metric is not available or inconclusive
                        except Exception: pass # Non-critical if check fails, proceed cautiously

                        if is_manifold_for_hole_closing:
                            try: ms_polish.meshing_close_holes(maxholesize=final_polish_max_hole_edges, newfaceselected=False)
                            except pymeshlab.PyMeshLabException as e_close_holes: print(f"INFO V{INTERNAL_VERSION_TRACKER}: Polish: meshing_close_holes failed: {e_close_holes}")
                        else: print(f"INFO V{INTERNAL_VERSION_TRACKER}: Polish: Skipping meshing_close_holes as mesh not determined to be sufficiently manifold.")
                        ms_polish.meshing_remove_unreferenced_vertices(); ms_polish.compute_normal_per_face()
                        ms_polish.save_current_mesh(temp_polish_out)
                        polished_mesh_loaded = trimesh.load_mesh(temp_polish_out, process=False)
                    
                    if MeshCleanProcess._is_mesh_valid_for_concat(polished_mesh_loaded, f"PolishedMeshFromPML_V{INTERNAL_VERSION_TRACKER}"):
                        final_mesh_output = polished_mesh_loaded; final_mesh_output.fix_normals()
                        print(f"INFO V{INTERNAL_VERSION_TRACKER}: Final polish step applied.")
                    else: print(f"WARN V{INTERNAL_VERSION_TRACKER}: Final polish resulted in invalid/empty mesh. Keeping pre-polish mesh.")
                except Exception as e_polish: print(f"WARN V{INTERNAL_VERSION_TRACKER}: Error during main final polish step: {e_polish}"); traceback.print_exc()
                finally:
                    if ms_polish is not None: del ms_polish

            if final_mesh_output is not None and MeshCleanProcess._is_mesh_valid_for_concat(final_mesh_output, f"MeshBeforeSpiderFilter_V{INTERNAL_VERSION_TRACKER}"):
                print(f"=== STEP 5.75: Filtering Spider-Web Triangles ===")
                target_faces_for_spider_filter = None
                if MeshCleanProcess._is_mesh_valid_for_concat(original_smplx_face_geom_tri, "OrigFaceForSpiderRefBounds"):
                    if len(final_mesh_output.faces) > 0 and len(original_smplx_face_geom_tri.vertices) > 0:
                        final_mesh_centroids = final_mesh_output.triangles_center
                        smplx_bounds_min, smplx_bounds_max = original_smplx_face_geom_tri.bounds
                        padding = 0.05 # Example padding
                        min_b = smplx_bounds_min - padding; max_b = smplx_bounds_max + padding
                        candidate_indices = [idx for idx, centroid in enumerate(final_mesh_centroids) if 
                                             (min_b[0] <= centroid[0] <= max_b[0] and
                                              min_b[1] <= centroid[1] <= max_b[1] and
                                              min_b[2] <= centroid[2] <= max_b[2])]
                        if candidate_indices: target_faces_for_spider_filter = np.array(candidate_indices, dtype=int)
                
                effective_max_edge_length = None
                if spider_filter_max_edge_len_factor is not None and spider_filter_max_edge_len_factor > 0:
                    if MeshCleanProcess._is_mesh_valid_for_concat(simplified_body_trimesh, "BodyForSpiderEdgeRef") and len(simplified_body_trimesh.edges) > 0:
                        body_edge_lengths = simplified_body_trimesh.edges_unique_length
                        if len(body_edge_lengths) > 0:
                            effective_max_edge_length = np.max(body_edge_lengths) * (1.0 + spider_filter_max_edge_len_factor)

                ref_for_stats = original_smplx_face_geom_tri if MeshCleanProcess._is_mesh_valid_for_concat(original_smplx_face_geom_tri, "RefForSpiderAreaStats") else None
                
                final_mesh_output = MeshCleanProcess._filter_large_triangles_from_fill(
                    final_mesh_output,
                    target_face_indices=target_faces_for_spider_filter,
                    max_allowed_edge_length=effective_max_edge_length,
                    max_allowed_area_factor=spider_filter_area_factor,
                    reference_mesh_for_stats=ref_for_stats,
                    mesh_name_for_debug=f"FinalOutput_V{INTERNAL_VERSION_TRACKER}"
                )
                if not MeshCleanProcess._is_mesh_valid_for_concat(final_mesh_output, f"MeshAfterSpiderFilter_V{INTERNAL_VERSION_TRACKER}"):
                    print(f"CRITICAL Error V{INTERNAL_VERSION_TRACKER}: Mesh invalid after spider-web filter. CANNOT SAVE."); return None
                final_mesh_output.fix_normals()

            if not MeshCleanProcess._is_mesh_valid_for_concat(final_mesh_output, f"Mesh Before Final Save V{INTERNAL_VERSION_TRACKER}"):
                print(f"CRITICAL Error V{INTERNAL_VERSION_TRACKER}: final_mesh_output is invalid or empty before final save. CANNOT SAVE."); return None

            final_mesh_output.export(output_path)
            print(f"--- Mesh Grafting V{INTERNAL_VERSION_TRACKER} Finished. Output: {output_path} ---")
            return final_mesh_output

        except Exception as e_main_vXXX:
            print(f"--- V{INTERNAL_VERSION_TRACKER} Failed (Outer Try-Except Block) --- {e_main_vXXX}")
            traceback.print_exc()
            return None
        finally:
            for temp_path in temp_files_to_clean:
                if temp_path and os.path.exists(temp_path):
                    try: os.remove(temp_path)
                    except OSError: pass