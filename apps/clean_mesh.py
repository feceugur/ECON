import pymeshlab


class MeshWatertightifier:
    def __init__(self, input_path, output_path):
        """
        Initialize the MeshWatertightifier class.

        Parameters:
            input_path (str): Path to the input mesh file.
            output_path (str): Path to save the watertight mesh.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.ms = pymeshlab.MeshSet()


    def load_mesh(self):
        """Loads the mesh into the MeshSet."""
        self.ms.load_new_mesh(self.input_path)

    def clean_mesh(self):
        """Cleans the mesh by removing duplicate vertices, faces, and merging close vertices."""
        self.ms.meshing_remove_duplicate_faces()
        self.ms.meshing_remove_duplicate_vertices()
        self.ms. meshing_repair_non_manifold_edges()
        self.ms.meshing_repair_non_manifold_vertices()
        self.ms.meshing_remove_unreferenced_vertices()


    def fill_holes(self):
        """
        Fills holes in the mesh.

        Parameters:
            max_hole_size (int): Maximum size of holes to fill.
        """
        self.ms.meshing_close_holes()

    def reconstruct_surface(self, method='poisson', **kwargs):
        """
        Reconstructs the surface to make the mesh watertight.

        Parameters:
            method (str): Reconstruction method ('poisson' or 'ball_pivoting').
            **kwargs: Additional parameters for the reconstruction method.
        """
        if method == 'poisson':
            depth = kwargs.get('depth', 8)
            self.ms.generate_surface_reconstruction_screened_poisson(depth=depth)
        else:
            raise ValueError("Unsupported reconstruction method. Use 'poisson' or 'ball_pivoting'.")

    def check_watertight(self):
        """
        Checks if the mesh is watertight.

        Returns:
            bool: True if the mesh is watertight, False otherwise.
        """
        geo_metrics = self.ms.get_geometric_measures()
        metrics = self.ms.get_topological_measures()
        return (
            metrics['boundary_edges'] == 0 and
            metrics['number_holes'] == 0 and
            metrics['is_mesh_two_manifold'] and
            metrics['non_two_manifold_edges'] == 0 and
            metrics['non_two_manifold_vertices'] == 0
        )

    def save_mesh(self):
        """Saves the processed mesh to the output path."""
        self.ms.save_current_mesh(self.output_path)

    def process(self, reconstruction_method='poisson', **kwargs):
        """
        Full pipeline to make the mesh watertight.

        Parameters:
            reconstruction_method (str): Reconstruction method ('poisson' or 'ball_pivoting').
            **kwargs: Additional parameters for the reconstruction method.

        Returns:
            bool: True if the processed mesh is watertight, False otherwise.
        """
        self.load_mesh()
        self.clean_mesh()
        self.fill_holes()
        self.reconstruct_surface(method=reconstruction_method, **kwargs)
        is_watertight = self.check_watertight()
        self.save_mesh()
        return is_watertight
