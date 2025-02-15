def count_polygons(obj_file_path):
    """
    Counts the number of polygons (faces) in a mesh stored in an .obj file.

    :param obj_file_path: Path to the .obj file
    :return: Number of polygons (faces) in the mesh
    """
    face_count = 0

    try:
        with open(obj_file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('f '):  # Face definition
                    face_count += 1
        return face_count
    except FileNotFoundError:
        print(f"Error: File '{obj_file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None
    

obj_files = [
    "/home/ubuntu/projects/induxr/ECON/results/econ/obj/fulden_tpose_f1_0_final.obj"
] 
for f in obj_files:
    counts = count_polygons(f)
    print(f)
    print(f'Polygons counts: {counts}')

