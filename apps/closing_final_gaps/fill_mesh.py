#!/usr/bin/env python3
import sys
import os
import numpy as np
import trimesh


def find_boundary_loops(edges):
    """
    Given an (m×2) array of boundary edges (vertex indices),
    return a list of loops, each a list of vertex indices in order.
    """
    adj = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    visited = set()
    loops = []
    for u, v in edges:
        if (u, v) in visited or (v, u) in visited:
            continue
        loop = [u, v]
        visited |= {(u, v), (v, u)}
        prev, curr = u, v
        while True:
            nbrs = [n for n in adj[curr] if n != prev]
            if not nbrs or nbrs[0] == loop[0]:
                break
            nxt = nbrs[0]
            loop.append(nxt)
            visited |= {(curr, nxt), (nxt, curr)}
            prev, curr = curr, nxt
        loops.append(loop)
    return loops


def loop_area(mesh, loop):
    """
    Compute approximate area of the hole defined by 'loop' on the mesh.
    Projects the loop onto its best-fit plane and uses the shoelace formula.
    """
    verts = mesh.vertices[np.array(loop)]  # (N,3)
    # best-fit plane via SVD
    centroid = verts.mean(axis=0)
    coords = verts - centroid
    # SVD on coords
    u, s, vh = np.linalg.svd(coords, full_matrices=False)
    # plane axes are first two columns of vh
    axis1, axis2 = vh[0], vh[1]
    # project points to 2D
    xy = np.vstack([coords.dot(axis1), coords.dot(axis2)]).T  # (N,2)
    # compute area via shoelace
    x, y = xy[:,0], xy[:,1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return area


def fill_loop(mesh, loop):
    """
    Fill a single boundary loop on a copy of the mesh and return the new mesh.
    """
    verts = mesh.vertices.tolist()
    faces = mesh.faces.tolist()

    coords = np.array([verts[i] for i in loop])
    centroid = coords.mean(axis=0)
    c_idx = len(verts)
    verts.append(centroid.tolist())

    for i in range(len(loop)):
        v1 = loop[i]
        v2 = loop[(i + 1) % len(loop)]
        faces.append([c_idx, v1, v2])

    new_mesh = trimesh.Trimesh(vertices=np.array(verts),
                               faces=np.array(faces),
                               process=False)
    # cleanup
    nondeg = new_mesh.nondegenerate_faces(height=1e-8)
    new_mesh.update_faces(nondeg)
    unique = new_mesh.unique_faces()
    new_mesh.update_faces(unique)
    new_mesh.remove_unreferenced_vertices()
    return new_mesh


def main():
    if len(sys.argv) < 2:
        print("Usage: python fill_mesh.py input.obj [output_dir]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.getcwd()
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_path))[0]

    mesh = trimesh.load(input_path, process=False)
    # initial cleanup
    nondeg_init = mesh.nondegenerate_faces(height=1e-8)
    mesh.update_faces(nondeg_init)
    unique_init = mesh.unique_faces()
    mesh.update_faces(unique_init)
    mesh.remove_unreferenced_vertices()

    # detect boundary loops
    unique_edges = mesh.edges_unique
    face_edges = mesh.faces_unique_edges
    counts = np.bincount(face_edges.reshape(-1), minlength=len(unique_edges))
    boundary_edges = unique_edges[counts == 1]
    loops = find_boundary_loops(boundary_edges)

    print(f"► Loaded '{input_path}'. Watertight? {mesh.is_watertight}")
    print(f"► Detected {len(loops)} hole(s).")

    # compute areas for each loop
    areas = [loop_area(mesh, loop) for loop in loops]
    for i, area in enumerate(areas, 1):
        print(f"   Hole {i}: {len(loops[i-1])} edges, area ≈ {area:.4f}")

    # skip the loop with the largest area (likely the open back)
    if areas:
        skip_idx = int(np.argmax(areas))
        print(f"► Skipping hole #{skip_idx+1} (largest area) with area ≈ {areas[skip_idx]:.4f}")
    else:
        skip_idx = None

    loops_to_fill = [loop for idx, loop in enumerate(loops) if idx != skip_idx]

    # export mesh with selected holes filled
    full_mesh = mesh.copy()
    for loop in loops_to_fill:
        full_mesh = fill_loop(full_mesh, loop)
    full_path = os.path.join(output_dir, f"{base_name}_filled_selected.obj")
    full_mesh.export(full_path, file_type='obj')
    print(f"► Exported selected holes filled as: {full_path}")

    # export individual fills, skipping largest
    for i, loop in enumerate(loops, 1):
        if skip_idx is not None and (i-1) == skip_idx:
            continue
        part_mesh = mesh.copy()
        part_filled = fill_loop(part_mesh, loop)
        hole_path = os.path.join(output_dir, f"{base_name}_filled_hole_{i}.obj")
        part_filled.export(hole_path, file_type='obj')
        print(f"► Exported hole {i} filled: {hole_path}")

if __name__ == "__main__":
    main()