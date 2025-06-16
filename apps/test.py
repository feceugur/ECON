# Assuming these imports are already at the top of your file
import torch
import trimesh
import os
from termcolor import colored
from tqdm import tqdm
import torchvision

# Assuming these are your custom or library functions, correctly imported
# from your_project.smpl import SMPLX_object
# from your_project.render import Renderer
# from your_project.losses import init_loss, update_mesh_shape_prior_losses
# from your_project.geometry import BNI, save_normal_tensor, register, part_removal, poisson, remesh, query_color
# from your_project.model import LocalAffine
# from your_project.utils import apply_homogeneous_transform, save_normal_comparison
# from pytorch3d.structures import Meshes
# from pytorch3d.ops import SubdivideMeshes

# This is the main function for the clothing refinement process
def refine_clothing_and_save(args, cfg, in_tensor, dataset, transform_manager, device):
    """
    The definitive clothing refinement pipeline. This function takes the initial data,
    fuses the partial scans into a clean point cloud, generates a smooth base mesh via
    Poisson reconstruction, and then refines it against the normal maps with balanced
    regularization to produce a high-quality, artifact-free result.
    """
    print(colored("Starting Definitive Clothing Refinement Pipeline...", "cyan"))

    # --- 1. Extract Partial Surfaces from Each View using BNI ---
    bni_mesh_list = []
    # Using a representative subset of views is often more stable than using all of them
    # For an 8-camera setup, views 0, 2, 4, 6 give good coverage (front, side, back, other side)
    view_indices_to_use = [0, 2, 4, 6] 
    print(f"Using views {view_indices_to_use} for BNI surface extraction.")

    for i in view_indices_to_use:
        view_data = in_tensor[f"view_{i}"]
        view_name = view_data["name"] # Use the specific view's name
        
        # BNI process to get a partial mesh from normals
        BNI_dict = save_normal_tensor(
            view_data,
            i, # Use view index for unique naming
            osp.join(args.out_dir, cfg.name, f"BNI/{view_name}"),
            cfg.bni.thickness,
        )
        BNI_object = BNI(
            dir_path=osp.join(args.out_dir, cfg.name, "BNI"),
            name=view_name,
            BNI_dict=BNI_dict,
            cfg=cfg.bni,
            device=device
        )
        BNI_object.extract_surface()
        
        # We need to transform this partial mesh back into the canonical space
        T_view_to_canonical = transform_manager.get_transform_to_target(int(view_name.split("_")[1]))
        
        # The transform from the view's space to the canonical world is the inverse
        T_view_to_canonical_np = T_view_to_canonical.cpu().numpy()
        BNI_object.F_B_trimesh.apply_transform(T_view_to_canonical_np)
        
        bni_mesh_list.append(BNI_object.F_B_trimesh)

    # --- 2. Robust Mesh Fusion via Point Cloud ---
    # This is the core fix: DO NOT concatenate meshes directly.
    # Instead, treat them as a single point cloud.
    print("Fusing partial scans into a unified point cloud...")
    all_bni_verts = np.concatenate([mesh.vertices for mesh in bni_mesh_list])
    all_bni_normals = np.concatenate([mesh.vertex_normals for mesh in bni_mesh_list])

    # Create a single Trimesh object representing the fused point cloud
    fused_point_cloud = trimesh.Trimesh(
        vertices=all_bni_verts,
        vertex_normals=all_bni_normals,
        process=False
    )
    # Optional: Save the intermediate point cloud for debugging
    fused_point_cloud.export(os.path.join(args.out_dir, cfg.name, "obj", f"{in_tensor['name']}_fused_points.ply"))
    print(colored(f"✅ Fused BNI surfaces into a single point cloud with {len(fused_point_cloud.vertices)} points.", "green"))

    # --- 3. Prepare a Smooth Base Mesh for Poisson Reconstruction ---
    # We use a clean, masked SMPL body as the "inside" guide for Poisson
    side_mesh_trimesh = in_tensor["smpl_obj_lst"][0].copy()
    side_mesh_trimesh = apply_vertex_mask(
        side_mesh_trimesh,
        (
            SMPLX_object.front_flame_vertex_mask + 
            SMPLX_object.smplx_mano_vertex_mask +
            SMPLX_object.eyeball_vertex_mask
        ).eq(0).float(),
    )

    # Combine the detailed BNI point cloud (outside) with the smooth SMPL back (inside)
    full_point_cloud_for_poisson = trimesh.Trimesh(
        vertices=np.concatenate([fused_point_cloud.vertices, side_mesh_trimesh.vertices]),
        vertex_normals=np.concatenate([fused_point_cloud.vertex_normals, side_mesh_trimesh.vertex_normals]),
        process=False
    )

    # --- 4. Generate a Clean, Watertight Mesh using Poisson ---
    print("Running Poisson Surface Reconstruction...")
    recon_obj = poisson(
        full_point_cloud_for_poisson,
        os.path.join(args.out_dir, cfg.name, "obj", f"{in_tensor['name']}_recon.obj"),
        cfg.bni.poisson_depth, # A higher depth (e.g., 10 or 11) can add more detail
    )

    # --- 5. Clean Up and Prepare for Final Refinement ---
    print("Remeshing Poisson output for clean topology...")
    verts_refine, faces_refine = remesh(
        recon_obj,
        os.path.join(args.out_dir, cfg.name, "obj", f"{in_tensor['name']}_remesh.obj"),
        device
    )
    mesh_pr = Meshes(verts=verts_refine, faces=faces_refine).to(device)

    # --- 6. Final Local Affine Refinement ---
    print("Starting final Local Affine Refinement...")
    local_affine_model = LocalAffine(
        mesh_pr.verts_padded().shape[1],
        mesh_pr.verts_padded().shape[0],
        mesh_pr.edges_packed()
    ).to(device)

    optimizer_cloth = torch.optim.Adam([{'params': local_affine_model.parameters()}], lr=1e-4)
    scheduler_cloth = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_cloth, mode="min", factor=0.5, patience=10)

    loop_cloth = tqdm(range(200), desc="Cloth Refinement")
    
    for i in loop_cloth:
        optimizer_cloth.zero_grad()

        deformed_verts, stiffness, rigid = local_affine_model(mesh_pr.verts_padded())
        
        # Create a new Meshes object with the deformed vertices
        deformed_mesh = Meshes(verts=deformed_verts, faces=mesh_pr.faces_padded())

        losses = init_loss() # Re-initialize losses at each step
        update_mesh_shape_prior_losses(deformed_mesh, losses)

        # --- Loss Calculation ---
        cloth_losses = []
        for view_idx in view_indices_to_use:
            view_data = in_tensor[f"view_{view_idx}"]
            T_canonical_to_view = torch.inverse(transform_manager.get_transform_to_target(int(view_data["name"].split("_")[1])))
            
            verts_view = apply_homogeneous_transform(deformed_mesh.verts_padded(), T_canonical_to_view)
            
            # Apply the crucial coordinate flip to match the GT normal rendering space
            verts_view_rendered = verts_view * torch.tensor([1.0, -1.0, -1.0], device=device)
            
            P_normal_F, _ = dataset.render_normal(verts_view_rendered, deformed_mesh.faces_padded())
            
            gt_normal_F = view_data["normal_F"]
            mask = torch.tensor(view_data["mask"]).to(device).float().unsqueeze(0)
            
            # Use Cosine Similarity for a more robust normal loss
            diff = (1.0 - F.cosine_similarity(P_normal_F, gt_normal_F, dim=1, eps=1e-6)).unsqueeze(0)
            cloth_loss_view = (diff * mask).sum() / mask.sum().clamp(min=1.0)
            cloth_losses.append(cloth_loss_view)
        
        losses["cloth"]["value"] = torch.stack(cloth_losses).mean()
        
        # Increase regularization to fight artifacts
        losses["stiff"]["weight"] = 5.0
        losses["rigid"]["weight"] = 5.0
        losses["stiff"]["value"] = torch.mean(stiffness)
        losses["rigid"]["value"] = torch.mean(rigid)

        # --- Total Loss and Optimization Step ---
        total_loss = torch.zeros(1, device=device)
        loss_msg = "Refining --- "
        for k, v in losses.items():
            if v["weight"] > 0.0:
                total_loss += v["value"] * v["weight"]
                loss_msg += f"{k}:{float(v['value'] * v['weight']):.5f} | "
        
        loop_cloth.set_description(loss_msg + f"Total: {total_loss.item():.5f}")
        total_loss.backward()
        optimizer_cloth.step()
        scheduler_cloth.step(total_loss)
        
        # Update the base mesh for the next iteration
        mesh_pr = deformed_mesh.detach()

    # --- 7. Finalize and Save Result ---
    print("Finalizing and texturing the mesh...")
    final_verts = mesh_pr.verts_packed().detach().cpu()
    final_faces = mesh_pr.faces_packed().detach().cpu()
    final_obj = trimesh.Trimesh(final_verts, final_faces, process=False, maintains_order=True)

    final_colors = query_color(final_verts, final_faces, in_tensor["image"], device=device)
    final_obj.visual.vertex_colors = final_colors

    refine_path = os.path.join(args.out_dir, cfg.name, f"obj/{in_tensor['name']}_final_refined.obj")
    final_obj.export(refine_path)
    
    print(colored(f"✅✅✅ Definitive Refined Mesh Saved to {refine_path} ✅✅✅", "blue"))