import os
import json
import numpy as np



class SMPLXJointAligner:
    def __init__(self, final_mesh):
        self.mesh = final_mesh
        self.kdtree = final_mesh.kdtree
        self.transform = final_mesh.principal_inertia_transform

    def align_with_kdtree(self, joints):
        aligned = []
        for joint in joints:
            _, idx = self.kdtree.query(joint)
            aligned.append(self.mesh.vertices[idx])
        return np.array(aligned)

    def align_with_inertia_transform(self, joints):
        joints_h = np.concatenate([joints, np.ones((joints.shape[0], 1))], axis=1)
        return (self.transform @ joints_h.T).T[:, :3]


class FaceRigExporter:
    def __init__(self, smplx_object, final_mesh=None, align_mode='smplx'):
        self.smplx_object = smplx_object
        self.face_vids = smplx_object.smplx_front_flame_vid
        self.final_mesh = final_mesh
        self.align_mode = align_mode
        if final_mesh and align_mode != 'smplx':
            self.aligner = SMPLXJointAligner(final_mesh)

    def align_joints(self, joints_np):
        if self.align_mode == 'smplx' or not self.final_mesh:
            return joints_np
        elif self.align_mode == 'kdtree':
            return self.aligner.align_with_kdtree(joints_np)
        elif self.align_mode == 'transform':
            return self.aligner.align_with_inertia_transform(joints_np)
        raise ValueError(f"Unknown align_mode: {self.align_mode}")

    def compute_facial_joint_weights(self, face_verts, facial_kpt3d):
        face_verts_np = face_verts[0].detach().cpu().numpy()
        joint_weights = {}
        for j_idx, joint in enumerate(facial_kpt3d):
            joint_np = np.array(joint)
            dists = np.linalg.norm(face_verts_np - joint_np, axis=1)
            weights = np.exp(-dists * 40)
            weights /= weights.sum()
            joint_weights[str(j_idx + 66)] = weights.tolist()
        return joint_weights

    def save_obj(self, verts_np, out_path, faces):
        with open(out_path, 'w') as f:
            for v in verts_np:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in faces + 1:  # OBJ is 1-indexed
                f.write(f"f {face[0]} {face[1]} {face[2]}\n")

    def export(self, data, smpl_verts, out_dir, cfg_name, expression_meshes):
        full_range = list(range(66, 137))
        excluded = [124, 127, 128, 129, 130, 131, 132, 133, 135]
        facial_joint_indices = [i for i in full_range if i not in excluded]

        facial_kpt3d_np = data["smplx_kpt3d"][0, facial_joint_indices].detach().cpu().numpy()
        facial_kpt3d_aligned = self.align_joints(facial_kpt3d_np)
        facial_kpt3d = facial_kpt3d_aligned.tolist()
        facial_kpt2d = data["smplx_kpt"][0, facial_joint_indices].detach().cpu().numpy().tolist()

        face_verts = smpl_verts[:, self.face_vids, :]
        weights_dict = self.compute_facial_joint_weights(face_verts, facial_kpt3d_aligned)

        base_face_np = face_verts[0].detach().cpu().numpy()
        displacements = []

        # Compute displacements for each expression
        for exp_mesh in expression_meshes:
            exp_face = exp_mesh[self.face_vids]
            delta = exp_face - base_face_np
            displacements.append(delta.tolist())

        # Output directory setup
        out_dir_rig = os.path.join(out_dir, cfg_name, "rig_params_json")
        os.makedirs(out_dir_rig, exist_ok=True)


        # Build JSON data
        rig_data_json = {
            "expression_params": data["exp"].detach().cpu().numpy().tolist(),
            "jaw_pose": data["jaw_pose"].detach().cpu().numpy().tolist(),
            "head_pose": data["head_pose"].detach().cpu().numpy().tolist(),
            "abs_head_pose": data["abs_head_pose"].detach().cpu().numpy().tolist(),
            "neck_pose": data["neck_pose"].detach().cpu().numpy().tolist(),
            "shape_params": data["shape"].detach().cpu().numpy().tolist(),
            "face_vertex_ids": self.face_vids.tolist(),
            "face_verts": base_face_np.tolist(),
            "smplx_kpt3d": data["smplx_kpt3d"][0].detach().cpu().numpy().tolist(),
            "smplx_kpt": data["smplx_kpt"][0].detach().cpu().numpy().tolist(),
            "facial_joint_indices": facial_joint_indices,
            "facial_kpt3d": facial_kpt3d,
            "facial_kpt2d": facial_kpt2d,
            "facial_vertex_weights": weights_dict,
            "blendshape_displacements": displacements,
            

        }

        json_path = os.path.join(out_dir_rig, f"{data['name']}_face_rig.json")
        with open(json_path, 'w') as f:
            json.dump(rig_data_json, f, indent=4)

        print(f"✅ Face rig (with blendshape data) saved to: {json_path}")
