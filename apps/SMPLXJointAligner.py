import numpy as np

class SMPLXJointAligner:
    def __init__(self, final_mesh):
        self.mesh = final_mesh
        self.kdtree = final_mesh.kdtree
        self.transform = final_mesh.principal_inertia_transform  # 4x4

    def align_with_kdtree(self, joints):
        aligned = []
        for joint in joints:
            _, idx = self.kdtree.query(joint)
            aligned.append(self.mesh.vertices[idx])
        return np.array(aligned)

    def align_with_inertia_transform(self, joints):
        joints_h = np.concatenate([joints, np.ones((joints.shape[0], 1))], axis=1)  # Nx4
        transformed = (self.transform @ joints_h.T).T[:, :3]
        return transformed

    def compare_alignment(self, joints):
        kdtree_aligned = self.align_with_kdtree(joints)
        transform_aligned = self.align_with_inertia_transform(joints)

        return {
            "kdtree": kdtree_aligned,
            "transform": transform_aligned
        }

    def save_json(self, joints_dict, base_path):
        for key, aligned_joints in joints_dict.items():
            output_path = base_path.replace(".json", f"_{key}_aligned.json")
            with open(output_path, "w") as f:
                json.dump(aligned_joints.tolist(), f, indent=4)
            print(f"✅ Saved {key} alignment to: {output_path}")
