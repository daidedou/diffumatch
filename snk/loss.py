from trimesh.graph import face_adjacency
import torch 
import torch.nn as nn

class PrismRegularizationLoss(nn.Module):
    """
    Calculate the loss based on the PriMo energy, as described in the paper:
    PriMo: Coupled Prisms for Intuitive Surface Modeling
    """
    def __init__(self, primo_h):
        super().__init__()
        self.h = primo_h

        # compute coefficient for the energy
        indices = torch.tensor([(i, j) for i in range(2) for j in range(2)])
        indices_A = indices.repeat_interleave(4, dim=0)
        indices_B = indices.repeat(4, 1)
        self.coeff = (torch.ones(1) * 2).pow(((indices_A - indices_B).abs() * -1).sum(dim=1))[None, :]

    def forward(self, transformed_prism, rotations, verts, faces, normals):
        # transformed_prism is (n_faces, 3, 3)
        # verts and faces are from the template (shape 2)
        # * for now assumes there is only one batch
        # todo add batch support
        bs = 1
        verts = verts.reshape(-1, 3)
        normals = normals.reshape(-1, 3)
        faces = faces

        # get the area of each face
        face_areas = self.get_face_areas(verts, faces)  # (n_faces,)

        # get list of edges and the faces that share each edge
        face_ids, edges = face_adjacency(faces.cpu().numpy(), return_edges=True)  # (n_edges, 2), (n_edges, 2)
        face_ids, edges = torch.from_numpy(face_ids).to(verts.device), torch.from_numpy(edges).to(verts.device)

        # normals and rotations of the faces that share each edge
        normals1, normals2 = normals[edges[:, 0]], normals[edges[:, 1]]  # (n_edges, 3), normals are per vertex
        rotations1, rotations2 = rotations[face_ids[:, 0]], rotations[face_ids[:, 1]]  # (n_edges, 3, 3), rotations are per face

        # computed normals from the transformed prism
        # normals = self.compute_normals(transformed_prism)

        # compute the loss
        face_id1, face_id2 = face_ids[:, 0], face_ids[:, 1]  # (n_edges,)
        faces_to_verts = self.get_verts_id_face(faces, edges, face_ids)  # (n_edges, 4)
        verts1_p1, verts2_p1 = transformed_prism[face_id1, faces_to_verts[:, 0]], transformed_prism[face_id1, faces_to_verts[:, 1]]  # (n_edges, 3)
        verts1_p2, verts2_p2 = transformed_prism[face_id2, faces_to_verts[:, 2]], transformed_prism[face_id2, faces_to_verts[:, 3]]  # (n_edges, 3)

        # get the normals per vertex
        # normals1, normals2 = normals[face_id1], normals[face_id2]  # (n_edges, 3)  # normals per face (NOT USED)
        prism1_n1, prism1_n2 = (normals1[:, None] @ rotations1).squeeze(1), (normals2[:, None] @ rotations1).squeeze(1)  # todo check if this is correct
        prism2_n1, prism2_n2 = (normals1[:, None] @ rotations2).squeeze(1), (normals2[:, None] @ rotations2).squeeze(1)

        # get the coordinates of the face of the prism
        # prism1 (1 -> 2)
        f_p1_00, f_p1_01 = verts1_p1 + prism1_n1 * self.h, verts2_p1 + prism1_n2 * self.h  # (n_edges, 3)
        f_p1_10, f_p1_11 = verts1_p1 - prism1_n1 * self.h, verts2_p1 - prism1_n2 * self.h  # (n_edges, 3)
        # prism2 (2 -> 1)
        f_p2_00, f_p2_01 = verts1_p2 + prism2_n1 * self.h, verts2_p2 + prism2_n2 * self.h  # (n_edges, 3)
        f_p2_10, f_p2_11 = verts1_p2 - prism2_n1 * self.h, verts2_p2 - prism2_n2 * self.h  # (n_edges, 3)

        # compute the energy
        A, B = torch.stack((f_p1_00, f_p1_01, f_p1_10, f_p1_11), dim=1), torch.stack((f_p2_00, f_p2_01, f_p2_10, f_p2_11), dim=1)  # (n_edges, 4, 3)
        energy = self.compute_energy(A - B, A - B)  # (n_edges,)

        # compute weight
        area1, area2 = face_areas[face_id1], face_areas[face_id2]  # (n_edges,)
        weight = torch.norm(verts[edges[:, 0]] - verts[edges[:, 1]], dim=1).square() / (area1 + area2)  # (n_edges,)
        # weight = torch.ones_like(weight).to(weight.device)  # todo remove
        energy = energy * weight  # (n_edges,)

        loss = energy.sum() / bs  # todo when batch enabled, need to divide by batch size
        return loss

    def compute_energy(self, A, B):
        """
        Computes the formula sum_{i,j,k,l=0}^{1} a_{ij}b_{kl} 2^{-|i - k| - |j - l|}.
        Assumes that A and B are tensors of size bs x 4 x 3, where bs is the batch size.
        """
        self.coeff = self.coeff.to(A.device)

        A_repeated = A.repeat_interleave(4, dim=1)
        B_repeated = B.repeat(1, 4, 1)

        energy = (A_repeated * B_repeated).sum(dim=-1)
        energy = (energy * self.coeff).sum(dim=1)
        energy = energy / 9

        return energy

    def get_face_areas(self, verts, faces):
        # get the area of each face
        v1, v2, v3 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        area = 0.5 * torch.cross(v2 - v1, v3 - v1, dim=-1).norm(dim=1)

        return area

    def get_verts_id_face(self, F, E, Q):
        e = E.shape[0]
        Z = torch.zeros((e, 4), dtype=torch.long)

        v1 = F[:, 0][Q[:, 0]]
        v2 = F[:, 1][Q[:, 0]]
        v3 = F[:, 2][Q[:, 0]]
        v4 = F[:, 0][Q[:, 1]]
        v5 = F[:, 1][Q[:, 1]]
        v6 = F[:, 2][Q[:, 1]]

        idx1 = torch.where(v1 == E[:, 0], 0, torch.where(v2 == E[:, 0], 1, torch.where(v3 == E[:, 0], 2, -1)))
        idx2 = torch.where(v1 == E[:, 1], 0, torch.where(v2 == E[:, 1], 1, torch.where(v3 == E[:, 1], 2, -1)))
        idx3 = torch.where(v4 == E[:, 0], 0, torch.where(v5 == E[:, 0], 1, torch.where(v6 == E[:, 0], 2, -1)))
        idx4 = torch.where(v4 == E[:, 1], 0, torch.where(v5 == E[:, 1], 1, torch.where(v6 == E[:, 1], 2, -1)))

        Z[:, 0:2] = torch.stack((idx1, idx2), dim=1)
        Z[:, 2:4] = torch.stack((idx3, idx4), dim=1)
        Z = Z.to(F.device)

        return Z