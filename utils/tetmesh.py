### adapted from https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/ops/conversions/tetmesh.py

# Copyright (c) 2021,22 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import torch

__all__ = ["marching_tetrahedra"]

triangle_table = torch.tensor(
    [
        [-1, -1, -1, -1, -1, -1],
        [1, 0, 2, -1, -1, -1],
        [4, 0, 3, -1, -1, -1],
        [1, 4, 2, 1, 3, 4],
        [3, 1, 5, -1, -1, -1],
        [2, 3, 0, 2, 5, 3],
        [1, 4, 0, 1, 5, 4],
        [4, 2, 5, -1, -1, -1],
        [4, 5, 2, -1, -1, -1],
        [4, 1, 0, 4, 5, 1],
        [3, 2, 0, 3, 5, 2],
        [1, 3, 5, -1, -1, -1],
        [4, 1, 2, 4, 3, 1],
        [3, 0, 4, -1, -1, -1],
        [2, 0, 1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1],
    ],
    dtype=torch.long,
)

num_triangles_table = torch.tensor([0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0], dtype=torch.long)
base_tet_edges = torch.tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long)
v_id = torch.pow(2, torch.arange(4, dtype=torch.long))


@torch.no_grad()
def _unbatched_marching_tetrahedra(tets, sdf, valids):
    device = tets.device
    occ_n = sdf > 0
    occ_fx4 = occ_n[tets]
    occ_sum = torch.sum(occ_fx4, -1)
    valid_fx4 = valids[tets]

    valid_tets = (occ_sum > 0) & (occ_sum < 4) & valid_fx4.all(dim=-1)

    all_edges = tets[valid_tets][:, base_tet_edges.to(device)].reshape(-1, 2)

    order = (all_edges[:, 0] > all_edges[:, 1]).bool()
    all_edges[order] = all_edges[order][:, [1, 0]]

    unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)

    unique_edges = unique_edges.long()
    mask_edges = occ_n[unique_edges].sum(-1) == 1
    mapping = torch.full((unique_edges.shape[0],), -1, dtype=torch.long, device=device)
    mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long, device=device)
    idx_map = mapping[idx_map]
    interp_v = unique_edges[mask_edges]
    idx_map = idx_map.reshape(-1, 6)
    tetindex = (occ_fx4[valid_tets] * v_id.to(device).unsqueeze(0)).sum(-1)
    num_triangles = num_triangles_table.to(device)[tetindex]
    triangle_table_device = triangle_table.to(device)

    # Generate triangle indices
    faces = torch.cat(
        (
            torch.gather(input=idx_map[num_triangles == 1], dim=1, index=triangle_table_device[tetindex[num_triangles == 1]][:, :3]).reshape(-1, 3),
            torch.gather(input=idx_map[num_triangles == 2], dim=1, index=triangle_table_device[tetindex[num_triangles == 2]][:, :6]).reshape(-1, 3),
        ),
        dim=0,
    )

    return faces, interp_v


def unbatched_marching_tetrahedra(vertices, tets, sdf, scales, valids):
    """unbatched marching tetrahedra.

    Refer to :func:`marching_tetrahedra`.
    """
    # construct a function to recycle the unused memory
    def inner_func():
        device = vertices.device

        # call by chunk
        chunk_size = 32 * 512 * 512

        keys_merged = None  # (M,2) edge ids in *storage order* (append-only)
        for tet_chunk in torch.chunk(tets, tets.shape[0] // chunk_size + 1):
            faces_new, ids_new_2 = _unbatched_marching_tetrahedra(tet_chunk, sdf, valids)
            torch.cuda.empty_cache()

            device = ids_new_2.device
            # faces_new = faces_new.long()

            # pack (E,2) -> (E,) int64
            a = ids_new_2[:, 0].to(torch.int64)
            b = ids_new_2[:, 1].to(torch.int64)

            keys_new = (a << 32) | (b & 0xFFFF_FFFF)  # (V,)

            if keys_merged is None:
                # initialize: keep merged keys sorted, and remap faces accordingly
                perm = torch.argsort(keys_new)
                keys_merged = keys_new[perm]  # (M,) sorted unique

                inv = torch.empty_like(perm)
                inv[perm] = torch.arange(perm.numel(), device=device)
                faces_merged = inv[faces_new]
                continue

            # -------- merge into existing (keys_merged sorted unique) --------
            M = keys_merged.numel()
            V = keys_new.numel()

            # find existing edges
            pos = torch.searchsorted(keys_merged, keys_new)  # (V,)
            pos_safe = pos.clamp_max(M - 1)
            exists = keys_merged[pos_safe] == keys_new

            # add truly new edges (keep sorted)
            add_mask = ~exists
            add_n = int(add_mask.sum().item())

            if add_n == 0:
                # only need to append faces, mapping each new edge to its merged index (=pos)
                map_edge = pos  # (V,)
                faces_merged = torch.cat([faces_merged, map_edge[faces_new]], dim=0)
                continue

            add_idx = torch.nonzero(add_mask, as_tuple=False).squeeze(1)  # (add_n,)
            add_keys = keys_new[add_idx]
            add_perm = torch.argsort(add_keys)
            add_idx = add_idx[add_perm]
            add_keys = add_keys[add_perm]  # sorted

            # insertion positions into old merged keys
            ins = torch.searchsorted(keys_merged, add_keys).to(torch.long)  # (add_n,) nondecreasing

            # compute how much each old index shifts right after insertions
            delta = torch.zeros((M + 1,), device=device, dtype=torch.int32)
            delta.scatter_add_(0, ins, torch.ones((add_n,), device=device, dtype=torch.int32))
            shift = torch.cumsum(delta, dim=0)[:-1].to(torch.long)  # (M,)

            old_to_new = torch.arange(M, device=device, dtype=torch.long) + shift  # (M,)
            # add_to_new = ins + torch.arange(add_n, device=device, dtype=torch.long)  # (add_n,)
            add_to_new = ins.add_(torch.arange(add_n, device=device, dtype=torch.long))

            U = M + add_n
            keys2 = torch.empty((U,), device=device, dtype=keys_merged.dtype)
            keys2[old_to_new] = keys_merged
            keys2[add_to_new] = add_keys
            keys_merged = keys2

            faces_merged.add_(shift[faces_merged])
            map_edge = torch.empty((V,), device=device, dtype=torch.long)
            map_edge[exists] = pos[exists] + shift[pos[exists]]
            map_edge[add_idx] = add_to_new  # note: add_idx is in original new-edge order
            faces_merged = torch.cat([faces_merged, map_edge[faces_new]], dim=0)
            torch.cuda.empty_cache()
        return keys_merged, ids_new_2, faces_merged

    keys_merged, ids_new_2, faces_merged = inner_func()
    gc.collect()
    torch.cuda.empty_cache()
    mask = torch.tensor(0xFFFF_FFFF, device=keys_merged.device, dtype=torch.int64)
    i = (keys_merged >> 32) & mask
    j = keys_merged & mask
    merged_verts_ids = torch.stack([i, j], dim=1).to(ids_new_2.dtype)  # (E,2)
    edges_to_interp = vertices[merged_verts_ids.reshape(-1)].reshape(-1, 2, 3)
    edges_to_interp_sdf = sdf[merged_verts_ids.reshape(-1)].reshape(-1, 2, 1)
    merged_scales = scales[merged_verts_ids.reshape(-1)].reshape(-1, 2, 1)
    merged_verts = (edges_to_interp, edges_to_interp_sdf)

    return merged_verts, merged_scales, faces_merged, merged_verts_ids


@torch.no_grad()
def marching_tetrahedra(vertices, tets, sdf, scales, valids):
    r"""Convert discrete signed distance fields encoded on tetrahedral grids to triangle
    meshes using marching tetrahedra algorithm as described in `An efficient method of
    triangulating equi-valued surfaces by using tetrahedral cells`_. The output surface is differentiable with respect to
    input vertex positions and the SDF values. For more details and example usage in learning, see
    `Deep Marching Tetrahedra\: a Hybrid Representation for High-Resolution 3D Shape Synthesis`_ NeurIPS 2021.


    Args:
        vertices (torch.tensor): batched vertices of tetrahedral meshes, of shape
                                 :math:`(\text{batch_size}, \text{num_vertices}, 3)`.
        tets (torch.tensor): unbatched tetrahedral mesh topology, of shape
                             :math:`(\text{num_tetrahedrons}, 4)`.
        sdf (torch.tensor): batched SDFs which specify the SDF value of each vertex, of shape
                            :math:`(\text{batch_size}, \text{num_vertices})`.

    Returns:
        (list[torch.Tensor], list[torch.LongTensor], (optional) list[torch.LongTensor]):

            - the list of vertices for mesh converted from each tetrahedral grid.
            - the list of faces for mesh converted from each tetrahedral grid.

    Example:
        >>> vertices = torch.tensor([[[0, 0, 0],
        ...               [1, 0, 0],
        ...               [0, 1, 0],
        ...               [0, 0, 1]]], dtype=torch.float)
        >>> tets = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        >>> sdf = torch.tensor([[-1., -1., 0.5, 0.5]], dtype=torch.float)
        >>> verts_list, faces_list, tet_idx_list = marching_tetrahedra(vertices, tets, sdf, True)
        >>> verts_list[0]
        tensor([[0.0000, 0.6667, 0.0000],
                [0.0000, 0.0000, 0.6667],
                [0.3333, 0.6667, 0.0000],
                [0.3333, 0.0000, 0.6667]])
        >>> faces_list[0]
        tensor([[3, 0, 1],
                [3, 2, 0]])
        >>> tet_idx_list[0]
        tensor([0, 0])

    .. _An efficient method of triangulating equi-valued surfaces by using tetrahedral cells:
        https://search.ieice.org/bin/summary.php?id=e74-d_1_214

    .. _Deep Marching Tetrahedra\: a Hybrid Representation for High-Resolution 3D Shape Synthesis:
            https://arxiv.org/abs/2111.04276
    """
    list_of_outputs = [unbatched_marching_tetrahedra(vertices[b], tets, sdf[b], scales[b], valids[b]) for b in range(vertices.shape[0])]
    return list(zip(*list_of_outputs))
