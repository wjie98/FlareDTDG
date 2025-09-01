import torch

from torch import Tensor
from typing import *

from pathlib import Path
from tqdm import trange, tqdm

import dgl


def get_node_types(g: dgl.DGLGraph, bins: int = 10) -> Tensor:
    w: Tensor = g.in_degrees() + g.out_degrees()
    w = w.float()

    b = torch.histogram(w, bins=bins).bin_edges
    ntypes = torch.bucketize(w, b[1:])

    c = 0
    for i in range(bins):
        x = torch.count_nonzero(ntypes == i).item()
        if x > 0:
            ntypes[ntypes == i] = c
            c += 1
    print(f"{ntypes.max().item() + 1}")
    return ntypes

def apply_inter_partition(g: dgl.DGLGraph, node_types: Tensor | None, k: int) -> Tensor:
    assert k < 100, f"k must be less than 100"
    print("Enter apply_inter_partition()")
    
    node_parts: Tensor = metis(g, k=k, balance_ntypes=node_types, balance_edges=True)
    assert node_parts.max().item() + 1 == k, f"node_parts.max().item() + 1 != k"
    return node_parts.type(torch.uint8)

def apply_intra_partition(g: dgl.DGLGraph, node_parts: Tensor, k: int) -> Tensor:
    assert k < 1000, f"k must be less than 1000"
    print("Enter apply_intra_partition()")

    num_parts = node_parts.max().item() + 1
    out = node_parts.type(torch.int32) << 16
    for i in range(num_parts):
        print(f"Enter apply_intra_partition():{i}/{num_parts}")
        p = dgl.node_subgraph(g, node_parts == i)
        x = metis(p, k=k, balance_edges=True)
        out[p.ndata[dgl.NID]] |= x.type(torch.int32) & 0xFFFF
    return out


if __name__ == "__main__":
    src_root = "~/DATA/FlareGraph/web"
    tgt_root = "~/DATA/FlareGraph/nparts"

    num_inter_parts = 4
    num_intra_parts = 128

    src_root = Path(src_root).expanduser().resolve()
    tgt_root = Path(tgt_root).expanduser().resolve()

    tgt_root.mkdir(parents=True, exist_ok=True)

    large_data_names = ["soc-bitcoin"]
    large_data_flags = False

    metis = dgl.distributed.partition.metis_partition_assignment
    for p in src_root.glob("*.pth"):
        name = p.stem

        if large_data_flags:
            if name not in large_data_names:
                continue
        else:
            if name in large_data_names:
                continue

        print(f"loading {name}...")
        state_dict = torch.load(str(p), mmap=True)
        
        num_nodes: int = state_dict["num_nodes"]
        dataset: List[Dict[str, Tensor]] = state_dict["dataset"]
        edge_index = torch.cat([data["edge_index"] for data in dataset], dim=1)
        
        print(f"building graph {name}...")
        src = edge_index[0]
        dst = edge_index[1]

        g = dgl.graph((src, dst), num_nodes=num_nodes, idtype=torch.int64)
        # ntypes = get_node_types(g, 3)
        ntypes = None

        g = dgl.to_simple(g)
        g = dgl.remove_self_loop(g)
        g = dgl.to_bidirected(g)

        print(f"partitioning {name} with k={num_inter_parts}...")
        inter_nparts = apply_inter_partition(g, node_types=ntypes, k=num_inter_parts)
        intra_nparts = apply_intra_partition(g, inter_nparts, k=num_intra_parts)
        
        inter_p = tgt_root / f"{name}_{num_inter_parts:03d}.pth"
        intra_p = tgt_root / f"{name}_{num_inter_parts:03d}_{num_intra_parts:04d}.pth"

        print(f"{name}: {num_nodes}")
        for i in range(num_inter_parts):
            m = inter_nparts == i
            num_part_nodes = torch.count_nonzero(m).item()
            x = intra_nparts[m] & 0xFFFF
            x = [torch.count_nonzero(x == j).item() for j in range(num_intra_parts)]
            x = torch.tensor(x).float()
            std, mean = torch.std_mean(x)
            print(f"  part {i}: {num_part_nodes} std={std.item():.2f} mean={mean.item():.2f}")

        print(f"saving {inter_p}...")
        torch.save(inter_nparts, str(inter_p))

        print(f"saving {intra_p}...")
        torch.save(intra_nparts, str(intra_p))
