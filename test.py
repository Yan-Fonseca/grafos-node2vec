import torch
import pyg_lib

rowptr = torch.tensor([0, 2, 3, 4], dtype=torch.int64, device="cuda")
col = torch.tensor([1, 2, 2, 0], dtype=torch.int64, device="cuda")
start = torch.tensor([0], dtype=torch.int64, device="cuda")

print("p=q=1")
print(torch.ops.pyg.random_walk(
    rowptr,
    col,
    start,
    5,
    1.0,
    1.0,
))

print("p=0.5")
print(torch.ops.pyg.random_walk(
    rowptr,
    col,
    start,
    5,
    0.5,
    1.0,
))
