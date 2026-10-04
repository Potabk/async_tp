#!/usr/bin/env python3
"""Minimal communication/GEMM overlap experiment.

CUDA, two GPUs (NCCL teaching implementation):
  torchrun --standalone --nproc-per-node=2 async_tp_minimal.py
  torchrun --standalone --nproc-per-node=2 async_tp_minimal.py --op rs

CUDA, symmetric-memory implementation used by vLLM (requires working P2P):
  torchrun --standalone --nproc-per-node=2 async_tp_minimal.py --impl symm
  torchrun --standalone --nproc-per-node=2 async_tp_minimal.py --impl symm --op rs

CPU numerical/collective check (does NOT verify CUDA overlap):
  torchrun --standalone --nproc-per-node=2 async_tp_minimal.py \
    --device cpu --dtype float32 --m 32 --k 16 --n 24 --iters 2 --warmup 1

--m is rows per rank, --n is local GEMM output width, --k is contraction width.
ag: all_gather(X_local) @ W_local -> [world*m, n]
rs: reduce_scatter_SUM(X_local @ W_local) -> [m, n], X_local=[world*m,k].
For rs every rank has a partial GEMM contribution; SUM, not AVG, is required.
The two ops are independent experiments, not a complete Transformer layer.

--trace trace_dir emits one Chrome trace per rank for the candidate path.
Use Nsight Systems for GPU stream scheduling and resource analysis.
Buffers are preallocated in chunked mode. Benchmarks include host launch costs,
AG output reorder copies, and (for symm) allocations inside the fused operators.
Performance can be slower than baseline: this is an experiment, not a claim.

Research snapshot: vLLM 0872ddf4b96d57a0745893478deef595b1994f2e (2026-10-04).
  vllm/compilation/passes/fusion/collective_fusion.py:
    AsyncTPPass, AllGatherGEMMPattern, GEMMReduceScatterPattern
  vllm/compilation/passes/fusion/sequence_parallelism.py
  tests/compile/passes/distributed/test_async_tp.py
PyTorch implementation inspected at tag v2.9.0:
  torch/distributed/_symmetric_memory/__init__.py:
    _pipelined_multi_all_gather_and_consume
    _pipelined_produce_and_all2all
    _fused_all_gather_matmul, _fused_matmul_reduce_scatter

The chunked path uses NCCL collectives (SM resources), one compute stream and
one communication stream. It is NOT the symmetric-memory/copy-engine,
alternating-stream production implementation. The symm path calls that family
of real fused operators directly, without _test_mode or a silent fallback.
Private symm APIs are version-sensitive; signatures checked against torch 2.9.
Validation here: syntax/CLI checks and CPU simulations of 2/3-rank collectives
passed for both ops, several chunk counts, and repeated buffer reuse. There
was no CUDA device; real Gloo tests were blocked by socket permissions. GPU
overlap, hardware support, and performance must be measured on your machine.
"""

import argparse
from contextlib import nullcontext
from datetime import timedelta
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist


class Experiment:
    def __init__(self, args):
        self.args = args
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.cuda = args.device == "cuda"
        self.device = torch.device("cuda", int(os.environ["LOCAL_RANK"])) if self.cuda else torch.device("cpu")
        self.m, self.k, self.n = args.m, args.k, args.n
        self.c = self.m // args.chunks
        dtype = getattr(torch, args.dtype)
        torch.manual_seed(1234 + self.rank)
        rows = self.m if args.op == "ag" else self.world * self.m
        self.x = torch.randn(rows, self.k, device=self.device, dtype=dtype)
        self.w = torch.randn(self.k, self.n, device=self.device, dtype=dtype) / math.sqrt(self.k)
        self.full = torch.empty(self.world * self.m, self.k, device=self.device, dtype=dtype) if args.op == "ag" else None
        self.partial = torch.empty(self.world * self.m, self.n, device=self.device, dtype=dtype) if args.op == "rs" else None
        out_rows = self.world * self.m if args.op == "ag" else self.m
        self.base_y = torch.empty(out_rows, self.n, device=self.device, dtype=dtype)
        self.y = torch.empty_like(self.base_y)
        chunked = args.impl == "chunked"
        self.comm = torch.cuda.Stream(device=self.device) if self.cuda and chunked else None
        self.ready = [torch.cuda.Event() for _ in range(args.chunks)] if self.cuda and chunked else None
        self.mm_bufs = [torch.empty(self.world * self.c, self.n, device=self.device, dtype=dtype) for _ in range(args.chunks)] if chunked else None
        self.ag_bufs = [torch.empty(self.world * self.c, self.k, device=self.device, dtype=dtype) for _ in range(args.chunks)] if args.op == "ag" and chunked else None
        self.group_name = dist.group.WORLD.group_name
        if args.impl == "symm":
            from torch.distributed._symmetric_memory import enable_symm_mem_for_group
            # All ranks must enable the same group; do not use _test_mode.
            enable_symm_mem_for_group(self.group_name)
        self.sync()

    def sync(self):
        if self.cuda:
            torch.cuda.synchronize(self.device)

    def baseline(self):
        if self.args.op == "ag":
            dist.all_gather_into_tensor(self.full, self.x)
            torch.mm(self.full, self.w, out=self.base_y)
        else:
            torch.mm(self.x, self.w, out=self.partial)
            dist.reduce_scatter_tensor(self.base_y, self.partial, op=dist.ReduceOp.SUM)
        return self.base_y

    def symm(self):
        if self.args.op == "ag":
            _, outputs = torch.ops.symm_mem.fused_all_gather_matmul(
                self.x, [self.w], gather_dim=0, group_name=self.group_name
            )
            return outputs[0]
        return torch.ops.symm_mem.fused_matmul_reduce_scatter(
            self.x, self.w, "sum", scatter_dim=0, group_name=self.group_name
        )

    def chunked_ag(self):
        """Gather chunk i while computing chunk i-1; retain all work/buffers."""
        works = []
        if self.cuda:
            self.comm.wait_stream(torch.cuda.current_stream(self.device))

        def consume(i):
            if self.cuda:
                torch.cuda.current_stream(self.device).wait_event(self.ready[i])
            else:
                works[i].wait()
            with torch.profiler.record_function(f"gemm_ag_chunk_{i}"):
                torch.mm(self.ag_bufs[i], self.w, out=self.mm_bufs[i])
                # Collective output order is rank-major within each chunk.
                # Restore the full all-gather's rank-major row order.
                lo = i * self.c
                self.y.view(self.world, self.m, self.n)[:, lo:lo + self.c].copy_(
                    self.mm_bufs[i].view(self.world, self.c, self.n)
                )

        for i in range(self.args.chunks):
            lo = i * self.c
            with torch.cuda.stream(self.comm) if self.cuda else nullcontext():
                with torch.profiler.record_function(f"all_gather_chunk_{i}"):
                    work = dist.all_gather_into_tensor(
                        self.ag_bufs[i], self.x[lo:lo + self.c], async_op=True
                    )
                    works.append(work)
                    if self.cuda:
                        # NCCL uses its own stream. wait() connects its completion
                        # to this comm stream; recording immediately after launch
                        # WITHOUT wait() would signal readiness too early.
                        work.wait()
                        self.ready[i].record(self.comm)
            if i > 0:
                consume(i - 1)
        consume(self.args.chunks - 1)
        return self.y

    def chunked_rs(self):
        """Produce chunk i+1 while reducing/scattering chunk i."""
        works = []
        for i in range(self.args.chunks):
            lo = i * self.c
            with torch.profiler.record_function(f"gemm_rs_chunk_{i}"):
                # Pack one stripe per destination rank into a contiguous buffer.
                for dst in range(self.world):
                    start = dst * self.m + lo
                    torch.mm(self.x[start:start + self.c], self.w,
                             out=self.mm_bufs[i][dst * self.c:(dst + 1) * self.c])
            if self.cuda:
                self.ready[i].record(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self.comm) if self.cuda else nullcontext():
                if self.cuda:
                    self.comm.wait_event(self.ready[i])
                with torch.profiler.record_function(f"reduce_scatter_chunk_{i}"):
                    work = dist.reduce_scatter_tensor(
                        self.y[lo:lo + self.c], self.mm_bufs[i],
                        op=dist.ReduceOp.SUM, async_op=True
                    )
                    works.append(work)
                    if self.cuda:
                        work.wait()
        if self.cuda:
            torch.cuda.current_stream(self.device).wait_stream(self.comm)
        else:
            for work in works:
                work.wait()
        return self.y

    def candidate(self):
        if self.args.impl == "symm":
            return self.symm()
        return self.chunked_ag() if self.args.op == "ag" else self.chunked_rs()

    def benchmark(self, fn):
        for _ in range(self.args.warmup):
            fn()
        self.sync()
        dist.barrier()
        self.sync()
        begin = time.perf_counter()
        for _ in range(self.args.iters):
            fn()
        self.sync()
        elapsed = torch.tensor((time.perf_counter() - begin) * 1000 / self.args.iters,
                               device=self.device, dtype=torch.float64)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        return elapsed.item()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impl", choices=["chunked", "symm"], default="chunked")
    p.add_argument("--op", choices=["ag", "rs"], default="ag")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--k", type=int, default=2048)
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--chunks", type=int, default=4)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--trace", type=Path)
    args = p.parse_args()
    if min(args.m, args.k, args.n, args.chunks, args.iters) <= 0 or args.warmup < 0:
        p.error("dimensions/chunks/iters must be positive; warmup must be nonnegative")
    if args.m % args.chunks:
        p.error("--m must be divisible by --chunks")
    if "LOCAL_RANK" not in os.environ:
        p.error("launch with torchrun --standalone --nproc-per-node=2")
    world = int(os.environ["WORLD_SIZE"])
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", world))
    if world < 2 or world != local_world:
        p.error("this example requires >=2 ranks, all on one host")
    if args.device == "cpu":
        if args.impl == "symm" or args.dtype != "float32":
            p.error("CPU check requires --impl chunked --dtype float32")
        torch.set_num_threads(1)
    else:
        local_rank = int(os.environ["LOCAL_RANK"])
        if not torch.cuda.is_available() or torch.cuda.device_count() < local_world:
            p.error("not enough visible CUDA GPUs for the local ranks")
        torch.cuda.set_device(local_rank)
        if args.impl == "symm":
            for peer in range(local_world):
                if peer != local_rank and not torch.cuda.can_device_access_peer(local_rank, peer):
                    p.error("symm needs GPU P2P access; try --impl chunked for this topology")
    dist.init_process_group("nccl" if args.device == "cuda" else "gloo",
                            timeout=timedelta(seconds=120))
    try:
        with torch.inference_mode():
            exp = Experiment(args)
            reference = exp.baseline().clone()
            got = exp.candidate()
            exp.sync()
            error = (reference.float() - got.float()).abs().max()
            atol = {"float32": 2e-5, "float16": 5e-3, "bfloat16": 5e-2}[args.dtype]
            ok = torch.allclose(got, reference, atol=atol, rtol=atol)
            passed = torch.tensor(int(ok), device=exp.device)
            dist.all_reduce(passed, op=dist.ReduceOp.MIN)
            dist.all_reduce(error, op=dist.ReduceOp.MAX)
            if not passed.item():
                raise AssertionError(f"numerical check failed: max_abs_error={error.item()}")
            base_ms = exp.benchmark(exp.baseline)
            test_ms = exp.benchmark(exp.candidate)
            if exp.rank == 0:
                print(f"torch={torch.__version__} device={args.device} ranks={exp.world}")
                print(f"op={args.op} impl={args.impl} local_m={args.m} k={args.k} n={args.n} chunks={args.chunks}")
                print(f"PASS max_abs_error={error.item():.6g}")
                print(f"baseline={base_ms:.3f} ms candidate={test_ms:.3f} ms ratio={base_ms/test_ms:.3f}x (max rank)")
                if exp.cuda and os.environ.get("TORCH_NCCL_BLOCKING_WAIT") == "1":
                    print("TORCH_NCCL_BLOCKING_WAIT=1 can block host work.wait() and prevent pipelining.")
                if args.device == "cpu":
                    print("CPU check verifies collectives, SUM and row order; no CUDA overlap claim.")
            if args.trace:
                args.trace.mkdir(parents=True, exist_ok=True)
                activities = [torch.profiler.ProfilerActivity.CPU]
                if exp.cuda:
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                dist.barrier()
                exp.sync()
                with torch.profiler.profile(activities=activities) as prof:
                    exp.candidate()
                    exp.sync()
                prof.export_chrome_trace(str(args.trace / f"rank{exp.rank}.json"))
            dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
