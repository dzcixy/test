#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import ssgetpy
from scipy.io import mmread
from scipy.sparse import csr_matrix


# ================== 通用工具 ==================

def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def run_cmd(
    cmd: List[str],
    cwd: str | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    print(f"[cmd] {' '.join(cmd)}")
    if capture:
        return subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    else:
        subprocess.run(cmd, cwd=cwd, check=True)
        return subprocess.CompletedProcess(cmd, 0)


# ================== 1. SuiteSparse 矩阵加载 ==================

def load_ss_matrix(selector: str, dest_dir: str) -> tuple[csr_matrix, ssgetpy.Matrix]:
    """
    selector:
      - 纯数字: ssgetpy id
      - group/name
      - name 子串搜索
    """
    if selector.isdigit():
        rs = ssgetpy.search(id=int(selector))
    elif "/" in selector:
        g, n = selector.split("/", 1)
        rs = ssgetpy.search(group=g, name=n)
    else:
        rs = ssgetpy.search(name=selector)

    if len(rs) == 0:
        raise ValueError(f"[ssgetpy] 找不到矩阵: {selector}")

    rec = rs[0]
    print(
        f"[ssgetpy] id={rec.id}  {rec.group}/{rec.name}  "
        f"{rec.rows}x{rec.cols}  nnz={rec.nnz}"
    )
    ensure_dir(dest_dir)
    tgz_path, extracted = rec.download(format="MM", destpath=dest_dir, extract=True)
    mtx_path = os.path.join(tgz_path, rec.name + ".mtx")
    print(f"[ssgetpy] Matrix Market file: {mtx_path}")
    A = mmread(mtx_path).tocsr()
    return A, rec


# ================== 2. 2D mesh 相关 ==================

def mesh_dims(num_nodes: int) -> Tuple[int, int]:
    """
    给 num_nodes 找一个 (p, q)，使得 p*q == num_nodes，且 |p-q| 尽量小。
    如果 num_nodes 是质数，就退化成 1 x num_nodes。
    """
    if num_nodes <= 0:
        return 1, 1
    p = int(np.floor(np.sqrt(num_nodes)))
    if p <= 0:
        p = 1
    while p > 1 and (num_nodes % p) != 0:
        p -= 1
    q = num_nodes // p
    return p, q


def generate_2d_mesh_metis(num_nodes: int, out_path: str) -> None:
    """
    生成 2D mesh 拓扑的 METIS 图（4-neighbor），用于 MtKaHyPar 的 -g。
    节点按 row-major 编号 0..(p*q-1)，总数等于 num_nodes。
    """
    p, q = mesh_dims(num_nodes)
    print(f"[info] Writing 2D mesh METIS graph: dims=({p},{q}), nodes={num_nodes} -> {out_path}")

    adj: List[List[int]] = [[] for _ in range(num_nodes)]

    def idx(r: int, c: int) -> int:
        return r * q + c

    for r in range(p):
        for c in range(q):
            v = idx(r, c)
            if v >= num_nodes:
                continue
            # 右邻居
            if c + 1 < q:
                u = idx(r, c + 1)
                if u < num_nodes:
                    adj[v].append(u)
                    adj[u].append(v)
            # 下邻居
            if r + 1 < p:
                u = idx(r + 1, c)
                if u < num_nodes:
                    adj[v].append(u)
                    adj[u].append(v)

    m = sum(len(set(neis)) for neis in adj) // 2

    with open(out_path, "w") as f:
        f.write(f"{num_nodes} {m}\n")
        for v in range(num_nodes):
            neighbors = sorted(set(adj[v]))
            line = " ".join(str(u + 1) for u in neighbors)  # METIS 1-based
            f.write(line + "\n")

    print("[info] METIS 2D mesh graph written.")


# ================== 3. 超图构造（带 B 行 nnz 权重） ==================

def build_gustavson_hypergraph_for_A(
    A: csr_matrix,
    B: csr_matrix,
    out_path: str,
    use_b_nnz_as_edge_weight: bool = True,
) -> None:
    """
    顶点: A 的每一行 (Gustavson row task)
    超边: A 每一列 k, 对应所有 A[i,k]!=0 的行 i

    如果 use_b_nnz_as_edge_weight=True，则使用 B 第 k 行的 nnz 作为 hyperedge 权重，
    以 hMETIS / Mt-KaHyPar 的格式:
      第一行: <num_hyperedges> <num_vertices> 1
      每条边: <w_e> v1 v2 ...
    """
    A_csc = A.tocsc()
    B_csr = B.tocsr()
    n_rows, n_cols = A.shape
    hyperedges: List[List[int]] = []
    weights: List[int] = []

    print("[info] Building Gustavson hypergraph for A (with B-row nnz weights) ...")

    for k in range(n_cols):
        start, end = A_csc.indptr[k], A_csc.indptr[k + 1]
        rows = A_csc.indices[start:end]
        if rows.size == 0:
            continue
        he = (rows + 1).tolist()  # 顶点 1-based
        hyperedges.append(he)

        if use_b_nnz_as_edge_weight:
            brs, bre = B_csr.indptr[k], B_csr.indptr[k + 1]
            w = bre - brs
            if w <= 0:
                w = 1
        else:
            w = 1
        weights.append(int(w))

    num_hyperedges = len(hyperedges)
    num_vertices = n_rows
    print(
        f"[info]   num_vertices={num_vertices}, num_hyperedges={num_hyperedges}"
    )

    with open(out_path, "w") as f:
        if use_b_nnz_as_edge_weight:
            f.write(f"{num_hyperedges} {num_vertices} 1\n")
            for w, he in zip(weights, hyperedges):
                f.write(str(w) + " " + " ".join(str(v) for v in he) + "\n")
        else:
            f.write(f"{num_hyperedges} {num_vertices}\n")
            for he in hyperedges:
                f.write(" ".join(str(v) for v in he) + "\n")

    print(f"[info] Hypergraph written to {out_path}")


# ================== 4. WAFERSPMM baseline + Mt-KaHyPar 分区 ==================

def build_wafer_partition(n_rows: int, num_parts: int) -> np.ndarray:
    """
    WAFERSPMM baseline: 简单的 rowblock 映射。
    """
    rows_per_part = int(np.ceil(n_rows / num_parts))
    part = np.arange(n_rows) // rows_per_part
    part = np.clip(part, 0, num_parts - 1)
    print(
        f"[info] WAFERSPMM (rowblock) partition built: n_rows={n_rows}, "
        f"num_parts={num_parts}, rows_per_part~{rows_per_part}"
    )
    return part.astype(int)


def run_mtkahypar(
    hgr_path: str,
    num_parts: int,
    out_partition_dir: str,
    mtk_bin: str,
    target_metis_path: str,
    eps: float = 0.03,
    threads: int = 8,
    preset_type: str = "quality",
) -> str:
    """
    调用 MtKaHyPar:
      MtKaHyPar -h <hgr> -k <k> -e <eps> -g <metis> -o steiner_tree ...
    """
    ensure_dir(out_partition_dir)
    print(f"[info] Running MtKaHyPar: k={num_parts}, eps={eps}")
    print(f"[info]   hypergraph: {hgr_path}")
    print(f"[info]   target graph (NoC): {target_metis_path}")

    cmd = [
        mtk_bin,
        "-h", hgr_path,
        "-k", str(num_parts),
        "-e", f"{eps:.3f}",
        "-g", target_metis_path,
        "-o", "steiner_tree",
        "-t", str(threads),
        "--preset-type", preset_type,
        "--write-partition-file=true",
        "--partition-output-folder", out_partition_dir,
        "--verbose=false",
    ]
    run_cmd(cmd)

    out_dir = Path(out_partition_dir)
    cand = list(out_dir.glob("*.part*")) + list(out_dir.glob("*.partition*"))
    if not cand:
        raise RuntimeError(
            f"[error] MtKaHyPar finished, but no partition file in {out_partition_dir}"
        )
    if len(cand) > 1:
        print("[warn] Multiple partition files found, using the first one:")
        for p in cand:
            print(f"        {p}")
    part_path = str(cand[0].resolve())
    print(f"[info] Partition file: {part_path}")
    return part_path


def load_partition(path: str, n_vertices: int) -> np.ndarray:
    print(f"[info] Loading partition: {path}")
    part = np.loadtxt(path, dtype=int)
    if part.ndim > 1:
        part = part.reshape(-1)
    if part.shape[0] != n_vertices:
        raise ValueError(
            f"[error] Partition length {part.shape[0]} != num_vertices {n_vertices}"
        )
    print(f"[info]   num_parts(from file) = {int(part.max()) + 1}")
    return part


def build_owner_B_wafer(n_rows_b: int, num_parts: int, mode: str) -> np.ndarray:
    """
    WAFERSPMM baseline 的 B 行 owner:
      - rowblock: 按行块
      - mod: k % num_parts
    """
    if mode == "rowblock":
        rows_per_part = int(np.ceil(n_rows_b / num_parts))
        owner = np.arange(n_rows_b) // rows_per_part
        owner = np.clip(owner, 0, num_parts - 1)
    elif mode == "mod":
        owner = np.arange(n_rows_b) % num_parts
    else:
        raise ValueError(f"Unknown b_owner mode: {mode}")
    print(f"[info] B-owner WAFERSPMM baseline ({mode}), n_rows_b={n_rows_b}, num_parts={num_parts}")
    return owner.astype(int)


def build_owner_B_hypergraph_center(
    A: csr_matrix,
    row_part: np.ndarray,
    num_parts: int,
) -> np.ndarray:
    """
    HyperWafer 场景下 B 行 owner:
      对于每一列 k (对应 B 的第 k 行):
        - 找到所有 A[i,k]!=0 的行 i；
        - 看这些行分配到了哪些 tiles (row_part[i])；
        - 统计每个 tile 上涉及该列的行数，选择频率最高的 tile 作为 owner_B[k]。
    """
    A_csc = A.tocsc()
    n_rows_b = A.shape[1]
    owner = np.zeros(n_rows_b, dtype=int)

    print("[info] Building HyperWafer-aware B owners (center by majority tile) ...")

    for k in range(n_rows_b):
        start, end = A_csc.indptr[k], A_csc.indptr[k + 1]
        rows = A_csc.indices[start:end]
        if rows.size == 0:
            owner[k] = k % num_parts
            continue
        tiles = row_part[rows]
        counts = np.bincount(tiles, minlength=num_parts)
        owner[k] = int(np.argmax(counts))

    return owner


# ================== 5. 2D mesh 上的精确 Bcast 仿真（oracle） ==================

def coords_2d(v: int, q: int) -> Tuple[int, int]:
    return divmod(v, q)


def shortest_path_mesh(src: int, dst: int, p: int, q: int):
    if src == dst:
        return []
    r1, c1 = coords_2d(src, q)
    r2, c2 = coords_2d(dst, q)
    edges = []
    cr, cc = r1, c1
    # 先行后列的 Manhattan path
    if r2 != cr:
        dr = 1 if r2 > cr else -1
        while cr != r2:
            nr = cr + dr
            u = cr * q + cc
            v = nr * q + cc
            e = (u, v) if u < v else (v, u)
            edges.append(e)
            cr = nr
    if c2 != cc:
        dc = 1 if c2 > cc else -1
        while cc != c2:
            nc = cc + dc
            u = cr * q + cc
            v = cr * q + nc
            e = (u, v) if u < v else (v, u)
            edges.append(e)
            cc = nc
    return edges


def simulate_bcast_on_mesh(
    A: csr_matrix,
    B: csr_matrix,
    row_part: np.ndarray,
    num_parts: int,
    elem_bytes: int,
    owner_B: np.ndarray,
) -> Tuple[int, int, int, Dict[Tuple[int, int], int],
           float, float, float, float]:
    """
    在 2D mesh 上精确统计 B 行多播的 per-link 流量和 GB-hop。
    """
    n_rows_a, n_cols_a = A.shape
    n_rows_b, n_cols_b = B.shape
    if n_cols_a != n_rows_b:
        raise ValueError(
            f"shape mismatch: A={A.shape}, B={B.shape} (need cols(A)==rows(B))"
        )

    p, q = mesh_dims(num_parts)
    A_csc = A.tocsc()
    B_csr = B.tocsr()

    edge_bytes: Dict[Tuple[int, int], int] = defaultdict(int)
    total_bytes = 0
    gb_hop = 0

    sum_tiles_all = 0
    sum_recv_all = 0
    cnt_rows_all = 0

    sum_tiles_comm = 0
    sum_recv_comm = 0
    cnt_rows_comm = 0

    print(f"[info] Simulating Bcast on 2D mesh (p={p}, q={q}) ...")

    for k in range(n_rows_b):
        col_start, col_end = A_csc.indptr[k], A_csc.indptr[k + 1]
        rows = A_csc.indices[col_start:col_end]
        if rows.size == 0:
            continue

        dest_tiles = np.unique(row_part[rows])
        src = int(owner_B[k])

        b_row_start, b_row_end = B_csr.indptr[k], B_csr.indptr[k + 1]
        nnz_b_row = b_row_end - b_row_start
        if nnz_b_row == 0:
            continue

        num_tiles_using = int(dest_tiles.size)
        num_recv = int(np.sum(dest_tiles != src))

        sum_tiles_all += num_tiles_using
        sum_recv_all += num_recv
        cnt_rows_all += 1

        if num_recv > 0:
            sum_tiles_comm += num_tiles_using
            sum_recv_comm += num_recv
            cnt_rows_comm += 1

        if num_recv <= 0:
            continue

        bytes_k = nnz_b_row * elem_bytes

        for d in dest_tiles:
            d = int(d)
            if d == src:
                continue
            path_edges = shortest_path_mesh(src, d, p, q)
            hops = len(path_edges)
            for e in path_edges:
                edge_bytes[e] += bytes_k
            total_bytes += bytes_k
            gb_hop += bytes_k * hops

    peak = max(edge_bytes.values()) if edge_bytes else 0

    avg_tiles_all = (sum_tiles_all / cnt_rows_all) if cnt_rows_all > 0 else 0.0
    avg_recv_all = (sum_recv_all / cnt_rows_all) if cnt_rows_all > 0 else 0.0
    avg_tiles_comm = (sum_tiles_comm / cnt_rows_comm) if cnt_rows_comm > 0 else 0.0
    avg_recv_comm = (sum_recv_comm / cnt_rows_comm) if cnt_rows_comm > 0 else 0.0

    print(f"[info]   total_bytes={total_bytes}, GB-hop={gb_hop}, peak_link={peak}")
    print(f"[info]   B-rows with work (A[:,k]!=0 & B[k,:]!=0): {cnt_rows_all}")
    print(f"[info]   avg tiles using each B-row (all rows)     = {avg_tiles_all:.3f}")
    print(f"[info]   avg remote tiles per B-row (all rows)    = {avg_recv_all:.3f}")
    print(f"[info]   B-rows with cross-tile comm: {cnt_rows_comm}")
    print(f"[info]   avg tiles using each B-row (comm rows)   = {avg_tiles_comm:.3f}")
    print(f"[info]   avg remote tiles per B-row (comm rows)  = {avg_recv_comm:.3f}")

    return (
        total_bytes,
        gb_hop,
        peak,
        edge_bytes,
        avg_tiles_all,
        avg_recv_all,
        avg_tiles_comm,
        avg_recv_comm,
    )


# ================== 6. 收集 per-B-row bytes ==================

def collect_bcast_row_bytes(
    A: csr_matrix,
    B: csr_matrix,
    row_part: np.ndarray,
    num_parts: int,
    elem_bytes: int,
    owner_B: np.ndarray,
) -> List[int]:
    """
    为每一行 B[k,:] 计算一次广播的总 bytes:
      bytes_k = nnz(B[k,:]) * elem_bytes * (#dest tiles - 1)
    """
    A_csc = A.tocsc()
    B_csr = B.tocsr()
    n_rows_b = B.shape[0]
    row_bytes: List[int] = []

    for k in range(n_rows_b):
        col_start, col_end = A_csc.indptr[k], A_csc.indptr[k + 1]
        rows = A_csc.indices[col_start:col_end]
        if rows.size == 0:
            continue

        dest_tiles = np.unique(row_part[rows])
        src = int(owner_B[k])

        b_row_start, b_row_end = B_csr.indptr[k], B_csr.indptr[k + 1]
        nnz_b_row = b_row_end - b_row_start
        if nnz_b_row == 0:
            continue

        num_dests = int(np.sum(dest_tiles != src))
        if num_dests <= 0:
            continue

        bytes_k = nnz_b_row * elem_bytes * num_dests
        row_bytes.append(bytes_k)

    return row_bytes


# ================== 7. MICRO phases 构造 ==================

def build_bcast_micro_phases_rows_batch(
    row_bytes: List[int],
    tag: str,
    rows_per_layer: int,
) -> List[Dict[str, object]]:
    """
    按“B 行数”进行 batching:
      - 给定 rows_per_layer，顺序遍历 row_bytes，累加到 rows_per_layer 条为一层；
      - 最后一层不足 rows_per_layer 也单独成层；
      - 不补 0，也不强行限制层数，适合你说的“batch 512 后 WAFERSPMM ~800 层、Hyper ~40 层”。
    """
    phases: List[Dict[str, object]] = []
    if rows_per_layer <= 0:
        rows_per_layer = 1

    if not row_bytes:
        name = f"SpGEMM_BcastB_{tag}_noop"
        phases.append({
            "name": name,
            "comm_type": "ALLGATHER",
            "bytes": 0,
        })
        print(f"[info] rows-batch MICRO (tag={tag}): no cross-tile rows, 1 noop layer.")
        return phases

    total_bytes = int(sum(row_bytes))
    acc = 0
    cnt = 0
    layer_idx = 0

    for b in row_bytes:
        acc += int(b)
        cnt += 1
        if cnt >= rows_per_layer:
            name = f"SpGEMM_BcastB_{tag}_phase{layer_idx:04d}"
            phases.append({
                "name": name,
                "comm_type": "ALLGATHER",
                "bytes": acc,
            })
            layer_idx += 1
            acc = 0
            cnt = 0

    if cnt > 0 or len(phases) == 0:
        name = f"SpGEMM_BcastB_{tag}_phase{layer_idx:04d}"
        phases.append({
            "name": name,
            "comm_type": "ALLGATHER",
            "bytes": acc,
        })
        layer_idx += 1

    sum_bytes = sum(ph["bytes"] for ph in phases)
    if sum_bytes != total_bytes:
        print(
            f"[warn] rows-batch MICRO (tag={tag}): sum phase bytes ({sum_bytes}) "
            f"!= total_bytes ({total_bytes})"
        )

    print(
        f"[info] rows-batch MICRO (tag={tag}): "
        f"rows_per_layer={rows_per_layer}, layers={len(phases)}, total_bytes={total_bytes}"
    )
    return phases


def write_micro_text(output_path: str, phases: List[Dict[str, object]]) -> None:
    """
    MICRO Text format:
      line1: MICRO
      line2: num_layers
      line3+: 每层:
        <name>  -1  0  NONE 0  0  NONE 0  0  <COMM_TYPE>   <size>  0
    """
    print(f"[info] Writing MICRO workload: {output_path}")
    with open(output_path, "w") as f:
        f.write("MICRO\n")
        f.write(f"{len(phases)}\n")
        for ph in phases:
            name = ph["name"]
            comm_type = ph["comm_type"]
            size = int(ph["bytes"])
            line = (
                f"{name}  -1  0  NONE 0  0  NONE 0  0  "
                f"{comm_type}   {size}  0\n"
            )
            f.write(line)
    print("[info] MICRO text written.")


# ================== 8. Chakra + AstraSim ==================

def run_chakra_converter(
    text_path: str,
    out_prefix: str,
    num_npus: int,
    num_passes: int,
    chakra_bin: str,
) -> None:
    print(f"[info] Running chakra_converter for {text_path}")
    cmd = [
        chakra_bin,
        "Text",
        "--input", text_path,
        "--output", out_prefix,
        "--num-npus", str(num_npus),
        "--num-passes", str(num_passes),
    ]
    run_cmd(cmd)


def run_astrasim_and_get_comm_time(
    workload_prefix: str,
    system_config: str,
    network_config: str,
    remote_mem_config: str,
    astrasim_bin: str,
) -> int:
    """
    调用 AstraSim Analytical backend，解析最大 Comm time。
    """
    print(f"[info] Running AstraSim with workload={workload_prefix}")
    cmd = [
        astrasim_bin,
        f"--workload-configuration={workload_prefix}",
        f"--system-configuration={system_config}",
        f"--remote-memory-configuration={remote_mem_config}",
        f"--network-configuration={network_config}",
    ]
    proc = run_cmd(cmd, capture=True)

    stdout = proc.stdout
    pattern = re.compile(
        r"sys\[(\d+)\], Comm time:\s*([0-9]+)", re.MULTILINE
    )
    comm_times = []
    for m in pattern.finditer(stdout):
        ct = int(m.group(2))
        comm_times.append(ct)

    if not comm_times:
        print("[warn] No Comm time lines found in AstraSim output.")
        return -1

    max_ct = max(comm_times)
    print(f"[info]   AstraSim Comm time (max over sys[]): {max_ct}")
    return max_ct


# ================== 9. 主流程 ==================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "SpGEMM communication pipeline (v8): "
            "SuiteSparse -> weighted hypergraph -> 2D mesh MtKaHyPar -> "
            "2D mesh oracle -> MICRO (rows-batch) -> Chakra -> AstraSim, "
            "compare WAFERSPMM vs HyperWafer mapping."
        )
    )

    # 矩阵 & 基础参数
    parser.add_argument(
        "--matrix-selector",
        required=True,
        help="SuiteSparse selector: id / group/name / name substring.",
    )
    parser.add_argument(
        "--ss-dest-dir",
        default="/data1/duzc/mat",
        help="SuiteSparse download/extract directory.",
    )

    parser.add_argument(
        "--num-parts",
        type=int,
        required=True,
        help="Number of tiles / NPUs / partitions.",
    )
    parser.add_argument(
        "--elem-bytes",
        type=int,
        default=4,
        help="Bytes per nonzero element in B (e.g., 4 for fp32).",
    )

    # Mt-KaHyPar 参数
    parser.add_argument("--mtk-bin", required=True, help="Path to MtKaHyPar binary.")
    parser.add_argument(
        "--mtk-eps",
        type=float,
        default=0.03,
        help="MtKaHyPar imbalance epsilon (-e).",
    )
    parser.add_argument(
        "--mtk-threads",
        type=int,
        default=8,
        help="MtKaHyPar threads (-t).",
    )
    parser.add_argument(
        "--mtk-preset",
        default="quality",
        help="MtKaHyPar preset (--preset-type).",
    )

    # Chakra / AstraSim
    parser.add_argument(
        "--chakra-bin",
        default="chakra_converter",
        help="chakra_converter binary.",
    )
    parser.add_argument(
        "--num-passes",
        type=int,
        default=1,
        help="Number of passes for Chakra ET.",
    )

    parser.add_argument(
        "--astrasim-bin",
        required=True,
        help="Path to AstraSim_Analytical_Congestion_Aware.",
    )
    parser.add_argument(
        "--system-config",
        required=True,
        help="AstraSim system config JSON.",
    )
    parser.add_argument(
        "--network-config",
        required=True,
        help="AstraSim network config (YAML/JSON).",
    )
    parser.add_argument(
        "--remote-mem-config",
        required=True,
        help="AstraSim remote memory config JSON.",
    )

    # 其他
    parser.add_argument(
        "--workdir",
        default="./spgemm_full_out_v8",
        help="Directory for all intermediate outputs.",
    )
    parser.add_argument(
        "--b-owner-mode",
        choices=["rowblock", "mod"],
        default="rowblock",
        help="Baseline B-row owner mode for WAFERSPMM (rowblock or mod).",
    )
    parser.add_argument(
        "--noc-metis",
        default=None,
        help=(
            "Optional METIS graph path for NoC. "
            "If not provided, a 2D mesh METIS is generated for MtKaHyPar -g."
        ),
    )

    # MICRO 行 batching 模式
    parser.add_argument(
        "--micro-mode",
        choices=["single", "auto_rows", "rows_manual"],
        default="auto_rows",
        help=(
            "MICRO construction mode: "
            "single      = one big ALLGATHER per mapping; "
            "auto_rows   = choose rows_per_layer so that WAFERSPMM has at most micro-max-layers layers, "
            "               HyperWafer uses the same rows_per_layer (所以它通常层数更少); "
            "rows_manual = use fixed micro-rows-per-layer for both mappings."
        ),
    )
    parser.add_argument(
        "--micro-max-layers",
        type=int,
        default=1000,
        help="Target maximum layers for WAFERSPMM in auto_rows mode.",
    )
    parser.add_argument(
        "--micro-rows-per-layer",
        type=int,
        default=512,
        help="Rows per MICRO layer if micro-mode=rows_manual.",
    )

    args = parser.parse_args()

    workdir = Path(args.workdir).absolute()
    ensure_dir(workdir)

    # 1. SuiteSparse 矩阵
    A, rec = load_ss_matrix(args.matrix_selector, args.ss_dest_dir)
    print(
        f"[info] Loaded A: {rec.group}/{rec.name}, shape={A.shape}, nnz={A.nnz}"
    )

    # B = A^T（之后你可以替换成真实 B）
    B = A.transpose().tocsr()
    print(f"[info] Using B = A^T, shape={B.shape}, nnz={B.nnz}")

    n_rows_a = A.shape[0]
    n_rows_b = B.shape[0]

    # 2. WAFERSPMM baseline 映射
    row_part_wafer = build_wafer_partition(n_rows_a, args.num_parts)
    owner_B_wafer = build_owner_B_wafer(
        n_rows_b=n_rows_b,
        num_parts=args.num_parts,
        mode=args.b_owner_mode,
    )

    # 3. HyperWafer 映射（超图 + Mt-KaHyPar, 目标图为 2D mesh，带 B-row nnz 权重）
    hgr_path = str(workdir / "hypergraph_weighted.hgr")
    build_gustavson_hypergraph_for_A(A, B, hgr_path, use_b_nnz_as_edge_weight=True)

    if args.noc_metis is None:
        noc_metis_path = str(workdir / f"noc_{args.num_parts}n_mesh.metis")
        generate_2d_mesh_metis(args.num_parts, noc_metis_path)
    else:
        noc_metis_path = args.noc_metis
        print(f"[info] Using user-supplied NoC METIS graph: {noc_metis_path}")

    part_dir = str(workdir / "mtk_parts")
    part_path = run_mtkahypar(
        hgr_path=hgr_path,
        num_parts=args.num_parts,
        out_partition_dir=part_dir,
        mtk_bin=args.mtk_bin,
        target_metis_path=noc_metis_path,
        eps=args.mtk_eps,
        threads=args.mtk_threads,
        preset_type=args.mtk_preset,
    )
    row_part_hyper = load_partition(part_path, n_vertices=n_rows_a)

    # 基于超图分区构造 B 行 owner（中心 tile = 使用该列任务最多的 tile）
    owner_B_hyper = build_owner_B_hypergraph_center(
        A=A,
        row_part=row_part_hyper,
        num_parts=args.num_parts,
    )

    # 4. 2D mesh oracle：WAFERSPMM vs HyperWafer
    print("\n[phase] 2D mesh oracle: WAFERSPMM baseline")
    (
        wafer_total,
        wafer_gbhop,
        wafer_peak,
        _,
        wafer_avg_tiles_all,
        wafer_avg_recv_all,
        wafer_avg_tiles_comm,
        wafer_avg_recv_comm,
    ) = simulate_bcast_on_mesh(
        A=A,
        B=B,
        row_part=row_part_wafer,
        num_parts=args.num_parts,
        elem_bytes=args.elem_bytes,
        owner_B=owner_B_wafer,
    )

    print("\n[phase] 2D mesh oracle: HyperWafer mapping")
    (
        hyper_total,
        hyper_gbhop,
        hyper_peak,
        _,
        hyper_avg_tiles_all,
        hyper_avg_recv_all,
        hyper_avg_tiles_comm,
        hyper_avg_recv_comm,
    ) = simulate_bcast_on_mesh(
        A=A,
        B=B,
        row_part=row_part_hyper,
        num_parts=args.num_parts,
        elem_bytes=args.elem_bytes,
        owner_B=owner_B_hyper,
    )

    print("\n[oracle-summary]")
    print(f"  WAFERSPMM:  total_bytes={wafer_total}, GB-hop={wafer_gbhop}, peak_link={wafer_peak}")
    print(f"              avg tiles using B-row (all/comm)   = {wafer_avg_tiles_all:.3f} / {wafer_avg_tiles_comm:.3f}")
    print(f"              avg remote tiles per B-row (all/comm) = {wafer_avg_recv_all:.3f} / {wafer_avg_recv_comm:.3f}")
    print(f"  HyperWafer: total_bytes={hyper_total}, GB-hop={hyper_gbhop}, peak_link={hyper_peak}")
    print(f"              avg tiles using B-row (all/comm)   = {hyper_avg_tiles_all:.3f} / {hyper_avg_tiles_comm:.3f}")
    print(f"              avg remote tiles per B-row (all/comm) = {hyper_avg_recv_all:.3f} / {hyper_avg_recv_comm:.3f}")
    if hyper_total > 0 and hyper_gbhop > 0 and hyper_peak > 0:
        print(f"  volume reduction (WAFERSPMM / HyperWafer)  = {wafer_total / hyper_total:.3f}x")
        print(f"  GB-hop reduction (WAFERSPMM / HyperWafer)  = {wafer_gbhop / hyper_gbhop:.3f}x")
        print(f"  peak-link reduction                        = {wafer_peak / hyper_peak:.3f}x")

    # 5. MICRO phases：single / auto_rows / rows_manual
    if args.micro_mode == "single":
        print("\n[phase] Build SINGLE MICRO phases (one big ALLGATHER per mapping)")
        phases_wafer = [{
            "name": "SpGEMM_BcastB_WAFERSPMM_total",
            "comm_type": "ALLGATHER",
            "bytes": wafer_total,
        }]
        phases_hyper = [{
            "name": "SpGEMM_BcastB_HyperWafer_total",
            "comm_type": "ALLGATHER",
            "bytes": hyper_total,
        }]
        rows_per_layer_effective = None
    else:
        print("\n[phase] Collect per-B-row bytes for WAFERSPMM")
        row_bytes_wafer = collect_bcast_row_bytes(
            A=A,
            B=B,
            row_part=row_part_wafer,
            num_parts=args.num_parts,
            elem_bytes=args.elem_bytes,
            owner_B=owner_B_wafer,
        )
        print(f"[info] WAFERSPMM: #active B-rows with cross-tile comm = {len(row_bytes_wafer)}")

        print("\n[phase] Collect per-B-row bytes for HyperWafer")
        row_bytes_hyper = collect_bcast_row_bytes(
            A=A,
            B=B,
            row_part=row_part_hyper,
            num_parts=args.num_parts,
            elem_bytes=args.elem_bytes,
            owner_B=owner_B_hyper,
        )
        print(f"[info] HyperWafer: #active B-rows with cross-tile comm = {len(row_bytes_hyper)}")

        if args.micro_mode == "auto_rows":
            Mw = len(row_bytes_wafer)
            if Mw > 0:
                rows_per_layer = max(1, int(np.ceil(Mw / args.micro_max_layers)))
            else:
                rows_per_layer = 1
            print(
                f"[info] auto_rows MICRO: "
                f"Mw={Mw}, target_max_layers={args.micro_max_layers}, "
                f"rows_per_layer={rows_per_layer}"
            )
        else:  # rows_manual
            rows_per_layer = max(1, args.micro_rows_per_layer)
            print(
                f"[info] rows_manual MICRO: rows_per_layer={rows_per_layer}"
            )

        rows_per_layer_effective = rows_per_layer

        print("\n[phase] Build rows-batch MICRO phases for WAFERSPMM")
        phases_wafer = build_bcast_micro_phases_rows_batch(
            row_bytes=row_bytes_wafer,
            tag="WAFERSPMM",
            rows_per_layer=rows_per_layer,
        )
        print("\n[phase] Build rows-batch MICRO phases for HyperWafer")
        phases_hyper = build_bcast_micro_phases_rows_batch(
            row_bytes=row_bytes_hyper,
            tag="HyperWafer",
            rows_per_layer=rows_per_layer,
        )

    total_wafer_micro = sum(ph["bytes"] for ph in phases_wafer)
    total_hyper_micro = sum(ph["bytes"] for ph in phases_hyper)
    print(f"[check] total Bcast bytes from MICRO phases (WAFERSPMM)  = {total_wafer_micro}")
    print(f"[check] total Bcast bytes from MICRO phases (HyperWafer) = {total_hyper_micro}")

    micro_wafer = str(workdir / "SpGEMM_WAFERSPMM.txt")
    micro_hyper = str(workdir / "SpGEMM_HyperWafer.txt")

    write_micro_text(micro_wafer, phases_wafer)
    write_micro_text(micro_hyper, phases_hyper)

    # 6. Chakra: MICRO -> ET
    wl_wafer_prefix = str(workdir / "SpGEMM_WAFERSPMM")
    wl_hyper_prefix = str(workdir / "SpGEMM_HyperWafer")

    run_chakra_converter(
        text_path=micro_wafer,
        out_prefix=wl_wafer_prefix,
        num_npus=args.num_parts,
        num_passes=args.num_passes,
        chakra_bin=args.chakra_bin,
    )
    run_chakra_converter(
        text_path=micro_hyper,
        out_prefix=wl_hyper_prefix,
        num_npus=args.num_parts,
        num_passes=args.num_passes,
        chakra_bin=args.chakra_bin,
    )

    # 7. AstraSim: WAFERSPMM / HyperWafer 的通信时间
    print("\n[phase] Run AstraSim for WAFERSPMM")
    comm_wafer = run_astrasim_and_get_comm_time(
        workload_prefix=wl_wafer_prefix,
        system_config=args.system_config,
        network_config=args.network_config,
        remote_mem_config=args.remote_mem_config,
        astrasim_bin=args.astrasim_bin,
    )

    print("\n[phase] Run AstraSim for HyperWafer")
    comm_hyper = run_astrasim_and_get_comm_time(
        workload_prefix=wl_hyper_prefix,
        system_config=args.system_config,
        network_config=args.network_config,
        remote_mem_config=args.remote_mem_config,
        astrasim_bin=args.astrasim_bin,
    )

    # 8. 总结对比
    print("\n========== SUMMARY (PIPELINE v8) ==========")
    print(f"Matrix: {rec.group}/{rec.name}")
    print(f"A: shape={A.shape}, nnz={A.nnz}")
    print(f"B: shape={B.shape}, nnz={B.nnz}")
    print(f"num_parts: {args.num_parts}")
    print(f"micro_mode: {args.micro_mode}")
    if args.micro_mode == "auto_rows":
        print(f"micro_max_layers (target for WAFERSPMM): {args.micro_max_layers}")
    if args.micro_mode == "rows_manual":
        print(f"micro_rows_per_layer: {args.micro_rows_per_layer}")
    if 'rows_per_layer_effective' in locals() and rows_per_layer_effective is not None:
        print(f"effective rows_per_layer used: {rows_per_layer_effective}")
    print("")
    print("WAFERSPMM baseline:")
    print(f"  Oracle total_bytes      = {wafer_total}")
    print(f"  Oracle GB-hop           = {wafer_gbhop}")
    print(f"  Oracle peak_link_bytes  = {wafer_peak}")
    print(f"  MICRO total_bytes       = {total_wafer_micro}")
    print(f"  MICRO num_layers        = {len(phases_wafer)}")
    print(f"  AstraSim Comm time      = {comm_wafer}")
    print("")
    print("HyperWafer mapping:")
    print(f"  Oracle total_bytes      = {hyper_total}")
    print(f"  Oracle GB-hop           = {hyper_gbhop}")
    print(f"  Oracle peak_link_bytes  = {hyper_peak}")
    print(f"  MICRO total_bytes       = {total_hyper_micro}")
    print(f"  MICRO num_layers        = {len(phases_hyper)}")
    print(f"  AstraSim Comm time      = {comm_hyper}")
    print("")
    if hyper_total > 0:
        print(f"Volume reduction (WAFERSPMM / HyperWafer):      {wafer_total / hyper_total:.3f}x")
    if hyper_gbhop > 0:
        print(f"GB-hop reduction (WAFERSPMM / HyperWafer):      {wafer_gbhop / hyper_gbhop:.3f}x")
    if hyper_peak > 0:
        print(f"Peak-link reduction (WAFERSPMM / HyperWafer):   {wafer_peak / hyper_peak:.3f}x")
    if comm_wafer > 0 and comm_hyper > 0:
        print(f"Comm-time speedup (AstraSim, WAFERSPMM/HyperWafer): {comm_wafer / comm_hyper:.3f}x")
    print("===========================================")


if __name__ == "__main__":
    main()
