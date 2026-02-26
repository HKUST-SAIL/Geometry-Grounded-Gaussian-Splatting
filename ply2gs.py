import argparse
import numpy as np
from plyfile import PlyData, PlyElement


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1.0 - p))


def _sorted_suffix(names, prefix):
    def keyfn(s):
        try:
            return int(s.split("_")[-1])
        except Exception:
            return 10**9
    return sorted([n for n in names if n.startswith(prefix)], key=keyfn)


def infer_sh_degree_from_frest_count(frest_count: int) -> int:
    # frest_count = 3 * ((deg+1)^2 - 1)
    if frest_count % 3 != 0:
        raise ValueError(f"f_rest_* count {frest_count} not divisible by 3")
    k = frest_count // 3
    # k = (deg+1)^2 - 1  => (deg+1)^2 = k+1
    m = k + 1
    r = int(round(np.sqrt(m)))
    if r * r != m:
        raise ValueError(f"Cannot infer SH degree from f_rest count={frest_count} (k={k}, m={m})")
    return r - 1


def main(args):

    ply = PlyData.read(args.in_ply)
    if "vertex" not in ply:
        raise ValueError("PLY has no 'vertex' element")

    v = ply["vertex"].data
    names = list(v.dtype.names)
    n = v.shape[0]

    # --- required basics ---
    for k in ["x", "y", "z", "opacity", "f_dc_0", "f_dc_1", "f_dc_2"]:
        if k not in names:
            raise ValueError(f"Missing required property '{k}'")

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

    # --- opacity ---
    opacity_raw = np.asarray(v["opacity"], dtype=np.float32).reshape(n, 1)
    opacity_act = sigmoid(opacity_raw)

    # --- filter_3D ---
    filter_3d = np.asarray(v["filter_3D"], dtype=np.float32).reshape(n, 1)

    # --- scales ---
    scale_names = _sorted_suffix(names, "scale_")
    if len(scale_names) == 0:
        raise ValueError("Missing scale_* properties")
    if len(scale_names) != 3:
        raise ValueError(f"Expected 3 scale_* props, got {len(scale_names)}: {scale_names}")

    scales_raw = np.stack([v[s] for s in scale_names], axis=1).astype(np.float32)
    scales = np.exp(scales_raw)

    # --- rotations ---
    rot_names = _sorted_suffix(names, "rot_")
    if len(rot_names) == 0:
        rot_names = _sorted_suffix(names, "rot")
    if len(rot_names) != 4:
        raise ValueError(f"Expected 4 rot_* props, got {len(rot_names)}: {rot_names}")
    rots = np.stack([v[rn] for rn in rot_names], axis=1).astype(np.float32)

    # --- SH features ---
    fdc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)

    frest_names = _sorted_suffix(names, "f_rest_")
    frest = np.stack([v[nm] for nm in frest_names], axis=1).astype(np.float32) if len(frest_names) else np.zeros((n, 0), np.float32)

    in_deg = infer_sh_degree_from_frest_count(frest.shape[1]) if frest.shape[1] > 0 else 0
    out_deg = in_deg if args.out_sh_degree == -1 else args.out_sh_degree

    out_rest = 3 * ((out_deg + 1) ** 2 - 1)  # e.g. deg3 => 45
    if frest.shape[1] >= out_rest:
        frest_out = frest[:, :out_rest]
    else:
        pad = np.zeros((n, out_rest - frest.shape[1]), dtype=np.float32)
        frest_out = np.concatenate([frest, pad], axis=1)

    # --- apply 3D filter ---
    scales2 = scales * scales  # (n,3)
    det1 = np.prod(scales2, axis=1)  # (n,)
    f2 = (filter_3d * filter_3d).astype(np.float32)  # (n,1)
    scales2_f = scales2 + f2  # (n,3)
    det2 = np.prod(scales2_f, axis=1)  # (n,)

    coef = np.sqrt(det1) / np.sqrt(det2)  # (n,)

    scales_new = np.sqrt(scales2_f)                 # (n,3)
    opacity_new = opacity_act * coef.reshape(n, 1)  # (n,1)

    # --- invert back to raw storage convention ---
    scales_new = np.clip(scales_new, 1e-20, None)
    scales_raw_new = np.log(scales_new).astype(np.float32)

    p = np.clip(opacity_new, 1e-6, 1.0 - 1e-6)
    opacity_raw_new = logit(p).astype(np.float32)

    out_dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
    ]
    out_dtype += [(f"f_rest_{i}", "f4") for i in range(out_rest)]
    out_dtype += [
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]

    out = np.empty(n, dtype=np.dtype(out_dtype))
    out["x"], out["y"], out["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    out["nx"], out["ny"], out["nz"] = 0.0, 0.0, 0.0

    out["f_dc_0"], out["f_dc_1"], out["f_dc_2"] = fdc[:, 0], fdc[:, 1], fdc[:, 2]
    for i in range(out_rest):
        out[f"f_rest_{i}"] = frest_out[:, i]

    out["opacity"] = opacity_raw_new[:, 0]
    out["scale_0"], out["scale_1"], out["scale_2"] = scales_raw_new[:, 0], scales_raw_new[:, 1], scales_raw_new[:, 2]
    out["rot_0"], out["rot_1"], out["rot_2"], out["rot_3"] = rots[:, 0], rots[:, 1], rots[:, 2], rots[:, 3]

    el = PlyElement.describe(out, "vertex")
    ply_out = PlyData([el])
    ply_out.write(args.out_ply)

    print(f"Done. in_deg={in_deg}, out_deg={out_deg}, N={n}")
    if "filter_3D" not in names:
        print("Note: input had no filter_3D; assumed 0.")


if __name__ == "__main__":
    # convert the output ply to the Gaussian Splatting format.
    # This script drops the spharical Gaussians and bakes 3D filter effect into scale and opacity.
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_ply", type=str)
    ap.add_argument("--out_ply", type=str)

    ap.add_argument("--out_sh_degree", type=int, default=3,
                    help="Supersplat commonly expects degree=3. Set -1 to keep inferred input degree.")
    args = ap.parse_args()
    main(args)