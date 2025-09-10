# tfrecord_explorer.py
# Quick visual explorer for AIST++ TFRecord shards
# pip install tensorflow matplotlib

import os
import glob
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# === CONFIG: change this to your path ===
# Example: r"C:\Users\kutay\Downloads\aist_generation_val_v2_tfrecord-*"
PATTERN = r"C:\path\to\aist_generation_*_v2_tfrecord-*"
MAX_PRINT = 3   # how many samples to print

def bytes_try_parse_tensor(b):
    """Try to decode tf.io.parse_tensor for bytes that contain a serialized Tensor."""
    try:
        return tf.io.parse_tensor(b, out_type=tf.float32).numpy()
    except Exception:
        try:
            return tf.io.parse_tensor(b, out_type=tf.int64).numpy()
        except Exception:
            return None

def describe_feature(feat):
    """Return (type, length, preview_numpy) for a tf.train.Feature."""
    f = feat
    if f.bytes_list.value:
        vals = list(f.bytes_list.value)
        if len(vals) == 1:
            # Sometimes a single bytes stores a serialized Tensor
            pt = bytes_try_parse_tensor(vals[0])
            if pt is not None:
                return ("bytes(serialized_tensor)", pt.size, pt)
            # Otherwise treat as raw bytes
            return ("bytes", len(vals[0]), None)
        else:
            return ("bytes_list", len(vals), None)
    if f.float_list.value:
        arr = np.array(f.float_list.value, dtype=np.float32)
        return ("float_list", arr.size, arr)
    if f.int64_list.value:
        arr = np.array(f.int64_list.value, dtype=np.int64)
        return ("int64_list", arr.size, arr)
    return ("unknown", 0, None)

def guess_shapes(name, arr):
    """Heuristics to guess useful shapes for visualization."""
    if arr is None:
        return None
    size = int(np.prod(arr.shape))
    flat = arr.reshape(-1)

    # Common AIST++ motion/pose shapes:
    # angles: (T, 8) -> e.g., 40*8 = 320
    for T in [20, 30, 40, 60, 80, 100, 120, 160]:
        if size % (T * 8) == 0:
            return (T, 8)

    # 2D keypoints: (T, 33, 2) or (T, 17, 2)
    for K in [33, 17, 25]:
        for T in [20, 30, 40, 60, 80, 100, 120, 160]:
            if size == T * K * 2:
                return (T, K, 2)

    # 3D keypoints: (T, 33, 3) or (T, 17, 3)
    for K in [33, 17, 25]:
        for T in [20, 30, 40, 60, 80, 100, 120, 160]:
            if size == T * K * 3:
                return (T, K, 3)

    # Generic: try time-major 2D if it’s “not too wide”
    # pick a D that divides size and is <= 256
    for D in [4, 6, 8, 10, 12, 16, 24, 32, 64, 128, 256]:
        if size % D == 0 and (size // D) >= 10:
            return (size // D, D)

    return None

def plot_heatmap(seq2d, title="Sequence heatmap (T x D)"):
    plt.figure()
    plt.imshow(seq2d, aspect='auto')  # DO NOT set colors explicitly (tooling rule)
    plt.title(title)
    plt.xlabel("Feature dim (D)")
    plt.ylabel("Time (T)")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

def main():
    files = sorted(glob.glob(PATTERN))
    if not files:
        print(f"No files matched pattern: {PATTERN}")
        return
    print(f"Found {len(files)} shard(s). Showing a quick peek from: {files[0]}")
    raw_ds = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)

    # Peek a few samples and print schema-like info
    printed = 0
    label_keys_candidates = ["label", "genre", "y", "class", "dance_label"]
    first_vis_done = False

    for raw in raw_ds.take(MAX_PRINT):
        ex = tf.train.Example()
        ex.ParseFromString(raw.numpy())
        feats = ex.features.feature
        print("\n=== SAMPLE ===")
        for k in sorted(feats.keys()):
            ftype, flen, arr = describe_feature(feats[k])
            preview = ""
            if isinstance(arr, np.ndarray):
                pr = np.array2string(arr[:min(12, arr.size)], precision=3, suppress_small=True)
                preview = f" | preview: {pr}"
            print(f"- {k:24s}  type={ftype:22s} len={flen:5d}{preview}")

        # Try to find and print a label if any
        label_val = None
        for lk in label_keys_candidates:
            if lk in feats and feats[lk].int64_list.value:
                label_val = int(feats[lk].int64_list.value[0])
                print(f"  -> Detected label key '{lk}': {label_val}")
                break

        # Try a visualization once: pick a float feature and reshape
        if not first_vis_done:
            candidate_key = None
            candidate_arr = None
            for k in feats.keys():
                ftype, flen, arr = describe_feature(feats[k])
                if ftype in ("float_list", "bytes(serialized_tensor)") and isinstance(arr, np.ndarray) and arr.size >= 40*6:
                    candidate_key = k
                    candidate_arr = arr
                    break
            if candidate_arr is not None:
                shp = guess_shapes(candidate_key, candidate_arr)
                if shp is not None:
                    try:
                        seq = candidate_arr.reshape(shp)
                        if seq.ndim == 2:
                            plot_heatmap(seq, title=f"{candidate_key}: {shp}")
                            first_vis_done = True
                        elif seq.ndim == 3:
                            # If (T,K,2 or 3), flatten landmarks dimension for heatmap
                            T = seq.shape[0]
                            D = int(np.prod(seq.shape[1:]))
                            plot_heatmap(seq.reshape(T, D), title=f"{candidate_key} flattened: {shp}")
                            first_vis_done = True
                    except Exception as e:
                        print(f"Could not reshape {candidate_key} to {shp}: {e}")

        printed += 1

    # Optional: quick label distribution (stream over shards)
    print("\n=== Quick label distribution (best-effort) ===")
    counts = {}
    def update_counts(lv):
        counts[lv] = counts.get(lv, 0) + 1

    for raw in tf.data.TFRecordDataset(files):
        ex = tf.train.Example()
        ex.ParseFromString(raw.numpy())
        feats = ex.features.feature
        got = False
        for lk in ["label", "genre", "y", "class", "dance_label"]:
            if lk in feats and feats[lk].int64_list.value:
                update_counts(int(feats[lk].int64_list.value[0]))
                got = True
                break
        if not got:
            continue

    if counts:
        total = sum(counts.values())
        print(f"Total labeled samples: {total}")
        for k in sorted(counts.keys()):
            pct = 100.0 * counts[k] / total
            print(f"  class {k}: {counts[k]} ({pct:.1f}%)")
    else:
        print("No obvious integer label key found (tried: label, genre, y, class, dance_label).")

if __name__ == "__main__":
    main()
