# ======================================================
# HPC READY — BATCH VERSION
#CONNECTIVITY AND SPATIAL STRUCTURE METRICS 
# ======================================================

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# --- HPC safe backend ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import networkx as nx

from scipy import ndimage as ndi
from skimage.morphology import dilation, disk

# ✅ ADDED
from networkx.algorithms import community
from scipy.spatial import cKDTree


# ==============================
# USER INPUTS (HPC)
# ==============================

# OPTION A: single image (works locally or HPC)
# Parent folder that contains day subfolders
INPUT_DIR = Path("/mnt/hpccs01/home/n11555581/Code/PARTIALRESULTS3")

# Disable single-image mode
IMAGE_PATH = None
IMAGE_GLOB = "*.png"

# Output root (HPC)
OUTPUT_ROOT = Path("./HPC_RESULTS3_INTERNAL_GRAPH")
DEBUG_DIR = OUTPUT_ROOT / "debug_images"
EXCEL_PATH = OUTPUT_ROOT / "results3QSUB .xlsx"

# Parameters
MIN_CRYSTAL_AREA = 1000
CONTACT_DILATION = 5

# Save dpi for plots
PLOT_DPI = 160


# ======================================================
# HELPERS
# ======================================================

def _ensure_dirs():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def _safe_stem(pathlike):
    p = Path(str(pathlike))
    # make stem filesystem-friendly
    s = p.stem
    s = "".join([c if c.isalnum() or c in ("-", "_", ".") else "_" for c in s])
    return s


def show(title, img, save_path=None):
    """
    HPC version:
    - keeps same signature behavior
    - saves instead of showing
    """
    plt.figure(figsize=(7, 7))
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(str(save_path), dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()


def save_hist(title, xlabel, ylabel, data, bins=30, save_path=None):
    plt.figure(figsize=(6, 5))
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(str(save_path), dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()


def save_scatter(title, xlabel, ylabel, x, y, save_path=None):
    plt.figure(figsize=(6, 5))
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(str(save_path), dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()


def annotate_multiline(img, lines):
    out = img.copy()
    h = 30 + len(lines) * 30
    cv2.rectangle(out, (10, 10), (650, h), (0, 0, 0), -1)

    y = 35
    for line in lines:
        cv2.putText(
            out,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 28
    return out


def keep_largest_component(binary_mask):
    lab, n = ndi.label(binary_mask)
    if n == 0:
        return binary_mask
    sizes = ndi.sum(binary_mask, lab, index=np.arange(1, n + 1))
    largest = 1 + int(np.argmax(sizes))
    return lab == largest


# ======================================================
# MAIN DEBUG FUNCTION (NOW RETURNS METRICS + SAVES DEBUG)
# ======================================================

def analyze_particle_debug(image_path):
    """
    Returns a dict of metrics for Excel.
    Also writes all debug images into DEBUG_DIR.
    """

    img_stem = _safe_stem(image_path)

    def dbg(name):
        # consistent file naming
        return DEBUG_DIR / f"{img_stem}__{name}.png"

    metrics = {
        "image_path": str(image_path),
        "image_stem": img_stem,
        "MIN_CRYSTAL_AREA": MIN_CRYSTAL_AREA,
        "CONTACT_DILATION": CONTACT_DILATION,
    }

    # -------------------------
    # LOAD IMAGE
    # -------------------------
    img = cv2.imread(str(image_path))
    if img is None:
        metrics["status"] = "ERROR: Image not found"
        return metrics

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    show("Original", img_rgb, save_path=dbg("01_original"))

    # -------------------------
    # PARTICLE MASK
    # -------------------------
    _, particle0 = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    particle = keep_largest_component(particle0.astype(bool))
    particle = ndi.binary_fill_holes(particle)
    particle = ndi.binary_opening(particle, structure=np.ones((5, 5)))

    show("Particle Mask", particle, save_path=dbg("02_particle_mask"))

    # -------------------------
    # RED BOUNDARIES
    # -------------------------
    lower_red = np.array([150, 0, 0])
    upper_red = np.array([255, 110, 110])

    boundaries = cv2.inRange(img_rgb, lower_red, upper_red).astype(bool)

    particle_dil = ndi.binary_dilation(particle, iterations=2)
    boundaries = boundaries & particle_dil

    boundaries = ndi.binary_dilation(boundaries, iterations=3, structure=np.ones((5, 5)))
    boundaries = ndi.binary_erosion(boundaries, iterations=3, structure=np.ones((5, 5)))

    show("Red Boundaries", boundaries, save_path=dbg("03_red_boundaries"))

    # -------------------------
    # CRYSTAL SEGMENTATION
    # -------------------------
    interior = particle & (~boundaries)
    interior = ndi.binary_opening(interior, structure=np.ones((3, 3)))

    labels, num = ndi.label(interior)

    keep = np.zeros_like(labels, dtype=bool)

    for lab in range(1, num + 1):
        if np.sum(labels == lab) >= MIN_CRYSTAL_AREA:
            keep |= (labels == lab)

    labels, num = ndi.label(keep)

    show(f"Detected Crystals (N={num})", labels, save_path=dbg("04_detected_crystals_labels"))

    metrics["num_detected_crystals"] = int(num)

    # ======================================================
    # THICKNESS ANALYSIS (GEOMETRIC HETEROGENEITY)
    # ======================================================
    distance_map = ndi.distance_transform_edt(interior)
    thickness_values = distance_map[interior] * 2  # radius -> diameter (pixels)

    if len(thickness_values) > 10:
        skew = pd.Series(thickness_values).skew()
        kurt = pd.Series(thickness_values).kurt()
        Thickness_bimodality = (skew**2 + 1) / (kurt + 3)
    else:
        Thickness_bimodality = 0

    metrics["Thickness_bimodality"] = float(Thickness_bimodality)
    metrics["Thickness_n"] = int(len(thickness_values))
    metrics["Thickness_mean_px"] = float(np.mean(thickness_values)) if len(thickness_values) else 0.0
    metrics["Thickness_std_px"] = float(np.std(thickness_values, ddof=1)) if len(thickness_values) > 1 else 0.0

    save_hist(
        title=f"Thickness Distribution\nBimodality = {Thickness_bimodality:.3f}",
        xlabel="Local Thickness (pixels)",
        ylabel="Frequency",
        data=thickness_values,
        bins=30,
        save_path=dbg("05_thickness_hist"),
    )

    # -------------------------
    # BUILD DATAFRAME
    # -------------------------
    records = []

    for lab in range(1, num + 1):
        region = labels == lab
        ys, xs = np.where(region)

        records.append(
            {
                "Label": lab,
                "Centroid_x": xs.mean() if len(xs) else 0.0,
                "Centroid_y": ys.mean() if len(ys) else 0.0,
            }
        )

    df = pd.DataFrame(records)

    metrics["Detected_crystals_df_len"] = int(len(df))

    # ======================================================
    # SPATIAL METRICS (VORONOI AREA CV, LOCAL DENSITY GRADIENT, RADIAL ANISOTROPY CV)
    # ======================================================

    if len(df) >= 2:

        # ---- centroids array (x,y) ----
        cent_xy = df[["Centroid_x", "Centroid_y"]].to_numpy(dtype=float)
        labs_arr = df["Label"].to_numpy(dtype=int)

        # ---- particle centroid (x,y) for radial normalization ----
        py, px = np.where(particle)
        part_cx = float(px.mean()) if len(px) else 0.0
        part_cy = float(py.mean()) if len(py) else 0.0

        # ======================================================
        # (A) Voronoi Area CV (Raster Voronoi within particle mask)
        # ======================================================
        tree = cKDTree(cent_xy)

        part_coords = np.column_stack([px.astype(float), py.astype(float)])

        nn_idx = tree.query(part_coords, k=1)[1]
        nn_labels = labs_arr[nn_idx]

        vor_map = np.zeros_like(labels, dtype=np.int32)
        vor_map[py, px] = nn_labels

        vor_areas = np.array([np.sum(vor_map == lab) for lab in labs_arr], dtype=float)
        vor_mean = float(vor_areas.mean()) if len(vor_areas) else 0.0
        Voronoi_area_CV = float(vor_areas.std(ddof=1) / vor_mean) if (len(vor_areas) > 1 and vor_mean > 0) else 0.0

        metrics["Voronoi_area_CV"] = float(Voronoi_area_CV)
        metrics["Voronoi_area_mean_px2"] = float(vor_mean)
        metrics["Voronoi_area_std_px2"] = float(vor_areas.std(ddof=1)) if len(vor_areas) > 1 else 0.0

        # ---- Voronoi debug image ----
        rng_v = np.random.default_rng(202)
        color_lut = {int(lab): rng_v.integers(50, 255, size=3, dtype=np.uint8) for lab in labs_arr}

        vor_rgb = img_rgb.copy()
        mask = vor_map > 0
        tmp = np.zeros_like(vor_rgb, dtype=np.uint8)
        for lab in labs_arr:
            tmp[vor_map == int(lab)] = color_lut[int(lab)]

        vor_rgb[mask] = (0.55 * vor_rgb[mask] + 0.45 * tmp[mask]).astype(np.uint8)

        vor_rgb = annotate_multiline(
            vor_rgb,
            [
                f"Voronoi_area_CV = {Voronoi_area_CV:.3f}",
                "Voronoi regions inside particle mask",
            ],
        )
        show("Voronoi Regions Debug", vor_rgb, save_path=dbg("06_voronoi_regions"))

        # ======================================================
        # (B) Local Density Gradient (density vs radius slope)
        # ======================================================
        k = int(min(7, len(cent_xy)))  # includes self
        dists, idxs = tree.query(cent_xy, k=k)

        nn_d = dists[:, 1:] if k > 1 else np.zeros((len(cent_xy), 0))
        mean_nn_d = np.mean(nn_d, axis=1) if nn_d.shape[1] > 0 else np.zeros(len(cent_xy))

        eps = 1e-9
        local_density = 1.0 / (mean_nn_d**2 + eps)

        r = np.sqrt((cent_xy[:, 0] - part_cx) ** 2 + (cent_xy[:, 1] - part_cy) ** 2)
        r_max = float(r.max()) if len(r) else 1.0
        r_norm = r / (r_max + eps)

        if len(r_norm) >= 2 and np.std(r_norm) > 0:
            Local_density_gradient = float(np.polyfit(r_norm, local_density, 1)[0])
        else:
            Local_density_gradient = 0.0

        metrics["Local_density_gradient"] = float(Local_density_gradient)
        metrics["Local_density_mean"] = float(np.mean(local_density)) if len(local_density) else 0.0
        metrics["Local_density_std"] = float(np.std(local_density, ddof=1)) if len(local_density) > 1 else 0.0

        save_scatter(
            title=f"Local Density vs Radius\nGradient = {Local_density_gradient:.4f}",
            xlabel="Normalized radius (0=center, 1=edge)",
            ylabel="Local density proxy (1/mean_kNN^2)",
            x=r_norm,
            y=local_density,
            save_path=dbg("07_local_density_vs_radius"),
        )

        # ======================================================
        # (C) Radial Anisotropy CV (directional bias of neighbor vectors)
        # ======================================================
        anis = np.zeros(len(cent_xy), dtype=float)

        for i in range(len(cent_xy)):
            neigh = idxs[i, 1:]  # exclude self
            if len(neigh) == 0:
                anis[i] = 0.0
                continue

            vec = cent_xy[neigh] - cent_xy[i]
            ang = np.arctan2(vec[:, 1], vec[:, 0])  # y, x
            ux = np.mean(np.cos(ang))
            uy = np.mean(np.sin(ang))
            R = float(np.sqrt(ux * ux + uy * uy))  # 1 aligned, 0 isotropic
            anis[i] = R

        anis_mean = float(anis.mean()) if len(anis) else 0.0
        Radial_anisotropy_CV = float(anis.std(ddof=1) / anis_mean) if (len(anis) > 1 and anis_mean > 0) else 0.0

        metrics["Radial_anisotropy_CV"] = float(Radial_anisotropy_CV)
        metrics["Radial_anisotropy_meanR"] = float(anis_mean)
        metrics["Radial_anisotropy_stdR"] = float(anis.std(ddof=1)) if len(anis) > 1 else 0.0

        a_min, a_max = float(anis.min()), float(anis.max())
        denom = (a_max - a_min) if (a_max > a_min) else 1.0

        anis_overlay = img_rgb.copy()
        for i, lab in enumerate(labs_arr):
            cx, cy = int(cent_xy[i, 0]), int(cent_xy[i, 1])
            t = (anis[i] - a_min) / denom  # 0..1
            col = (int(255 * (1 - t)), int(0), int(255 * t))
            cv2.circle(anis_overlay, (cx, cy), 8, col, -1)

        anis_overlay = annotate_multiline(
            anis_overlay,
            [
                f"Radial_anisotropy_CV = {Radial_anisotropy_CV:.3f}",
                f"Anisotropy range (R): {a_min:.2f}–{a_max:.2f}",
                "Node color encodes anisotropy R",
            ],
        )
        show("Radial Anisotropy Debug", anis_overlay, save_path=dbg("08_radial_anisotropy"))

    else:
        metrics["Voronoi_area_CV"] = 0.0
        metrics["Local_density_gradient"] = 0.0
        metrics["Radial_anisotropy_CV"] = 0.0

    # ======================================================
    # CONTACT NETWORK (PHYSICALLY CORRECT)
    # ======================================================

    G = nx.Graph()

    for lab in df["Label"]:
        G.add_node(int(lab))

    # ---- safer dilation (no overwrite issue) ----
    labels_dil = np.zeros_like(labels)

    for lab in df["Label"]:
        reg = dilation(labels == int(lab), disk(CONTACT_DILATION))
        labels_dil[reg & (labels_dil == 0)] = int(lab)

    # ---- detect contacts along RED boundaries ----
    by, bx = np.where(boundaries)

    for y, x in zip(by, bx):

        y0, y1 = max(0, y - 1), min(labels.shape[0], y + 2)
        x0, x1 = max(0, x - 1), min(labels.shape[1], x + 2)

        neigh = labels_dil[y0:y1, x0:x1]
        touching = np.unique(neigh[neigh > 0])

        if len(touching) >= 2:
            for i in range(len(touching)):
                for j in range(i + 1, len(touching)):
                    G.add_edge(int(touching[i]), int(touching[j]))

    # -------------------------
    # GRAPH METRICS
    # -------------------------
    degrees = np.array([d for _, d in G.degree()])

    Coord_mean = degrees.mean() if len(degrees) > 0 else 0
    Coord_var = degrees.var() if len(degrees) > 0 else 0
    Cycle_count = len(nx.cycle_basis(G)) if len(G.nodes) > 0 else 0
    Graph_density = nx.density(G) if len(G.nodes) > 1 else 0

    metrics["Coord_mean"] = float(Coord_mean)
    metrics["Coord_var"] = float(Coord_var)
    metrics["Cycle_count"] = int(Cycle_count)
    metrics["Graph_density"] = float(Graph_density)
    metrics["Graph_nodes"] = int(G.number_of_nodes())
    metrics["Graph_edges"] = int(G.number_of_edges())

    # ===== CONNECTIVITY METRICS =====
    if len(G.nodes) > 0:
        Num_components = nx.number_connected_components(G)
        components = list(nx.connected_components(G))
        largest_cc = max(components, key=len)
        LCC_fraction = len(largest_cc) / len(G.nodes)
    else:
        Num_components = 0
        components = []
        largest_cc = []
        LCC_fraction = 0

    metrics["Connected_components"] = int(Num_components)
    metrics["Largest_CC_fraction"] = float(LCC_fraction)
    metrics["Largest_CC_size"] = int(len(largest_cc)) if isinstance(largest_cc, (set, list, tuple)) else 0

    # ======================================================
    # COMMUNITY STRUCTURE METRICS
    # ======================================================
    if len(G.nodes) > 1 and G.number_of_edges() > 0:
        communities = list(community.greedy_modularity_communities(G))
        Community_count = len(communities)
        Modularity = community.modularity(G, communities)
    else:
        communities = []
        Community_count = 0
        Modularity = 0

    metrics["Community_count"] = int(Community_count)
    metrics["Modularity"] = float(Modularity)

    # -------------------------
    # GRAPH VISUAL
    # -------------------------
    overlay = img_rgb.copy()

    centroids = {}
    for _, row in df.iterrows():
        centroids[int(row["Label"])] = (int(row["Centroid_x"]), int(row["Centroid_y"]))

    # draw edges
    for (i, j) in G.edges():
        cv2.line(overlay, centroids[i], centroids[j], (255, 255, 255), 1)

    # draw nodes + degree
    for n in G.nodes():
        cx, cy = centroids[n]
        deg = G.degree[n]

        cv2.circle(overlay, (cx, cy), 6, (0, 255, 0), -1)
        cv2.putText(
            overlay,
            str(deg),
            (cx + 6, cy - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

    overlay = annotate_multiline(
        overlay,
        [
            f"Coordination mean = {Coord_mean:.2f}",
            f"Coordination variance = {Coord_var:.2f}",
            f"Cycle count = {Cycle_count}",
            f"Graph density = {Graph_density:.3f}",
            f"Connected components = {Num_components}",
            f"Largest CC fraction = {LCC_fraction:.2f}",
            f"Community count = {Community_count}",
            f"Modularity = {Modularity:.3f}",
            f"Thickness bimodality = {metrics['Thickness_bimodality']:.3f}",
            f"Voronoi_area_CV = {metrics.get('Voronoi_area_CV', 0.0):.3f}",
            f"Local_density_gradient = {metrics.get('Local_density_gradient', 0.0):.4f}",
            f"Radial_anisotropy_CV = {metrics.get('Radial_anisotropy_CV', 0.0):.3f}",
            "Number near node = degree",
        ],
    )

    show("Graph Metrics", overlay, save_path=dbg("09_graph_metrics_overlay"))

    # -------------------------
    # LARGEST CC DEBUG IMAGE
    # -------------------------
    if len(largest_cc) > 0:

        lcc_overlay = img_rgb.copy()

        for n in G.nodes():
            cx, cy = centroids[n]
            if n in largest_cc:
                color = (0, 255, 0)      # green = in largest CC
            else:
                color = (255, 0, 0)      # red = fragmented node
            cv2.circle(lcc_overlay, (cx, cy), 8, color, -1)

        lcc_overlay = annotate_multiline(
            lcc_overlay,
            [
                f"Largest CC fraction = {LCC_fraction:.2f}",
                f"Connected components = {Num_components}",
                "Green = largest component",
                "Red = disconnected crystals",
            ],
        )

        show("Largest Connected Component Debug", lcc_overlay, save_path=dbg("10_largest_cc_debug"))

    # -------------------------
    # FULL CONNECTED COMPONENTS DEBUG IMAGE
    # -------------------------
    if Num_components > 0:

        cc_overlay = img_rgb.copy()

        rng = np.random.default_rng(42)
        colors = rng.integers(50, 255, size=(Num_components, 3))

        for comp_id, comp in enumerate(components):
            color = tuple(int(c) for c in colors[comp_id])
            for n in comp:
                cx, cy = centroids[n]
                cv2.circle(cc_overlay, (cx, cy), 8, color, -1)

        comp_sizes = [len(c) for c in components]

        cc_overlay = annotate_multiline(
            cc_overlay,
            [
                f"Connected components = {Num_components}",
                f"Component sizes = {comp_sizes}",
                "Each color = one connected component",
            ],
        )

        show("Connected Components Debug", cc_overlay, save_path=dbg("11_connected_components_debug"))

    # -------------------------
    # COMMUNITY DEBUG IMAGE
    # -------------------------
    if Community_count > 0:

        comm_overlay = img_rgb.copy()

        rng = np.random.default_rng(123)
        colors = rng.integers(50, 255, size=(Community_count, 3))

        for cid, comm in enumerate(communities):
            color = tuple(int(c) for c in colors[cid])
            for n in comm:
                cx, cy = centroids[int(n)]
                cv2.circle(comm_overlay, (cx, cy), 8, color, -1)

        comm_overlay = annotate_multiline(
            comm_overlay,
            [
                f"Community count = {Community_count}",
                f"Modularity = {Modularity:.3f}",
                "Each color = one community",
            ],
        )

        show("Community Structure Debug", comm_overlay, save_path=dbg("12_community_debug"))

    metrics["status"] = "OK"
    return metrics


# ======================================================
# RUN (BATCH + EXCEL + DEBUG FOLDER)
# ======================================================

def main():

    _ensure_dirs()

    # Create debug root
    debug_root = OUTPUT_ROOT / "debug_images"
    debug_root.mkdir(parents=True, exist_ok=True)

    # Find all subfolders (days)
    day_folders = sorted([p for p in INPUT_DIR.iterdir() if p.is_dir()])

    if len(day_folders) == 0:
        raise RuntimeError("No subfolders found inside INPUT_DIR")

    print(f"Found {len(day_folders)} day folders")

    # Open Excel writer ONCE
    with pd.ExcelWriter(str(EXCEL_PATH), engine="openpyxl") as writer:

        for day_path in day_folders:

            day_name = day_path.name
            print(f"\nProcessing folder: {day_name}")

            # Create debug folder for this day
            global DEBUG_DIR
            DEBUG_DIR = debug_root / day_name
            DEBUG_DIR.mkdir(parents=True, exist_ok=True)

            images = sorted(day_path.glob(IMAGE_GLOB))

            if len(images) == 0:
                print(f"⚠ No images in {day_name}")
                continue

            all_metrics = []

            for p in images:
                try:
                    m = analyze_particle_debug(p)
                except Exception as e:
                    m = {
                        "image_path": str(p),
                        "image_stem": _safe_stem(p),
                        "MIN_CRYSTAL_AREA": MIN_CRYSTAL_AREA,
                        "CONTACT_DILATION": CONTACT_DILATION,
                        "status": f"ERROR: {repr(e)}",
                    }
                all_metrics.append(m)

            df_day = pd.DataFrame(all_metrics)

            # Excel sheet name max length = 31
            sheet_name = day_name[:31]

            df_day.to_excel(writer, index=False, sheet_name=sheet_name)

            print(f"Saved sheet: {sheet_name} ({len(images)} images)")

    print("\nAll folders processed successfully.")
    print(f"Excel saved at: {EXCEL_PATH}")

if __name__ == "__main__":
    main()