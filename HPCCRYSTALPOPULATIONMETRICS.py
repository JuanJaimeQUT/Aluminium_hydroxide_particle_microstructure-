# ======================================================
# HPC CRYSTAL POPULATION + SIZE HIERARCHY
# PAPER 2 — CRYSTAL POPULATION METRICS
# ======================================================

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from scipy import ndimage as ndi


# ======================================================
# PARAMETERS
# ======================================================

PIXEL_SIZE_UM = 0.18
MIN_CRYSTAL_AREA = 1000
SAVE_DEBUG = True


# ======================================================
# DEBUG SAVE
# ======================================================

def save_debug(path, img, cmap=None):

    plt.figure(figsize=(6,6))

    if img.dtype == bool:
        plt.imshow(img, cmap="gray")
    elif len(img.shape)==2:
        plt.imshow(img, cmap=cmap if cmap else "gray")
    else:
        plt.imshow(img)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ======================================================
# GINI COEFFICIENT
# ======================================================

def gini(x):

    x = np.sort(x)
    n = len(x)

    return (2*np.arange(1,n+1)-n-1).dot(x)/(n*x.sum())


# ======================================================
# CORE ANALYSIS
# ======================================================

def analyze_particle(image_path, debug_dir):

    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    stem = image_path.stem


    # ======================================================
    # PARTICLE SEGMENTATION
    # ======================================================

    _, particle = cv2.threshold(gray,20,255,cv2.THRESH_BINARY)

    particle = particle.astype(bool)
    particle = ndi.binary_fill_holes(particle)

    labeled_particle, num = ndi.label(particle)

    if num>1:

        sizes = ndi.sum(particle,labeled_particle,
                        index=np.arange(1,num+1))

        particle = labeled_particle == (1 + np.argmax(sizes))


    if SAVE_DEBUG:
        save_debug(debug_dir / f"{stem}_particle.png", particle)


    # ======================================================
    # RED CRYSTAL BOUNDARIES
    # ======================================================

    lower_red = np.array([150,0,0])
    upper_red = np.array([255,100,100])

    boundaries = cv2.inRange(img_rgb,lower_red,upper_red).astype(bool)

    particle_dil = ndi.binary_dilation(particle,iterations=2)
    boundaries = boundaries & particle_dil

    boundaries = ndi.binary_dilation(boundaries,iterations=3,
                                     structure=np.ones((5,5)))

    boundaries = ndi.binary_erosion(boundaries,iterations=3,
                                    structure=np.ones((5,5)))


    if SAVE_DEBUG:
        save_debug(debug_dir / f"{stem}_boundaries.png", boundaries)


    # ======================================================
    # INTERNAL CRYSTALS
    # ======================================================

    interior = particle & (~boundaries)
    interior = ndi.binary_opening(interior,structure=np.ones((3,3)))

    labels, num = ndi.label(interior)


    for lab in range(1,num+1):

        if np.sum(labels==lab) < MIN_CRYSTAL_AREA:
            labels[labels==lab] = 0

    labels,_ = ndi.label(labels>0)


    if SAVE_DEBUG:
        save_debug(debug_dir / f"{stem}_crystals.png", labels, cmap="tab20")


    # ======================================================
    # CRYSTAL MEASUREMENTS
    # ======================================================

    records = []

    for lab in np.unique(labels):

        if lab==0:
            continue

        region = labels==lab
        area_px = region.sum()

        ys,xs = np.where(region)

        records.append({

            "Label":lab,
            "Area_px":area_px,
            "Area_um2":area_px * PIXEL_SIZE_UM**2,
            "Centroid_x":xs.mean(),
            "Centroid_y":ys.mean()

        })

    df = pd.DataFrame(records)


    # ======================================================
    # PARTICLE SIZE
    # ======================================================

    particle_area_px = particle.sum()

    Particle_ECD_um = np.sqrt(
        4*(particle_area_px*PIXEL_SIZE_UM**2)/np.pi
    )


    # ======================================================
    # MAIN CRYSTAL
    # ======================================================

    main_idx = df["Area_px"].idxmax()

    main_label = df.loc[main_idx,"Label"]

    main_region = labels==main_label

    A_main = df.loc[main_idx,"Area_px"]

    contours,_ = cv2.findContours(
        main_region.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    P = cv2.arcLength(contours[0],True)

    Main_circularity = 4*np.pi*A_main/(P**2)

    hull = cv2.convexHull(contours[0])
    hull_area = cv2.contourArea(hull)

    Main_solidity = A_main/hull_area


    Main_ECD_um = np.sqrt(
        4*(A_main*PIXEL_SIZE_UM**2)/np.pi
    )

        # particle centroid
    py, px = np.where(particle)
    particle_cx = px.mean()
    particle_cy = py.mean()

    # main crystal centroid
    my, mx = np.where(main_region)
    main_cx = mx.mean()
    main_cy = my.mean()

    # centroid distance
    main_offset_px = np.sqrt((main_cx - particle_cx)**2 + (main_cy - particle_cy)**2)

    # equivalent particle radius
    particle_area_px = np.sum(particle)
    particle_radius_px = np.sqrt(particle_area_px / np.pi)

    Main_crystal_centrality = main_offset_px / particle_radius_px

    # ======================================================
    # CORE / RIM FRACTION OF MAIN CRYSTAL
    # ======================================================

    dist_to_edge = ndi.distance_transform_edt(particle)

    dist_norm = dist_to_edge / dist_to_edge.max()

    core_mask = (dist_norm > 0.5) & particle
    rim_mask  = (dist_norm < 0.2) & particle

    Main_core_fraction = np.sum(main_region & core_mask) / np.sum(main_region) #how much of the main crystal lies in the core region core occupancy of main unit
    Main_rim_fraction  = np.sum(main_region & rim_mask) / np.sum(main_region)
    Main_core_fraction_particle = np.sum(main_region & core_mask) / np.sum(particle) #core fraction within main unit
    # ======================================================
# DEBUG MAIN CRYSTAL POSITION
# ======================================================

    if SAVE_DEBUG:

        overlay = img_rgb.copy()

        # main crystal
        overlay[main_region] = [255,0,0]

        # particle centroid
        cv2.circle(
            overlay,
            (int(particle_cx), int(particle_cy)),
            6,
            (0,255,0),
            -1
        )

        # main crystal centroid
        cv2.circle(
            overlay,
            (int(main_cx), int(main_cy)),
            6,
            (0,0,255),
            -1
        )

        save_debug(
            debug_dir / f"{stem}_main_crystal_position.png",
            overlay
        )
            # ======================================================
    # POPULATION METRICS
    # ======================================================

    total_area = df["Area_px"].sum()

    N_crystals = len(df)
    N_satellites = N_crystals-1

    Main_fraction = A_main/total_area
    Satellite_fraction = 1-Main_fraction


    # ======================================================
    # CRYSTAL PSD
    # ======================================================

    df["ECD_um"] = np.sqrt(
        4*df["Area_um2"]/np.pi
    )

    d = df["ECD_um"].values
    N_tot = len(d)


    bins = {

        "lt5": d<5,
        "5_10":(d>=5)&(d<10),
        "10_22":(d>=10)&(d<22),
        "22_44":(d>=22)&(d<44),
        "44_88":(d>=44)&(d<88),
        "88_150":(d>=88)&(d<150),
        "gt150":d>=150

    }

    bin_counts = {k:int(np.sum(v)) for k,v in bins.items()}

    bin_fractions = {

        f"Frac_{k}":bin_counts[k]/N_tot
        for k in bin_counts

    }

    area_bins = {
        "Area_lt5": df.loc[d < 5, "Area_um2"].sum(),
        "Area_5_10": df.loc[(d >= 5) & (d < 10), "Area_um2"].sum(),
        "Area_10_22": df.loc[(d >= 10) & (d < 22), "Area_um2"].sum(),
        "Area_22_44": df.loc[(d >= 22) & (d < 44), "Area_um2"].sum(),
        "Area_44_88": df.loc[(d >= 44) & (d < 88), "Area_um2"].sum(),
        "Area_88_150": df.loc[(d >= 88) & (d < 150), "Area_um2"].sum(),
        "Area_gt150": df.loc[d >= 150, "Area_um2"].sum()
    }

    total_area = df["Area_um2"].sum()

    area_fractions = {
        f"FracArea_{k}": v / total_area
        for k, v in area_bins.items()
    }

        # ======================================================
    # SATELLITE SIZE METRICS (NEW)
    # ======================================================

    # remove the main crystal
    satellites = df[df["Label"] != main_label]

    sat_sizes = satellites["ECD_um"].values

    if len(sat_sizes) > 0:

        Satellite_mean_ECD = np.mean(sat_sizes)
        Satellite_median_ECD = np.median(sat_sizes)

    else:

        Satellite_mean_ECD = np.nan
        Satellite_median_ECD = np.nan


    # dominant satellite size bin (by count)

    dominant_bin = max(bin_counts, key=bin_counts.get)

    bin_center = {

        "lt5": 2.5,
        "5_10": 7.5,
        "10_22": 16,
        "22_44": 33,
        "44_88": 66,
        "88_150": 120,
        "gt150": 175

    }

    Dominant_satellite_size = bin_center[dominant_bin]

    # ======================================================
# SATELLITE TO MAIN SIZE RATIO
# ======================================================

    Satellite_to_main_ratio = Satellite_median_ECD / Main_ECD_um if Main_ECD_um > 0 else np.nan

    # ======================================================
# AGGLOMERATION DOMINANCE INDEX
# ======================================================

    satellite_area = total_area - A_main

    Agglomeration_dominance_index = A_main / satellite_area if satellite_area > 0 else np.nan

    # ======================================================
# CRYSTAL SIZE HETEROGENEITY INDEX
# ======================================================

    Crystal_size_std = np.std(df["ECD_um"].values)
    Crystal_size_mean = np.mean(df["ECD_um"].values)

    Crystal_size_heterogeneity = (
        Crystal_size_std / Crystal_size_mean
        if Crystal_size_mean > 0 else np.nan
    )
    # ======================================================
    # GINI
    # ======================================================

    Gini_area = gini(df["Area_px"].values)


    # ======================================================
    # FINAL RESULTS
    # ======================================================

    results = {

        "Particle_ECD_um":Particle_ECD_um,

        "N_crystals":N_crystals,
        "N_satellites":N_satellites,

        "Main_fraction":Main_fraction,
        "Satellite_fraction":Satellite_fraction,

        "Main_ECD_um":Main_ECD_um,
        "Main_solidity":Main_solidity,
        "Main_circularity":Main_circularity,

        "Gini_area":Gini_area,

        "Main_crystal_centrality": Main_crystal_centrality,
        "Main_core_fraction": Main_core_fraction,
        "Main_rim_fraction": Main_rim_fraction,

        # NEW SATELLITE SIZE METRICS
        "Satellite_mean_ECD": Satellite_mean_ECD,
        "Satellite_median_ECD": Satellite_median_ECD,
        "Dominant_satellite_size": Dominant_satellite_size,

                # satellite structure metrics
        "Satellite_to_main_ratio": Satellite_to_main_ratio,
        "Agglomeration_dominance_index": Agglomeration_dominance_index,

        "Crystal_size_heterogeneity": Crystal_size_heterogeneity,

        **bin_counts,
        **bin_fractions,
        **area_bins,
        **area_fractions

    }

    return pd.DataFrame([results])

# ======================================================
# BATCH PROCESSING
# ======================================================

INPUT_DIR = Path("/mnt/hpccs01/home/n11555581/Code/PARTIALRESULTS4")

OUTPUT_DIR = Path("/mnt/hpccs01/home/n11555581/Code/OUTPUT_CRYSTAL_POPULATION_RESULTS4")

OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_EXCEL = OUTPUT_DIR / "HPC_crystal_population_metrics_RESULTS4.xlsx"


with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:

    for folder in sorted([f for f in INPUT_DIR.iterdir() if f.is_dir()]):

        print(f"\nProcessing folder: {folder.name}")

        # debug images go into OUTPUT folder, not input
        debug_dir = OUTPUT_DIR / f"_DEBUG_{folder.name}"
        debug_dir.mkdir(exist_ok=True)

        results = []

        images = sorted(
            [p for p in folder.iterdir()
             if p.suffix.lower() in [".png",".jpg",".jpeg",".bmp",".tif"]]
        )

        for img_path in images:

            try:

                df_particle = analyze_particle(img_path, debug_dir)

                df_particle["Day"] = folder.name
                df_particle["Image"] = img_path.name

                results.append(df_particle)

            except Exception as e:

                print("Skipped", img_path.name, e)

        if results:

            day_df = pd.concat(results, ignore_index=True)

            day_df.to_excel(
                writer,
                sheet_name=folder.name[:31],
                index=False
            )

            print(f"✔ Saved {len(day_df)} particles")

print("\nDONE")
print("Results saved to:", OUTPUT_EXCEL)