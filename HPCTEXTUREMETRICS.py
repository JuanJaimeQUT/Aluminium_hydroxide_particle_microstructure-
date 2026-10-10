#HPCTEXTUREMETRICS

# ======================================================
# SEGMENT ANYTHING — TEXTURE & SPATIAL STRUCTURE
# ======================================================

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from scipy import ndimage as ndi
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from skimage.feature import graycomatrix, graycoprops


# ======================================================
# PARAMETERS
# ======================================================

MIN_CRYSTAL_AREA = 1000
PIXEL_SIZE_UM = 0.18
SAVE_DEBUG = True


# ======================================================
# DEBUG SAVE FUNCTION
# ======================================================

def save_debug(path, img, cmap=None):

    plt.figure(figsize=(6,6))

    if img.dtype == bool:
        plt.imshow(img, cmap="gray")

    elif len(img.shape) == 2:
        plt.imshow(img, cmap=cmap if cmap else "gray")

    else:
        plt.imshow(img)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ======================================================
# FRACTAL DIMENSION
# ======================================================

def fractal_dimension(Z):

    Z = Z.astype(bool)

    def boxcount(Z, k):

        S = np.add.reduceat(
            np.add.reduceat(Z,
                            np.arange(0, Z.shape[0], k),
                            axis=0),
                            np.arange(0, Z.shape[1], k),
                            axis=1)

        return len(np.where(S > 0)[0])

    sizes = 2**np.arange(1,6)

    counts = []

    for size in sizes:

        if size < min(Z.shape):
            counts.append(boxcount(Z,size))

    if len(counts) > 1:

        coeffs = np.polyfit(
            np.log(sizes[:len(counts)]),
            np.log(counts),
            1
        )

        return -coeffs[0]

    return np.nan


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

    labeled_particle, num_part = ndi.label(particle)

    if num_part > 1:

        sizes = ndi.sum(particle,
                        labeled_particle,
                        index=np.arange(1,num_part+1))

        particle = labeled_particle == (1 + np.argmax(sizes))

    if SAVE_DEBUG:
        save_debug(debug_dir / f"{stem}_01_particle.png", particle)

    # ======================================================
    # RED BOUNDARIES
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
        save_debug(debug_dir / f"{stem}_02_boundaries.png", boundaries)

    # ======================================================
    # INTERIOR CRYSTAL REGIONS
    # ======================================================

    interior = particle & (~boundaries)
    interior = ndi.binary_opening(interior, structure=np.ones((3,3)))

    labels, num = ndi.label(interior)

    for lab in range(1,num+1):
        if np.sum(labels == lab) < MIN_CRYSTAL_AREA:
            labels[labels == lab] = 0

    labels,_ = ndi.label(labels > 0)

    if SAVE_DEBUG:
        save_debug(debug_dir / f"{stem}_03_labels.png", labels, cmap="tab20")

    # ======================================================
    # SAFE INTRACRYSTALLINE MASK
    # ======================================================

    interior_mask = particle & (~boundaries)
    interior_mask = ndi.binary_erosion(interior_mask, iterations=3)

    if np.sum(interior_mask) < 50:
        interior_mask = particle & (~boundaries)

    if np.sum(interior_mask) < 50:
        interior_mask = particle

    if SAVE_DEBUG:
        save_debug(debug_dir / f"{stem}_04_interior_mask.png", interior_mask)

    # ======================================================
    # LOCAL TEXTURE VARIANCE
    # ======================================================

    local_mean = cv2.blur(gray.astype(np.float64),(7,7))
    local_sqmean = cv2.blur((gray.astype(np.float64))**2,(7,7))
    local_var = local_sqmean - local_mean**2

    Internal_texture_variance = np.mean(local_var[interior_mask])

    if SAVE_DEBUG:

        tex = np.zeros_like(gray,dtype=float)
        tex[interior_mask] = local_var[interior_mask]

        save_debug(debug_dir / f"{stem}_texture_variance.png",
                   tex, cmap="inferno")

    # ======================================================
    # LAPLACIAN ENERGY (intracrystalline only)
    # ======================================================

    lap = cv2.Laplacian(gray,cv2.CV_64F,ksize=3)

    Laplacian_texture_energy = np.mean((lap[interior_mask])**2)

    if SAVE_DEBUG:

        lap_map = np.zeros_like(gray)
        lap_map[interior_mask] = np.abs(lap[interior_mask])

        save_debug(debug_dir / f"{stem}_laplacian_texture.png",
                   lap_map, cmap="magma")

# ======================================================
# CRYSTAL FACE SMOOTHNESS INDEX (CFSI)
# ======================================================

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, 3)

    grad_mag = np.sqrt(gx**2 + gy**2)

    # avoid division by zero
    norm_intensity = gray.astype(float) + 1e-6

    CFSI_map = grad_mag / norm_intensity

    Crystal_face_smoothness_index = np.mean(CFSI_map[interior_mask])


    if SAVE_DEBUG:

        cfsi_debug = np.zeros_like(gray, dtype=float)

        cfsi_debug[interior_mask] = CFSI_map[interior_mask]

        save_debug(
            debug_dir / f"{stem}_crystal_face_smoothness.png",
            cfsi_debug,
            cmap="magma"
        )

    # ======================================================
    # GLCM TEXTURE METRICS (intracrystalline)
    # ======================================================

    # normalize grayscale to 8-bit levels
    gray_quant = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray_quant = gray_quant.astype(np.uint8)

    # mask outside interior
    texture_img = np.zeros_like(gray_quant)
    texture_img[interior_mask] = gray_quant[interior_mask]

    # compute GLCM
    glcm = graycomatrix(
        texture_img,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True,
        normed=True
    )

    # texture properties
    GLCM_contrast = graycoprops(glcm, 'contrast').mean()
    GLCM_homogeneity = graycoprops(glcm, 'homogeneity').mean()
    GLCM_energy = graycoprops(glcm, 'energy').mean()

    # entropy
    glcm_prob = glcm / glcm.sum()
    GLCM_entropy = -np.sum(glcm_prob * np.log(glcm_prob + 1e-12))


    if SAVE_DEBUG:

        save_debug(
            debug_dir / f"{stem}_glcm_texture_region.png",
            texture_img,
            cmap="gray"
        )

    # ======================================================
    # FRACTAL TEXTURE (intracrystalline rough patches)
    # ======================================================

    var_thresh = np.percentile(local_var[interior_mask],60)

    texture_mask = (local_var > var_thresh) & interior_mask
    texture_mask = ndi.binary_opening(texture_mask, structure=np.ones((3,3)))

    Fractal_dimension_texture = fractal_dimension(texture_mask)

    if SAVE_DEBUG:
        save_debug(debug_dir / f"{stem}_texture_mask.png", texture_mask)

    # ======================================================
    # CURVATURE ENERGY
    # ======================================================

    cnts,_ = cv2.findContours(particle.astype(np.uint8),
                              cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_NONE)

    outer = max(cnts,key=cv2.contourArea).squeeze()

    x,y = outer[:,0],outer[:,1]

    def curvature(x,y):

        dx = np.gradient(x)
        dy = np.gradient(y)

        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        return np.abs(dx*ddy - dy*ddx)/(dx*dx+dy*dy)**1.5

    scales = [1,3,6,12]
    curv_energy=[]

    for s in scales:

        xs = gaussian_filter1d(x,s)
        ys = gaussian_filter1d(y,s)

        k = curvature(xs,ys)
        curv_energy.append(np.nanmean(k**2))

    Curvature_energy_decay = curv_energy[-1]/curv_energy[0]

    if SAVE_DEBUG:

        plt.figure()
        plt.plot(scales,curv_energy,"o-")
        plt.xlabel("scale")
        plt.ylabel("curvature energy")

        plt.savefig(debug_dir / f"{stem}_curvature_decay.png")
        plt.close()

    # ======================================================
    # OUTER TO INNER ROUGHNESS
    # ======================================================

    gx = cv2.Sobel(gray,cv2.CV_64F,1,0,3)
    gy = cv2.Sobel(gray,cv2.CV_64F,0,1,3)

    inner_energy = np.mean((gx[interior_mask]**2 + gy[interior_mask]**2))
    outer_energy = curv_energy[0]

    Outer_to_inner_roughness = outer_energy / inner_energy if inner_energy > 0 else np.nan

    if SAVE_DEBUG:

        rough_map = np.zeros_like(gray)
        rough_map[interior_mask] = np.sqrt(gx[interior_mask]**2 + gy[interior_mask]**2)

        save_debug(debug_dir / f"{stem}_inner_gradient_energy.png",
                   rough_map, cmap="magma")

    # ======================================================
    # RIM ANALYSIS
    # ======================================================

    dist_to_edge = ndi.distance_transform_edt(particle)
    dist_norm = dist_to_edge / dist_to_edge.max()

    rim = (dist_norm < 0.15) & particle
    core = (dist_norm > 0.7) & particle

    Rim_enrichment_index = gray[rim].mean() / gray[core].mean()

    if SAVE_DEBUG:

        rim_img = img_rgb.copy()
        rim_img[rim] = [255,0,0]
        rim_img[core] = [0,0,255]

        save_debug(debug_dir / f"{stem}_rim_regions.png", rim_img)

    # ======================================================
    # RADIAL TRANSITION
    # ======================================================

    rvals = dist_norm[particle]
    Ivals = gray[particle]

    rbins = np.linspace(0,1,50)

    Irad = [
        Ivals[(rvals>=rbins[i])&(rvals<rbins[i+1])].mean()
        if np.sum((rvals>=rbins[i])&(rvals<rbins[i+1]))>10 else np.nan
        for i in range(len(rbins)-1)
    ]

    rmid = 0.5*(rbins[:-1]+rbins[1:])

    def sigmoid(r,a,b,c,d):
        return a + b/(1+np.exp(-(r-c)/d))

    try:

        popt,_ = curve_fit(sigmoid,
                           rmid[~np.isnan(Irad)],
                           np.array(Irad)[~np.isnan(Irad)])

        Rim_transition_depth = popt[2]
        Rim_transition_width = popt[3]

    except:

        Rim_transition_depth = np.nan
        Rim_transition_width = np.nan

    if SAVE_DEBUG:

        plt.figure()
        plt.plot(rmid,Irad)

        plt.xlabel("radius")
        plt.ylabel("intensity")

        plt.savefig(debug_dir / f"{stem}_rim_profile.png")
        plt.close()

    # ======================================================
    # FINAL RESULTS
    # ======================================================

    results = {

        "Internal_texture_variance": Internal_texture_variance,
        "Laplacian_texture_energy": Laplacian_texture_energy,
        "Fractal_dimension_texture": Fractal_dimension_texture,

        "Outer_to_inner_roughness": Outer_to_inner_roughness,
        "Curvature_energy_decay": Curvature_energy_decay,

        "Rim_enrichment_index": Rim_enrichment_index,
        "Rim_transition_depth": Rim_transition_depth,
        "Rim_transition_width": Rim_transition_width,
        "Crystal_face_smoothness_index": Crystal_face_smoothness_index,
        "GLCM_contrast": GLCM_contrast,
        "GLCM_homogeneity": GLCM_homogeneity,
        "GLCM_energy": GLCM_energy,
        "GLCM_entropy": GLCM_entropy
    }

    return pd.DataFrame([results])


# ======================================================
# BATCH PROCESSING
# ======================================================

# ======================================================
# BATCH PROCESSING
# ======================================================

INPUT_ROOT = Path("/mnt/hpccs01/home/n11555581/Code/PARTIALRESULTS4")

OUTPUT_ROOT = Path("/mnt/hpccs01/home/n11555581/Code/OUTPUT_TEXTURE_RESULTS4")
OUTPUT_ROOT.mkdir(exist_ok=True)

DEBUG_ROOT = OUTPUT_ROOT / "DEBUG"
DEBUG_ROOT.mkdir(exist_ok=True)

OUTPUT_EXCEL = OUTPUT_ROOT / "HPC_Texture_metrics_results1.xlsx"


with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:

    for folder in sorted([f for f in INPUT_ROOT.iterdir() if f.is_dir()]):

        print(f"\nProcessing folder: {folder.name}")

        debug_dir = DEBUG_ROOT / folder.name
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

print("DONE")