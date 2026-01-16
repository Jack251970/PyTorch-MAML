#!/usr/bin/env python3
"""Plot turbine relative positions in meters from a CSV with Latitude/Longitude.

Produces:
 - .materials/turbine_positions.csv (id, lat, lon, x_m, y_m)
 - .materials/turbine_positions.png (scatter plot, IDs in legend)

Usage:
    python plot_turbine_positions.py --input .materials/Penmanshiel_WT_static.csv --output .materials/turbine_positions.png

"""
import argparse
import math
import os
import sys
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


# Reuse detection helper from calculate_coordinate if available
def detect_columns(df: pd.DataFrame):
    cols = list(df.columns)
    lower = {c: c.lower() for c in cols}
    id_col = None
    for k in ["alternative title", "alternative_title", "title", "identity", "id", "turbine"]:
        for c in cols:
            if c.lower() == k or k in c.lower():
                id_col = c
                break
        if id_col:
            break
    lat_col = None
    lon_col = None
    for c in cols:
        cl = c.lower()
        if "lat" in cl and lat_col is None:
            lat_col = c
        if ("lon" in cl or "long" in cl) and lon_col is None:
            lon_col = c
    return id_col, lat_col, lon_col


METERS_PER_DEG_LAT = 111132.954  # approximate


def deg_to_meters(lat_ref: float, lat: float, lon: float):
    """Convert (lat, lon) to local meters relative to reference lat_ref/lon_ref.

    Returns (x_m, y_m) where x is Easting (meters), y is Northing (meters).
    Uses simple equirectangular approximation with latitude scaling.
    """
    lat_ref_rad = math.radians(lat_ref)
    meters_per_deg_lon = (111412.84 * math.cos(lat_ref_rad) - 93.5 * math.cos(3 * lat_ref_rad))  # approx
    dx = (lon) * meters_per_deg_lon
    dy = (lat) * METERS_PER_DEG_LAT
    return dx, dy


def compute_relative_positions(df: pd.DataFrame, id_col: Optional[str], lat_col: str, lon_col: str):
    # ensure numeric
    df = df.copy()
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col])
    if df.empty:
        raise ValueError("No valid lat/lon values found")

    # reference is centroid (mean)
    lat0 = df[lat_col].mean()
    lon0 = df[lon_col].mean()

    # compute meters relative to reference
    xs = []
    ys = []
    for lat, lon in zip(df[lat_col].tolist(), df[lon_col].tolist()):
        # compute dx,dy relative to ref
        dx_total, dy_total = deg_to_meters(lat0, lat - lat0, lon - lon0)
        # Note: deg_to_meters expects absolute lat/lon differences in degrees but uses lat_ref for scaling
        # Recompute using direct formulas for small deltas
        lat_rad = math.radians(lat0)
        meters_per_deg_lon = (111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3 * lat_rad))
        dx = (lon - lon0) * meters_per_deg_lon
        dy = (lat - lat0) * METERS_PER_DEG_LAT
        xs.append(dx)
        ys.append(dy)

    df = df.reset_index(drop=True)
    df["x_m"] = xs
    df["y_m"] = ys

    # determine id labels
    if id_col and id_col in df.columns:
        labels = df[id_col].astype(str).tolist()
    else:
        # try other probable columns
        for cand in ["Alternative Title", "Title", "Identity", "Alternative_Title"]:
            if cand in df.columns:
                id_col = cand
                labels = df[id_col].astype(str).tolist()
                break
        else:
            labels = [str(i) for i in range(len(df))]

    return df, labels


def plot_and_save(df: pd.DataFrame, labels, out_image: str):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot each turbine separately to get legend entries per id
    for i, (x, y, lab) in enumerate(zip(df["x_m"], df["y_m"], labels)):
        ax.scatter(x, y, label=lab)
        # annotate near point, slight offset
        ax.text(x + 2, y + 2, lab, fontsize=9)

    ax.set_xlabel("Easting (m) relative to centroid")
    ax.set_ylabel("Northing (m) relative to centroid")
    ax.set_title("Penmanshiel Turbine Relative Positions (meters)")
    ax.grid(True)
    ax.legend(title="Turbine ID", bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_image) or ".", exist_ok=True)
    fig.savefig(out_image, dpi=200)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Compute relative turbine positions and plot them")
    p.add_argument("--input", "-i", default=".materials/Penmanshiel_WT_static.csv", help="Input CSV path")
    p.add_argument("--output", "-o", default=".materials/turbine_positions.png", help="Output image path")
    p.add_argument("--csv-out", default=".materials/turbine_positions.csv", help="Output CSV with x_m,y_m")
    p.add_argument("--id-col", help="ID column override")
    p.add_argument("--lat-col", help="Latitude column override")
    p.add_argument("--lon-col", help="Longitude column override")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        sys.exit(2)

    df = pd.read_csv(args.input)
    if df.empty:
        print("Input CSV is empty")
        sys.exit(2)

    detected_id, detected_lat, detected_lon = detect_columns(df)

    id_col = args.id_col or detected_id
    lat_col = args.lat_col or detected_lat
    lon_col = args.lon_col or detected_lon

    if lat_col is None or lon_col is None:
        print("Could not detect lat/lon columns. Available columns:", list(df.columns))
        sys.exit(2)

    df_pos, labels = compute_relative_positions(df, id_col, lat_col, lon_col)

    # Save CSV
    out_csv = args.csv_out
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df_pos[[lat_col, lon_col, "x_m", "y_m"]].to_csv(out_csv, index=False)
    print(f"Saved positions CSV to {out_csv}")

    # Keep original data in output CSV
    if id_col and id_col in df.columns:
        df_pos[id_col] = df[id_col]

    # Plot
    plot_and_save(df_pos, labels, args.output)
    print(f"Saved plot image to {args.output}")


if __name__ == "__main__":
    main()
