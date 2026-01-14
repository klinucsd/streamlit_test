import streamlit as st
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import pynhd
import py3dep
import pygeoutils as geoutils
from scipy.spatial import KDTree
from shapely.ops import linemerge
from shapely.geometry import MultiLineString, LineString, Point

st.set_page_config(page_title="Relative Elevation Model (REM)", layout="wide")
st.title("Relative Elevation Model (REM) Generator")
st.markdown("""
This app creates a **Relative Elevation Model (REM)** by detrending a Digital Elevation Model (DEM) 
using the elevation of the nearest river point.  
Very useful for floodplain mapping and flood visualization.
""")

# ──── SIDEBAR ───────────────────────────────────────────────────────────────
st.sidebar.header("Parameters")

# River selection (NEW - main way to choose river)
st.sidebar.subheader("River Selection")
river_name_input = st.sidebar.text_input(
    "River name (e.g. Colorado, Snake, Missouri, Arkansas)",
    value="",
    help="Partial name works. Leave empty to use bounding box + automatic mainstem detection."
)

use_river_name = bool(river_name_input.strip())

# Area of Interest
st.sidebar.subheader("Area of Interest")
use_default = st.sidebar.checkbox("Use default area (Tar River, NC)", value=not use_river_name)

if use_default:
    bbox = (-77.75, 35.7, -77.25, 36.1)
else:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        min_lon = st.number_input("Min Longitude", value=-77.75)
        min_lat = st.number_input("Min Latitude", value=35.7)
    with col2:
        max_lon = st.number_input("Max Longitude", value=-77.25)
        max_lat = st.number_input("Max Latitude", value=36.1)
    bbox = (min_lon, min_lat, max_lon, max_lat)

# Processing parameters
st.sidebar.subheader("Processing Settings")
dem_resolution = st.sidebar.selectbox("DEM Resolution (m)", [10, 30], index=1)
river_spacing = st.sidebar.slider("River Profile Spacing (m)", 10, 200, 30, step=5)
num_neighbors = st.sidebar.slider("Number of IDW Neighbors", 5, 120, 40, step=5)
rem_span_max = st.sidebar.slider("REM Visualization Span Max (m)", 1, 25, 8)

if st.sidebar.button("Generate REM", type="primary"):
    with st.spinner("Processing... This may take several minutes."):
        try:
            # ── 1. Get DEM ───────────────────────────────────────────────────────
            st.subheader("1. Retrieving Digital Elevation Model")
            progress_bar = st.progress(0)

            dem = py3dep.get_map("DEM", bbox, resolution=dem_resolution, crs="EPSG:5070")
            
            if not hasattr(dem, 'rio'):
                import rioxarray
                dem = dem.rio.write_crs("EPSG:5070")

            st.success(f"✓ DEM retrieved — shape: {dem.shape}")
            progress_bar.progress(15)

            # ── 2. Get river flowlines ──────────────────────────────────────────
            st.subheader("2. Extracting River Flowlines")
            wd = pynhd.WaterData("nhdflowline_network")

            if use_river_name:
                # ── River name search mode ──────────────────────────────────────
                river_name_clean = river_name_input.strip().title()
                st.info(f"Searching for rivers containing: **{river_name_clean}**")

                flw_all = wd.bybox(bbox)
                
                flw = flw_all[
                    flw_all["gnis_name"].str.contains(river_name_clean, case=False, na=False)
                ].copy()

                if len(flw) == 0:
                    st.error(f"No flowlines found containing '{river_name_input}' in this area.")
                    st.stop()

                # Summarize found rivers
                summary = (
                    flw.groupby("gnis_name", as_index=False)
                    .agg({
                        "comid": "count",
                        "gnis_id": "first",
                        "streamorde": "max",
                        "geometry": lambda g: g.length.sum() / 1000
                    })
                    .rename(columns={"comid": "# segments", "geometry": "Length (km)"})
                    .sort_values("Length (km)", ascending=False)
                )

                st.subheader("Select the desired river")
                st.dataframe(summary[["gnis_name", "# segments", "streamorde", "Length (km)"]])

                selected_name = st.selectbox(
                    "Choose river",
                    options=summary["gnis_name"].tolist(),
                    format_func=lambda x: f"{x}  ({summary.loc[summary['gnis_name']==x, 'Length (km)'].values[0]:.1f} km)"
                )

                flw = flw[flw["gnis_name"] == selected_name].copy()
                st.success(f"Selected: **{selected_name}** — {len(flw)} segments, ≈ {flw.geometry.length.sum()/1000:.1f} km")

            else:
                # ── Original bounding box + mainstem logic ──────────────────────
                flw = wd.bybox(bbox)

                try:
                    flw = pynhd.prepare_nhdplus(flw, 0, 0, 0, remove_isolated=True)
                except Exception as e:
                    st.warning(f"NHDPlus preparation failed: {e}. Using raw flowlines.")

                if 'levelpathi' in flw.columns and flw['levelpathi'].notna().any():
                    main_levelpath = flw['levelpathi'].value_counts().idxmax()
                    flw = flw[flw['levelpathi'] == main_levelpath]
                elif 'streamorde' in flw.columns:
                    max_order = flw['streamorde'].max()
                    flw = flw[flw['streamorde'] == max_order]
                else:
                    flw['length'] = flw.geometry.length
                    flw = flw.nlargest(1, 'length')

                if len(flw) == 0:
                    raise ValueError("No suitable main flowline found in the area")

                st.success(f"✓ Selected main flowline — {len(flw)} segments")

            progress_bar.progress(35)

            # ── 3. Create continuous river line & sample points ─────────────────
            st.subheader("3. Building River Elevation Profile")

            # Merge geometries
            geoms = []
            for geom in flw.geometry:
                if isinstance(geom, MultiLineString):
                    geoms.extend(geom.geoms)
                elif isinstance(geom, LineString):
                    geoms.append(geom)

            line = linemerge(geoms)
            if isinstance(line, MultiLineString):
                line = max(line.geoms, key=lambda x: x.length)

            total_length = line.length
            target_spacing = river_spacing
            num_points = max(30, min(12000, int(total_length / target_spacing) + 1))

            st.info(f"River length ≈ {total_length/1000:.1f} km → sampling {num_points} points "
                    f"(≈ {total_length/num_points:.0f} m spacing)")

            distances = np.linspace(0, total_length, num_points)
            points = [line.interpolate(d) for d in distances]
            coords = np.array([[p.x, p.y] for p in points])

            # Sample elevations
            elevations = []
            for x, y in coords:
                x_idx = np.argmin(np.abs(dem.x.values - x))
                y_idx = np.argmin(np.abs(dem.y.values - y))
                elev = float(dem.values[y_idx, x_idx])
                elevations.append(elev)

            river_elev = np.column_stack([coords, elevations])
            st.success(f"✓ River profile created with {len(river_elev)} points")
            progress_bar.progress(55)

            # ── 4. IDW interpolation of river elevation surface ────────────────
            st.subheader("4. Computing Interpolated River Surface (IDW)")
            st.info("Processing in chunks...")

            x_coords = dem.x.values
            y_coords = dem.y.values
            xx, yy = np.meshgrid(x_coords, y_coords)
            dem_points = np.column_stack([xx.ravel(), yy.ravel()])

            tree = KDTree(river_elev[:, :2])

            chunk_size = 15000
            n_points = len(dem_points)
            elevation_interpolated = np.zeros(n_points)

            progress_text = st.empty()
            chunk_progress = st.progress(0)

            for i in range(0, n_points, chunk_size):
                end = min(i + chunk_size, n_points)
                chunk = dem_points[i:end]

                distances, idxs = tree.query(chunk, k=num_neighbors)
                distances = np.maximum(distances, 1e-8)
                weights = 1.0 / (distances ** 2)
                weights /= weights.sum(axis=1, keepdims=True)

                elevation_interpolated[i:end] = np.sum(weights * river_elev[idxs, 2], axis=1)

                progress = int((end / n_points) * 100)
                progress_text.text(f"IDW progress: {progress}%")
                chunk_progress.progress(end / n_points)

            progress_text.empty()
            chunk_progress.empty()

            elevation = xr.DataArray(
                elevation_interpolated.reshape((len(y_coords), len(x_coords))),
                dims=("y", "x"),
                coords={"x": x_coords, "y": y_coords}
            )

            rem = dem - elevation
            st.success("✓ REM calculation complete")
            progress_bar.progress(80)

            # ── 5. Visualizations ───────────────────────────────────────────────
            st.subheader("5. Visualizations")

            fig, axes = plt.subplots(2, 3, figsize=(18, 12))

            # DEM + flowline
            dem.plot(ax=axes[0,0], cmap="terrain", robust=True, add_colorbar=True)
            flw.plot(ax=axes[0,0], color="red", linewidth=2)
            axes[0,0].set_title("DEM + Selected River")

            # Interpolated river surface
            elevation.plot(ax=axes[0,1], cmap="viridis", add_colorbar=True)
            flw.plot(ax=axes[0,1], color="red", linewidth=2)
            axes[0,1].set_title("Interpolated River Level")

            # REM
            rem.plot(ax=axes[0,2], cmap="RdYlBu_r", robust=True, add_colorbar=True)
            flw.plot(ax=axes[0,2], color="black", linewidth=2)
            axes[0,2].set_title("Relative Elevation Model (REM)")

            # Flood zone highlight
            rem_masked = rem.where(rem < rem_span_max)
            im = axes[1,0].imshow(
                rem_masked.values, cmap="YlOrRd", vmin=0, vmax=rem_span_max,
                extent=[float(dem.x.min()), float(dem.x.max()), float(dem.y.min()), float(dem.y.max())],
                origin='upper'
            )
            flw.plot(ax=axes[1,0], color="blue", linewidth=2)
            axes[1,0].set_title(f"Potential Flood Area (< {rem_span_max}m)")
            plt.colorbar(im, ax=axes[1,0], label="Relative elevation (m)")

            plt.tight_layout()
            st.pyplot(fig)

            progress_bar.progress(100)

            # Statistics
            st.subheader("Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("DEM Min", f"{float(dem.min()):.2f} m")
                st.metric("DEM Max", f"{float(dem.max()):.2f} m")
            with col2:
                st.metric("REM Min", f"{float(rem.min()):.2f} m")
                st.metric("REM Max", f"{float(rem.max()):.2f} m")
            with col3:
                st.metric("REM Mean", f"{float(rem.mean()):.2f} m")
                st.metric("REM Std", f"{float(rem.std()):.2f} m")

        except Exception as e:
            st.error(f"Processing failed: {str(e)}")
            st.exception(e)

# ──── Footer / Info ─────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info("""
Uses HyRiver stack (pynhd, py3dep)  
Data: USGS 3DEP DEM + NHDPlus HR flowlines  
Main improvement: River name search & selection  
""")
