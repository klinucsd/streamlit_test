import streamlit as st
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pynhd
import py3dep
from scipy.spatial import KDTree
from shapely.ops import linemerge
from shapely.geometry import MultiLineString, LineString

st.set_page_config(page_title="Relative Elevation Model (REM)", layout="wide")
st.title("Relative Elevation Model (REM) Generator")

st.markdown("""
This app creates a **Relative Elevation Model (REM)** by subtracting an interpolated river surface  
from a DEM — useful for visualizing floodplains and low-lying areas.
""")

# ──── SIDEBAR ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    # River selection
    st.subheader("River Selection")
    river_name_input = st.text_input(
        "River name (partial ok)",
        value="",
        help="e.g. Colorado, Snake, Missouri, Tar, Arkansas..."
    )
    use_river_name = bool(river_name_input.strip())

    # Area of interest
    st.subheader("Area of Interest")
    use_default = st.checkbox("Use default area (Tar River, NC)", value=not use_river_name)

    if use_default:
        bbox = (-77.75, 35.7, -77.25, 36.1)
    else:
        col1, col2 = st.columns(2)
        with col1:
            min_lon = st.number_input("Min Lon", value=-77.75)
            min_lat = st.number_input("Min Lat", value=35.7)
        with col2:
            max_lon = st.number_input("Max Lon", value=-77.25)
            max_lat = st.number_input("Max Lat", value=36.1)
        bbox = (min_lon, min_lat, max_lon, max_lat)

    # Processing parameters
    st.subheader("Processing")
    dem_res = st.selectbox("DEM resolution (m)", [10, 30], index=1)
    spacing_m = st.slider("River point spacing (m)", 10, 250, 40)
    n_neighbors = st.slider("IDW neighbors", 3, 60, 20)
    rem_max_display = st.slider("REM display max (m)", 2, 20, 8)

    generate_btn = st.button("Generate REM", type="primary")

# ──── MAIN PROCESSING ───────────────────────────────────────────────────────
if generate_btn:
    with st.spinner("Processing (may take 1–5 minutes)..."):
        try:
            # 1. Load DEM
            st.subheader("1. Loading DEM")
            dem = py3dep.get_map("DEM", bbox, resolution=dem_res, crs="EPSG:5070")
            dem = dem.rio.write_crs("EPSG:5070", inplace=True)
            st.success(f"DEM loaded — {dem.sizes['x']} × {dem.sizes['y']}")

            # 2. Get river network
            st.subheader("2. Getting river flowlines")
            wd = pynhd.WaterData("nhdflowline_network")

            if use_river_name:
                name_clean = river_name_input.strip().title()
                flw_box = wd.bybox(bbox)
                flw = flw_box[
                    flw_box["gnis_name"].str.contains(name_clean, case=False, na=False)
                ].copy()

                if len(flw) == 0:
                    st.error(f"No features found containing '{river_name_input}'")
                    st.stop()

                # Show selection
                summary = (
                    flw.groupby("gnis_name", as_index=False)
                    .agg(length_km=("geometry", lambda g: g.length.sum()/1000),
                         segments=("comid", "count"),
                         max_order=("streamorde", "max"))
                    .sort_values("length_km", ascending=False)
                )

                st.subheader("Select river")
                st.dataframe(summary.style.format({"length_km": "{:.1f}"}), use_container_width=True)

                selected_name = st.selectbox(
                    "Choose river",
                    options=summary["gnis_name"].tolist(),
                    format_func=lambda x: f"{x}  —  {summary.set_index('gnis_name').loc[x, 'length_km']:.1f} km"
                )

                flw = flw[flw["gnis_name"] == selected_name].copy()

            else:
                # Fallback — try to get main stem
                flw = wd.bybox(bbox)
                try:
                    flw = pynhd.prepare_nhdplus(flw, min_network_area=0.02)
                except:
                    pass  # continue with raw

                if 'levelpathi' in flw.columns and flw['levelpathi'].notna().any():
                    main_lp = flw['levelpathi'].value_counts().idxmax()
                    flw = flw[flw['levelpathi'] == main_lp]
                elif 'streamorde' in flw.columns:
                    flw = flw[flw['streamorde'] == flw['streamorde'].max()]
                else:
                    flw = flw.loc[[flw.geometry.length.idxmax()]]

            if len(flw) == 0:
                st.error("No suitable river found in the area.")
                st.stop()

            st.success(f"Working with {len(flw)} river segment(s)")

            # 3. Create single river line & sample points
            st.subheader("3. River profile")

            geoms = []
            for g in flw.geometry:
                if isinstance(g, MultiLineString):
                    geoms.extend(g.geoms)
                else:
                    geoms.append(g)

            line = linemerge(geoms)
            if isinstance(line, MultiLineString):
                line = max(line.geoms, key=lambda g: g.length)

            length_m = line.length
            n_points = max(50, min(10000, int(length_m / spacing_m) + 1))

            st.info(f"River length ≈ {length_m/1000:.1f} km  →  {n_points} sample points")

            distances = np.linspace(0, length_m, n_points)
            points = [line.interpolate(d) for d in distances]
            coords = np.array([[p.x, p.y] for p in points])

            # Sample DEM elevations
            elevs = []
            for x, y in coords:
                xi = np.argmin(np.abs(dem.x - x))
                yi = np.argmin(np.abs(dem.y - y))
                elevs.append(float(dem[yi, xi]))

            river_points = np.column_stack([coords, elevs])  # [x, y, z]

            # 4. IDW interpolation
            st.subheader("4. River surface interpolation (IDW)")
            st.info("Processing in chunks...")

            xx, yy = np.meshgrid(dem.x, dem.y)
            grid_points = np.column_stack([xx.ravel(), yy.ravel()])

            tree = KDTree(river_points[:, :2])

            chunk_size = 15000
            river_surface = np.zeros(len(grid_points))

            prog_text = st.empty()
            prog_bar = st.progress(0)

            n_total = len(grid_points)
            k_use = min(n_neighbors, len(river_points))

            for i in range(0, n_total, chunk_size):
                end = min(i + chunk_size, n_total)
                chunk = grid_points[i:end]

                dists, idxs = tree.query(chunk, k=k_use)

                # Ensure 2D even if k=1
                if dists.ndim == 1:
                    dists = dists[:, None]
                    idxs = idxs[:, None]

                dists = np.maximum(dists, 1e-8)
                weights = 1.0 / (dists ** 2)
                weights /= weights.sum(axis=1, keepdims=True)

                river_surface[i:end] = np.sum(weights * river_points[idxs, 2], axis=1)

                prog_bar.progress(end / n_total)
                prog_text.text(f"{end/n_total:3.0%}")

            prog_text.empty()

            river_surf_da = xr.DataArray(
                river_surface.reshape(dem.shape),
                dims=dem.dims,
                coords=dem.coords
            )

            rem = dem - river_surf_da

            st.success("REM calculation complete ✓")

            # 5. Visualization
            st.subheader("Results")

            fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharex=True, sharey=True)

            dem.plot(ax=axes[0,0], cmap='terrain', robust=True)
            flw.plot(ax=axes[0,0], color='red', linewidth=1.8, alpha=0.9)
            axes[0,0].set_title("DEM + River")

            river_surf_da.plot(ax=axes[0,1], cmap='viridis')
            flw.plot(ax=axes[0,1], color='darkred', linewidth=1.5)
            axes[0,1].set_title("Interpolated River Surface")

            rem.plot(ax=axes[1,0], cmap='RdYlBu_r', robust=True)
            axes[1,0].set_title("Relative Elevation Model (REM)")

            # Simple flood highlight
            rem.where(rem <= rem_max_display).plot(
                ax=axes[1,1], cmap='YlOrRd', vmin=0, vmax=rem_max_display,
                add_colorbar=True, cbar_kwargs={'label':'m above river'}
            )
            axes[1,1].set_title(f"Low areas (< {rem_max_display} m)")

            for ax in axes.flat:
                ax.set_aspect('equal')

            plt.tight_layout()
            st.pyplot(fig)

            # Basic stats
            st.subheader("Statistics")
            cols = st.columns(4)
            cols[0].metric("DEM min/max", f"{float(dem.min()):.1f} – {float(dem.max()):.1f} m")
            cols[1].metric("REM min/max", f"{float(rem.min()):.1f} – {float(rem.max()):.1f} m")
            cols[2].metric("REM mean", f"{float(rem.mean()):.1f} m")
            cols[3].metric("River length", f"{length_m/1000:.1f} km")

        except Exception as e:
            st.error("Processing failed")
            st.exception(e)
