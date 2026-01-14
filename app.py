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
Create a **Relative Elevation Model** (REM) by subtracting an interpolated river water surface  
from a DEM — useful for floodplain visualization and flood risk assessment.
""")

# ──── SIDEBAR ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    # River selection
    st.subheader("River Selection")
    river_name = st.text_input(
        "River name (partial match ok)",
        "",
        help="Examples: Colorado, Snake, Missouri, Tar, Arkansas, Platte..."
    )
    use_name_search = bool(river_name.strip())

    # Area
    st.subheader("Area of Interest")
    use_default = st.checkbox("Use default area (Tar River, NC)", value=not use_name_search)

    if use_default:
        bbox = (-77.75, 35.7, -77.25, 36.1)
    else:
        col1, col2 = st.columns(2)
        with col1:
            min_lon = st.number_input("Min Lon", value=-78.0, step=0.05)
            min_lat = st.number_input("Min Lat", value=35.5, step=0.05)
        with col2:
            max_lon = st.number_input("Max Lon", value=-77.0, step=0.05)
            max_lat = st.number_input("Max Lat", value=36.2, step=0.05)
        bbox = (min_lon, min_lat, max_lon, max_lat)

    # Processing parameters
    st.subheader("Processing Parameters")
    dem_resolution = st.selectbox("DEM resolution (m)", [10, 30], index=1)
    point_spacing_m = st.slider("River sample spacing (m)", 10, 250, 40)
    idw_neighbors = st.slider("IDW number of neighbors", 3, 60, 20)
    rem_display_max = st.slider("REM display max (m)", 2, 20, 8)

    run_button = st.button("Generate REM", type="primary")


# ──── MAIN ──────────────────────────────────────────────────────────────────
if run_button:
    with st.spinner("Processing... (can take 1–6 minutes depending on area size)"):
        try:
            # 1. Get DEM
            st.subheader("1. Loading DEM")
            dem = py3dep.get_map("DEM", bbox, resolution=dem_resolution, crs="EPSG:5070")

            # Ensure we have coordinate arrays
            x_coords = dem.x.values
            y_coords = dem.y.values

            st.success(f"DEM loaded — {dem.sizes['x']} × {dem.sizes['y']} cells")

            # 2. Get river flowlines
            st.subheader("2. Loading river network")
            wd = pynhd.WaterData("nhdflowline_network")

            if use_name_search:
                flw_box = wd.bybox(bbox)
                name_pattern = river_name.strip().title()
                flw = flw_box[
                    flw_box["gnis_name"].str.contains(name_pattern, case=False, na=False)
                ].copy()

                if len(flw) == 0:
                    st.error(f"No river features found containing: **{river_name}**")
                    st.stop()

                # Summarize options
                summary = (
                    flw.groupby("gnis_name", as_index=False)
                    .agg(
                        length_km=("geometry", lambda g: g.length.sum() / 1000),
                        segments=("comid", "count"),
                        max_order=("streamorde", "max")
                    )
                    .sort_values("length_km", ascending=False)
                )

                st.subheader("Select desired river")
                st.dataframe(summary.style.format(precision=1), use_container_width=True)

                selected = st.selectbox(
                    "River",
                    options=summary["gnis_name"].tolist(),
                    format_func=lambda n: f"{n}  —  {summary.set_index('gnis_name').loc[n, 'length_km']:.1f} km"
                )

                flw = flw[flw["gnis_name"] == selected].copy()

            else:
                # Classic main stem selection
                flw = wd.bybox(bbox)

                try:
                    flw = pynhd.prepare_nhdplus(flw, min_network_area=0.02)
                except:
                    st.info("NHDPlus preparation skipped – using raw flowlines")

                if 'levelpathi' in flw.columns and flw['levelpathi'].notna().any():
                    flw = flw[flw['levelpathi'] == flw['levelpathi'].value_counts().idxmax()]
                elif 'streamorde' in flw.columns:
                    flw = flw[flw['streamorde'] == flw['streamorde'].max()]
                else:
                    flw = flw.loc[[flw.geometry.length.idxmax()]]

            if len(flw) == 0:
                st.error("No suitable main river flowline found in selected area.")
                st.stop()

            st.success(f"Using {len(flw)} river segment(s)")

            # 3. Create single river line & sample points
            st.subheader("3. River profile creation")

            geoms = []
            for geom in flw.geometry:
                if isinstance(geom, MultiLineString):
                    geoms.extend(geom.geoms)
                elif isinstance(geom, LineString):
                    geoms.append(geom)

            line = linemerge(geoms)
            if isinstance(line, MultiLineString):
                line = max(line.geoms, key=lambda g: g.length)

            length_m = line.length
            n_points = max(60, min(10000, int(length_m / point_spacing_m) + 1))

            st.info(f"River length ≈ {length_m/1000:.1f} km → using {n_points} sample points")

            distances = np.linspace(0, length_m, n_points)
            points_along = [line.interpolate(d) for d in distances]
            river_xy = np.array([[p.x, p.y] for p in points_along])

            # Sample elevations — SAFE VERSION
            river_elev = []
            for px, py in river_xy:
                # Option A: nearest neighbor index (robust)
                xi = np.argmin(np.abs(x_coords - px))
                yi = np.argmin(np.abs(y_coords - py))
                xi = np.clip(xi, 0, len(x_coords)-1)
                yi = np.clip(yi, 0, len(y_coords)-1)
                z = float(dem.values[yi, xi])
                river_elev.append(z)

                # Alternative B (often cleaner): xarray selection
                # z = dem.sel(x=px, y=py, method="nearest").item()

            river_points = np.column_stack([river_xy, river_elev])  # shape: (n, 3) → x,y,z

            # 4. IDW interpolation
            st.subheader("4. Interpolating river surface (IDW)")
            st.info("Processing in chunks...")

            xx, yy = np.meshgrid(x_coords, y_coords)
            grid_points = np.column_stack([xx.ravel(), yy.ravel()])

            tree = KDTree(river_points[:, :2])

            chunk_size = 15000
            river_surface_flat = np.zeros(len(grid_points))

            k_actual = min(idw_neighbors, len(river_points))

            progress = st.progress(0)
            status = st.empty()

            for i in range(0, len(grid_points), chunk_size):
                end = min(i + chunk_size, len(grid_points))
                chunk = grid_points[i:end]

                dists, idxs = tree.query(chunk, k=k_actual)

                if dists.ndim == 1:  # when k=1
                    dists = dists[:, None]
                    idxs = idxs[:, None]

                dists = np.maximum(dists, 1e-8)
                weights = 1.0 / (dists ** 2)
                weights /= weights.sum(axis=1, keepdims=True)

                river_surface_flat[i:end] = np.sum(weights * river_points[idxs, 2], axis=1)

                progress.progress(end / len(grid_points))
                status.text(f"{end / len(grid_points):.0%}")

            status.empty()

            river_surface = xr.DataArray(
                river_surface_flat.reshape(dem.shape),
                dims=dem.dims,
                coords=dem.coords
            )

            rem = dem - river_surface

            st.success("Calculation finished ✓")

            # 5. Visualization
            st.subheader("Visualization")

            fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)

            dem.plot(ax=axes[0,0], cmap='terrain', robust=True)
            flw.plot(ax=axes[0,0], color='red', linewidth=1.8, alpha=0.9)
            axes[0,0].set_title("DEM + River centerline")

            river_surface.plot(ax=axes[0,1], cmap='viridis')
            flw.plot(ax=axes[0,1], color='darkred', linewidth=1.5)
            axes[0,1].set_title("Interpolated river surface")

            rem.plot(ax=axes[1,0], cmap='RdYlBu_r', robust=True)
            axes[1,0].set_title("Relative Elevation Model (REM)")

            rem.where(rem <= rem_display_max).plot(
                ax=axes[1,1], cmap='YlOrRd', vmin=0, vmax=rem_display_max,
                add_colorbar=True, cbar_kwargs={'label': 'meters above river'}
            )
            axes[1,1].set_title(f"Low-lying areas (< {rem_display_max} m)")

            plt.tight_layout()
            st.pyplot(fig)

        except Exception as e:
            st.error("Processing failed")
            st.exception(e)
