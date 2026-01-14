import streamlit as st
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pynhd
import py3dep
import pygeoutils as geoutils
from scipy.spatial import KDTree
import opt_einsum as oe
import datashader as ds
import datashader.transfer_functions as tf
from datashader.colors import Greys9, inferno
import xrspatial as xs

st.set_page_config(page_title="Relative Elevation Model (REM)", layout="wide")

st.title("Relative Elevation Model (REM) Generator")
st.markdown("""
This app creates a Relative Elevation Model (REM) which detrends a Digital Elevation Model (DEM) 
by subtracting the elevation of the nearest river point. This is useful for flood modeling and 
identifying floodplain areas.
""")

# Sidebar for parameters
st.sidebar.header("Parameters")

# Define area of interest
st.sidebar.subheader("Area of Interest")
use_default = st.sidebar.checkbox("Use default area (Tar River, NC)", value=True)

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

# Other parameters
dem_resolution = st.sidebar.selectbox("DEM Resolution (m)", [10, 30], index=0)
river_spacing = st.sidebar.slider("River Profile Spacing (m)", 5, 50, 10)
num_neighbors = st.sidebar.slider("Number of IDW Neighbors", 50, 500, 200, step=50)
rem_span_max = st.sidebar.slider("REM Visualization Span Max", 1, 20, 7)

if st.sidebar.button("Generate REM", type="primary"):
    with st.spinner("Processing... This may take a few minutes."):
        try:
            # Step 1: Get DEM
            st.subheader("Step 1: Retrieving Digital Elevation Model")
            progress_bar = st.progress(0)
            
            dem = py3dep.get_map("DEM", bbox, resolution=dem_resolution, crs="EPSG:5070")
            st.success(f"✓ Retrieved DEM with shape: {dem.shape}")
            progress_bar.progress(20)
            
            # Step 2: Get river flowlines
            st.subheader("Step 2: Extracting River Flowlines")
            wd = pynhd.WaterData("nhdflowline_network")
            flw = wd.bybox(bbox)
            flw = pynhd.prepare_nhdplus(flw, 0, 0, 0, remove_isolated=True)
            
            # Find main flowline
            flw = flw[flw.levelpathi == flw.levelpathi.min()].to_crs(dem.rio.crs).copy()
            st.success(f"✓ Extracted main flowline with {len(flw)} segments")
            progress_bar.progress(40)
            
            # Step 3: Get elevation profile along river
            st.subheader("Step 3: Computing River Elevation Profile")
            line = geoutils.geometry_list(flw)
            line = geoutils.smooth_linestring(line, smoothing=river_spacing)
            line = gpd.GeoDataFrame(geometry=[line], crs=flw.crs)
            river_elev = py3dep.elevation_profile(line, dem)
            st.success(f"✓ Generated elevation profile with {len(river_elev)} points")
            progress_bar.progress(60)
            
            # Step 4: Compute REM using IDW
            st.subheader("Step 4: Computing Relative Elevation Model")
            distances, idxs = KDTree(river_elev[:, :2]).query(
                np.dstack(np.meshgrid(dem.x, dem.y)).reshape(-1, 2),
                k=num_neighbors,
                workers=-1,
            )
            
            w = np.reciprocal(np.power(distances, 2) + np.isclose(distances, 0))
            w_sum = np.sum(w, axis=1)
            w_norm = oe.contract(
                "ij,i->ij",
                w,
                np.reciprocal(w_sum + np.isclose(w_sum, 0)),
                optimize="optimal"
            )
            elevation = oe.contract(
                "ij,ij->i",
                w_norm,
                river_elev[idxs, 2],
                optimize="optimal"
            )
            elevation = elevation.reshape((dem.sizes["y"], dem.sizes["x"]))
            elevation = xr.DataArray(
                elevation,
                dims=("y", "x"),
                coords={"x": dem.x, "y": dem.y}
            )
            
            rem = dem - elevation
            st.success("✓ REM computation complete")
            progress_bar.progress(80)
            
            # Step 5: Visualizations
            st.subheader("Step 5: Generating Visualizations")
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: DEM with flowline
            dem.plot(ax=axes[0, 0], cmap="terrain", robust=True)
            flw.plot(ax=axes[0, 0], color="red", linewidth=2)
            axes[0, 0].set_title("Digital Elevation Model with Main Flowline")
            axes[0, 0].set_xlabel("X (m)")
            axes[0, 0].set_ylabel("Y (m)")
            
            # Plot 2: River elevation interpolation
            elevation.plot(ax=axes[0, 1], cmap="viridis")
            flw.plot(ax=axes[0, 1], color="red", linewidth=2)
            axes[0, 1].set_title("Interpolated River Elevation Surface")
            axes[0, 1].set_xlabel("X (m)")
            axes[0, 1].set_ylabel("Y (m)")
            
            # Plot 3: REM
            rem.plot(ax=axes[1, 0], cmap="RdYlBu_r", robust=True)
            flw.plot(ax=axes[1, 0], color="black", linewidth=2)
            axes[1, 0].set_title("Relative Elevation Model (REM)")
            axes[1, 0].set_xlabel("X (m)")
            axes[1, 0].set_ylabel("Y (m)")
            
            # Plot 4: REM with custom colormap (flood zones)
            rem_masked = rem.where(rem < rem_span_max)
            rem_masked.plot(ax=axes[1, 1], cmap="YlOrRd", vmin=0, vmax=rem_span_max)
            flw.plot(ax=axes[1, 1], color="blue", linewidth=2)
            axes[1, 1].set_title(f"REM - Potential Flood Zone (< {rem_span_max}m)")
            axes[1, 1].set_xlabel("X (m)")
            axes[1, 1].set_ylabel("Y (m)")
            
            plt.tight_layout()
            st.pyplot(fig)
            progress_bar.progress(100)
            
            # Statistics
            st.subheader("Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("DEM Min Elevation", f"{float(dem.min()):.2f} m")
                st.metric("DEM Max Elevation", f"{float(dem.max()):.2f} m")
            with col2:
                st.metric("REM Min", f"{float(rem.min()):.2f} m")
                st.metric("REM Max", f"{float(rem.max()):.2f} m")
            with col3:
                st.metric("REM Mean", f"{float(rem.mean()):.2f} m")
                st.metric("REM Std Dev", f"{float(rem.std()):.2f} m")
            
            # Download option
            st.subheader("Download Results")
            if st.button("Prepare GeoTIFF Downloads"):
                # Save REM as GeoTIFF
                rem.rio.to_raster("rem_output.tif")
                st.success("✓ Files ready for download")
                st.info("REM saved as 'rem_output.tif'")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)

# Information section
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info("""
This app uses the HyRiver software stack to create Relative Elevation Models.

**Key Parameters:**
- **DEM Resolution**: Higher resolution = more detail, longer processing
- **River Spacing**: Distance between elevation profile points
- **IDW Neighbors**: More neighbors = smoother interpolation
- **REM Span**: Maximum elevation for flood zone visualization

**Data Sources:**
- DEM: USGS 3DEP
- Flowlines: NHDPlus
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Citation:**  
Chegini, T., Li, H., & Leung, L. R. (2021). HyRiver: Hydroclimate Data Retriever. 
*Journal of Open Source Software*, 6(66), 3175.
""")
