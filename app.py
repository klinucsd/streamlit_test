import streamlit as st
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import pynhd
import py3dep
import pygeoutils as geoutils
from scipy.spatial import KDTree
import opt_einsum as oe

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
dem_resolution = st.sidebar.selectbox("DEM Resolution (m)", [10, 30], index=1)
river_spacing = st.sidebar.slider("River Profile Spacing (m)", 10, 100, 30)
num_neighbors = st.sidebar.slider("Number of IDW Neighbors", 10, 100, 50, step=10)
rem_span_max = st.sidebar.slider("REM Visualization Span Max", 1, 20, 7)

if st.sidebar.button("Generate REM", type="primary"):
    with st.spinner("Processing... This may take a few minutes."):
        try:
            # Step 1: Get DEM
            st.subheader("Step 1: Retrieving Digital Elevation Model")
            progress_bar = st.progress(0)
            
            dem = py3dep.get_map("DEM", bbox, resolution=dem_resolution, crs="EPSG:5070")
            
            # Ensure DEM has proper attributes
            if not hasattr(dem, 'rio'):
                import rioxarray
                dem = dem.rio.write_crs("EPSG:5070")
            
            st.success(f"✓ Retrieved DEM with shape: {dem.shape}")
            progress_bar.progress(20)
            
            # Step 2: Get river flowlines
            st.subheader("Step 2: Extracting River Flowlines")
            wd = pynhd.WaterData("nhdflowline_network")
            flw = wd.bybox(bbox)
            
            # Try to prepare NHDPlus, but handle cases where no terminal is found
            try:
                flw = pynhd.prepare_nhdplus(flw, 0, 0, 0, remove_isolated=True)
            except Exception as e:
                st.warning(f"Could not prepare NHDPlus network: {e}. Using raw flowlines.")
            
            # Find main flowline - use levelpathi if available, otherwise use streamorde
            if 'levelpathi' in flw.columns and not flw.levelpathi.isna().all():
                flw = flw[flw.levelpathi == flw.levelpathi.min()].to_crs(dem.rio.crs).copy()
            elif 'streamorde' in flw.columns:
                # Use highest stream order (typically the main river)
                max_order = flw.streamorde.max()
                flw = flw[flw.streamorde == max_order].to_crs(dem.rio.crs).copy()
            else:
                # Just use the longest flowline
                flw['length'] = flw.geometry.length
                flw = flw.nlargest(1, 'length').to_crs(dem.rio.crs).copy()
            
            if len(flw) == 0:
                raise ValueError("No suitable flowlines found in the selected area")
            
            st.success(f"✓ Extracted main flowline with {len(flw)} segments")
            progress_bar.progress(40)
            
            # Step 3: Get elevation profile along river
            st.subheader("Step 3: Computing River Elevation Profile")
            
            # Combine all flowline segments into a single line
            from shapely.ops import linemerge
            from shapely.geometry import MultiLineString, LineString
            import pyproj
            
            # Get all geometries and flatten any MultiLineStrings
            geoms = []
            for geom in flw.geometry:
                if isinstance(geom, MultiLineString):
                    geoms.extend(list(geom.geoms))
                elif isinstance(geom, LineString):
                    geoms.append(geom)
            
            # Merge into a single line
            if len(geoms) == 1:
                line = geoms[0]
            else:
                line = linemerge(geoms)
            
            # If linemerge returns a MultiLineString, take the longest segment
            if isinstance(line, MultiLineString):
                line = max(line.geoms, key=lambda x: x.length)
            
            # Create points along the line at regular intervals
            from shapely.geometry import Point
            
            # Calculate number of points based on line length and spacing
            line_length = line.length
            num_points = max(int(line_length / river_spacing), 10)
            
            # Sample points along the line
            distances = np.linspace(0, line_length, num_points)
            points = [line.interpolate(distance) for distance in distances]
            
            # Extract coordinates from DEM at these points
            coords = np.array([[p.x, p.y] for p in points])
            
            # Get elevation values at each point
            elevations = []
            for x, y in coords:
                # Find nearest DEM pixel
                x_idx = np.argmin(np.abs(dem.x.values - x))
                y_idx = np.argmin(np.abs(dem.y.values - y))
                elev = float(dem.values[y_idx, x_idx])
                elevations.append(elev)
            
            # Create river elevation array [x, y, z]
            river_elev = np.column_stack([coords, elevations])
            
            st.success(f"✓ Generated elevation profile with {len(river_elev)} points")
            progress_bar.progress(60)
            
            # Step 4: Compute REM using IDW
            st.subheader("Step 4: Computing Relative Elevation Model")
            st.info("Processing in chunks to conserve memory...")
            
            # Create coordinate grid
            x_coords = dem.x.values
            y_coords = dem.y.values
            xx, yy = np.meshgrid(x_coords, y_coords)
            dem_points = np.column_stack([xx.ravel(), yy.ravel()])
            
            # Build KDTree from river elevation points
            tree = KDTree(river_elev[:, :2])
            
            # Process in chunks to reduce memory usage
            chunk_size = 10000  # Process 10k pixels at a time
            n_points = len(dem_points)
            elevation_interpolated = np.zeros(n_points)
            
            progress_text = st.empty()
            chunk_progress = st.progress(0)
            
            for i in range(0, n_points, chunk_size):
                end_idx = min(i + chunk_size, n_points)
                chunk = dem_points[i:end_idx]
                
                # Query nearest neighbors for this chunk
                distances, idxs = tree.query(chunk, k=num_neighbors, workers=-1)
                
                # IDW calculation for this chunk
                # Avoid division by zero
                distances = np.maximum(distances, 1e-10)
                weights = 1.0 / (distances ** 2)
                weight_sum = np.sum(weights, axis=1, keepdims=True)
                weights_normalized = weights / weight_sum
                
                # Calculate weighted elevation
                elevation_interpolated[i:end_idx] = np.sum(
                    weights_normalized * river_elev[idxs, 2], 
                    axis=1
                )
                
                # Update progress
                progress = int((end_idx / n_points) * 100)
                progress_text.text(f"Processing: {progress}% complete")
                chunk_progress.progress(end_idx / n_points)
            
            progress_text.empty()
            chunk_progress.empty()
            
            # Reshape back to 2D grid
            elevation = elevation_interpolated.reshape((len(y_coords), len(x_coords)))
            elevation = xr.DataArray(
                elevation,
                dims=("y", "x"),
                coords={"x": dem.x, "y": dem.y}
            )
            
            # Calculate REM
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
