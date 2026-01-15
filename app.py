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
aoi_method = st.sidebar.radio(
    "Selection Method",
    ["Example Location", "Custom Bounding Box", "River Name + Length"]
)

if aoi_method == "Example Location":
    example = st.sidebar.selectbox(
        "Choose Example",
        ["Tar River, NC (Small)", "Tar River, NC (Original)", "Custom"]
    )
    if example == "Tar River, NC (Small)":
        bbox = (-77.6, 35.85, -77.4, 36.0)
        st.sidebar.info("📍 Tar River near Rocky Mount, NC (Small section)")
    elif example == "Tar River, NC (Original)":
        bbox = (-77.75, 35.7, -77.25, 36.1)
        st.sidebar.info("📍 Tar River near Rocky Mount, NC (Full area)")
        st.sidebar.warning("⚠️ This is a larger area - may use significant memory")
    else:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            min_lon = st.number_input("Min Longitude", value=-77.6, format="%.4f")
            min_lat = st.number_input("Min Latitude", value=35.85, format="%.4f")
        with col2:
            max_lon = st.number_input("Max Longitude", value=-77.4, format="%.4f")
            max_lat = st.number_input("Max Latitude", value=36.0, format="%.4f")
        bbox = (min_lon, min_lat, max_lon, max_lat)

elif aoi_method == "Custom Bounding Box":
    st.sidebar.info("💡 Tip: Keep area < 0.2° x 0.2° to avoid memory issues")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        min_lon = st.number_input("Min Longitude", value=-77.6, format="%.4f")
        min_lat = st.number_input("Min Latitude", value=35.85, format="%.4f")
    with col2:
        max_lon = st.number_input("Max Longitude", value=-77.4, format="%.4f")
        max_lat = st.number_input("Max Latitude", value=36.0, format="%.4f")
    bbox = (min_lon, min_lat, max_lon, max_lat)
    
    # Calculate bbox size
    width = max_lon - min_lon
    height = max_lat - min_lat
    if width > 0.3 or height > 0.3:
        st.sidebar.warning("⚠️ Large area! This may require significant memory and processing time.")
    
    st.sidebar.metric("Area Size", f"{width:.2f}° × {height:.2f}°")

else:  # River Name + Length
    st.sidebar.info("💡 Search for a river and specify analysis length")
    river_name = st.sidebar.text_input("River Name", value="Tar River")
    river_state = st.sidebar.text_input("State (optional)", value="NC", help="Helps narrow search")
    river_length_km = st.sidebar.slider("Analysis Length (km)", 5, 100, 30, step=5)
    
    st.sidebar.warning(f"⚠️ Will analyze ~{river_length_km} km of river. Longer = more memory.")
    
    # Note: bbox will be determined after finding the river
    bbox = None

# Other parameters
dem_resolution = st.sidebar.selectbox("DEM Resolution (m)", [10, 30], index=1)
river_spacing = st.sidebar.slider("River Profile Spacing (m)", 10, 100, 30)
num_neighbors = st.sidebar.slider("Number of IDW Neighbors", 10, 100, 50, step=10)
rem_span_max = st.sidebar.slider("REM Visualization Span Max", 1, 20, 7)

if st.sidebar.button("Generate REM", type="primary"):
    with st.spinner("Processing... This may take a few minutes."):
        try:
            # Handle River Name search first if needed
            if aoi_method == "River Name + Length" and bbox is None:
                st.subheader("Step 0: Finding River")
                progress_bar = st.progress(0)
                
                try:
                    # Search for river by name
                    from pynhd import NLDI
                    
                    # Try to find the river
                    nldi = NLDI()
                    
                    # Create a rough search area based on state
                    state_centers = {
                        "NC": (-79.0, 35.5),
                        "OH": (-82.9, 40.4),
                        "VA": (-78.6, 37.4),
                        "TN": (-86.5, 35.8),
                        "KY": (-84.9, 37.8),
                        "PA": (-77.8, 40.9),
                        "NY": (-75.5, 43.0),
                        "CA": (-119.4, 36.7),
                        "TX": (-99.9, 31.9),
                    }
                    
                    # Get approximate center
                    if river_state.upper() in state_centers:
                        center_lon, center_lat = state_centers[river_state.upper()]
                    else:
                        center_lon, center_lat = -95.7, 37.1  # Center of US
                    
                    # Create search bbox (1 degree around center)
                    search_bbox = (center_lon - 2, center_lat - 2, center_lon + 2, center_lat + 2)
                    
                    # Get flowlines
                    wd = pynhd.WaterData("nhdflowline_network")
                    flw_search = wd.bybox(search_bbox)
                    
                    # Search for river name in the GNIS_NAME field
                    if 'gnis_name' in flw_search.columns:
                        matches = flw_search[flw_search.gnis_name.str.contains(river_name, case=False, na=False)]
                    elif 'name' in flw_search.columns:
                        matches = flw_search[flw_search.name.str.contains(river_name, case=False, na=False)]
                    else:
                        st.error("Could not find name field in flowline data")
                        matches = None
                    
                    if matches is not None and len(matches) > 0:
                        # Get the main stem (highest stream order or longest)
                        if 'streamorde' in matches.columns:
                            main_river = matches.nlargest(1, 'streamorde')
                        else:
                            matches['length'] = matches.geometry.length
                            main_river = matches.nlargest(1, 'length')
                        
                        # Get centroid and create bbox
                        centroid = main_river.geometry.centroid.iloc[0]
                        
                        # Convert km to degrees (approximate)
                        deg_per_km = 0.009  # roughly 1 km = 0.009 degrees at mid-latitudes
                        half_size = (river_length_km * deg_per_km) / 2
                        
                        bbox = (
                            centroid.x - half_size,
                            centroid.y - half_size,
                            centroid.x + half_size,
                            centroid.y + half_size
                        )
                        
                        st.success(f"✓ Found {river_name}! Analyzing {river_length_km} km section.")
                        st.info(f"Bbox: {bbox}")
                        progress_bar.progress(10)
                    else:
                        st.error(f"Could not find '{river_name}' in {river_state}. Try adjusting the name or using Custom Bounding Box.")
                        st.stop()
                        
                except Exception as e:
                    st.error(f"Error finding river: {str(e)}")
                    st.info("💡 Try using 'Custom Bounding Box' method instead.")
                    st.stop()
            else:
                progress_bar = st.progress(0)
            
            # Validate bbox size
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height
            
            if area > 0.25:
                st.warning(f"⚠️ Area is {area:.2f} deg² - this may take several minutes and use significant memory!")
            
            # Estimate memory usage
            if dem_resolution == 10:
                pixels_per_deg = 11132  # roughly
            else:
                pixels_per_deg = 3711
            
            estimated_pixels = int(width * pixels_per_deg * height * pixels_per_deg)
            estimated_mb = (estimated_pixels * 8 * num_neighbors) / (1024**2)  # rough estimate
            
            proceed = True
            
            if estimated_mb > 800:
                st.error(f"⚠️ Estimated memory: {estimated_mb:.0f} MB - This will likely crash! Please reduce area, resolution, or number of neighbors.")
                proceed = st.checkbox("I understand the risks, proceed anyway", key="risk_800")
            elif estimated_mb > 400:
                st.warning(f"⚠️ Estimated memory: {estimated_mb:.0f} MB - This may crash or be very slow!")
                st.info("💡 Suggestions: Reduce area size, use 30m resolution, or reduce IDW neighbors to 30")
                proceed = st.checkbox("I understand the risks, proceed anyway", key="risk_400")
            elif estimated_mb > 200:
                st.info(f"ℹ️ Estimated memory: {estimated_mb:.0f} MB - May take a few minutes.")
            
            if not proceed:
                st.stop()
            
            # Continue with existing processing...
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
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Plot 1: DEM with flowline
            dem.plot(ax=axes[0, 0], cmap="terrain", robust=True, add_colorbar=True)
            flw.plot(ax=axes[0, 0], color="red", linewidth=2)
            axes[0, 0].set_title("Digital Elevation Model with Main Flowline", fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel("X (m)")
            axes[0, 0].set_ylabel("Y (m)")
            
            # Plot 2: River elevation interpolation
            elevation.plot(ax=axes[0, 1], cmap="viridis", add_colorbar=True)
            flw.plot(ax=axes[0, 1], color="red", linewidth=2)
            axes[0, 1].set_title("Interpolated River Elevation Surface (IDW)", fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel("X (m)")
            axes[0, 1].set_ylabel("Y (m)")
            
            # Plot 3: REM
            rem.plot(ax=axes[0, 2], cmap="RdYlBu_r", robust=True, add_colorbar=True)
            flw.plot(ax=axes[0, 2], color="black", linewidth=2)
            axes[0, 2].set_title("Relative Elevation Model (REM)", fontsize=12, fontweight='bold')
            axes[0, 2].set_xlabel("X (m)")
            axes[0, 2].set_ylabel("Y (m)")
            
            # Plot 4: REM with custom colormap (flood zones)
            rem_masked = rem.where(rem < rem_span_max)
            im = axes[1, 0].imshow(
                rem_masked.values, 
                cmap="YlOrRd", 
                vmin=0, 
                vmax=rem_span_max,
                extent=[float(dem.x.min()), float(dem.x.max()), 
                        float(dem.y.min()), float(dem.y.max())],
                origin='upper'
            )
            flw.plot(ax=axes[1, 0], color="blue", linewidth=2)
            axes[1, 0].set_title(f"REM - Potential Flood Zone (< {rem_span_max}m)", fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel("X (m)")
            axes[1, 0].set_ylabel("Y (m)")
            plt.colorbar(im, ax=axes[1, 0], label="Elevation (m)")
            
            # Plot 5: Hillshade
            st.info("Computing hillshade...")
            try:
                from scipy.ndimage import generic_filter
                
                # Calculate hillshade
                altitude = 10  # degrees
                azimuth = 90  # degrees
                
                # Calculate gradients
                x, y = np.gradient(dem.values)
                
                # Convert angles to radians
                azimuth_rad = np.radians(azimuth)
                altitude_rad = np.radians(altitude)
                
                # Calculate slope and aspect
                slope = np.arctan(np.sqrt(x**2 + y**2))
                aspect = np.arctan2(-x, y)
                
                # Calculate hillshade
                hillshade = np.sin(altitude_rad) * np.sin(slope) + \
                           np.cos(altitude_rad) * np.cos(slope) * \
                           np.cos(azimuth_rad - aspect)
                
                hillshade = (hillshade - hillshade.min()) / (hillshade.max() - hillshade.min())
                
                axes[1, 1].imshow(
                    hillshade,
                    cmap="gray",
                    extent=[float(dem.x.min()), float(dem.x.max()), 
                            float(dem.y.min()), float(dem.y.max())],
                    origin='upper'
                )
                axes[1, 1].set_title("Hillshade (Illumination)", fontsize=12, fontweight='bold')
                axes[1, 1].set_xlabel("X (m)")
                axes[1, 1].set_ylabel("Y (m)")
            except Exception as e:
                axes[1, 1].text(0.5, 0.5, f"Hillshade error:\n{str(e)}", 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title("Hillshade (Error)", fontsize=12, fontweight='bold')
            
            # Plot 6: Combined visualization (DEM + Hillshade + REM overlay)
            st.info("Creating composite visualization...")
            try:
                # Create RGB composite
                # Normalize DEM for grayscale base
                dem_norm = (dem.values - np.nanmin(dem.values)) / (np.nanmax(dem.values) - np.nanmin(dem.values))
                
                # Apply hillshade shading
                shaded = dem_norm * (hillshade * 0.5 + 0.5)
                
                # Create REM overlay with transparency
                rem_norm = np.clip(rem.values / rem_span_max, 0, 1)
                rem_colored = plt.cm.inferno_r(rem_norm)
                
                # Composite: blend hillshaded DEM with REM
                alpha = 0.6 * (rem_norm < 1)  # Transparency based on REM value
                composite = shaded[..., np.newaxis] * (1 - alpha[..., np.newaxis]) + \
                           rem_colored[..., :3] * alpha[..., np.newaxis]
                
                axes[1, 2].imshow(
                    composite,
                    extent=[float(dem.x.min()), float(dem.x.max()), 
                            float(dem.y.min()), float(dem.y.max())],
                    origin='upper'
                )
                flw.plot(ax=axes[1, 2], color="cyan", linewidth=2, alpha=0.8)
                axes[1, 2].set_title("Composite: Hillshade + REM Overlay", fontsize=12, fontweight='bold')
                axes[1, 2].set_xlabel("X (m)")
                axes[1, 2].set_ylabel("Y (m)")
            except Exception as e:
                axes[1, 2].text(0.5, 0.5, f"Composite error:\n{str(e)}", 
                               ha='center', va='center', transform=axes[1, 2].transAxes)
                axes[1, 2].set_title("Composite Visualization (Error)", fontsize=12, fontweight='bold')
            
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

**Tips for Best Results:**
- Start with small areas (0.15° x 0.15° or ~17km x 17km)
- Use 30m resolution for faster processing  
- Reduce IDW neighbors to 30-40 for large areas
- For long rivers, analyze sections rather than entire length
- Urban areas may have less accurate flowline data

**Memory Usage Guide:**
- Small area (0.15° x 0.15°, 30m, 50 neighbors): ~150 MB
- Medium area (0.25° x 0.25°, 30m, 50 neighbors): ~400 MB  
- Large area (0.5° x 0.5°, 30m, 50 neighbors): ~1500 MB ⚠️
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Citation:**  
Chegini, T., Li, H., & Leung, L. R. (2021). HyRiver: Hydroclimate Data Retriever. 
*Journal of Open Source Software*, 6(66), 3175.
""")
