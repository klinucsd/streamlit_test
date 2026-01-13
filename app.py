import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import colors
from matplotlib.collections import LineCollection
from typing import Literal
import traceback

# Page config
st.set_page_config(page_title="Ridge Map Generator", layout="wide")

st.title("🏔️ Ridge Map Generator")
st.markdown("""
This app creates beautiful ridge maps (2D representations of 3D terrain) using elevation data.
Based on the [HyRiver Ridge Map tutorial](https://docs.hyriver.io/examples/notebooks/ridges.html).
""")

# Import dependencies with error handling
try:
    import py3dep
    from pynhd import NLDI
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    st.error("⚠️ Required packages not installed: `py3dep` and `pynhd`. Install with: `pip install py3dep pynhd`")

def plot_ridges(
    elevation: xr.DataArray,
    label: str | None = None,
    label_x: float = 0.62,
    label_y: float = 0.15,
    label_verticalalignment: str = "bottom",
    label_size: int = 60,
    line_color: str | colors.Colormap = "black",
    kind: Literal["gradient", "elevation"] = "gradient",
    linewidth: int = 2,
    background_color: tuple[float, float, float] = (0.9255, 0.9098, 0.9255),
    size_scale: int = 20,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the ridge map.
    
    Modified version from the ridge_map package to work with xarray.DataArray.
    """
    if kind not in {"gradient", "elevation"}:
        raise TypeError("Argument `kind` must be one of 'gradient' or 'elevation'")

    if not isinstance(elevation, xr.DataArray):
        raise TypeError("Argument `elevation` must be an xarray.DataArray")

    xmin, ymin, xmax, ymax = elevation.rio.bounds()
    ratio = (ymax - ymin) / (xmax - xmin)
    fig, ax = plt.subplots(figsize=(size_scale, size_scale * ratio))

    values = elevation.to_numpy()
    x = np.arange(values.shape[1])
    norm = colors.Normalize(np.nanmin(values), np.nanmax(values))
    
    for idx, row in enumerate(values):
        y_base = -6 * idx * np.ones_like(row)
        y = row + y_base
        
        if callable(line_color) and kind == "elevation":
            points = np.array([x, y]).T.reshape((-1, 1, 2))
            segments = np.concatenate([points[:-1], points[1:]], axis=1).tolist()
            lines = LineCollection(segments, cmap=line_color, zorder=idx + 1, norm=norm)
            lines.set_array(row)
            lines.set_linewidth(linewidth)
            ax.add_collection(lines)
        else:
            if callable(line_color) and kind == "gradient":
                color = line_color(idx / values.shape[0])
            else:
                color = line_color

            ax.plot(x, y, "-", color=color, zorder=idx, lw=linewidth)
        ax.fill_between(x, y_base, y, color=background_color, zorder=idx)

    if label:
        ax.text(
            label_x,
            label_y,
            label,
            transform=ax.transAxes,
            size=label_size,
            verticalalignment=label_verticalalignment,
            bbox={"facecolor": background_color, "alpha": 1, "linewidth": 0},
            zorder=len(values) + 10,
        )

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor(background_color)
    
    return fig, ax

# Initialize session state for caching
if 'last_station_id' not in st.session_state:
    st.session_state.last_station_id = None
if 'cached_dem' not in st.session_state:
    st.session_state.cached_dem = None
if 'cached_geometry' not in st.session_state:
    st.session_state.cached_geometry = None
if 'last_resolution' not in st.session_state:
    st.session_state.last_resolution = None
if 'error_count' not in st.session_state:
    st.session_state.error_count = 0

# Sidebar controls
st.sidebar.header("🎨 Map Settings")

# Input method
input_method = st.sidebar.radio(
    "Input Method",
    ["USGS Station ID", "Coordinates (Coming Soon)"],
    help="Choose how to define your area of interest"
)

if input_method == "USGS Station ID":
    station_id = st.sidebar.text_input(
        "Station ID",
        value="01031500",
        help="Enter a USGS station ID to get its watershed basin"
    )
else:
    st.sidebar.info("Coordinate-based input will be available in a future update")
    station_id = None

# DEM resolution
resolution = st.sidebar.slider(
    "DEM Resolution (meters)",
    min_value=10,
    max_value=1000,
    value=90,
    step=10,
    help="Higher resolution = more detail but slower processing. Try larger values (200-500m) if downloads fail."
)

# Warn about known problematic stations (after resolution is defined)
if input_method == "USGS Station ID" and station_id:
    LARGE_BASIN_STATIONS = {
        "09380000": "Colorado River (130,000 sq mi) - Use resolution ≥500m",
        "08279500": "Rio Grande (Very large basin) - Use resolution ≥500m",
    }
    
    if station_id in LARGE_BASIN_STATIONS:
        st.sidebar.warning(f"⚠️ {LARGE_BASIN_STATIONS[station_id]}")
        if resolution < 500:
            st.sidebar.error(f"❌ Resolution too fine for this basin! Set to ≥500m")

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Visual Style")

# Color scheme
color_scheme = st.sidebar.selectbox(
    "Color Scheme",
    ["terrain", "viridis", "plasma", "inferno", "magma", "cividis", "twilight", "black"],
    help="Colormap for the ridge lines"
)

# Color by
color_by = st.sidebar.radio(
    "Color By",
    ["elevation", "gradient"],
    help="'elevation' colors by actual height, 'gradient' colors by line position"
)

# Line width
linewidth = st.sidebar.slider(
    "Line Width",
    min_value=1,
    max_value=5,
    value=2,
    help="Thickness of ridge lines"
)

# Map label
add_label = st.sidebar.checkbox("Add Label", value=False)
if add_label:
    label_text = st.sidebar.text_input("Label Text", value="Ridge Map")
    label_size = st.sidebar.slider("Label Size", 20, 100, 60)
else:
    label_text = None
    label_size = 60

# Size scale
size_scale = st.sidebar.slider(
    "Figure Size",
    min_value=10,
    max_value=30,
    value=20,
    help="Larger = higher quality but more memory"
)

# Advanced options
with st.sidebar.expander("⚙️ Advanced Options"):
    force_refresh = st.checkbox(
        "Force Refresh Data",
        value=False,
        help="Clear cache and re-download data"
    )

# Main content area
if DEPENDENCIES_AVAILABLE and station_id:
    if st.button("🗺️ Generate Ridge Map", type="primary"):
        
        # Clear cache if force refresh or station changed
        if force_refresh or station_id != st.session_state.last_station_id:
            st.session_state.cached_dem = None
            st.session_state.cached_geometry = None
            st.session_state.last_station_id = station_id
            st.session_state.last_resolution = None
        
        # Check if we can use cached data
        use_cached = (
            st.session_state.cached_dem is not None and 
            st.session_state.cached_geometry is not None and
            st.session_state.last_resolution == resolution and
            station_id == st.session_state.last_station_id
        )
        
        try:
            # Get basin geometry
            if use_cached:
                st.info("ℹ️ Using cached data")
                geometry = st.session_state.cached_geometry
                dem = st.session_state.cached_dem
            else:
                # Fetch basin with error handling
                with st.spinner(f"Fetching basin geometry for station {station_id}..."):
                    try:
                        nldi = NLDI()
                        basins_gdf = nldi.get_basins(station_id)
                        
                        if basins_gdf is None or len(basins_gdf) == 0:
                            st.error(f"❌ No basin found for station {station_id}. Please check the station ID.")
                            st.stop()
                        
                        geometry = basins_gdf.geometry.iloc[0]
                        st.session_state.cached_geometry = geometry
                        
                        # Calculate basin area to warn users
                        try:
                            # Convert to appropriate projection for area calculation
                            from shapely.ops import transform
                            import pyproj
                            
                            # Calculate area in square kilometers
                            geod = pyproj.Geod(ellps='WGS84')
                            area_m2 = abs(geod.geometry_area_perimeter(geometry)[0])
                            area_km2 = area_m2 / 1_000_000
                            area_sq_miles = area_km2 * 0.386102
                            
                            st.success(f"✅ Found basin for station {station_id} (~{area_sq_miles:,.0f} sq mi)")
                            
                            # Warn if basin is very large
                            if area_sq_miles > 10000:
                                st.warning(f"⚠️ Large basin detected ({area_sq_miles:,.0f} sq mi). DEM download may take time or fail.")
                                if resolution < 500:
                                    st.error(f"❌ Resolution {resolution}m is too fine for a basin this large. Try ≥500m to avoid crashes.")
                                    st.info("💡 Click the button again with resolution ≥500m")
                                    st.stop()
                        except Exception:
                            # If area calculation fails, just continue
                            st.success(f"✅ Found basin for station {station_id}")
                        
                        
                    except Exception as e:
                        st.error(f"❌ Failed to fetch basin: {str(e)}")
                        st.warning("💡 Tips: Make sure the station ID is valid and the NLDI service is accessible.")
                        with st.expander("🔍 Full Error"):
                            st.code(traceback.format_exc())
                        st.stop()
                
                # Get DEM data with robust error handling
                with st.spinner(f"Downloading DEM data at {resolution}m resolution..."):
                    try:
                        dem = py3dep.get_dem(geometry, resolution)
                        
                        # Validate DEM
                        if dem is None:
                            raise ValueError("DEM download returned None")
                        
                        if dem.shape[0] == 0 or dem.shape[1] == 0:
                            raise ValueError("DEM has zero dimensions")
                        
                        # Check for all NaN
                        if np.all(np.isnan(dem.values)):
                            raise ValueError("DEM contains only NaN values")
                        
                        st.session_state.cached_dem = dem
                        st.session_state.last_resolution = resolution
                        st.session_state.error_count = 0  # Reset error counter on success
                        
                        st.success(f"✅ Downloaded DEM: {dem.shape[0]} x {dem.shape[1]} pixels")
                            
                    except Exception as e:
                        st.session_state.error_count += 1
                        error_msg = str(e)
                        
                        st.error(f"❌ Failed to download DEM: {error_msg}")
                        
                        # Provide helpful suggestions based on error
                        st.warning("💡 Troubleshooting suggestions:")
                        suggestions = [
                            "• Try a different resolution (200-500m often works better)",
                            "• The basin might be too large - try a different station",
                            "• Check your internet connection",
                            "• The USGS 3DEP service might be temporarily unavailable",
                            "• Try enabling 'Force Refresh Data' in Advanced Options"
                        ]
                        
                        if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                            suggestions.insert(0, "• The request timed out - try a larger resolution value")
                        
                        if "memory" in error_msg.lower():
                            suggestions.insert(0, "• Reduce resolution or figure size to use less memory")
                        
                        for suggestion in suggestions:
                            st.markdown(suggestion)
                        
                        with st.expander("🔍 Full Error Details"):
                            st.code(traceback.format_exc())
                        
                        # Suggest alternative stations if multiple failures
                        if st.session_state.error_count >= 2:
                            st.info("💡 Try these reliable stations: 01031500, 14211010, 09380000, 08279500")
                        
                        st.stop()
            
            # Create ridge map
            with st.spinner("Generating ridge map..."):
                try:
                    # Get colormap
                    if color_scheme == "black":
                        cmap = "black"
                    else:
                        cmap = plt.get_cmap(color_scheme)
                    
                    # Generate the plot
                    fig, ax = plot_ridges(
                        dem,
                        label=label_text,
                        line_color=cmap,
                        kind=color_by,
                        linewidth=linewidth,
                        label_size=label_size,
                        size_scale=size_scale
                    )
                    
                    # Display in Streamlit
                    st.pyplot(fig, use_container_width=True)
                    
                    # Close figure to free memory
                    plt.close(fig)
                    
                    # Download button
                    import io
                    
                    # Recreate figure for download
                    fig_download, _ = plot_ridges(
                        dem,
                        label=label_text,
                        line_color=cmap,
                        kind=color_by,
                        linewidth=linewidth,
                        label_size=label_size,
                        size_scale=size_scale
                    )
                    
                    buf = io.BytesIO()
                    fig_download.savefig(buf, format='png', bbox_inches='tight', dpi=300)
                    buf.seek(0)
                    plt.close(fig_download)
                    
                    st.download_button(
                        label="📥 Download High-Res PNG",
                        data=buf,
                        file_name=f"ridge_map_{station_id}.png",
                        mime="image/png"
                    )
                    
                    # Show metadata
                    with st.expander("📊 DEM Information"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Width", f"{dem.shape[1]} pixels")
                        with col2:
                            st.metric("Height", f"{dem.shape[0]} pixels")
                        with col3:
                            st.metric("Resolution", f"{resolution}m")
                        
                        st.write("**Elevation Statistics:**")
                        st.write(f"- Min: {float(dem.min()):.1f}m")
                        st.write(f"- Max: {float(dem.max()):.1f}m")
                        st.write(f"- Mean: {float(dem.mean()):.1f}m")
                        st.write(f"- Std: {float(dem.std()):.1f}m")
                        
                        # Show bounds
                        bounds = dem.rio.bounds()
                        st.write("**Geographic Bounds:**")
                        st.write(f"- West: {bounds[0]:.4f}°")
                        st.write(f"- South: {bounds[1]:.4f}°")
                        st.write(f"- East: {bounds[2]:.4f}°")
                        st.write(f"- North: {bounds[3]:.4f}°")
                
                except Exception as e:
                    st.error(f"❌ Error generating ridge map: {str(e)}")
                    with st.expander("🔍 Error Details"):
                        st.code(traceback.format_exc())
                    
        except Exception as e:
            # Catch-all for any unexpected errors
            st.error(f"❌ Unexpected error: {str(e)}")
            with st.expander("🔍 Full Error Details"):
                st.code(traceback.format_exc())
            st.warning("💡 Try refreshing the page or using a different station ID")
            
else:
    # Show example
    st.info("👆 Configure settings in the sidebar and click 'Generate Ridge Map' to create your visualization")
    
    # Show examples
    st.subheader("📸 Example Ridge Maps")
    st.markdown("""
    **Recommended Stations (tested and reliable):**
    - **01031500** - Penobscot River, Maine (~3,000 sq mi, use 90-200m resolution)
    - **14211010** - Columbia River, Oregon (Medium basin, use 200m resolution)
    
    **Large Basin Stations (require high resolution ≥500m):**
    - **09380000** - Colorado River at Lees Ferry, AZ (~130,000 sq mi, **MUST use ≥500m**)
    - **08279500** - Rio Grande, NM (Large basin, **MUST use ≥500m**)
    
    **Why do large basins need higher resolution values?**
    - Counterintuitively, a "higher resolution" number (500m vs 90m) means **lower detail**
    - 500m resolution = fewer pixels = smaller download = won't crash
    - 90m resolution = more pixels = huge download = crashes for large basins
    
    **Tips for Success:**
    - **Small/medium basins (<10,000 sq mi)**: Use 90-200m resolution for detail
    - **Large basins (>10,000 sq mi)**: **MUST use ≥500m** resolution to prevent crashes
    - App will automatically warn you if basin is too large for chosen resolution
    - If download fails, try **increasing** the resolution number (200→500→1000)
    - Some stations may not have elevation data available
    - Use "Force Refresh Data" if you encounter issues
    
    **Visual Styles:**
    - **terrain** colormap - Natural, earth-toned appearance
    - **viridis/plasma** - Modern, high-contrast gradients
    - **black** - Classic minimalist look
    - Try **"Color By: elevation"** for topographic accuracy or **"gradient"** for artistic effect
    """)

# Footer
st.markdown("---")
st.markdown("""
**About Ridge Maps**: A ridge map is a 2D visualization technique that represents 3D terrain by showing elevation 
contours as stacked lines. Each line represents an elevation profile, creating a distinctive "ridge" effect.

**Data Sources**: 
- Elevation data from [USGS 3DEP](https://www.usgs.gov/3d-elevation-program)
- Basin boundaries from [NLDI](https://labs.waterdata.usgs.gov/about-nldi/)

**Built with**: [HyRiver](https://docs.hyriver.io/) • [Streamlit](https://streamlit.io/)

**Note**: Data download reliability depends on USGS web services. If you experience crashes, try different stations or resolution settings.
""")
