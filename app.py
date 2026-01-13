import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import colors
from matplotlib.collections import LineCollection
from typing import Literal

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
    help="Higher resolution = more detail but slower processing"
)

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

# Main content area
if DEPENDENCIES_AVAILABLE and station_id:
    if st.button("🗺️ Generate Ridge Map", type="primary"):
        with st.spinner(f"Fetching basin geometry for station {station_id}..."):
            try:
                # Get basin geometry
                nldi = NLDI()
                geometry = nldi.get_basins(station_id).geometry.iloc[0]
                st.success(f"✅ Found basin for station {station_id}")
                
                # Get DEM data
                with st.spinner(f"Downloading DEM data at {resolution}m resolution..."):
                    dem = py3dep.get_dem(geometry, resolution)
                    st.success(f"✅ Downloaded DEM: {dem.shape[0]} x {dem.shape[1]} pixels")
                
                # Create ridge map
                with st.spinner("Generating ridge map..."):
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
                    
                    # Download button
                    import io
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
                    buf.seek(0)
                    
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
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                with st.expander("🔍 Error Details"):
                    import traceback
                    st.code(traceback.format_exc())
else:
    # Show example
    st.info("👆 Configure settings in the sidebar and click 'Generate Ridge Map' to create your visualization")
    
    # Show examples
    st.subheader("📸 Example Ridge Maps")
    st.markdown("""
    Try these USGS station IDs:
    - **01031500** - Penobscot River, Maine
    - **14211010** - Columbia River at The Dalles, Oregon
    - **09380000** - Colorado River, Arizona
    - **08279500** - Rio Grande, New Mexico
    
    **Tips:**
    - Lower resolution (90-200m) processes faster and works well for large basins
    - Higher resolution (10-30m) shows more detail for smaller areas
    - Try different color schemes - 'terrain' is great for natural look, 'viridis' for modern aesthetic
    - 'elevation' coloring shows actual terrain height, 'gradient' creates abstract patterns
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
""")
