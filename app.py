"""
NHD Data using HyRiver - Streamlit App

This app demonstrates capabilities of HyRiver for accessing National Hydrography 
Dataset (NHDPlus MR and HR) and Watershed Boundary Dataset (WBD).

Authors: Taher Chegini (Purdue University), Dave Blodgett (USGS)
"""

import streamlit as st

# Check for required packages
try:
    import folium
    from streamlit_folium import st_folium
    import matplotlib
    import mapclassify
    from pygeohydro import NWIS, WBD
    from pynhd import HP3D, NLDI, GeoConnex, NHDPlusHR, WaterData
except ImportError as e:
    st.error(f"Missing required package: {e}")
    st.info("""
    Please install all required packages:
    
    ```bash
    pip install streamlit streamlit-folium pygeohydro pynhd folium matplotlib mapclassify
    ```
    
    Or using conda:
    
    ```bash
    conda install -c conda-forge streamlit streamlit-folium pygeohydro pynhd "folium>=0.12" matplotlib mapclassify
    ```
    """)
    st.stop()

st.set_page_config(page_title="NHD Data Explorer", layout="wide")

st.title("🌊 NHD Data using HyRiver")
st.markdown("""
This app demonstrates capabilities of HyRiver for accessing:
- **National Hydrography Dataset** (NHDPlus MR and HR)
- **Watershed Boundary Dataset** (WBD)

Using web services: NLDI, Water Data, 3DHP, GeoConnex, and NHDPlusHR
""")

# Sidebar for configuration
st.sidebar.header("Configuration")
site_id = st.sidebar.text_input("USGS Station ID", value="04074950", 
                                help="Enter a USGS station ID (e.g., 04074950 for Wolf River at Langlade, WI)")

# Initialize web services
@st.cache_resource
def init_services():
    """Initialize all web service classes"""
    return {
        'nldi': NLDI(),
        'nhd_mr': WaterData("nhdflowline_network"),
        'h4_wd': WaterData("wbd04"),
        'h4_wbd': WBD("huc4"),
        'nhd_hr': NHDPlusHR("flowline"),
        'nwis': NWIS(),
        'hp3d': HP3D("flowline"),
        'gcx': GeoConnex()
    }

services = init_services()

# Fix for Folium marker icon issue
def create_map(gdf, **kwargs):
    """Create a folium map with proper icon configuration"""
    m = gdf.explore(**kwargs)
    # Fix icon paths
    for child in m._children.values():
        if isinstance(child, folium.Marker):
            child.icon = folium.Icon()
    return m

# Helper function to add GeoJson with proper styling
def add_geojson_layer(m, gdf, name, style_function=None, tooltip=None):
    """Add GeoJSON layer to map with optional styling"""
    if tooltip:
        folium.GeoJson(
            gdf,
            name=name,
            tooltip=tooltip,
            marker=folium.CircleMarker(radius=5, fill=True)
        ).add_to(m)
    elif style_function:
        folium.GeoJson(
            gdf,
            name=name,
            style_function=style_function
        ).add_to(m)
    else:
        folium.GeoJson(gdf, name=name).add_to(m)
    return m

# Main content tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Station & Network", 
    "Basin & MR Flowlines", 
    "HR Flowlines",
    "3DHP Flowlines",
    "GeoConnex Data",
    "Watershed Boundaries"
])

# Tab 1: Station and upstream network
with tab1:
    st.header("Station and Upstream Network")
    
    if st.button("Load Station and Network", key="tab1_load"):
        with st.spinner("Fetching station and upstream network data..."):
            try:
                site_feature = services['nldi'].getfeature_byid("nwissite", f"USGS-{site_id}")
                upstream_network = services['nldi'].navigate_byid(
                    "nwissite", f"USGS-{site_id}", "upstreamMain", "flowlines", distance=9999
                )
                
                st.session_state['site_feature'] = site_feature
                st.session_state['upstream_network'] = upstream_network
                st.session_state['tab1_loaded'] = True
                
                st.success(f"Loaded upstream network for station {site_id}")
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    
    # Display map if data is loaded
    if st.session_state.get('tab1_loaded', False):
        try:
            site_feature = st.session_state['site_feature']
            upstream_network = st.session_state['upstream_network']
            
            # Create map
            m = folium.Map(
                location=[upstream_network.geometry.centroid.y.mean(), 
                         upstream_network.geometry.centroid.x.mean()],
                zoom_start=10
            )
            
            # Add upstream network
            folium.GeoJson(
                upstream_network,
                name="Upstream Network",
                style_function=lambda x: {"color": "blue", "weight": 2}
            ).add_to(m)
            
            # Add station marker
            site_coords = [site_feature.geometry.y.iloc[0], site_feature.geometry.x.iloc[0]]
            folium.CircleMarker(
                location=site_coords,
                radius=8,
                popup=f"Station: {site_feature['identifier'].iloc[0]}",
                tooltip=f"Station: {site_feature['identifier'].iloc[0]}",
                color='red',
                fill=True,
                fillColor='red',
                fillOpacity=0.7
            ).add_to(m)
            
            folium.LayerControl().add_to(m)
            
            st_folium(m, width=1000, height=600, key="map_tab1")
            
        except Exception as e:
            st.error(f"Error displaying map: {str(e)}")

# Tab 2: Basin and MR flowlines
with tab2:
    st.header("Basin and NHDPlus MR Flowlines")
    
    if st.button("Load Basin and MR Flowlines", key="tab2_load"):
        with st.spinner("Fetching basin and flowline data..."):
            try:
                basin = services['nldi'].get_basins(site_id)
                subset = services['nhd_mr'].bygeom(basin.geometry.iloc[0], basin.crs)
                
                st.session_state['basin'] = basin
                st.session_state['subset'] = subset
                st.session_state['tab2_loaded'] = True
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Basin Area (km²)", f"{basin.to_crs(5070).area.iloc[0] * 1e-6:.2f}")
                with col2:
                    st.metric("Number of MR Flowlines", len(subset))
                
                st.success("Basin and MR flowlines loaded successfully!")
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    
    # Display map if data is loaded
    if st.session_state.get('tab2_loaded', False):
        try:
            basin = st.session_state['basin']
            subset = st.session_state['subset']
            
            # Create map
            m = folium.Map(
                location=[basin.geometry.centroid.y.iloc[0], 
                         basin.geometry.centroid.x.iloc[0]],
                zoom_start=10
            )
            
            # Add basin
            folium.GeoJson(
                basin,
                name="Basin",
                style_function=lambda x: {"fillColor": "gray", "color": "black", "weight": 2, "fillOpacity": 0.3}
            ).add_to(m)
            
            # Add flowlines
            folium.GeoJson(
                subset,
                name="MR Flowlines",
                style_function=lambda x: {"color": "blue", "weight": 2}
            ).add_to(m)
            
            folium.LayerControl().add_to(m)
            
            st_folium(m, width=1000, height=600, key="map_tab2")
            
        except Exception as e:
            st.error(f"Error displaying map: {str(e)}")

# Tab 3: HR Flowlines
with tab3:
    st.header("NHDPlus HR Flowlines")
    
    if 'basin' not in st.session_state:
        st.warning("Please load the basin data in Tab 2 first.")
    else:
        if st.button("Load HR Flowlines", key="tab3_load"):
            with st.spinner("Fetching high-resolution flowlines..."):
                try:
                    basin = st.session_state['basin']
                    flw_hr = services['nhd_hr'].bygeom(basin.geometry.iloc[0], basin.crs)
                    st.session_state['flw_hr'] = flw_hr
                    st.session_state['tab3_loaded'] = True
                    
                    st.metric("Number of HR Flowlines", len(flw_hr))
                    st.success("HR flowlines are more detailed than MR flowlines!")
                    
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
        
        # Display map if data is loaded
        if st.session_state.get('tab3_loaded', False):
            try:
                basin = st.session_state['basin']
                flw_hr = st.session_state['flw_hr']
                
                # Create map
                m = folium.Map(
                    location=[basin.geometry.centroid.y.iloc[0], 
                             basin.geometry.centroid.x.iloc[0]],
                    zoom_start=10
                )
                
                # Add basin
                folium.GeoJson(
                    basin,
                    name="Basin",
                    style_function=lambda x: {"fillColor": "gray", "color": "black", "weight": 2, "fillOpacity": 0.3}
                ).add_to(m)
                
                # Add HR flowlines
                folium.GeoJson(
                    flw_hr,
                    name="HR Flowlines",
                    style_function=lambda x: {"color": "blue", "weight": 2}
                ).add_to(m)
                
                folium.LayerControl().add_to(m)
                
                st_folium(m, width=1000, height=600, key="map_tab3")
                
            except Exception as e:
                st.error(f"Error displaying map: {str(e)}")
        
        # Station matching section
        st.subheader("Find Matching HR Flowline for Station")
        if st.button("Match Station to HR Flowline", key="tab3_match"):
            with st.spinner("Finding matching flowline..."):
                try:
                    site_info = services['nwis'].get_info({"site": site_id}, expanded=True)
                    sqmi_to_sqkm = 2.59
                    area_sqkm = site_info["drain_area_va"].iloc[0] * sqmi_to_sqkm
                    
                    potential_matches = services['nhd_hr'].bygeom(
                        site_info.geometry.iloc[0], 
                        site_info.crs, 
                        distance=2000
                    )
                    
                    match = potential_matches.iloc[
                        [potential_matches.totdasqkm.sub(area_sqkm).abs().idxmin()]
                    ]
                    
                    st.session_state['match'] = match
                    st.session_state['site_info'] = site_info
                    st.session_state['area_sqkm'] = area_sqkm
                    st.session_state['tab3_match_loaded'] = True
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Station Drainage Area (km²)", f"{area_sqkm:.2f}")
                    with col2:
                        st.metric("Matched Flowline Drainage Area (km²)", 
                                f"{match.totdasqkm.iloc[0]:.2f}")
                    
                except Exception as e:
                    st.error(f"Error matching flowline: {str(e)}")
        
        # Display match map if loaded
        if st.session_state.get('tab3_match_loaded', False):
            try:
                match = st.session_state['match']
                site_info = st.session_state['site_info']
                
                # Create map
                m = folium.Map(
                    location=[match.geometry.centroid.y.iloc[0], 
                             match.geometry.centroid.x.iloc[0]],
                    zoom_start=12
                )
                
                # Add matched flowline
                folium.GeoJson(
                    match,
                    name="Matched Flowline",
                    style_function=lambda x: {"color": "blue", "weight": 3}
                ).add_to(m)
                
                # Add station marker
                site_coords = [site_info.geometry.y.iloc[0], site_info.geometry.x.iloc[0]]
                folium.CircleMarker(
                    location=site_coords,
                    radius=8,
                    popup=f"Station: {site_info['site_no'].iloc[0]}",
                    tooltip=f"Station: {site_info['site_no'].iloc[0]}",
                    color='red',
                    fill=True,
                    fillColor='red',
                    fillOpacity=0.7
                ).add_to(m)
                
                folium.LayerControl().add_to(m)
                
                st_folium(m, width=1000, height=600, key="map_tab3_match")
                
            except Exception as e:
                st.error(f"Error displaying map: {str(e)}")

# Tab 4: 3DHP Flowlines
with tab4:
    st.header("3DHP (3D Hydrography Program) Flowlines")
    
    if 'basin' not in st.session_state:
        st.warning("Please load the basin data in Tab 2 first.")
    else:
        if st.button("Load 3DHP Flowlines", key="tab4_load"):
            with st.spinner("Fetching 3DHP flowlines..."):
                try:
                    basin = st.session_state['basin']
                    flw_3dhp = services['hp3d'].bygeom(basin.union_all(), basin.crs)
                    st.session_state['flw_3dhp'] = flw_3dhp
                    st.session_state['tab4_loaded'] = True
                    
                    st.metric("Number of 3DHP Flowlines", len(flw_3dhp))
                    st.info("Blue: 3DHP flowlines, Red: NHDPlus HR flowlines")
                    st.success("Some flowlines in 3DHP are not in NHDPlus HR!")
                    
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
        
        # Display map if data is loaded
        if st.session_state.get('tab4_loaded', False):
            try:
                basin = st.session_state['basin']
                flw_3dhp = st.session_state['flw_3dhp']
                
                # Create comparison map
                m = folium.Map(
                    location=[basin.geometry.centroid.y.iloc[0], 
                             basin.geometry.centroid.x.iloc[0]],
                    zoom_start=10
                )
                
                # Add basin
                folium.GeoJson(
                    basin,
                    name="Basin",
                    style_function=lambda x: {"fillColor": "gray", "color": "black", "weight": 2, "fillOpacity": 0.3}
                ).add_to(m)
                
                # Add 3DHP flowlines
                folium.GeoJson(
                    flw_3dhp,
                    name="3DHP Flowlines",
                    style_function=lambda x: {"color": "blue", "weight": 2}
                ).add_to(m)
                
                # Add HR flowlines if available
                if 'flw_hr' in st.session_state:
                    folium.GeoJson(
                        st.session_state['flw_hr'],
                        name="HR Flowlines",
                        style_function=lambda x: {"color": "red", "weight": 2}
                    ).add_to(m)
                
                folium.LayerControl().add_to(m)
                
                st_folium(m, width=1000, height=600, key="map_tab4")
                
            except Exception as e:
                st.error(f"Error displaying map: {str(e)}")

# Tab 5: GeoConnex Data
with tab5:
    st.header("GeoConnex Reference Data")
    
    st.markdown("""
    GeoConnex provides access to community-contributed reference datasets including:
    - Stream gages
    - Mainstem rivers
    - Dams, aquifers, and more
    """)
    
    # Show available layers
    with st.expander("Available GeoConnex Layers"):
        st.code(str(services['gcx']))
    
    if 'basin' not in st.session_state:
        st.warning("Please load the basin data in Tab 2 first.")
    else:
        if st.button("Load GeoConnex Data", key="tab5_load"):
            with st.spinner("Fetching GeoConnex data..."):
                try:
                    basin = st.session_state['basin']
                    bounds = basin.to_crs(4326).total_bounds
                    
                    # Get gages
                    services['gcx'].item = "gages"
                    wolf_gages = services['gcx'].bybox(bounds)
                    wolf_gages = wolf_gages[wolf_gages.within(basin.union_all())]
                    
                    # Get mainstems
                    services['gcx'].item = "mainstems"
                    wolf_mainstems = services['gcx'].bybox(bounds)
                    wolf_mainstems = wolf_mainstems[wolf_mainstems.intersects(basin.union_all())]
                    
                    st.session_state['wolf_gages'] = wolf_gages
                    st.session_state['wolf_mainstems'] = wolf_mainstems
                    st.session_state['tab5_loaded'] = True
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Number of Gages", len(wolf_gages))
                    with col2:
                        st.metric("Number of Mainstems", len(wolf_mainstems))
                    
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
        
        # Display map if data is loaded
        if st.session_state.get('tab5_loaded', False):
            try:
                basin = st.session_state['basin']
                wolf_gages = st.session_state['wolf_gages']
                wolf_mainstems = st.session_state['wolf_mainstems']
                
                # Create map
                m = folium.Map(
                    location=[basin.geometry.centroid.y.iloc[0], 
                             basin.geometry.centroid.x.iloc[0]],
                    zoom_start=10
                )
                
                # Add basin
                folium.GeoJson(
                    basin,
                    name="Basin",
                    style_function=lambda x: {"fillColor": "gray", "color": "black", "weight": 2, "fillOpacity": 0.3}
                ).add_to(m)
                
                # Add mainstems
                folium.GeoJson(
                    wolf_mainstems,
                    name="Mainstems",
                    style_function=lambda x: {"color": "blue", "weight": 3}
                ).add_to(m)
                
                # Add gage markers
                for idx, gage in wolf_gages.iterrows():
                    folium.CircleMarker(
                        location=[gage.geometry.y, gage.geometry.x],
                        radius=6,
                        popup=f"Gage: {gage.get('provider_id', 'N/A')}",
                        tooltip=f"Gage: {gage.get('provider_id', 'N/A')}",
                        color='red',
                        fill=True,
                        fillColor='red',
                        fillOpacity=0.7
                    ).add_to(m)
                
                folium.LayerControl().add_to(m)
                
                st_folium(m, width=1000, height=600, key="map_tab5")
                
            except Exception as e:
                st.error(f"Error displaying map: {str(e)}")

# Tab 6: Watershed Boundaries
with tab6:
    st.header("Watershed Boundary Dataset (WBD)")
    
    if 'basin' not in st.session_state:
        st.warning("Please load the basin data in Tab 2 first.")
    else:
        if st.button("Load HUC4 Boundaries", key="tab6_load"):
            with st.spinner("Fetching watershed boundaries..."):
                try:
                    basin = st.session_state['basin']
                    wolf_huc4 = services['h4_wd'].bygeom(basin.union_all(), basin.crs)
                    st.session_state['wolf_huc4'] = wolf_huc4
                    st.session_state['tab6_loaded'] = True
                    
                    st.success("HUC4 boundaries loaded successfully!")
                    
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
        
        # Display map if data is loaded
        if st.session_state.get('tab6_loaded', False):
            try:
                basin = st.session_state['basin']
                wolf_huc4 = st.session_state['wolf_huc4']
                
                # Create map
                m = folium.Map(
                    location=[wolf_huc4.geometry.centroid.y.mean(), 
                             wolf_huc4.geometry.centroid.x.mean()],
                    zoom_start=9
                )
                
                # Add HUC4
                folium.GeoJson(
                    wolf_huc4,
                    name="HUC4",
                    style_function=lambda x: {"fillColor": "lightgray", "color": "black", "weight": 2, "fillOpacity": 0.4}
                ).add_to(m)
                
                # Add station basin
                folium.GeoJson(
                    basin,
                    name="Station Basin",
                    style_function=lambda x: {"fillColor": "blue", "color": "blue", "weight": 2, "fillOpacity": 0.3}
                ).add_to(m)
                
                folium.LayerControl().add_to(m)
                
                st_folium(m, width=1000, height=600, key="map_tab6")
                
            except Exception as e:
                st.error(f"Error displaying map: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
**About this app:** This Streamlit app is based on the [HyRiver nhd_demo notebook](https://docs.hyriver.io/examples/notebooks/nhd_demo.html).
HyRiver is a suite of Python packages for retrieving geospatial/temporal data from various web services.

**Authors:** Taher Chegini (Purdue University), Dave Blodgett (USGS)

**Required packages:**
```bash
pip install streamlit streamlit-folium pygeohydro pynhd folium matplotlib mapclassify geopandas
```
""")
