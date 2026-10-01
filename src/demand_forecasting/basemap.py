"""Generate the HCM street-map background with Folium; no API key is needed."""

from functools import lru_cache

import folium
from branca.element import MacroElement, Template


HCM_CENTER = (10.7769, 106.7009)
# A viewing envelope around HCM, not an administrative boundary.
HCM_VIEW = {"south": 10.25, "west": 106.20, "north": 11.70, "east": 107.80}
STREET_TILES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Street_Map/MapServer/tile/{z}/{y}/{x}"
)
ATTRIBUTION = (
    'Tiles &copy; <a href="https://www.esri.com/" target="_blank" '
    'rel="noopener noreferrer">Esri</a> &mdash; Esri, HERE, Garmin, USGS, '
    "Intermap, increment P, NRCAN, METI, TomTom, GIS User Community"
)


@lru_cache(maxsize=1)
def render_hcm_basemap() -> str:
    """Return a cached Leaflet page used behind the existing SVG H3 layer.

    Folium generates the page locally. The browser loads only the visible
    street tiles; Python never downloads a whole city or the whole world.
    Pan/zoom are owned by the dashboard so both layers share one viewport.
    """
    basemap = folium.Map(
        location=HCM_CENTER,
        tiles=None,
        zoom_start=11,
        min_zoom=9,
        max_zoom=16,
        zoom_control=False,
        attribution_control=True,
        dragging=False,
        scroll_wheel_zoom=False,
        double_click_zoom=False,
        touch_zoom=False,
        box_zoom=False,
        keyboard=False,
        zoom_animation=False,
        fade_animation=False,
        marker_zoom_animation=False,
    )
    # The background needs only Leaflet, not Folium's optional marker/UI assets.
    basemap.default_js = [
        ("leaflet", "https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js")
    ]
    basemap.default_css = [
        ("leaflet_css", "https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css")
    ]
    tiles = folium.TileLayer(
        tiles=STREET_TILES,
        attr=ATTRIBUTION,
        name="TP.HCM · đường phố",
        min_zoom=9,
        max_zoom=16,
        no_wrap=True,
        control=False,
        keep_buffer=2,
        update_when_idle=True,
    ).add_to(basemap)
    # Keep the attribution outside the SVG overlay, where its link is usable.
    basemap.get_root().header.add_child(folium.Element("""
        <style>
          .leaflet-control-attribution { display: none; }
          .leaflet-container { background: #dce7ea; }
        </style>
    """))
    bridge = MacroElement()
    bridge.tile_layer_name = tiles.get_name()
    bridge.view = HCM_VIEW
    bridge._template = Template("""
        {% macro script(this, kwargs) %}
        (() => {
          const map = {{ this._parent.get_name() }};
          const origin = window.location.origin;
          const report = (type) => window.parent.postMessage({ type }, origin);
          const tileLayer = {{ this.tile_layer_name }};
          let tileErrors = 0;
          tileLayer.on('tileerror', () => {
            tileErrors += 1;
            if (tileErrors >= 3) report('hcm-basemap-unavailable');
          });
          tileLayer.on('tileload', () => {
            tileErrors = 0;
            report('hcm-basemap-loaded');
          });
          window.addEventListener('message', (event) => {
            if (event.source !== window.parent || event.origin !== origin) return;
            const view = event.data;
            if (!view) return;
            if (view.type === 'hcm-basemap-request-ready') {
              report('hcm-basemap-ready');
              return;
            }
            if (view.type !== 'hcm-basemap-view') return;
            const { latitude, longitude, zoom } = view;
            if (![latitude, longitude, zoom].every(Number.isFinite)) return;
            if (latitude < {{ this.view.south }} || latitude > {{ this.view.north }} || longitude < {{ this.view.west }} || longitude > {{ this.view.east }}) return;
            if (!Number.isInteger(zoom) || zoom < 9 || zoom > 16) return;
            map.invalidateSize({ pan: false, animate: false });
            map.setView([latitude, longitude], zoom, { animate: false });
          });
          window.addEventListener('resize', () => map.invalidateSize({ pan: false }));
          report('hcm-basemap-ready');
        })();
        {% endmacro %}
    """)
    bridge.add_to(basemap)
    return basemap.get_root().render()
