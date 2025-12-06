#!/usr/bin/env python3
"""
Parse itineraries from the Fourth Climate OCR results and create updated map.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple


# Expanded Iberian locations with coordinates
IBERIAN_LOCATIONS = {
    # Major cities
    'cordoue': {'lat': 37.8882, 'lon': -4.7794, 'name': 'Córdoba'},
    'cordoba': {'lat': 37.8882, 'lon': -4.7794, 'name': 'Córdoba'},
    'cordova': {'lat': 37.8882, 'lon': -4.7794, 'name': 'Córdoba'},
    'grenade': {'lat': 37.1773, 'lon': -3.5986, 'name': 'Granada'},
    'granada': {'lat': 37.1773, 'lon': -3.5986, 'name': 'Granada'},
    'séville': {'lat': 37.3891, 'lon': -5.9845, 'name': 'Sevilla'},
    'seville': {'lat': 37.3891, 'lon': -5.9845, 'name': 'Sevilla'},
    'sevilla': {'lat': 37.3891, 'lon': -5.9845, 'name': 'Sevilla'},
    'toledo': {'lat': 39.8628, 'lon': -4.0273, 'name': 'Toledo'},
    'tolède': {'lat': 39.8628, 'lon': -4.0273, 'name': 'Toledo'},
    'valence': {'lat': 39.4699, 'lon': -0.3763, 'name': 'Valencia'},
    'valencia': {'lat': 39.4699, 'lon': -0.3763, 'name': 'Valencia'},
    'barcelone': {'lat': 41.3851, 'lon': 2.1734, 'name': 'Barcelona'},
    'barcelona': {'lat': 41.3851, 'lon': 2.1734, 'name': 'Barcelona'},
    'lisbonne': {'lat': 38.7223, 'lon': -9.1393, 'name': 'Lisboa'},
    'lisbon': {'lat': 38.7223, 'lon': -9.1393, 'name': 'Lisboa'},
    'lisboa': {'lat': 38.7223, 'lon': -9.1393, 'name': 'Lisboa'},
    'madrid': {'lat': 40.4168, 'lon': -3.7038, 'name': 'Madrid'},
    'saragosse': {'lat': 41.6488, 'lon': -0.8891, 'name': 'Zaragoza'},
    'zaragoza': {'lat': 41.6488, 'lon': -0.8891, 'name': 'Zaragoza'},
    'malaga': {'lat': 36.7213, 'lon': -4.4214, 'name': 'Málaga'},
    'málaga': {'lat': 36.7213, 'lon': -4.4214, 'name': 'Málaga'},
    'almeria': {'lat': 36.8414, 'lon': -2.4637, 'name': 'Almería'},
    'almería': {'lat': 36.8414, 'lon': -2.4637, 'name': 'Almería'},
    'murcie': {'lat': 37.9922, 'lon': -1.1307, 'name': 'Murcia'},
    'murcia': {'lat': 37.9922, 'lon': -1.1307, 'name': 'Murcia'},
    'jaen': {'lat': 37.7796, 'lon': -3.7849, 'name': 'Jaén'},
    'badajoz': {'lat': 38.8794, 'lon': -6.9706, 'name': 'Badajoz'},
    'cáceres': {'lat': 39.4753, 'lon': -6.3724, 'name': 'Cáceres'},
    'mérida': {'lat': 38.9167, 'lon': -6.3433, 'name': 'Mérida'},
    'merida': {'lat': 38.9167, 'lon': -6.3433, 'name': 'Mérida'},
    'cadix': {'lat': 36.5297, 'lon': -6.2929, 'name': 'Cádiz'},
    'cadiz': {'lat': 36.5297, 'lon': -6.2929, 'name': 'Cádiz'},
    'cádiz': {'lat': 36.5297, 'lon': -6.2929, 'name': 'Cádiz'},
    'tarragone': {'lat': 41.1189, 'lon': 1.2445, 'name': 'Tarragona'},
    'tarragona': {'lat': 41.1189, 'lon': 1.2445, 'name': 'Tarragona'},
    'santarem': {'lat': 39.2369, 'lon': -8.6869, 'name': 'Santarém'},
    'santarém': {'lat': 39.2369, 'lon': -8.6869, 'name': 'Santarém'},
    'coimbra': {'lat': 40.2033, 'lon': -8.4103, 'name': 'Coimbra'},
    'coïmbra': {'lat': 40.2033, 'lon': -8.4103, 'name': 'Coimbra'},
    'porto': {'lat': 41.1579, 'lon': -8.6291, 'name': 'Porto'},
    'évora': {'lat': 38.5714, 'lon': -7.9093, 'name': 'Évora'},
    'evora': {'lat': 38.5714, 'lon': -7.9093, 'name': 'Évora'},
    'salamanque': {'lat': 40.9701, 'lon': -5.6635, 'name': 'Salamanca'},
    'salamanca': {'lat': 40.9701, 'lon': -5.6635, 'name': 'Salamanca'},
    'burgos': {'lat': 42.3439, 'lon': -3.6969, 'name': 'Burgos'},
    'león': {'lat': 42.5987, 'lon': -5.5671, 'name': 'León'},
    'leon': {'lat': 42.5987, 'lon': -5.5671, 'name': 'León'},
    'pampelune': {'lat': 42.8125, 'lon': -1.6458, 'name': 'Pamplona'},
    'pamplona': {'lat': 42.8125, 'lon': -1.6458, 'name': 'Pamplona'},
    'gibraltar': {'lat': 36.1408, 'lon': -5.3536, 'name': 'Gibraltar'},
    'algeciras': {'lat': 36.1408, 'lon': -5.4553, 'name': 'Algeciras'},
    'algésiras': {'lat': 36.1408, 'lon': -5.4553, 'name': 'Algeciras'},
    'carthagène': {'lat': 37.6256, 'lon': -0.9962, 'name': 'Cartagena'},
    'cartagena': {'lat': 37.6256, 'lon': -0.9962, 'name': 'Cartagena'},
    'tortose': {'lat': 40.8125, 'lon': 0.5208, 'name': 'Tortosa'},
    'tortosa': {'lat': 40.8125, 'lon': 0.5208, 'name': 'Tortosa'},
    'alicante': {'lat': 38.3452, 'lon': -0.4815, 'name': 'Alicante'},
    'alcira': {'lat': 39.1500, 'lon': -0.4333, 'name': 'Alzira'},
    'alzira': {'lat': 39.1500, 'lon': -0.4333, 'name': 'Alzira'},
    'denia': {'lat': 38.8408, 'lon': 0.1059, 'name': 'Denia'},
    'dénia': {'lat': 38.8408, 'lon': 0.1059, 'name': 'Denia'},
    'játiva': {'lat': 38.9900, 'lon': -0.5200, 'name': 'Xàtiva'},
    'xativa': {'lat': 38.9900, 'lon': -0.5200, 'name': 'Xàtiva'},
    'xàtiva': {'lat': 38.9900, 'lon': -0.5200, 'name': 'Xàtiva'},
    'cuenca': {'lat': 40.0703, 'lon': -2.1374, 'name': 'Cuenca'},
    'huesca': {'lat': 42.1401, 'lon': -0.4086, 'name': 'Huesca'},
    'lérida': {'lat': 41.6176, 'lon': 0.6200, 'name': 'Lleida'},
    'lleida': {'lat': 41.6176, 'lon': 0.6200, 'name': 'Lleida'},
    'gerona': {'lat': 41.9794, 'lon': 2.8214, 'name': 'Girona'},
    'girona': {'lat': 41.9794, 'lon': 2.8214, 'name': 'Girona'},
    'tarifa': {'lat': 36.0133, 'lon': -5.6067, 'name': 'Tarifa'},
    'ronda': {'lat': 36.7425, 'lon': -5.1672, 'name': 'Ronda'},
    'antequera': {'lat': 37.0193, 'lon': -4.5589, 'name': 'Antequera'},
    'ubeda': {'lat': 38.0156, 'lon': -3.3708, 'name': 'Úbeda'},
    'baeza': {'lat': 37.9927, 'lon': -3.4697, 'name': 'Baeza'},
    'écija': {'lat': 37.5428, 'lon': -5.0817, 'name': 'Écija'},
    'ecija': {'lat': 37.5428, 'lon': -5.0817, 'name': 'Écija'},
    'carmona': {'lat': 37.4708, 'lon': -5.6419, 'name': 'Carmona'},
}


def extract_route_patterns(text: str) -> List[Dict]:
    """
    Extract route information using multiple pattern matching strategies.

    Patterns to match:
    - "de X à Y, N milles/journées"
    - "entre X et Y, N milles"
    - "X est à N milles de Y"
    - "depuis X jusqu'à Y, N journées"
    """
    routes = []

    # Pattern 1: de X à Y, distance
    pattern1 = r'de\s+([A-ZÀ-ÿa-zà-ÿ\-\'\s]+?)\s+[àa]\s+([A-ZÀ-ÿa-zà-ÿ\-\'\s]+?)[\s,;:]+(\d+)\s*(milles?|lieues?|journées?|parasanges?)'

    # Pattern 2: entre X et Y, distance
    pattern2 = r'entre\s+([A-ZÀ-ÿa-zà-ÿ\-\'\s]+?)\s+et\s+([A-ZÀ-ÿa-zà-ÿ\-\'\s]+?)[\s,;:]+(\d+)\s*(milles?|lieues?|journées?|parasanges?)'

    # Pattern 3: distance de X à Y
    pattern3 = r'(\d+)\s*(milles?|lieues?|journées?|parasanges?)\s+de\s+([A-ZÀ-ÿa-zà-ÿ\-\'\s]+?)\s+[àa]\s+([A-ZÀ-ÿa-zà-ÿ\-\'\s]+?)[,\.\s]'

    # Pattern 4: depuis X jusqu'à Y, distance
    pattern4 = r'depuis\s+([A-ZÀ-ÿa-zà-ÿ\-\'\s]+?)\s+jusqu[\'']?[àa]\s+([A-ZÀ-ÿa-zà-ÿ\-\'\s]+?)[\s,;:]+(\d+)\s*(milles?|lieues?|journées?)'

    for pattern in [pattern1, pattern2, pattern3, pattern4]:
        matches = re.finditer(pattern, text, re.IGNORECASE)

        for match in matches:
            if pattern == pattern3:
                distance = match.group(1)
                unit = match.group(2)
                from_loc = match.group(3)
                to_loc = match.group(4)
            else:
                from_loc = match.group(1)
                to_loc = match.group(2)
                distance = match.group(3)
                unit = match.group(4)

            # Clean location names
            from_loc = from_loc.strip().rstrip(',;.')
            to_loc = to_loc.strip().rstrip(',;.')

            # Skip if names are too long (likely not cities)
            if len(from_loc) > 30 or len(to_loc) > 30:
                continue

            routes.append({
                'from': from_loc.lower(),
                'to': to_loc.lower(),
                'distance': int(distance),
                'unit': unit.lower()
            })

    return routes


def match_location(location_name: str) -> Dict:
    """Try to match location name to known coordinates."""
    location_lower = location_name.lower().strip()

    # Direct match
    if location_lower in IBERIAN_LOCATIONS:
        return IBERIAN_LOCATIONS[location_lower]

    # Partial match
    for known_name, coords in IBERIAN_LOCATIONS.items():
        if known_name in location_lower or location_lower in known_name:
            return coords

    return None


def parse_ocr_file(json_file: str) -> List[Dict]:
    """Parse OCR JSON file and extract routes."""

    if not Path(json_file).exists():
        print(f"Error: {json_file} not found")
        print("Run ocr_fourth_climate.py first!")
        return []

    print(f"\nLoading OCR data from: {json_file}")

    with open(json_file, 'r', encoding='utf-8') as f:
        pages_data = json.load(f)

    print(f"Loaded {len(pages_data)} pages")

    all_routes = []

    for page in pages_data:
        page_num = page['page_num']
        text = page.get('text', '')

        if not text:
            continue

        # Extract routes from this page
        routes = extract_route_patterns(text)

        for route in routes:
            # Try to geocode
            from_coords = match_location(route['from'])
            to_coords = match_location(route['to'])

            if from_coords and to_coords:
                all_routes.append({
                    'page': page_num,
                    'from_name': from_coords['name'],
                    'from_lat': from_coords['lat'],
                    'from_lon': from_coords['lon'],
                    'to_name': to_coords['name'],
                    'to_lat': to_coords['lat'],
                    'to_lon': to_coords['lon'],
                    'distance': route['distance'],
                    'unit': route['unit'],
                    'matched': True,
                    'source': 'al-Idrisi Fourth Climate'
                })
            else:
                # Save unmatched for manual review
                all_routes.append({
                    'page': page_num,
                    'from_name': route['from'],
                    'to_name': route['to'],
                    'distance': route['distance'],
                    'unit': route['unit'],
                    'matched': False
                })

    return all_routes


def create_updated_map(routes: List[Dict], output_file: str = "iberian_routes_map_updated.html"):
    """Create updated map with extracted routes."""

    # Filter matched routes
    matched_routes = [r for r in routes if r.get('matched', False)]

    if not matched_routes:
        print("\nNo matched routes to map!")
        return

    # Get all unique locations
    locations = {}
    for route in matched_routes:
        locations[route['from_name']] = {
            'lat': route['from_lat'],
            'lon': route['from_lon']
        }
        locations[route['to_name']] = {
            'lat': route['to_lat'],
            'lon': route['to_lon']
        }

    # Calculate center
    avg_lat = sum(loc['lat'] for loc in locations.values()) / len(locations)
    avg_lon = sum(loc['lon'] for loc in locations.values()) / len(locations)

    # Generate HTML (similar to demo but with real data)
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Al-Idrisi's Iberian Routes - From Original Text</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; padding: 0; font-family: 'Georgia', serif; }}
        #map {{ position: absolute; top: 60px; bottom: 0; width: 100%; }}
        .header {{
            position: absolute; top: 0; left: 0; right: 0; height: 60px;
            background: rgba(139, 90, 43, 0.95); color: white;
            display: flex; align-items: center; justify-content: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3); z-index: 1000;
        }}
        .header h1 {{ margin: 0; font-size: 24px; font-weight: normal; }}
        .header .subtitle {{ font-size: 14px; opacity: 0.9; margin-top: 3px; }}
        .info {{
            padding: 10px 15px; font: 14px/18px Arial; background: rgba(255,255,255,0.95);
            box-shadow: 0 0 15px rgba(0,0,0,0.2); border-radius: 8px; border-left: 4px solid #8b5a2b;
        }}
        .info h4 {{ margin: 0 0 8px; color: #8b5a2b; font-size: 16px; }}
        .info p {{ margin: 5px 0; font-size: 13px; color: #555; }}
        .legend {{ line-height: 22px; color: #555; background: white; padding: 10px; border-radius: 8px; box-shadow: 0 0 15px rgba(0,0,0,0.2); }}
        .legend h4 {{ margin: 0 0 8px; color: #8b5a2b; }}
        .legend i {{ width: 20px; height: 20px; float: left; margin-right: 8px; opacity: 0.8; }}
        .legend .line {{ width: 30px; height: 3px; }}
        .city-label {{
            font-size: 12px; font-weight: bold; color: #333;
            text-shadow: -1px -1px 0 white, 1px -1px 0 white, -1px 1px 0 white, 1px 1px 0 white, 2px 2px 3px rgba(0,0,0,0.3);
            font-family: 'Georgia', serif;
        }}
        .distance-label {{
            font-size: 11px; background: rgba(255,255,255,0.9); padding: 3px 6px;
            border-radius: 4px; border: 1px solid #8b5a2b; color: #8b5a2b; font-weight: 500;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>⁂ Al-Idrisi's Routes of Medieval Iberia - Extracted from Original Text ⁂</h1>
            <div class="subtitle">Fourth Climate Section - Pages 197-266 (1866 Edition)</div>
        </div>
    </div>
    <div id="map"></div>
    <script>
        var map = L.map('map').setView([{avg_lat}, {avg_lon}], 6);

        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
            maxZoom: 18
        }}).addTo(map);

        var info = L.control({{position: 'topright'}});
        info.onAdd = function (map) {{
            this._div = L.DomUtil.create('div', 'info');
            this._div.innerHTML =
                '<h4>Abu Abdullah al-Idrisi (1100-1165)</h4>' +
                '<p><em>Routes extracted from OCR of original text</em></p>' +
                '<p>These routes were found in the French translation<br/>' +
                'of al-Idrisi\\'s geographic work describing<br/>' +
                'the Iberian Peninsula in the 12th century</p>' +
                '<p style="font-size:11px; margin-top:10px; color:#888;">' +
                'Distances: milles (miles) or journées (days of travel)</p>';
            return this._div;
        }};
        info.addTo(map);

        var locations = {json.dumps({name: coords for name, coords in locations.items()})};

        for (var name in locations) {{
            var loc = locations[name];
            L.circleMarker([loc.lat, loc.lon], {{
                radius: 7, fillColor: "#d4af37", color: "#8b5a2b",
                weight: 2, opacity: 1, fillOpacity: 0.9
            }})
            .bindPopup("<b>" + name + "</b><br/><em>Medieval Iberian city</em>")
            .addTo(map);

            L.marker([loc.lat, loc.lon], {{
                icon: L.divIcon({{
                    className: 'label',
                    html: '<div class="city-label">' + name + '</div>',
                    iconSize: [100, 20], iconAnchor: [50, -10]
                }})
            }}).addTo(map);
        }}

        var routes = {json.dumps(matched_routes)};

        routes.forEach(function(route) {{
            var latlngs = [[route.from_lat, route.from_lon], [route.to_lat, route.to_lon]];

            var polyline = L.polyline(latlngs, {{
                color: '#8b5a2b', weight: 3, opacity: 0.7, dashArray: '10, 5'
            }}).addTo(map);

            polyline.bindPopup(
                "<div style='font-family: Georgia, serif;'>" +
                "<b style='color:#8b5a2b;'>Route:</b> " + route.from_name + " → " + route.to_name + "<br/>" +
                "<b style='color:#8b5a2b;'>Distance:</b> " + route.distance + " " + route.unit + "<br/>" +
                "<b style='color:#8b5a2b;'>Source:</b> Page " + route.page + "<br/>" +
                "<em style='font-size:11px; color:#666;'>From al-Idrisi's original text</em>" +
                "</div>"
            );

            var midLat = (route.from_lat + route.to_lat) / 2;
            var midLon = (route.from_lon + route.to_lon) / 2;
            L.marker([midLat, midLon], {{
                icon: L.divIcon({{
                    className: 'distance-label-marker',
                    html: '<div class="distance-label">' + route.distance + ' ' + route.unit + '</div>',
                    iconSize: [70, 20]
                }})
            }}).addTo(map);
        }});

        var legend = L.control({{position: 'bottomright'}});
        legend.onAdd = function (map) {{
            var div = L.DomUtil.create('div', 'legend');
            div.innerHTML =
                '<h4>Legend</h4>' +
                '<i style="background:#d4af37; border-radius: 50%; border: 2px solid #8b5a2b;"></i> Cities<br>' +
                '<i class="line" style="background:#8b5a2b;"></i> Trade Routes<br>' +
                '<hr style="margin: 8px 0;">' +
                '<div style="font-size:12px; color:#666;">' +
                '<strong>Routes:</strong> ' + routes.length + '<br>' +
                '<strong>Cities:</strong> ' + Object.keys(locations).length + '<br>' +
                '<strong>Source:</strong> OCR Extraction' +
                '</div>';
            return div;
        }};
        legend.addTo(map);
    </script>
</body>
</html>"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n✓ Updated map created: {output_file}")
    print(f"  - {len(matched_routes)} routes mapped")
    print(f"  - {len(locations)} unique locations")


def main():
    print("\n" + "="*70)
    print("PARSING ITINERARIES FROM FOURTH CLIMATE OCR")
    print("="*70 + "\n")

    # Parse OCR file
    ocr_file = "fourth_climate_pages_197-266.json"
    routes = parse_ocr_file(ocr_file)

    if not routes:
        print("No routes found. Make sure OCR file exists.")
        return

    matched = [r for r in routes if r.get('matched', False)]
    unmatched = [r for r in routes if not r.get('matched', False)]

    print(f"\n{'='*70}")
    print("EXTRACTION RESULTS")
    print(f"{'='*70}")
    print(f"Total route references: {len(routes)}")
    print(f"Matched to coordinates: {len(matched)}")
    print(f"Unmatched (need review): {len(unmatched)}")

    # Save all routes
    with open('extracted_routes_fourth_climate.json', 'w', encoding='utf-8') as f:
        json.dump(routes, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Saved routes to: extracted_routes_fourth_climate.json")

    # Create updated map if we have matches
    if matched:
        create_updated_map(routes)

        # Print route list
        print(f"\n{'='*70}")
        print("MAPPED ROUTES:")
        print(f"{'='*70}")
        for route in matched[:30]:
            print(f"Page {route['page']:3}: {route['from_name']:15} → {route['to_name']:15} : "
                  f"{route['distance']:3} {route['unit']:10}")

        if len(matched) > 30:
            print(f"... and {len(matched) - 30} more routes")

    # Print unmatched
    if unmatched:
        print(f"\n{'='*70}")
        print("UNMATCHED LOCATIONS (first 20):")
        print(f"{'='*70}")
        unmatched_locs = set()
        for route in unmatched:
            unmatched_locs.add(route['from_name'])
            unmatched_locs.add(route['to_name'])

        for loc in sorted(unmatched_locs)[:20]:
            print(f"  - {loc}")


if __name__ == '__main__':
    main()
