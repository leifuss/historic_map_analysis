#!/usr/bin/env python3
"""
Extract location names and distances from itinerary sections,
then create an interactive Leaflet map.
"""

import re
import json
from typing import List, Dict, Tuple
from pathlib import Path


# Known Iberian locations with approximate coordinates
KNOWN_LOCATIONS = {
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
    'coimbra': {'lat': 40.2033, 'lon': -8.4103, 'name': 'Coimbra'},
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
    'carthagène': {'lat': 37.6256, 'lon': -0.9962, 'name': 'Cartagena'},
    'cartagena': {'lat': 37.6256, 'lon': -0.9962, 'name': 'Cartagena'},
    'tortose': {'lat': 40.8125, 'lon': 0.5208, 'name': 'Tortosa'},
    'tortosa': {'lat': 40.8125, 'lon': 0.5208, 'name': 'Tortosa'},
}


def extract_distances_from_text(text: str) -> List[Dict]:
    """
    Extract distance information from text using pattern matching.

    Looks for patterns like:
    - "de X à Y, N milles"
    - "entre X et Y: N journées"
    - "X to Y is N miles"
    """
    routes = []

    # Pattern 1: "de X à Y" with distance
    # Example: "de Cordoue à Grenade, 20 milles"
    pattern1 = r'de\s+([A-ZÀ-ÿa-zà-ÿ\-\']+)\s+[àa]\s+([A-ZÀ-ÿa-zà-ÿ\-\']+)[\s,]+(\d+)\s*(milles?|lieues?|journées?|parasanges?)'

    # Pattern 2: "entre X et Y"
    pattern2 = r'entre\s+([A-ZÀ-ÿa-zà-ÿ\-\']+)\s+et\s+([A-ZÀ-ÿa-zà-ÿ\-\']+)[\s,:]+(\d+)\s*(milles?|lieues?|journées?|parasanges?)'

    # Pattern 3: Distance before locations
    pattern3 = r'(\d+)\s*(milles?|lieues?|journées?|parasanges?)\s+de\s+([A-ZÀ-ÿa-zà-ÿ\-\']+)\s+[àa]\s+([A-ZÀ-ÿa-zà-ÿ\-\']+)'

    for pattern in [pattern1, pattern2, pattern3]:
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

            routes.append({
                'from': from_loc.lower().strip(),
                'to': to_loc.lower().strip(),
                'distance': int(distance),
                'unit': unit.lower()
            })

    return routes


def match_location(location_name: str) -> Dict:
    """
    Try to match a location name to known coordinates.

    Args:
        location_name: Name to match

    Returns:
        Location dict with coordinates or None
    """
    location_lower = location_name.lower().strip()

    # Direct match
    if location_lower in KNOWN_LOCATIONS:
        return KNOWN_LOCATIONS[location_lower]

    # Partial match
    for known_name, coords in KNOWN_LOCATIONS.items():
        if known_name in location_lower or location_lower in known_name:
            return coords

    return None


def parse_itinerary_pages(json_file: str) -> List[Dict]:
    """
    Parse itinerary pages and extract routes with distances.

    Args:
        json_file: Path to itinerary pages JSON

    Returns:
        List of route dictionaries
    """
    if not Path(json_file).exists():
        print(f"Error: {json_file} not found")
        return []

    with open(json_file, 'r', encoding='utf-8') as f:
        pages_data = json.load(f)

    all_routes = []

    for page in pages_data:
        page_num = page['page_num']
        text = page['text']

        # Extract distances
        routes = extract_distances_from_text(text)

        for route in routes:
            # Try to geocode locations
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
                    'matched': True
                })
            else:
                # Store even if not matched for review
                all_routes.append({
                    'page': page_num,
                    'from_name': route['from'],
                    'to_name': route['to'],
                    'distance': route['distance'],
                    'unit': route['unit'],
                    'matched': False
                })

    return all_routes


def create_leaflet_map(routes: List[Dict], output_file: str = "iberian_routes_map.html"):
    """
    Create an interactive Leaflet map showing routes and distances.

    Args:
        routes: List of route dictionaries
        output_file: Output HTML file path
    """
    # Filter matched routes only
    matched_routes = [r for r in routes if r.get('matched', False)]

    if not matched_routes:
        print("No matched routes to map!")
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

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Al-Idrisi's Iberian Routes</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
        }}
        #map {{
            position: absolute;
            top: 0;
            bottom: 0;
            width: 100%;
        }}
        .info {{
            padding: 6px 8px;
            font: 14px/16px Arial, Helvetica, sans-serif;
            background: white;
            background: rgba(255,255,255,0.9);
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
            border-radius: 5px;
        }}
        .info h4 {{
            margin: 0 0 5px;
            color: #777;
        }}
        .legend {{
            line-height: 18px;
            color: #555;
        }}
        .legend i {{
            width: 18px;
            height: 18px;
            float: left;
            margin-right: 8px;
            opacity: 0.7;
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    <script>
        // Initialize map
        var map = L.map('map').setView([{avg_lat}, {avg_lon}], 6);

        // Add OpenStreetMap tiles
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
            maxZoom: 18
        }}).addTo(map);

        // Add custom info control
        var info = L.control();

        info.onAdd = function (map) {{
            this._div = L.DomUtil.create('div', 'info');
            this.update();
            return this._div;
        }};

        info.update = function (props) {{
            this._div.innerHTML = '<h4>Al-Idrisi\\'s Iberian Routes</h4>' +
                'Extracted from "Description de l\\'Afrique et de l\\'Espagne" (12th century)<br/>' +
                '<small>Hover over routes for details</small>';
        }};

        info.addTo(map);

        // Add locations (cities)
        var locations = {json.dumps({name: coords for name, coords in locations.items()})};

        for (var name in locations) {{
            var loc = locations[name];
            L.circleMarker([loc.lat, loc.lon], {{
                radius: 6,
                fillColor: "#ff7800",
                color: "#000",
                weight: 1,
                opacity: 1,
                fillOpacity: 0.8
            }})
            .bindPopup("<b>" + name + "</b>")
            .addTo(map);

            // Add label
            L.marker([loc.lat, loc.lon], {{
                icon: L.divIcon({{
                    className: 'label',
                    html: '<div style="font-size: 11px; font-weight: bold; color: #333; text-shadow: 1px 1px 1px white;">' + name + '</div>',
                    iconSize: [100, 20]
                }})
            }}).addTo(map);
        }}

        // Add routes (polylines)
        var routes = {json.dumps(matched_routes)};

        routes.forEach(function(route) {{
            var latlngs = [
                [route.from_lat, route.from_lon],
                [route.to_lat, route.to_lon]
            ];

            var polyline = L.polyline(latlngs, {{
                color: 'blue',
                weight: 3,
                opacity: 0.6
            }}).addTo(map);

            polyline.bindPopup(
                "<b>Route:</b> " + route.from_name + " → " + route.to_name + "<br/>" +
                "<b>Distance:</b> " + route.distance + " " + route.unit + "<br/>" +
                "<b>Source:</b> Page " + route.page
            );

            // Add distance label at midpoint
            var midLat = (route.from_lat + route.to_lat) / 2;
            var midLon = (route.from_lon + route.to_lon) / 2;

            L.marker([midLat, midLon], {{
                icon: L.divIcon({{
                    className: 'distance-label',
                    html: '<div style="font-size: 10px; background: rgba(255,255,255,0.8); padding: 2px 4px; border-radius: 3px;">' +
                          route.distance + ' ' + route.unit + '</div>',
                    iconSize: [50, 20]
                }})
            }}).addTo(map);
        }});

        // Add legend
        var legend = L.control({{position: 'bottomright'}});

        legend.onAdd = function (map) {{
            var div = L.DomUtil.create('div', 'info legend');
            div.innerHTML =
                '<i style="background:#ff7800; border-radius: 50%;"></i> Cities<br>' +
                '<i style="background:blue; width: 30px; height: 3px;"></i> Routes<br>' +
                '<small>Total routes: ' + routes.length + '</small>';
            return div;
        }};

        legend.addTo(map);
    </script>
</body>
</html>"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n✓ Map created: {output_file}")
    print(f"  - {len(matched_routes)} routes mapped")
    print(f"  - {len(locations)} unique locations")


def main():
    print("\n" + "="*70)
    print("EXTRACTING ITINERARIES AND CREATING MAP")
    print("="*70 + "\n")

    # Parse itinerary pages
    print("Step 1: Parsing itinerary pages...")
    routes = parse_itinerary_pages("itinerary_pages.json")

    if not routes:
        print("No routes found. Run search_itineraries.py first.")
        return

    print(f"Found {len(routes)} total route references")

    matched = [r for r in routes if r.get('matched', False)]
    unmatched = [r for r in routes if not r.get('matched', False)]

    print(f"  - {len(matched)} matched to coordinates")
    print(f"  - {len(unmatched)} unmatched")

    # Save all routes
    with open('extracted_routes.json', 'w', encoding='utf-8') as f:
        json.dump(routes, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Saved routes to: extracted_routes.json")

    # Create map
    if matched:
        print("\nStep 2: Creating Leaflet map...")
        create_leaflet_map(routes)

        # Print route list
        print("\n" + "="*70)
        print("MAPPED ROUTES:")
        print("="*70)
        for route in matched[:20]:  # Show first 20
            print(f"{route['from_name']:15} → {route['to_name']:15} : "
                  f"{route['distance']:3} {route['unit']:10} (page {route['page']})")

        if len(matched) > 20:
            print(f"... and {len(matched) - 20} more routes")
    else:
        print("\nNo matched routes to map.")

    # Print unmatched for reference
    if unmatched:
        print("\n" + "="*70)
        print("UNMATCHED LOCATIONS (need coordinates):")
        print("="*70)
        unmatched_locations = set()
        for route in unmatched:
            unmatched_locations.add(route['from_name'])
            unmatched_locations.add(route['to_name'])

        for loc in sorted(unmatched_locations)[:20]:
            print(f"  - {loc}")


if __name__ == '__main__':
    main()
