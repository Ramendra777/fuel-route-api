import os
import csv
import json
import math
import requests
import numpy as np
from django.conf import settings
from scipy.spatial import KDTree

# Set of 50 US state codes
US_STATES = {
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
}

# Earth radius in miles
EARTH_RADIUS_MILES = 3958.8

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in miles between two points
    on the earth (specified in decimal degrees).
    """
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2.0)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0)**2
    c = 2.0 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_MILES * c

def haversine_vectorized(s_lat, s_lon, route_lats_rad, route_lons_rad):
    """
    Vectorized haversine: compute distances from a single point (s_lat, s_lon)
    to every point in route_lats_rad / route_lons_rad (already in radians).
    Returns a numpy array of distances in miles.
    """
    s_lat_r = math.radians(s_lat)
    s_lon_r = math.radians(s_lon)
    dlat = route_lats_rad - s_lat_r
    dlon = route_lons_rad - s_lon_r
    a = np.sin(dlat / 2.0)**2 + math.cos(s_lat_r) * np.cos(route_lats_rad) * np.sin(dlon / 2.0)**2
    return EARTH_RADIUS_MILES * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

# ── In-memory caches ───────────────────────────────────────────────────────────
_stations = None        # list of station dicts
_kdtree = None          # SciPy KDTree over station coords

_geocode_cache = {}     # query string -> (lat, lon)
_route_cache = {}       # (start_query, finish_query) -> full result dict
# ──────────────────────────────────────────────────────────────────────────────

def get_stations_and_tree():
    """
    Lazy-load fuel stations once at startup. Filters to US states, strips
    whitespace, deduplicates, geocodes via local DB, and builds a KDTree.
    """
    global _stations, _kdtree
    if _stations is not None:
        return _stations, _kdtree

    fuel_csv_path    = os.path.join(settings.BASE_DIR, 'fuel-prices-for-be-assessment.csv')
    uscities_csv_path = os.path.join(settings.BASE_DIR, 'uscities.csv')
    cache_json_path  = os.path.join(settings.BASE_DIR, 'geocoded_cities_cache.json')

    # 1. Load US cities database
    uscities = {}
    if os.path.exists(uscities_csv_path):
        with open(uscities_csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                city_key = (row['city_ascii'].lower().strip(), row['state_id'].upper().strip())
                uscities[city_key] = (float(row['lat']), float(row['lng']))

    # 2. Load geocoding cache
    cache = {}
    if os.path.exists(cache_json_path):
        with open(cache_json_path, 'r', encoding='utf-8') as f:
            try:
                raw_cache = json.load(f)
                for k, v in raw_cache.items():
                    city, state = k.split(',')
                    cache[(city.lower().strip(), state.lower().strip())] = v
            except Exception:
                pass

    # 3. Read and process fuel prices CSV
    loaded_stations = []
    seen_keys = set()

    if os.path.exists(fuel_csv_path):
        with open(fuel_csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                opis_id = int(row.get('OPIS Truckstop ID', 0))
                name    = row.get('Truckstop Name', '').strip()
                address = row.get('Address', '').strip()
                city    = row.get('City', '').strip()
                state   = row.get('State', '').strip().upper()

                # Filter out Canadian provinces; keep only 50 US states
                if state not in US_STATES:
                    continue

                try:
                    price = float(row.get('Retail Price', 0.0))
                except ValueError:
                    price = 0.0

                # Clean city name (strips trailing whitespace)
                cleaned_city  = city.lower().strip()
                cleaned_state = state.lower().strip()

                # Deduplicate by (City, State, Retail Price)
                dedup_key = (cleaned_city, cleaned_state, price)
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)

                # Lookup coordinates: simplemaps DB first, then geocoded cache
                lat, lng = None, None
                city_key = (cleaned_city, state)
                if city_key in uscities:
                    lat, lng = uscities[city_key]
                elif (cleaned_city, cleaned_state) in cache:
                    lat, lng = cache[(cleaned_city, cleaned_state)]

                if lat is not None and lng is not None:
                    loaded_stations.append({
                        'opis_id': opis_id,
                        'name':    name,
                        'address': address,
                        'city':    city,
                        'state':   state,
                        'price':   price,
                        'lat':     lat,
                        'lng':     lng,
                    })

    # 4. Build KDTree indexed by (lat, lng)
    coords   = np.array([[s['lat'], s['lng']] for s in loaded_stations]) if loaded_stations else np.empty((0, 2))
    _kdtree  = KDTree(coords)
    _stations = loaded_stations

    return _stations, _kdtree


def geocode_location(query):
    """
    Geocode a query string (e.g. "Chicago, IL") using OSM Nominatim.
    Results are cached in memory so repeated calls are instant.
    Returns (lat, lon) or None.
    """
    global _geocode_cache

    # Return cached result immediately if available
    key = query.strip().lower()
    if key in _geocode_cache:
        return _geocode_cache[key]

    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q":      query,
        "format": "json",
        "limit":  1,
    }
    headers = {
        "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if data:
                result = (float(data[0]['lat']), float(data[0]['lon']))
                _geocode_cache[key] = result
                return result
    except Exception:
        pass
    return None


def get_driving_route(lat1, lon1, lat2, lon2):
    """
    Call OSRM to get the route geometry, distance in miles, and duration in hours.
    Returns a dict with route info or None.
    """
    url = (
        f"http://router.project-osrm.org/route/v1/driving/"
        f"{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
    )
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            data = r.json()
            if data.get("routes"):
                route          = data["routes"][0]
                distance_miles = route["distance"] / 1609.344
                duration_hours = route["duration"] / 3600.0
                geometry       = route["geometry"]
                return {
                    "distance":    distance_miles,
                    "duration":    duration_hours,
                    "coordinates": geometry["coordinates"],  # list of [lon, lat]
                }
    except Exception:
        pass
    return None


def find_optimal_stops(route_length, stations):
    """
    Dynamic Programming solver.
    MPG = 10, Range = 500 miles.
    stations: list of dicts with 'distance' and 'price', sorted by distance along route.
    Returns (stops_list, total_cost).
    """
    if route_length <= 500:
        return [], 0.0

    n  = len(stations)
    dp = {}

    # Base cases: stations reachable directly from start (within first 500 miles)
    for i in range(n):
        s = stations[i]
        if s['distance'] <= 500:
            cost    = (s['distance'] / 10.0) * s['price'] + 1e-5  # tiny penalty per stop
            dp[i]   = (cost, -1)
        else:
            dp[i]   = (float('inf'), -1)

    # Transitions: from station j to station i
    for i in range(n):
        s_i = stations[i]
        for j in range(i):
            s_j  = stations[j]
            dist = s_i['distance'] - s_j['distance']
            if dist <= 500 and dp[j][0] != float('inf'):
                cost_transition = (dist / 10.0) * s_i['price'] + 1e-5
                total_cost      = dp[j][0] + cost_transition
                if total_cost < dp[i][0]:
                    dp[i] = (total_cost, j)

    # Find the best last station from which the destination is reachable
    best_cost         = float('inf')
    best_last_station = -1

    for i in range(n):
        s            = stations[i]
        dist_to_dest = route_length - s['distance']
        if dist_to_dest <= 500 and dp[i][0] != float('inf'):
            final_leg_cost = (dist_to_dest / 10.0) * s['price']
            total_cost     = dp[i][0] + final_leg_cost
            if total_cost < best_cost:
                best_cost         = total_cost
                best_last_station = i

    if best_last_station == -1:
        return None, float('inf')

    # Reconstruct stop sequence
    stops = []
    curr  = best_last_station
    while curr != -1:
        stops.append(stations[curr])
        curr = dp[curr][1]

    stops.reverse()
    return stops, best_cost


def optimize_fuel_route(start_query, finish_query):
    """
    Orchestrate route calculation and optimal fuel stop identification.
    Results are cached in memory — repeated identical requests return instantly.
    """
    global _route_cache

    # ── Cache hit: return immediately ─────────────────────────────────────────
    cache_key = (start_query.strip().lower(), finish_query.strip().lower())
    if cache_key in _route_cache:
        return _route_cache[cache_key]

    # ── Step 1: Geocode start and finish (each cached after first call) ───────
    start_coords = geocode_location(start_query)
    if not start_coords:
        return {"error": f"Could not geocode start location: {start_query}"}

    # FIX: removed time.sleep(1.0) — no forced 1-second pause
    finish_coords = geocode_location(finish_query)
    if not finish_coords:
        return {"error": f"Could not geocode finish location: {finish_query}"}

    # ── Step 2: Single OSRM call for full route geometry ─────────────────────
    lat_start, lon_start = start_coords
    lat_finish, lon_finish = finish_coords
    route_info = get_driving_route(lat_start, lon_start, lat_finish, lon_finish)
    if not route_info:
        return {"error": "Could not calculate driving route between locations"}

    route_coords = route_info["coordinates"]  # list of [lon, lat]
    route_length = route_info["distance"]

    # ── Step 3: Cumulative distances along route ──────────────────────────────
    cum_distances = [0.0]
    total_dist    = 0.0
    for i in range(1, len(route_coords)):
        lon1, lat1 = route_coords[i - 1]
        lon2, lat2 = route_coords[i]
        total_dist += haversine(lon1, lat1, lon2, lat2)
        cum_distances.append(total_dist)

    # ── Step 4: KDTree lookup for nearby stations ─────────────────────────────
    stations, tree = get_stations_and_tree()
    if not stations:
        return {"error": "Fuel station data is empty or not loaded"}

    # Subsample route points every ~2 miles to limit KDTree queries
    subsampled_indices = [0]
    last_dist = 0.0
    for i in range(1, len(route_coords)):
        if cum_distances[i] - last_dist >= 2.0 or i == len(route_coords) - 1:
            subsampled_indices.append(i)
            last_dist = cum_distances[i]

    nearby_station_indices = set()
    for idx in subsampled_indices:
        lon, lat = route_coords[idx]
        nearby_station_indices.update(tree.query_ball_point([lat, lon], r=0.15))

    # ── Step 5: Vectorized projection — station → closest route point ─────────
    # Pre-compute route coordinates as radians arrays (done once, outside station loop)
    route_arr      = np.array(route_coords)          # shape (N, 2): [lon, lat]
    route_lats_rad = np.radians(route_arr[:, 1])     # latitudes in radians
    route_lons_rad = np.radians(route_arr[:, 0])     # longitudes in radians
    cum_dist_arr   = np.array(cum_distances)

    route_stations = []
    for s_idx in nearby_station_indices:
        station = stations[s_idx]

        # Single vectorized call replaces the inner Python for-loop
        dists           = haversine_vectorized(station['lat'], station['lng'], route_lats_rad, route_lons_rad)
        closest_idx     = int(np.argmin(dists))
        min_dist        = float(dists[closest_idx])

        # Only include stations within 10 miles of the actual highway
        if min_dist <= 10.0:
            station_with_dist = station.copy()
            station_with_dist['distance']             = float(cum_dist_arr[closest_idx])
            station_with_dist['perpendicular_distance'] = min_dist
            route_stations.append(station_with_dist)

    # ── Step 6: Sort and deduplicate by route distance ────────────────────────
    route_stations.sort(key=lambda s: s['distance'])

    unique_dist_stations = []
    for rs in route_stations:
        if not unique_dist_stations or unique_dist_stations[-1]['distance'] != rs['distance']:
            unique_dist_stations.append(rs)
        else:
            if rs['price'] < unique_dist_stations[-1]['price']:
                unique_dist_stations[-1] = rs

    # ── Step 7: DP solver ─────────────────────────────────────────────────────
    stops, total_cost = find_optimal_stops(route_length, unique_dist_stations)

    if stops is None:
        return {
            "error": "The route is longer than 500 miles and no sequence of fuel stops within 500-mile ranges could be found."
        }

    # Reformat route coords [lon, lat] → [lat, lon] for Leaflet/JSON consumers
    map_route_coords = [[pt[1], pt[0]] for pt in route_coords]

    result = {
        "start":               start_query,
        "finish":              finish_query,
        "start_coords":        start_coords,
        "finish_coords":       finish_coords,
        "total_distance":      route_length,
        "total_duration_hours": route_info["duration"],
        "total_fuel_cost":     total_cost,
        "total_fuel_gallons":  route_length / 10.0,
        "fuel_stops":          stops,
        "route_coordinates":   map_route_coords,
    }

    # Cache full result for this (start, finish) pair
    _route_cache[cache_key] = result
    return result
