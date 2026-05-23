import os
import csv
import json
import math
import requests
import time
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
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2.0)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0)**2
    c = 2.0 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_MILES * c

# Globals for lazy loading
_stations = None
_kdtree = None

def get_stations_and_tree():
    """
    Lazy load and geocode the fuel stations, filter by US states,
    deduplicate, and build a SciPy KDTree for spatial indexing.
    """
    global _stations, _kdtree
    if _stations is not None:
        return _stations, _kdtree
    
    fuel_csv_path = os.path.join(settings.BASE_DIR, 'fuel-prices-for-be-assessment.csv')
    uscities_csv_path = os.path.join(settings.BASE_DIR, 'uscities.csv')
    cache_json_path = os.path.join(settings.BASE_DIR, 'geocoded_cities_cache.json')
    
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
                
    # 3. Read and process fuel prices
    loaded_stations = []
    seen_keys = set() # For deduplication
    
    if os.path.exists(fuel_csv_path):
        with open(fuel_csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                opis_id = int(row.get('OPIS Truckstop ID', 0))
                name = row.get('Truckstop Name', '').strip()
                address = row.get('Address', '').strip()
                city = row.get('City', '').strip()
                state = row.get('State', '').strip().upper()
                
                # Filter out Canadian provinces and keep only 50 US States
                if state not in US_STATES:
                    continue
                
                # Parse Retail Price
                try:
                    price = float(row.get('Retail Price', 0.0))
                except ValueError:
                    price = 0.0
                    
                # Clean city name (stripping trailing whitespace)
                cleaned_city = city.lower().strip()
                cleaned_state = state.lower().strip()
                
                # Deduplication by (City, State, Retail Price)
                dedup_key = (cleaned_city, cleaned_state, price)
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)
                
                # Lookup coordinates
                lat, lng = None, None
                # First try simplemaps
                city_key = (cleaned_city, state)
                if city_key in uscities:
                    lat, lng = uscities[city_key]
                # Then try geocoded cache
                elif (cleaned_city, cleaned_state) in cache:
                    lat, lng = cache[(cleaned_city, cleaned_state)]
                    
                if lat is not None and lng is not None:
                    loaded_stations.append({
                        'opis_id': opis_id,
                        'name': name,
                        'address': address,
                        'city': city,
                        'state': state,
                        'price': price,
                        'lat': lat,
                        'lng': lng
                    })
                    
    # Build KDTree using (lat, lng)
    coords = np.array([[s['lat'], s['lng']] for s in loaded_stations]) if loaded_stations else np.empty((0, 2))
    _kdtree = KDTree(coords)
    _stations = loaded_stations
    
    return _stations, _kdtree

def geocode_location(query):
    """
    Geocode a query string (e.g. "Chicago, IL") using OSM Nominatim.
    Returns (lat, lon) or None.
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "json",
        "limit": 1
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    }
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if data:
                return float(data[0]['lat']), float(data[0]['lon'])
    except Exception:
        pass
    return None

def get_driving_route(lat1, lon1, lat2, lon2):
    """
    Call OSRM to get the route geometry, distance in miles, and duration in hours.
    Returns a dict with route info or None.
    """
    url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            data = r.json()
            if data.get("routes"):
                route = data["routes"][0]
                distance_miles = route["distance"] / 1609.344
                duration_hours = route["duration"] / 3600.0
                geometry = route["geometry"] # GeoJSON Linestring
                return {
                    "distance": distance_miles,
                    "duration": duration_hours,
                    "coordinates": geometry["coordinates"] # List of [lon, lat]
                }
    except Exception:
        pass
    return None

def find_optimal_stops(route_length, stations):
    """
    Find optimal fuel stop locations and calculate total cost.
    MPG = 10, Range = 500 miles.
    stations is a list of dicts: {'distance': float, 'price': float, 'name': str, ...}
    sorted by distance along the route.
    """
    if route_length <= 500:
        return [], 0.0
        
    n = len(stations)
    # dp[i] = (min_cost, parent_index) to reach station i and fill up the tank
    dp = {}
    
    # Base cases: stations reachable directly from start
    for i in range(n):
        s = stations[i]
        if s['distance'] <= 500:
            # Add a tiny stop penalty (1e-5) to break ties in favor of fewer stops
            cost = (s['distance'] / 10.0) * s['price'] + 1e-5
            dp[i] = (cost, -1)
        else:
            dp[i] = (float('inf'), -1)
            
    # Transitions
    for i in range(n):
        s_i = stations[i]
        for j in range(i):
            s_j = stations[j]
            dist = s_i['distance'] - s_j['distance']
            if dist <= 500 and dp[j][0] != float('inf'):
                # Add a tiny stop penalty (1e-5) to break ties
                cost_transition = (dist / 10.0) * s_i['price'] + 1e-5
                total_cost = dp[j][0] + cost_transition
                if total_cost < dp[i][0]:
                    dp[i] = (total_cost, j)
                    
    # Find best last station to reach destination
    best_cost = float('inf')
    best_last_station = -1
    
    for i in range(n):
        s = stations[i]
        dist_to_dest = route_length - s['distance']
        if dist_to_dest <= 500 and dp[i][0] != float('inf'):
            final_leg_cost = (dist_to_dest / 10.0) * s['price']
            total_cost = dp[i][0] + final_leg_cost
            if total_cost < best_cost:
                best_cost = total_cost
                best_last_station = i
                
    if best_last_station == -1:
        # Destination is unreachable!
        return None, float('inf')
        
    # Reconstruct stops
    stops = []
    curr = best_last_station
    while curr != -1:
        stops.append(stations[curr])
        curr = dp[curr][1]
        
    stops.reverse()
    return stops, best_cost

def optimize_fuel_route(start_query, finish_query):
    """
    Orchestrate route calculation and optimal fuel stop identification.
    """
    # 1. Geocode start and finish locations (respect rate limit with a brief sleep)
    start_coords = geocode_location(start_query)
    if not start_coords:
        return {"error": f"Could not geocode start location: {start_query}"}
        
    time.sleep(1.0)
    finish_coords = geocode_location(finish_query)
    if not finish_coords:
        return {"error": f"Could not geocode finish location: {finish_query}"}
        
    # 2. Get route from OSRM
    lat_start, lon_start = start_coords
    lat_finish, lon_finish = finish_coords
    route_info = get_driving_route(lat_start, lon_start, lat_finish, lon_finish)
    if not route_info:
        return {"error": "Could not calculate driving route between locations"}
        
    route_coords = route_info["coordinates"] # List of [lon, lat]
    route_length = route_info["distance"]
    
    # 3. Calculate cumulative distances along route coordinates
    # OSRM coordinates are [lon, lat]
    cum_distances = [0.0]
    total_dist = 0.0
    for i in range(1, len(route_coords)):
        lon1, lat1 = route_coords[i-1]
        lon2, lat2 = route_coords[i]
        seg_dist = haversine(lon1, lat1, lon2, lat2)
        total_dist += seg_dist
        cum_distances.append(total_dist)
        
    # 4. Filter stations near the route using KDTree
    stations, tree = get_stations_and_tree()
    if not stations:
        return {"error": "Fuel station data is empty or not loaded"}
        
    # Subsample route points to speed up KDTree query (e.g. at least 2 miles apart)
    subsampled_indices = [0]
    last_dist = 0.0
    for i in range(1, len(route_coords)):
        if cum_distances[i] - last_dist >= 2.0 or i == len(route_coords) - 1:
            subsampled_indices.append(i)
            last_dist = cum_distances[i]
            
    # Find unique stations near the subsampled route points
    nearby_station_indices = set()
    # 0.15 degrees is roughly 10 miles
    for idx in subsampled_indices:
        lon, lat = route_coords[idx]
        indices = tree.query_ball_point([lat, lon], r=0.15)
        nearby_station_indices.update(indices)
        
    # 5. Project nearby stations onto the route to find their route distances and clean filtering
    route_stations = []
    for s_idx in nearby_station_indices:
        station = stations[s_idx]
        
        # Find closest point on the full route
        min_dist = float('inf')
        closest_route_idx = -1
        for i in range(len(route_coords)):
            lon_r, lat_r = route_coords[i]
            d = haversine(station['lng'], station['lat'], lon_r, lat_r)
            if d < min_dist:
                min_dist = d
                closest_route_idx = i
                
        # Perpendicular distance must be <= 10 miles to be considered along route
        if min_dist <= 10.0 and closest_route_idx != -1:
            station_with_dist = station.copy()
            station_with_dist['distance'] = cum_distances[closest_route_idx]
            station_with_dist['perpendicular_distance'] = min_dist
            route_stations.append(station_with_dist)
            
    # Sort stations by distance along route
    route_stations.sort(key=lambda s: s['distance'])
    
    # Clean up stations to ensure strict strictly increasing distance along route
    # (If two stations project to exactly the same coordinate index, keep the cheaper one)
    unique_dist_stations = []
    for rs in route_stations:
        if not unique_dist_stations or unique_dist_stations[-1]['distance'] != rs['distance']:
            unique_dist_stations.append(rs)
        else:
            # Overlap: keep the cheaper station
            if rs['price'] < unique_dist_stations[-1]['price']:
                unique_dist_stations[-1] = rs
                
    # 6. Run DP solver
    stops, total_cost = find_optimal_stops(route_length, unique_dist_stations)
    
    if stops is None:
        return {
            "error": "The route is longer than 500 miles and no sequence of fuel stops within 500-mile ranges could be found."
        }
        
    # Reformat route coords for Leaflet/JSON (from [lon, lat] to [lat, lon])
    map_route_coords = [[pt[1], pt[0]] for pt in route_coords]
    
    return {
        "start": start_query,
        "finish": finish_query,
        "start_coords": start_coords,
        "finish_coords": finish_coords,
        "total_distance": route_length,
        "total_duration_hours": route_info["duration"],
        "total_fuel_cost": total_cost,
        "total_fuel_gallons": route_length / 10.0,
        "fuel_stops": stops,
        "route_coordinates": map_route_coords
    }
