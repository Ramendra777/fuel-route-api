# Fuel Route Optimizer API

A Django-based REST API that calculates the optimal, most cost-effective fuel stops for any driving route within the United States. 

The application assumes the vehicle has a maximum range of **500 miles** and achieves a fuel economy of **10 miles per gallon (MPG)**.

---

## Datasets

The application relies on three dataset files located at the root of the project:
1. **`fuel-prices-for-be-assessment.csv`**: The primary dataset containing retail fuel prices for 8,151 truckstops across the US and Canada.
2. **`uscities.csv`**: A local database containing geographical coordinates (latitude and longitude) for U.S. cities, used to geocode fuel stations offline.
   - *Source*: [Simplemaps US Cities Database (Basic Version)](https://simplemaps.com/data/us-cities)
3. **`geocoded_cities_cache.json`**: A precomputed JSON cache mapping remaining cities (including small towns and CDPs not covered by the Simplemaps database) to coordinates. This was compiled using Nominatim during development to guarantee 100% offline coverage and zero external geocoding calls for fuel stations at runtime.

---

## Key Features & Optimization

- **US State Filtering**: Filters out Canadian provinces at CSV load time, focusing strictly on the 50 US State codes.
- **Whitespace Cleaning**: Automatic `.strip()` cleaning on city names to ensure perfect match rates.
- **Deduplication**: Deduplicates redundant truckstop coordinates by `(City, State, Retail Price)`.
- **High-Performance Spatial Indexing**: Loads stations into memory at startup and builds a SciPy `KDTree`, allowing instant $O(\log N)$ spatial queries.
- **Mathematically Exact DP Solver**: Uses a Dynamic Programming (DP) algorithm to find the absolute lowest fuel cost for the trip, with a tiny stop penalty (`1e-5`) to eliminate redundant stops when prices are identical.
- **Low Latency**: Uses only 3 keyless, free API calls at runtime (2 geocoding calls to Nominatim for start/finish coordinates, and 1 routing call to OSRM).

---

## Setup & Running

### Prerequisites
- Python 3.12+

### 1. Installation
Clone the repository, create a virtual environment, and install the required dependencies:
```powershell
# Create virtual environment
python -m venv venv

# Install requirements
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Running the Server
Start the Django development server:
```powershell
.\venv\Scripts\python.exe manage.py runserver
```
The server will start at `http://127.0.0.1:8000/`.

---

## API Documentation

### Optimize Route Endpoint
Returns the route geometry (for rendering maps), trip statistics, and the sequence of optimal fuel stops.

- **URL**: `/api/route/`
- **Method**: `GET`
- **Query Parameters**:
  - `start` (string, required): Start location (e.g. `Chicago, IL`)
  - `finish` (string, required): Destination (e.g. `Los Angeles, CA`)

#### Example Request
```http
GET http://127.0.0.1:8000/api/route/?start=Chicago,IL&finish=Los+Angeles,CA
```

#### Example Response
```json
{
  "start": "Chicago, IL",
  "finish": "Los Angeles, CA",
  "start_coords": [41.8781136, -87.6297982],
  "finish_coords": [34.0522342, -118.2436849],
  "total_distance": 2018.137,
  "total_duration_hours": 35.34,
  "total_fuel_cost": 628.737,
  "total_fuel_gallons": 201.814,
  "fuel_stops": [
    {
      "opis_id": 1567,
      "name": "KUM & GO #0267",
      "address": "I-80, EXIT 267 & SR-38",
      "city": "Tipton",
      "state": "IA",
      "price": 2.98233333,
      "lat": 41.7456075,
      "lng": -91.1304308,
      "distance": 202.495
    },
    ...
  ],
  "route_coordinates": [
    [41.8781, -87.6298],
    ...
  ]
}
```

## Automated Tests

Run the automated test suite verifying distance calculations, preprocessing loaders, DP optimization cases, and REST API views:
```powershell
.\venv\Scripts\python.exe manage.py test
```
