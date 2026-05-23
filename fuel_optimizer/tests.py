from django.test import TestCase, Client
from django.urls import reverse
from unittest.mock import patch
from .utils import (
    haversine, 
    find_optimal_stops, 
    get_stations_and_tree,
    US_STATES
)

class FuelOptimizerUtilsTestCase(TestCase):
    
    def test_haversine_distance(self):
        # Coordinates for Chicago, IL and Los Angeles, CA
        # Great circle distance is approx 1742 miles
        chicago_lat, chicago_lon = 41.8781, -87.6298
        la_lat, la_lon = 34.0522, -118.2437
        dist = haversine(chicago_lon, chicago_lat, la_lon, la_lat)
        self.assertAlmostEqual(dist, 1742.33, delta=10.0)

    def test_station_loading_and_filtering(self):
        stations, tree = get_stations_and_tree()
        self.assertIsNotNone(stations)
        self.assertIsNotNone(tree)
        self.assertGreater(len(stations), 0)
        
        # Verify that all loaded stations belong to the 50 US States
        for s in stations:
            self.assertIn(s['state'], US_STATES)
            # Verify whitespace is cleaned
            self.assertEqual(s['city'].strip(), s['city'])

    def test_dp_solver_under_range(self):
        # Route is less than or equal to 500 miles, should return no stops and 0 cost
        stops, cost = find_optimal_stops(400.0, [])
        self.assertEqual(stops, [])
        self.assertEqual(cost, 0.0)

    def test_dp_solver_unreachable(self):
        # Route is 1200 miles but there are no stations, should return None and inf cost
        stops, cost = find_optimal_stops(1200.0, [])
        self.assertIsNone(stops)
        self.assertEqual(cost, float('inf'))

    def test_dp_solver_optimal_stops(self):
        # Mock stations sorted by distance
        mock_stations = [
            {'distance': 100.0, 'price': 3.00, 'name': 'Station A', 'city': 'City A', 'state': 'ST'},
            {'distance': 250.0, 'price': 2.50, 'name': 'Station B', 'city': 'City B', 'state': 'ST'},
            {'distance': 450.0, 'price': 3.50, 'name': 'Station C', 'city': 'City C', 'state': 'ST'},
            {'distance': 600.0, 'price': 2.20, 'name': 'Station D', 'city': 'City D', 'state': 'ST'},
            {'distance': 800.0, 'price': 3.10, 'name': 'Station E', 'city': 'City E', 'state': 'ST'},
            {'distance': 950.0, 'price': 2.80, 'name': 'Station F', 'city': 'City F', 'state': 'ST'},
        ]
        stops, cost = find_optimal_stops(1200.0, mock_stations)
        
        self.assertIsNotNone(stops)
        # Expected optimal stops: Station B, Station D, Station F
        self.assertEqual(len(stops), 3)
        self.assertEqual(stops[0]['name'], 'Station B')
        self.assertEqual(stops[1]['name'], 'Station D')
        self.assertEqual(stops[2]['name'], 'Station F')
        # Cost check:
        # Stop B (25.0 gallons @ 2.5) = 62.5
        # Stop D (35.0 gallons @ 2.2) = 77.0
        # Stop F (35.0 gallons @ 2.8) = 98.0
        # Final leg (25.0 gallons @ 2.8) = 70.0
        # Total = 307.50
        self.assertAlmostEqual(cost, 307.50, delta=0.001)


class FuelOptimizerAPITestCase(TestCase):
    
    def setUp(self):
        self.client = Client()

    def test_api_missing_parameters(self):
        # Call API without start/finish params
        url = reverse('fuel_optimizer:api_optimize_route')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.json())

    def test_api_missing_start(self):
        url = reverse('fuel_optimizer:api_optimize_route')
        response = self.client.get(url, {'finish': 'Los Angeles, CA'})
        self.assertEqual(response.status_code, 400)

    @patch('fuel_optimizer.views.optimize_fuel_route')
    def test_api_successful_response(self, mock_optimize):
        # Mock successful return from utils
        mock_optimize.return_value = {
            "start": "Chicago, IL",
            "finish": "Los Angeles, CA",
            "total_distance": 2000.0,
            "total_fuel_cost": 600.0,
            "fuel_stops": []
        }
        url = reverse('fuel_optimizer:api_optimize_route')
        response = self.client.get(url, {'start': 'Chicago, IL', 'finish': 'Los Angeles, CA'})
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['start'], 'Chicago, IL')
        self.assertEqual(data['total_fuel_cost'], 600.0)

    def test_homepage_render(self):
        url = reverse('fuel_optimizer:index')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'fuel_optimizer/index.html')
