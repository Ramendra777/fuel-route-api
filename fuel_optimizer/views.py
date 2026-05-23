from django.http import JsonResponse
from django.shortcuts import render
from .utils import optimize_fuel_route

def api_optimize_route(request):
    """
    API endpoint that accepts start and finish locations,
    calculates the route, and returns optimal fuel stops and total fuel cost.
    """
    start = request.GET.get('start', '').strip()
    finish = request.GET.get('finish', '').strip()
    
    if not start or not finish:
        return JsonResponse({
            "error": "Both 'start' and 'finish' parameters are required."
        }, status=400)
        
    try:
        result = optimize_fuel_route(start, finish)
        if "error" in result:
            return JsonResponse(result, status=400)
        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({
            "error": f"An unexpected error occurred: {str(e)}"
        }, status=500)

def index(request):
    """
    Frontend page serving the map UI.
    """
    return render(request, 'fuel_optimizer/index.html')
