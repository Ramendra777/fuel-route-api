from django.urls import path
from . import views

app_name = 'fuel_optimizer'

urlpatterns = [
    path('', views.index, name='index'),
    path('api/route/', views.api_optimize_route, name='api_optimize_route'),
]
