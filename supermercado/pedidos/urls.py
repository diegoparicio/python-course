from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # Página principal
    path('calcular_compra/', views.calcular_compra, name='calcular_compra'),
    path('resultado/', views.resultado, name='resultado'),
    path('vaciar_cesta/', views.vaciar_cesta, name='vaciar_cesta'),
]