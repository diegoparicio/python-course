from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # Página principal
    path('calcular_compra/', views.calcular_compra, name='calcular_compra'),
    path('resultado/', views.resultado, name='resultado'),
    path('vaciar_cesta/', views.vaciar_cesta, name='vaciar_cesta'),
    path('pagar_compra/', views.pagar_compra, name='pagar_compra'),
    path('reiniciar_compra/', views.reiniciar_compra, name='reiniciar_compra'),
]