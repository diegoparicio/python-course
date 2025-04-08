from django.contrib import admin
from .models import Tickets

@admin.register(Tickets)
class PedidoAdmin(admin.ModelAdmin):
    list_display = ('id_pedido', 'id_cliente', 'fecha')  # Campos que se mostrarán en la lista