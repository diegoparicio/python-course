from django.db import models

class Tickets(models.Model):
    id = models.AutoField(primary_key=True)
    id_pedido = models.IntegerField(blank=True, null=True)
    id_cliente = models.IntegerField(blank=True, null=True)
    fecha = models.TextField(blank=True, null=True)  # This field type is a guess.
    hora = models.IntegerField(blank=True, null=True)
    id_departamento = models.IntegerField(blank=True, null=True)
    id_seccion = models.IntegerField(blank=True, null=True)
    id_producto = models.IntegerField(blank=True, null=True)
    nombre_producto = models.TextField(blank=True, null=True)
    precio_unitario = models.FloatField(blank=True, null=True)
    cantidad = models.IntegerField(blank=True, null=True)
    precio_total = models.FloatField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'tickets'

    def __str__(self):
        return f"Pedido {self.id_pedido} - {self.nombre_producto}"

'''
class Facturas(models.Model):
    fecha_factura = models.TextField(blank=True, null=True)
    proveedor = models.TextField(blank=True, null=True)
    concepto = models.TextField(blank=True, null=True)
    importe = models.TextField(blank=True, null=True)  # This field type is a guess.

    class Meta:
        managed = False
        db_table = 'facturas''
'''

