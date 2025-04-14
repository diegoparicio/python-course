from django.shortcuts import render
from .models import Tickets
from itertools import groupby
from operator import itemgetter
import openpyxl

def index(request):
    # Ruta al archivo Excel
    # secciones_path = 'C:/Users/kskme/Desktop/Python/python-course/supermercado/secciones.xlsx'
    secciones_path = '/home/supermercado/python-course/supermercado/secciones.xlsx'
    workbook = openpyxl.load_workbook(secciones_path)
    sheet = workbook.active

    # Crear un diccionario para mapear id_seccion a nombre_seccion e imagen
    secciones_dict = {}
    for row in sheet.iter_rows(min_row=2, values_only=True):  # Saltar la fila de encabezados
        id_seccion, nombre_seccion, imagen = row
        secciones_dict[id_seccion] = {'nombre': nombre_seccion, 'imagen': imagen}

    # Obtener productos únicos y ordenarlos por sección
    productos = Tickets.objects.values('id_producto', 'nombre_producto', 'precio_unitario', 'id_seccion').distinct()
    productos_ordenados = sorted(productos, key=itemgetter('id_seccion'))  # Ordenar por sección

    # Agrupar productos por sección
    productos_por_seccion = {}
    for seccion, items in groupby(productos_ordenados, key=itemgetter('id_seccion')):
        productos_por_seccion[seccion] = {
            'nombre': secciones_dict.get(seccion, {}).get('nombre', f"Sección {seccion}"),
            'imagen': secciones_dict.get(seccion, {}).get('imagen', ''),
            'productos': list(items)
        }

    return render(request, 'pedidos/index.html', {'productos_por_seccion': productos_por_seccion})

def calcular_compra(request):
    if request.method == 'POST':
        total = 0
        recomendaciones = {}

        # Recuperar los productos seleccionados previamente de la sesión
        productos_seleccionados = request.session.get('productos_seleccionados', [])

        # Importar métricas desde el archivo CSV
        metricas = importar_metricas()

        # Iterar sobre los datos enviados en el formulario
        productos_comprados = []
        for key, value in request.POST.items():
            if key.startswith('cantidad_') and value.isdigit():
                cantidad = int(value)
                if cantidad > 0:
                    # Obtener el identificador del producto (clave dinámica)
                    identificador = key.split('_', 1)[1]

                    # Determinar si el identificador es un número (id_producto) o un nombre (nombre_producto)
                    if identificador.isdigit():
                        # Buscar por ID del producto
                        producto = Tickets.objects.filter(
                            id_producto=identificador
                        ).values(
                            'id_producto', 'nombre_producto', 'precio_unitario'
                        ).first()
                    else:
                        # Buscar por nombre del producto
                        producto = Tickets.objects.filter(
                            nombre_producto=identificador
                        ).values(
                            'id_producto', 'nombre_producto', 'precio_unitario'
                        ).first()

                    if producto:
                        # Calcular el subtotal y redondearlo a 2 decimales
                        subtotal = round(producto['precio_unitario'] * cantidad, 2)
                        total += subtotal
                        # Añadir el producto a la lista de productos seleccionados
                        productos_seleccionados.append({
                            'id_producto': producto['id_producto'],
                            'nombre': producto['nombre_producto'],
                            'cantidad': cantidad,
                            'precio_unitario': round(producto['precio_unitario'], 2),
                            'subtotal': subtotal
                        })
                        # Añadir el nombre del producto comprado a la lista
                        productos_comprados.append(producto['nombre_producto'])
                    else:
                        print(f"Producto no encontrado: {identificador}")  # Depuración

        # Redondear el total a 2 decimales
        total = round(sum(p['subtotal'] for p in productos_seleccionados), 2)

        # Generar recomendaciones basadas en las métricas
        for producto_nombre in productos_comprados:
            recomendaciones_producto = []
            for m in metricas:
                if m['antecedente'] == producto_nombre and m['lift'] >= 1.5:
                    # Buscar el precio del producto recomendado en la base de datos
                    producto_recomendado = Tickets.objects.filter(
                        nombre_producto=m['consecuente']
                    ).values('precio_unitario').first()

                    # Si se encuentra el producto, añadirlo a las recomendaciones
                    if producto_recomendado:
                        recomendaciones_producto.append({
                            'consecuente': m['consecuente'],
                            'precio': producto_recomendado['precio_unitario'],  # Añadir el precio
                            'lift': m['lift']
                        })

            # Ordenar las recomendaciones por lift y obtener las 3 mejores
            recomendaciones_producto = sorted(recomendaciones_producto, key=lambda x: x['lift'], reverse=True)[:3]

            # Añadir las recomendaciones al diccionario si hay resultados
            if recomendaciones_producto:
                recomendaciones[producto_nombre] = recomendaciones_producto

        # Guardar los productos seleccionados en la sesión
        request.session['productos_seleccionados'] = productos_seleccionados

        return render(request, 'pedidos/cesta.html', {
            'productos_seleccionados': productos_seleccionados,
            'total': total,
            'recomendaciones': recomendaciones  # Pasar las recomendaciones al template
        })

    # Si el método no es POST, redirigir al índice o devolver una respuesta adecuada
    # return redirect('calcular_compra')

import csv

def importar_metricas():
    # Ruta al archivo CSV
    # csv_path = 'C:/Users/kskme/Desktop/Python/python-course/supermercado/reglas.csv'
    csv_path = '/home/supermercado/python-course/supermercado/reglas.csv'

    # Lista para almacenar las métricas
    metricas = []

    # Leer el archivo CSV
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')  # Usa ';' como delimitador
        for row in reader:
            metricas.append({
                'antecedente': row['antecedente'],  # Mantener como cadena
                'consecuente': row['consecuente'],  # Mantener como cadena
                'soporte_a': float(row['soporte_a'].replace(',', '.')),
                'confianza': float(row['confianza'].replace(',', '.')),
                'lift': float(row['lift'].replace(',', '.')),
                'id_producto_a': int(row['id_producto_a']),
                'id_seccion_a': int(row['id_seccion_a']),
                'id_departamento_a': int(row['id_departamento_a']),
            })

    return metricas

from django.shortcuts import redirect

def vaciar_cesta(request):
    if request.method == 'POST':
        # Eliminar los productos seleccionados de la sesión
        if 'productos_seleccionados' in request.session:
            del request.session['productos_seleccionados']
        # También puedes limpiar toda la sesión si es necesario:
        # request.session.flush()
    return redirect('resultado')  # Redirigir a la página de resultados


def resultado(request):
    # Recuperar los productos seleccionados y el total de la sesión
    productos_seleccionados = request.session.get('productos_seleccionados', [])
    total = round(sum(p['subtotal'] for p in productos_seleccionados), 2) if productos_seleccionados else 0

    # Renderizar la plantilla de resultados
    return render(request, 'pedidos/cesta.html', {
        'productos_seleccionados': productos_seleccionados,
        'total': total,
        'recomendaciones': {}  # Puedes pasar recomendaciones vacías si no son necesarias
    })

def pagar_compra(request):
    if request.method == 'POST':
        # Lógica para procesar el pago
         # Recuperar los productos seleccionados y el total de la sesión
        productos_seleccionados = request.session.get('productos_seleccionados', [])
        total = round(sum(p['subtotal'] for p in productos_seleccionados), 2) if productos_seleccionados else 0

        # Renderizar la plantilla de pago
        return render(request, 'pedidos/pago.html', {
            'productos_seleccionados': productos_seleccionados,
            'total': total
        })
    
def reiniciar_compra(request):
    if request.method == 'POST':
        # Eliminar los productos seleccionados de la sesión
        if 'productos_seleccionados' in request.session:
            del request.session['productos_seleccionados']
        # También puedes limpiar toda la sesión si es necesario:
        # request.session.flush()
    return redirect('index')  # Redirigir a la página de resultados