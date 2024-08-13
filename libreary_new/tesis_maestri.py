
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog
import os

import torch
import torchvision
from torchvision import transforms as torchtrans  
import torchvision.transforms as transforms

import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from PIL import Image
import sqlite3

from datetime import datetime

from ultralytics import YOLO
from tensorflow import keras

from PIL import Image
from PIL.TiffTags import TAGS
import re
import rasterio
import rasterio.features
import rasterio.warp
from rasterio.transform import Affine
import rasterio.sample
import rasterio.vrt



import numpy as np 
import cv2
import csv  # libreria para convertir a csv
import shapefile # libreria para shape

np.set_printoptions(precision=8)
"""/// Listado de funciones\\\
    self.ruta_variable => Variable que guarda la dir de folder
    self.ver_formato
"""

# Funcion para leer la dirección de carpeta
def ver_formato(ruta):
    lista= os. listdir(ruta)
    #print(lista)
    lis_jpg_tif=[]
    onl_jpg=[]
    con=0
    for nombres in lista:
            if nombres.split(".")[-1].upper() in ["jpg","JPG","TIF","tif"]:
                lis_jpg_tif.append(nombres)
            if nombres.split(".")[-1].upper() in ["jpg","JPG"]:
                onl_jpg.append(nombres)
        #variable nombres tiene solo los jpg y los tif
        #verificamos que por lo menos 1 tenga el formato
    #print(onl_jpg)
    for archivos in onl_jpg:
        if (os.path.exists(ruta+'/'+archivos[0:7]+'1.TIF')) and (os.path.exists(ruta+'/'+archivos[0:7]+'2.TIF')) and (os.path.exists(ruta+'/'+archivos[0:7]+'3.TIF')) and (os.path.exists(ruta+'/'+archivos[0:7]+'4.TIF')) and (os.path.exists(ruta+'/'+archivos[0:7]+'5.TIF')):
             con+=1
    return onl_jpg,con


#sección para realizar la predicción del modelo

def prediccion(ruta_imagen,model,lista_imagenes):
    # leemos las diferentes rutas de las imagenes en la función
    # Es decir, la ruta de cada imagen es ruta_imagen+lista_imagenes

    conn = sqlite3.connect('./library_new/test.db')
    cursor = conn.cursor()

    d = str(datetime.now())
        # Insertar datos en la tabla
    cursor.execute("INSERT INTO registro_carpeta (numero_de_imagenes,ruta_carpeta,fecha) VALUES (?, ?, ?)",
                    (str(len(lista_imagenes)),ruta_imagen, d))

        # Guardar los cambios y cerrar la conexión
    conn.commit()
    conn.close()


    # Verifivamos la id en la que esta 

    conn = sqlite3.connect('./library_new/test.db')
    cursor = conn.cursor()

    # Ejecutar la consulta SELECT *
    cursor.execute("SELECT * FROM registro_carpeta")

    # Obtener los resultados de la consulta
    resultados = cursor.fetchall()
    

    # Recorrer los resultados e imprimir los valores
    
    for indice,recorrido in enumerate(resultados):
        if (resultados[indice][3]==str(d)):
            id_datos=int(resultados[indice][0]) #ID de FK
    # Cerrar la conexión
    conn.close()
    #**********************************************# BASE DE DATOS TT
    for individual in lista_imagenes:
        np.set_printoptions(precision=8)
        ruta_total=ruta_imagen+"/"+individual
        ruta_rojatif=ruta_imagen+"/"+individual[0:7]+"1.TIF"
        #print(ruta_total)
        image = Image.open(ruta_total)
        """ transform = transforms.Compose([
            transforms.Resize((1300, 1600)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalización típica de ImageNet
        ])
        input_image = transform(image).unsqueeze(0)  # Añade una dimensión adicional para el lote (batch)
        with torch.no_grad():  # Desactiva el cálculo de gradientes
            o """
 
        output = model(image)

        
        #llenamos base de datos
        conn = sqlite3.connect('./library_new/test.db')
        cursor = conn.cursor()

        d = str(datetime.now())
            # Insertar datos en la tabla
        cursor.execute("INSERT INTO tabla_imagenes (nombre_imagen,cantidad_detect,id_registro_carpeta) VALUES (?, ?, ?)",
                        (str(individual),int(len(output[0])),int(id_datos)))

            # Guardar los cambios y cerrar la conexión
        conn.commit()
        conn.close()

#__ Id de segunda tabla (filtramos por ir de carpeta y nombre)

        conexion = sqlite3.connect('./library_new/test.db')
        cursor = conexion.cursor()


        # Consultar el id de la fila con nombre_imagen y id_registro_carpeta especificados
        cursor.execute("SELECT id FROM tabla_imagenes WHERE nombre_imagen = ? AND id_registro_carpeta = ?", (str(individual), int(id_datos)))
        resultado = cursor.fetchone()

        if resultado is not None:
            id_fila = resultado[0] #id_fila tiene la id de guardado
        # Cerrar la conexión
        conexion.close()


        # Finalizacion llenado tabla 2 base de datos
        imagen_etiquetada=cv2.imread(ruta_total,1)
        imagen_etiquetada=cv2.cvtColor(imagen_etiquetada,cv2.COLOR_BGR2RGB)
        for detection in output:
            boxes = detection.boxes.xyxy
            labels = detection.boxes.cls
            scores = detection.boxes.conf
            # Recorre cada detección y dibuja el cuadro delimitador
            for box, label, score in zip(boxes, labels, scores):
                x, y, x2, y2 = box.tolist()
                
                cx=int(x+((x2-x)//2))
                cy=int(y+((y2-y)//2))

                lat,longi =convertgps(cx,cy,ruta_rojatif)

                conn = sqlite3.connect('./library_new/test.db')
                cursor = conn.cursor()
                
                d = str(datetime.now())
                    # Insertar datos en la tabla
                cursor.execute("INSERT INTO resultado_imagen (pixel_min,pixel_max,latitud,longitud,score,id_tabla_imagenes) VALUES (?, ?, ?, ?, ?, ?)",
                                (str([int(x),int(y)]),str([int(x2),int(y2)]),str(longi),str(lat),str(score.item()),id_fila))

                    # Guardar los cambios y cerrar la conexión
                conn.commit()
                conn.close()
                # FIN DE LLENADO DE TABLA 3
                #Nota: Tal vez lo de abajo ya no este

                
                # Crea un objeto Rectangle para el cuadro delimitador
                """ rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
                
                # Añade el cuadro delimitador a la imagen
                ax.add_patch(rect)
                
                # Añade una etiqueta con la clase y la puntuación
                label_str = f'Clase: {label.item()}, score: {score.item():.2f}'
                ax.text(x, y, label_str, fontsize=8, color='r', verticalalignment='top') """
        
    



# conversion de pixel a gps ////////////////////////////

def metadata(imagen):
    img = Image.open(imagen)
# obtenemos los Tags de la metadata y se almacenan en un diccionario
    meta_dict = {TAGS[key]: img.tag[key] for key in img.tag_v2}
    # Se imprime el diccionario para obtener la composición de los datos
    # for rec in meta_dict:
    #     print(rec, ":", meta_dict[rec])

    # Extraemos el indicador xmp
    p = meta_dict.get("XMP")
    s = p.decode("UTF-8")
    # dividimos por el salto de linea y obtenemos una lista
    div = s.split("\n")

    #eliminamos los espacios vacios 

    for ind,recorrido in enumerate(div):
        div[ind]=recorrido.strip() 
    usar = div[17]

    result = re.search(":(.*)=", div[17])
    result.group(1)

    result2 = re.search("\"(.*)\"", div[17])
    result2.group(1)

    metadiccionario = {}
    for ind,recorrido in enumerate(div):
        try:
            metadiccionario[re.search(":(.*)=", div[ind]).group(1)] = re.search("\"(.*)\"", div[ind]).group(1)
        except:
            pass

    metadiccionario = {}
    for ind,recorrido in enumerate(div):
        try:
            metadiccionario[re.search(":(.*)=", div[ind]).group(1)] = re.search("\"(.*)\"", div[ind]).group(1)
        except:
            pass

    #print(metadiccionario)
    return metadiccionario

def TransfromRaster(img_path):
    metadiccionario = metadata(img_path)
    
    altura_vuelo = float(metadiccionario["RelativeAltitude"])
    distancia_focal = float(metadiccionario["CalibratedFocalLength"])

    resolucion = altura_vuelo/distancia_focal
   
    min_lon = (float(metadiccionario["GpsLongitude"])) - (
        float(metadiccionario["CalibratedOpticalCenterX"]) * resolucion) / 111111
    max_lon = (float(metadiccionario["GpsLongitude"])) + (
        float(metadiccionario["CalibratedOpticalCenterX"]) * resolucion) / 111111
    min_lat = (float(metadiccionario["GpsLatitude"])) - (
        float(metadiccionario["CalibratedOpticalCenterY"]) * resolucion) / 111111
    max_lat = (float(metadiccionario["GpsLatitude"])) + (
        float(metadiccionario["CalibratedOpticalCenterY"]) * resolucion) / 111111

    img_data = rasterio.open(img_path, 'r')

    bands = [1] #Se especifica las cantidades de canales que tiene la imagen.
    count_bands = len(bands)
    data = img_data.read(bands)
    _, width, height = data.shape 

    crs = {'init': 'epsg:4326'}

    transform = rasterio.transform.from_bounds(
            min_lon, min_lat, max_lon, max_lat, height, width)
    
    return transform

def convertgps(x,y,path_red):
    
    trans = TransfromRaster(path_red)
    transfor = np.array([[trans.a, trans.b, trans.c], 
                     [trans.d, trans.e, trans.f],
                     [0, 0, 1]], dtype =np.float64)

    xy = np.array([[x], #x
               [y], #y
               [1]], dtype=np.float64)

    x=np.dot(transfor, xy)   
    return x[0],x[1]


# conversion de pixel a gps //////////////////////////// dibujar imagen con respecto a la base de datos
def imagen_etiquetada(ruta,nombre_imagen):

    imagen_etiquetada=cv2.imread(ruta+"/"+nombre_imagen,1)
    imagen_etiquetada=cv2.cvtColor(imagen_etiquetada,cv2.COLOR_BGR2RGB)
    conexion = sqlite3.connect('./library_new/test.db')
    cursor = conexion.cursor()

    # Nombre de la imagen y ruta a buscar

    # Ejecutar la consulta
    cursor.execute("SELECT * FROM tabla_imagenes AS ti JOIN resultado_imagen AS ri ON ti.id = ri.id_tabla_imagenes WHERE ti.nombre_imagen = ? AND ti.id_registro_carpeta IN (SELECT id FROM registro_carpeta WHERE ruta_carpeta = ?)", (nombre_imagen, ruta))
    rows = cursor.fetchall()
    #print(len(rows))
    # Recorrer y mostrar los resultados
    for row in rows:
        # Acceder a los valores de las columnas
        id_tabla_imagenes = row[0]
        # ...
        # Acceder a los valores de las columnas de resultado_imagen
        pixel_min = row[5]
        pixel_max = row[6]
        latitud = row[7]
        longitud = row[8]
        score = row[9]
        #print(f"{pixel_max} - {pixel_min} - {latitud} - {longitud} - {score}")

        numbers = pixel_min.strip('[]').split(',')
        # Convertir los números de cadena a enteros
        x= int(numbers[0])
        y = int(numbers[1])

        numbers2= pixel_max.strip('[]').split(',')
        # Convertir los números de cadena a enteros
        x2= int(numbers2[0])
        y2 = int(numbers2[1])

        cv2.rectangle(imagen_etiquetada, (int(x), int(y)), (int(x2), int(y2)), (255, 0, 0), 2)
        label_str = f'Score: {score:.2f}'
        cv2.putText(imagen_etiquetada, label_str, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        # Hacer algo con los valores obtenidos

    # Cerrar la conexión
    conexion.close()
    return imagen_etiquetada
    


#### consulta de b dicionario para tablas

def consulta_tablas1(ruta_carpeta):
   
    lista_dic =[]
    conexion = sqlite3.connect('./library_new/test.db')  # Reemplaza con el nombre de tu base de datos
    cursor = conexion.cursor()

        # Consulta SQL para obtener los nombres de las imágenes y la cantidad de detecciones
    consulta = '''
            SELECT i.nombre_imagen, i.cantidad_detect
            FROM tabla_imagenes AS i
            INNER JOIN registro_carpeta AS rc ON i.id_registro_carpeta = rc.id
            WHERE rc.ruta_carpeta = ?
        '''

        # Ejecutar la consulta y obtener los resultados
    cursor.execute(consulta, (ruta_carpeta,))
    resultados = cursor.fetchall()
    conexion.close()

        # Recorrer los resultados e imprimir los nombres de las imágenes y la cantidad de detecciones
    for nombre_imagen, cantidad_detect in resultados:
            #print(f'Imagen: {nombre_imagen}, Detecciones: {cantidad_detect}')
            diccionario = {"nombre": nombre_imagen, "n_detection": cantidad_detect}
            lista_dic.append(diccionario)
    return lista_dic


def actualizar_tabla2(ruta_carpeta,nombre_imagen):
    
    conexion = sqlite3.connect('./library_new/test.db')  # Reemplaza con el nombre de tu base de datos
    cursor = conexion.cursor()
    # Consulta SQL para obtener los datos de resultado_imagen relacionados con nombre_imagen y ruta_carpeta
    consulta = '''
        SELECT ri.*
        FROM resultado_imagen AS ri
        INNER JOIN tabla_imagenes AS ti ON ri.id_tabla_imagenes = ti.id
        INNER JOIN registro_carpeta AS rc ON ti.id_registro_carpeta = rc.id
        WHERE ti.nombre_imagen = ? AND rc.ruta_carpeta = ?
    '''

    # Ejecutar la consulta y obtener los resultados
    cursor.execute(consulta, (nombre_imagen, ruta_carpeta))
    resultados = cursor.fetchall()
    conexion.close()
    lista_dic=[]
    # Recorrer los resultados y procesar los datos
    for resultado in resultados:
        # Acceder a los campos de la tabla resultado_imagen
        #id_resultado = resultado[0]
        pixel_min = resultado[1]
        pixel_max = resultado[2]
        latitud = resultado[3]
        longitud = resultado[4]
        # Hacer algo con los datos...
        diccionario = {"pixel_min": pixel_min, "pixel_max": pixel_max, "lat":latitud,"long":longitud}
        lista_dic.append(diccionario)

    return lista_dic

def obtener_latitudylongitudes(ruta_carpeta):
    conexion = sqlite3.connect("./library_new/test.db")  # Reemplaza con el nombre de tu base de datos
    cursor = conexion.cursor()

    # Realizar la consulta utilizando JOIN para combinar las tablas
    consulta = '''SELECT r.latitud, r.longitud
                  FROM resultado_imagen r
                  INNER JOIN tabla_imagenes t ON r.id_tabla_imagenes = t.id
                  INNER JOIN registro_carpeta rc ON t.id_registro_carpeta = rc.id
                  WHERE rc.ruta_carpeta = ?'''
    
    cursor.execute(consulta, (ruta_carpeta,))
    resultados = cursor.fetchall()

    conexion.close()

    # Crear una lista con las latitudes y longitudes
    latitudes_longitudes = [(latitud, longitud) for latitud, longitud in resultados]

    return latitudes_longitudes

def convertir_a_shapefile(ruta):

    latitudes_longitudes=obtener_latitudylongitudes(ruta)
    nombre_archivo=crear_carpeta(ruta)
    w = shapefile.Writer(nombre_archivo+"/resultados.shp", shapeType=shapefile.POINT)
    w.field('Latitud', 'F', decimal=10)
    w.field('Longitud', 'F', decimal=10)

    for latitud, longitud in latitudes_longitudes:
        latitud = float(latitud.strip('[]'))
        longitud = float(longitud.strip('[]'))
        w.point(longitud, latitud)
        w.record(latitud, longitud)

    w.close()

def enumerar_en_csv(ruta):
    
    latitudes_longitudes=obtener_latitudylongitudes(ruta)
    nombre_archivo=crear_carpeta(ruta)
    with open(nombre_archivo+"/resultado.csv", 'w', newline='') as archivo_csv:
        writer = csv.writer(archivo_csv)
        writer.writerow(['Latitud', 'Longitud'])
        for latitud, longitud in latitudes_longitudes:
            writer.writerow([latitud.strip('[]'), longitud.strip('[]')])

# No he definido la creacion de carpeta bien :D lo mas probable es que sea con la fecha
def crear_carpeta(direccion_carpeta):
    # Establecer conexión con la base de datos
    conexion = sqlite3.connect('./library_new/test.db')  # Reemplaza con el nombre de tu base de datos

    # Crear un cursor
    cursor = conexion.cursor()

    # Ejecutar la consulta para obtener la fecha asociada a la dirección de carpeta
    cursor.execute("SELECT fecha FROM registro_carpeta WHERE ruta_carpeta = ?", (direccion_carpeta,))

    # Obtener el resultado de la consulta
    resultado = cursor.fetchone()

    # Cerrar la conexión con la base de datos
    conexion.close()
    if resultado:
        fecha = resultado[0]

        # Modificar el formato de la fecha
        fecha_valida = re.sub(r'[^0-9a-zA-Z]+', '_', fecha)

        # Crear la carpeta con el nombre de la fecha válida
        carpeta_fecha = os.path.join(direccion_carpeta, fecha_valida)

        # Verificar si la carpeta ya existe
        contador = 1
        carpeta_alternativa = carpeta_fecha
        while os.path.exists(carpeta_alternativa):
            carpeta_alternativa = f"{carpeta_fecha}_v{contador}"
            contador += 1

        # Crear la carpeta alternativa
        os.makedirs(carpeta_alternativa)
        #print(f"Se ha creado la carpeta {carpeta_alternativa}")
    

    return carpeta_alternativa

def consulta_porfecha(fecha):
   
    conexion = sqlite3.connect('./library_new/test.db')  # Reemplaza con el nombre de tu base de datos

    # Crear un cursor
    cursor = conexion.cursor()

    # Ejecutar la consulta
    cursor.execute("""
        SELECT tabla_imagenes.cantidad_detect, tabla_imagenes.nombre_imagen
        FROM tabla_imagenes
        JOIN registro_carpeta ON tabla_imagenes.id_registro_carpeta = registro_carpeta.id
        WHERE registro_carpeta.fecha = ?
        """, (str(fecha),))


    # Obtener los resultados de la consulta
    resultados = cursor.fetchall()

    # Cerrar la conexión con la base de datos
    conexion.close()
    lista_dic=[]
    # Recorrer los resultados y procesar los datos
    for cantidad,nombre in resultados:
        diccionario={"nombre": nombre, "cant": cantidad}
        lista_dic.append(diccionario)
    return lista_dic

def borrar_porfecha(fecha):
    conexion = sqlite3.connect('./library_new/test.db')  # Reemplaza con el nombre de tu base de datos

    # Crear un cursor
    cursor = conexion.cursor()
    cursor.execute("SELECT id FROM registro_carpeta WHERE fecha = ?", (fecha,))
    registro_carpeta_id = cursor.fetchone()[0]

    # Delete rows from the resultado_imagen table associated with the registro_carpeta_id
    cursor.execute("DELETE FROM resultado_imagen WHERE id_tabla_imagenes IN (SELECT id FROM tabla_imagenes WHERE id_registro_carpeta = ?)", (registro_carpeta_id,))

    # Delete rows from the tabla_imagenes table associated with the registro_carpeta_id
    cursor.execute("DELETE FROM tabla_imagenes WHERE id_registro_carpeta = ?", (registro_carpeta_id,))

    # Delete the row from the registro_carpeta table
    cursor.execute("DELETE FROM registro_carpeta WHERE fecha = ?", (fecha,))

    # Commit the changes to the database
    conexion.commit()
    conexion.close()


# para proceso de clasificacion
def clasificacion(ruta_imagen,model,lista_imagenes):

    # leemos las diferentes rutas de las imagenes en la función
    # Es decir, la ruta de cada imagen es ruta_imagen+lista_imagenes
    np.set_printoptions(precision=8)
    conn = sqlite3.connect('./library_new/test.db')
    cursor = conn.cursor()

    d = str(datetime.now())
        # Insertar datos en la tabla
    cursor.execute("INSERT INTO registro_carpeta (numero_de_imagenes,ruta_carpeta,fecha) VALUES (?, ?, ?)",
                    (str(len(lista_imagenes)),ruta_imagen, d))

        # Guardar los cambios y cerrar la conexión
    conn.commit()
    conn.close()


    # Verifivamos la id en la que esta 

    conn = sqlite3.connect('./library_new/test.db')
    cursor = conn.cursor()

    # Ejecutar la consulta SELECT *
    cursor.execute("SELECT * FROM registro_carpeta")

    # Obtener los resultados de la consulta
    resultados = cursor.fetchall()
    

    # Recorrer los resultados e imprimir los valores
    
    for indice,recorrido in enumerate(resultados):
        if (resultados[indice][3]==str(d)):
            id_datos=int(resultados[indice][0]) #ID de FK
    # Cerrar la conexión
    conn.close()
    #**********************************************# BASE DE DATOS TT
    for individual in lista_imagenes:
        ruta_total=ruta_imagen+"/"+individual
        ruta_rojatif=ruta_imagen+"/"+individual[0:7]+"3.TIF"
        ruta_regtif=ruta_imagen+"/"+individual[0:7]+"4.TIF"
        ruta_niratif=ruta_imagen+"/"+individual[0:7]+"5.TIF"
        
        img_RGB=cv2.imread(ruta_total,1)
        img_RED=cv2.imread(ruta_rojatif,0)
        img_REG=cv2.imread(ruta_regtif,0)
        img_NIR=cv2.imread(ruta_niratif,0)
        width,height,_=img_RGB.shape
        b_RGB = cv2.resize(img_RGB, (width, height), interpolation=cv2.INTER_LINEAR)
        base_NIR = cv2.resize(img_NIR, (width, height), interpolation=cv2.INTER_LINEAR)
        b_RED = cv2.resize(img_RED, (width, height), interpolation=cv2.INTER_LINEAR)
        b_REG = cv2.resize(img_REG, (width, height), interpolation=cv2.INTER_LINEAR)

        
        stb_RED = estabilizador_imagen(b_RED, b_RGB)
        stb_REG = estabilizador_imagen(b_REG, b_RGB)
        stb_NIR = estabilizador_imagen(base_NIR, b_RGB)
        
        nueva=cv2.merge([stb_RED,stb_REG,stb_NIR])
        
        # Aqui debo hacer clasificacion
        
        tam_x,tam_y,tamz=nueva.shape
        # División de total de pixeles/ número de divisiones
        # Se obtiene el tamaño de cada sección
        div_x=round(tam_x/10)
        div_y=round(tam_y/10)
        copia=nueva.copy()
        output=[]
        boxes=[]
        scores=[]
        for j in range(10):
            for i in range (10):
                if (div_x*(i+1)<=tam_x):
                    cuadro=copia[div_x*i:div_x*(i+1),j*div_y:div_y*(j+1)]
                    cuadro=cv2.resize(cuadro,(224,224),interpolation = cv2.INTER_AREA)
                    #cuadro=cuadro/255
                    matriz_cuatro_dimensiones = np.expand_dims(cuadro, axis=0)
                    result=model.predict(matriz_cuatro_dimensiones)
                    max_value = np.max(result)
                    max_index = np.argmax(result)
                    if (max_index==0):
                        boxes.append([div_x*i,j*div_y,div_x*(i+1),div_y*(j+1)])
                        scores.append(max_value)
        output=[{"boxes":boxes, "scores": scores}]
        #llenamos base de datos
        conn = sqlite3.connect('./library_new/test.db')
        cursor = conn.cursor()

        d = str(datetime.now())
            # Insertar datos en la tabla
        cursor.execute("INSERT INTO tabla_imagenes (nombre_imagen,cantidad_detect,id_registro_carpeta) VALUES (?, ?, ?)",
                        (str(individual),int(len(output[0]['scores'])),int(id_datos)))

            # Guardar los cambios y cerrar la conexión
        conn.commit()
        conn.close()

#__ Id de segunda tabla (filtramos por ir de carpeta y nombre)

        conexion = sqlite3.connect('./library_new/test.db')
        cursor = conexion.cursor()


        # Consultar el id de la fila con nombre_imagen y id_registro_carpeta especificados
        cursor.execute("SELECT id FROM tabla_imagenes WHERE nombre_imagen = ? AND id_registro_carpeta = ?", (str(individual), int(id_datos)))
        resultado = cursor.fetchone()

        if resultado is not None:
            id_fila = resultado[0] #id_fila tiene la id de guardado
        # Cerrar la conexión
        conexion.close()

        
        # Finalizacion llenado tabla 2 base de datos
        
        for detection in output:
            boxes = detection['boxes']
            
            scores = detection['scores']
            
            
            # Recorre cada detección y dibuja el cuadro delimitador
            for box, score in zip(boxes,  scores):
                x, y, x2, y2 = box
                
                cx=int(x+((x2-x)//2))
                cy=int(y+((y2-y)//2))
                lat,longi =convertgps(cx,cy,ruta_rojatif)
                #llenado de tabla 3
                conn = sqlite3.connect('./library_new/test.db')
                cursor = conn.cursor()

                d = str(datetime.now())
                    # Insertar datos en la tabla
                cursor.execute("INSERT INTO resultado_imagen (pixel_min,pixel_max,latitud,longitud,score,id_tabla_imagenes) VALUES (?, ?, ?, ?, ?, ?)",
                                (str([int(x),int(y)]),str([int(x2),int(y2)]),str(longi),str(lat),str(score.item()),id_fila))

                    # Guardar los cambios y cerrar la conexión
                conn.commit()
                conn.close()
                # FIN DE LLENADO DE TABLA 3
                #Nota: Tal vez lo de abajo ya no este

                
                # Crea un objeto Rectangle para el cuadro delimitador
                """ rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
                
                # Añade el cuadro delimitador a la imagen
                ax.add_patch(rect)
                
                # Añade una etiqueta con la clase y la puntuación
                label_str = f'Clase: {label.item()}, score: {score.item():.2f}'
                ax.text(x, y, label_str, fontsize=8, color='r', verticalalignment='top') """
                
def estabilizador_imagen(imagen_base, imagen_a_estabilizar, radio = 0.75, error_reproyeccion = 4.0, coincidencias = False):
        """Esta clase devuelve una secuencia de imágenes tomadas de la cámara estabilizada con respecto a la primera imagen"""
        
        # Se obtienen los puntos deinterés
        
        (kpsBase, featuresBase) = obtener_puntos_interes(imagen_base)
        (kpsAdicional, featuresAdicional) = obtener_puntos_interes(imagen_a_estabilizar)
        # Se buscan las coincidencias        
        
        M = encontrar_coincidencias(imagen_base, imagen_a_estabilizar, kpsBase, kpsAdicional, featuresBase, featuresAdicional, radio)
        
        if M is None:

            return None
        
        if len(M) > 4:
            # construct the two sets of points
            # M2 = cv2.getPerspectiveTransform(ptsA,ptsB)
            (H, status) = encontrar_H_RANSAC_Estable(M, kpsBase, kpsAdicional, error_reproyeccion)
            estabilizada = cv2.warpPerspective(imagen_base,H,(imagen_base.shape[1],imagen_base.shape[0]))
            return estabilizada

        return None
    #--------------------------------------------------------------------------
def obtener_puntos_interes(imagen):
    f =cv2.SIFT_create()
    (kps, features) =f.detectAndCompute(imagen, None)
    return kps, features

def encontrar_coincidencias(img1, img2, kpsA, kpsB, featuresA, featuresB, ratio):
        """Metodo para estimar la homografia"""
        
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []
#        
#        # loop over the raw matches
        for m in rawMatches:
#            # ensure the distance is within a certain ratio of each
#            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        
#        print (matches)
        return matches
    
def encontrar_H_RANSAC_Estable( matches, kpsA, kpsB, reprojThresh):
        """Metodo para aplicar una H a una imagen y obtener la proyectividad"""
        
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i].pt for (_, i) in matches])
            ptsB = np.float32([kpsB[i].pt for (i, _) in matches])
    
            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
            
            return (H, status)