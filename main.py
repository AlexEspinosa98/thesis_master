
from PyQt6.QtWidgets import QMainWindow, QApplication,QLineEdit,QMessageBox,QTableWidget,QFileDialog,QTableWidgetItem

from PyQt6.QtGui import QGuiApplication,QIcon,QImage,QPixmap
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import QPropertyAnimation, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import sys
from PyQt6.uic import loadUi
#import recursos

import recursos_iconos
import cv2


from library_new.tesis_maestri import *

from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog

import os

import torch
import torchvision
from torchvision import transforms as torchtrans  
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from PIL import Image
from tensorflow import keras

from datetime import datetime
import re
import rasterio
import rasterio.features
import rasterio.warp
from rasterio.transform import Affine
import rasterio.sample
import rasterio.vrt
np.set_printoptions(precision=8)

import sqlite3
from ultralytics import YOLO

"""/// Listado de variable  utilizadas y funciones\\
    self.ruta_variable => Variable que guarda la dir de folder
    self.list_image    => Variable que contiene la lista de imagenes
    self.imgproyectada => variable para determinar que imagen se esta visualizando
    self.pag = dice en que pagina esta y reinicia
    timage= saber si es la imagen original o la procesada
    """



class mainUI(QMainWindow):
    def __init__(self):
        super(mainUI,self).__init__()
        loadUi('Main_GUI.ui',self)
        self.ruta_carpeta=None
        self.list_images=[]
        self.imgproyectada=-1
        self.pag=0
        self.timage=0
        self.hact=0
        self.detect_p=0
        self.b_home.clicked.connect(lambda: self.pages.setCurrentWidget(self.p_home))	 #botones para cambiar de pagina
        self.b_home.clicked.connect(self.fun_home)
        self.b_model1.clicked.connect(lambda: self.pages.setCurrentWidget(self.p_model1))
        self.b_history.clicked.connect(lambda: self.pages.setCurrentWidget(self.p_history))
        self.b_history.clicked.connect(self.historial)
        self.b_model1.clicked.connect(self.identificador)
        self.b_model2.clicked.connect(lambda: self.pages.setCurrentWidget(self.p_model2))
        self.b_model2.clicked.connect(self.identificador2)
        self.b1_select_3.clicked.connect(self.leer_direc)  # Boton para seleccionar la carpeta
        self.b1_select_2.clicked.connect(self.leer_direc)
        self.b3_left_3.clicked.connect(self.pasarimage)   #boton izquierda para pasar datos
        self.b4_right_3.clicked.connect(self.pasarimage2)
        self.b3_left_2.clicked.connect(self.pasarimage)   #boton izquierda para pasar datos
        self.b4_right_2.clicked.connect(self.pasarimage2)
        self.b_model2.clicked.connect(lambda: self.pages.setCurrentWidget(self.p_model2))
        
        self.b_help.clicked.connect(lambda: self.pages.setCurrentWidget(self.p_help))
        #boton para pdf
        self.b_information.clicked.connect(self.open_pdf)
        
        # Botones para 
        self.b5_csv_2.clicked.connect(self.downloadcsv)
        self.b6_shape_2.clicked.connect(self.downloadshape)

        self.b5_csv_3.clicked.connect(self.downloadcsv)
        self.b6_shape_3.clicked.connect(self.downloadshape)
        #borrado de base de dato
        self.b_borrarbase.clicked.connect(self.borrartodo)

        #programando si es original o procesada para muestra de imagen
        self.b_original.clicked.connect(self.original)
        self.b_process.clicked.connect(self.procesada)
        # ventana clasification
        self.b2_process_3.clicked.connect(self.pro_clasification)


        #boton de seleccionar el historial
        self.list_combo.currentIndexChanged.connect(self.mostrar_estadisticas)
        self.b_csv_history.clicked.connect(self.downloadcsv)
        self.b_shape_history.clicked.connect(self.downloadshape)
        self.b_delete_selec.clicked.connect(self.borrarselect)
        self.b2_process_2.clicked.connect(self.procesar_detection)
        # modelo de deteccion
    
        self.modelc = keras.models.load_model("./library_new/modelo/model_class.h5")

        
        self.modeld= YOLO("./library_new/modelo/best.pt")

    def open_pdf(self):
        os.startfile("Manual_de_usuario.pdf")
    
    def fun_home(self):
        self.hact=0
    
    def original(self):
        if (self.ruta_carpeta):
            self.timage=0
            self.proyect_image()

    def procesada(self):
        if (self.timage or self.detect_p):
            #print("hola1")
            self.timage=1
            self.proyect_image()
        else: 
            mensaje = "the folder has not been process"
            QMessageBox.critical(self, "Error", mensaje)
        # falta comprobacion si ya procesaron :D
        

    def identificador(self):
        
        self.hact=0
        self.l_image_3.clear() 
        self.tabla_d2_2.clearContents()
        self.tabla2_3.clearContents()
        self.tabla_d2_2.setRowCount(0)
        self.timage=0
        self.ruta_carpeta=None
        self.detect_p=0
        self.pag=1
        self.imgproyectada=0 
        #self.proyect_image()
        self.label_7.clear()    

    def identificador2(self):
        
        self.hact=0
        self.detect_p=0
        self.l_image_2.clear()
        self.tabla_d2.clearContents()
        self.tabla_r.clearContents()
        self.tabla_d2.setRowCount(0)
        self.timage=0
        self.ruta_carpeta=None
        self.pag=0
        self.imgproyectada=0
        self.label_8.clear() 
        #self.proyect_image()

    def leer_direc(self):
        self.ruta_carpeta = QFileDialog.getExistingDirectory(self,"Select Folder")
        if (self.ruta_carpeta):
            self.pruebafun()
    
    
    def pruebafun(self):
        try:
            con=0
            self.list_images,con=ver_formato(self.ruta_carpeta)
            if (con>0):
                mensaje = "files uploaded successfully"
                QMessageBox.information(self, "information", mensaje)
                self.imgproyectada=0
                self.proyect_image()
                
            else:
                mensaje = "The folder does not contain the Phantom 4 format"
                QMessageBox.critical(self, "Error", mensaje)
        except:
            mensaje = "Folder not found"
            QMessageBox.critical(self, "Error", mensaje)
        
    def proyect_image(self):

        if (self.timage):

            self.image=imagen_etiquetada(self.ruta_carpeta,self.list_images[int(self.imgproyectada)])
            print(self.ruta_carpeta,self.list_images[int(self.imgproyectada)])
           # Aqui debe proyectarse la imagen con las etiquetas de la base de datos
            
        else: 
            self.image=cv2.imread(self.ruta_carpeta+'/'+self.list_images[int(self.imgproyectada)],1)
            self.image=cv2.cvtColor(self.image,cv2.COLOR_BGR2RGB)
            
        #self.image=cv2.imread(self.ruta_carpeta+'/'+self.list_images[int(self.imgproyectada)],1)
        self.actualizartabla1()
        self.actualizartabla2()
        self.image= cv2.resize(self.image, (520, 414), interpolation=cv2.INTER_LINEAR)
        qformat=QImage.Format.Format_BGR888
        img = QImage(self.image,self.image.shape[1],
                        self.image.shape[0],
                        self.image.strides[0],qformat)
        img= img.rgbSwapped()
        
        if (self.pag):
            self.l_image_3.setPixmap(QPixmap.fromImage(img))
            #aqui agrego label clsificactyion
            self.label_7.setText(self.list_images[int(self.imgproyectada)])
        else:
            self.l_image_2.setPixmap(QPixmap.fromImage(img))
            self.label_8.setText(self.list_images[int(self.imgproyectada)])
            #aqui agrego label2 detection
            
            
        
    def pasarimage(self):
        try:
                if self.imgproyectada==0:
                    self.imgproyectada=int(len(self.list_images))-1
                else:
                    self.imgproyectada-=1
                self.proyect_image()
        except : 
            mensaje = "The folder has not been selected"
            QMessageBox.critical(self, "Error", mensaje)

    def pasarimage2(self):
        try:
            if self.imgproyectada==int(len(self.list_images))-1:
                self.imgproyectada=0
            else:
                self.imgproyectada+=1
            self.proyect_image()
        except:
            mensaje = "The folder has not been selected"
            QMessageBox.critical(self, "Error", mensaje)
    
    def procesar_detection(self):
        try:    
            #necesitamos direccion para ller la variable
            # variable de detección
            self.detect_p=1
            prediccion(self.ruta_carpeta,self.modeld,self.list_images)
            self.timage=1
            self.proyect_image()
            self.llenartabla2()
            mensaje = "complete"
            QMessageBox.information(self, "information", mensaje)

        except:
            mensaje = "the folder has not been selected"
            QMessageBox.critical(self, "Error", mensaje)


    def historial(self):
        self.list_combo.clear()
        self.label_history.clear()
        connection = sqlite3.connect("./library_new/test.db")  # Replace "your_database.db" with your actual database file name

        # Create a cursor
        cursor = connection.cursor()

        # Execute the query to fetch all the dates from the first table
        cursor.execute("SELECT fecha,ruta_carpeta FROM registro_carpeta")

        # Fetch all the results
        dates = cursor.fetchall()
        cursor.close()
        connection.close()
        self.tabla_d2_3.clearContents()
        self.list_combo.addItem("--------------------")

        if len(dates)>0:
            for fechas in dates:
                
                self.list_combo.addItem(fechas[0]+"_ FOLDER:"+fechas[1].split("/")[-2]+"/"+fechas[1].split("/")[-1])
    
    
    #funcion para borrar las tablas de bdd
    def borrartodo(self):
        conexion = sqlite3.connect('./library_new/test.db')
        cursor = conexion.cursor()
        # Borrar todos los datos de la tabla "registro_carpeta"
        cursor.execute("DELETE FROM registro_carpeta")
        # Borrar todos los datos de la tabla "tabla_imagenes"
        cursor.execute("DELETE FROM tabla_imagenes")
        # Borrar todos los datos de la tabla "resultado_imagen"
        cursor.execute("DELETE FROM resultado_imagen")
        # Confirmar los cambios
        conexion.commit()
        conexion.close()
        self.historial()
        mensaje = "Deleted"
        QMessageBox.information(self, "information", mensaje)

    def llenartabla2(self):
        if (self.timage):
            dic=consulta_tablas1(self.ruta_carpeta)
            
            self.tabla_d2.setRowCount(len(dic))
            for indice,imagenes in enumerate(dic):
                self.tabla_d2.setItem(indice,0,QtWidgets.QTableWidgetItem(str(imagenes["nombre"])))
                self.tabla_d2.setItem(indice,1,QtWidgets.QTableWidgetItem(str(imagenes["n_detection"])))
        else:
            mensaje = "the folder has not been process"
            QMessageBox.critical(self, "Error", mensaje)

    def llenartabla3(self):
        if (self.timage):
            dic=consulta_tablas1(self.ruta_carpeta)
            
            self.tabla_d2_2.setRowCount(len(dic))
            for indice,imagenes in enumerate(dic):
                self.tabla_d2_2.setItem(indice,0,QtWidgets.QTableWidgetItem(str(imagenes["nombre"])))
                self.tabla_d2_2.setItem(indice,1,QtWidgets.QTableWidgetItem(str(imagenes["n_detection"])))
        else:
            mensaje = "the folder has not been process"
            QMessageBox.critical(self, "Error", mensaje) 
            
    def actualizartabla2(self):
        if (self.timage):
            dic2=actualizar_tabla2(self.ruta_carpeta,self.list_images[int(self.imgproyectada)])
        else:
            dic2=[]
        self.tabla2_3.clearContents()
        self.tabla2_3.setRowCount(len(dic2))
        #print(dic2)
        for indice,imagenes in enumerate(dic2):
            
            self.tabla2_3.setItem(indice,0,QtWidgets.QTableWidgetItem(str(imagenes["pixel_min"])))
            self.tabla2_3.setItem(indice,1,QtWidgets.QTableWidgetItem(str(imagenes["pixel_max"])))
            self.tabla2_3.setItem(indice,2,QtWidgets.QTableWidgetItem(str(imagenes["lat"])))
            self.tabla2_3.setItem(indice,3,QtWidgets.QTableWidgetItem(str(imagenes["long"])))

    def actualizartabla1(self):
        if (self.timage):
            dic2=actualizar_tabla2(self.ruta_carpeta,self.list_images[int(self.imgproyectada)])
        else:
            dic2=[]
        self.tabla_r.clearContents()
        self.tabla_r.setRowCount(len(dic2))
        #print(dic2)
        for indice,imagenes in enumerate(dic2):
            
            self.tabla_r.setItem(indice,0,QtWidgets.QTableWidgetItem(str(imagenes["pixel_min"])))
            self.tabla_r.setItem(indice,1,QtWidgets.QTableWidgetItem(str(imagenes["pixel_max"])))
            self.tabla_r.setItem(indice,2,QtWidgets.QTableWidgetItem(str(imagenes["lat"])))
            self.tabla_r.setItem(indice,3,QtWidgets.QTableWidgetItem(str(imagenes["long"])))

    def downloadshape(self):
        if (self.timage or self.hact):
            #print("hola1")
            convertir_a_shapefile(self.ruta_carpeta)
            mensaje = "Generated Shape"
            QMessageBox.information(self, "information", mensaje)
        else: 
            mensaje = "the folder has not been process"
            QMessageBox.critical(self, "Error", mensaje)
            
    def downloadcsv(self):
        #print("hola2")
        if (self.timage or self.hact):
            enumerar_en_csv(self.ruta_carpeta)
            mensaje = "Generated CSV"
            QMessageBox.information(self, "information", mensaje)
        else: 
            mensaje = "the folder has not been process"
            QMessageBox.critical(self, "Error", mensaje)

    def mostrar_estadisticas(self):
        historial=self.list_combo.currentText()
        historial=historial.split("_")[0]
        if (historial!="--------------------"):
            registro=consulta_porfecha(historial)
            self.tabla_d2_3.clearContents()
            self.tabla_d2_3.setRowCount(len(registro))
            for indice,regist in enumerate(registro):
                self.tabla_d2_3.setItem(indice,0,QtWidgets.QTableWidgetItem(regist["nombre"]))
                self.tabla_d2_3.setItem(indice,1,QtWidgets.QTableWidgetItem(str(regist["cant"])))
            # Graficar 
            if (len(registro)>0):
                self.label_history.clear()
                valores = [dato['cant'] for dato in registro]
                etiquetas = ['Not Sigatoka', 'Sigatoka']
                comparar_cero = lambda valor: valor == 0
                cantidad_ceros = len(list(filter(comparar_cero, valores)))
                comparar_ncero = lambda valor: valor > 0
                cantidad_nceros = len(list(filter(comparar_ncero, valores)))
                v_g=[cantidad_ceros,cantidad_nceros]
                fig, ax = plt.subplots()
                wedges, texts, autotexts = ax.pie(v_g, labels=etiquetas, colors=['#ffcc99', '#99ff99'], autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct/100*sum(v_g), 0))})",
                                    textprops={'fontsize': 10}, startangle=90)
                plt.setp(autotexts, size=8, weight='bold')
                ax.set_title('Areas with Sigatoka')
                canvas = FigureCanvas(fig)
                canvas.draw()
                pixmal=canvas.grab()
                pixmal=pixmal.scaled(450,259, transformMode=Qt.TransformationMode.SmoothTransformation)
                self.label_history.setPixmap(pixmal)

            conn = sqlite3.connect('./library_new/test.db')
            cursor = conn.cursor()

            # Obtener la ruta de carpeta por fecha en la primera tabla
            fecha = str(historial)
            cursor.execute("SELECT ruta_carpeta FROM registro_carpeta WHERE fecha = ?", (fecha,))
            resultado = cursor.fetchone()

            conn.close()
            if (resultado):
                self.ruta_carpeta=resultado[0]
            self.hact=1
        else:
            self.tabla_d2_3.clearContents()
            self.tabla_d2_3.setRowCount(0)
            self.label_history.clear()

    def borrarselect(self):
        historial=self.list_combo.currentText()
        historial=historial.split("_")[0]
        if (historial!="--------------------"):
            borrar_porfecha(historial)
            self.historial()

    def pro_clasification(self):
    
        if (self.ruta_carpeta):    
            #necesitamos direccion para ller la variable
            # variable de detección
            self.detect_p=1
            clasificacion(self.ruta_carpeta,self.modelc,self.list_images)
            self.timage=1
            self.proyect_image()
            self.llenartabla3()
            mensaje = "complete"
            QMessageBox.information(self, "information", mensaje)

        else:
            mensaje = "the folder has not been selected"
            QMessageBox.critical(self, "Error", mensaje)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = mainUI()
    ui.show()
    app.exec()



"""  # Para llamar desde el otro codigo que genera automaticamente puic6
 class MiApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow() 
        self.ui.setupUi(self)

if __name__ == "__main__":
     app = QtWidgets.QApplication(sys.argv)
     mi_app = MiApp()
     mi_app.show()
     app.exec() """