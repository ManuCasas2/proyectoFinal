import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
 
path = "ImagenesAsistencia"                                                                         #hacer una lista con la cant de fotos que tengo en carpeta, ordenarlas y archivarlas para usarlas despues
imagenes = []                                                                                       #lista de imagenes
nombrefotos = []                                                                                    #lista con los nombres de las imagenes(sin .jpg)
miLista = os.listdir(path)                                                                          #lista con los nombres de las imagenes(con .jpg)
#print(miLista)
for nf in miLista:                                             
    ftactual = cv2.imread(f'{path}/{nf}')              
    imagenes.append(ftactual)                                                                       #agrega la imagen en la lista imagenes
    nombrefotos.append(os.path.splitext(nf)[0])                                                     #Saca el .jpg y deja solo el nombre de la gente en la lista nombrefotos. Va tomando de a un elemento
print(nombrefotos)
 
def encontrarCoordenadas(imagenes):                                  
    ListaCoordenadas =[]                                                                            #lista vacia que va a tener las coordenadas de cada cara
    for img in imagenes:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        codificar = face_recognition.face_encodings(img)[0]                                         #busca las coordenadas de c/foto
        ListaCoordenadas.append(codificar)                                                          #guarda estas coordenadas en la lista
    return ListaCoordenadas
 
cap = cv2.VideoCapture(0)

def marcarAsistencia(nombre):
    with open('listaAsistencia.csv','r+') as f:                                                     #r+ es para escribir y leer al mismo tiempo en el archivo excel
        listaDatos = f.readlines()
        listaNombre = []
        for linea in listaDatos:
            entrada = linea.split(",")
            listaNombre.append(entrada[0])                                                          #entrada[0] representa el 1er elemento de la lista que es el nombre
        if nombre not in listaNombre:                                                               #agrega a la persona al archivo si todavia no la agregó
            ahora = datetime.now()
            fechaString = ahora.strftime("%H:%M:%S")                                                #solo toma la hora del dia, sin la fecha
            f.writelines(f'\n{nombre},{fechaString}')         

ListaCoordenadasConocidas = encontrarCoordenadas(imagenes)
print('Codificado completo')

while True:
   success, img = cap.read()
   TmImg = cv2.resize(img,(0,0),None,0.25,0.25)                                                     #Achica el cuadrado de lectura para una mejor definicion
   TmImg = cv2.cvtColor(TmImg, cv2.COLOR_BGR2RGB)
    
   CarasEnPantalla = face_recognition.face_locations(TmImg)                                         #convertir el frame del video en una imagen
   CoordenadaEnPantalla = face_recognition.face_encodings(TmImg,CarasEnPantalla)                    #busca las coordenadas de la nueva imagen
 
   for coordenadaCara,lugarCara in zip(CoordenadaEnPantalla,CarasEnPantalla):                  
       coincidencia = face_recognition.compare_faces(ListaCoordenadasConocidas,coordenadaCara)      #compara los valores de la lista con los valores de la camara
       disCara = face_recognition.face_distance(ListaCoordenadasConocidas,coordenadaCara)           #comparar la distancia entre las facciones de las caras de la lista con las de la camara
       #print(disCara)
       indiceCoincidencia = np.argmin(disCara)                                                      #busca el menor valor de la lista que dio en el paso anterior y esa es la mejor coincidencia
       if coincidencia[indiceCoincidencia]:
           nombre = nombrefotos[indiceCoincidencia].upper()
           #print(nombre)
           y1,x2,y2,x1 = lugarCara
           y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4                                                     #reagranda la foto a su tamaño original
           cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)                                           #hace un rectangulo con la localizacion de la cara de la imagen
           cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)                               #Hace un rectangulo en la parte inferior del rectangulo anterior donde se va a mostrar el nombre de la persona
           cv2.putText(img,nombre,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)           #Escribe el nombre en el rectangulo
           marcarAsistencia(nombre)
       cv2.imshow('Webcam',img)
       cv2.waitKey(1)