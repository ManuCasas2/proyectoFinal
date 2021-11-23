import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
 
path = "ImagenesAsistencia"
imagenes = []
nombrefotos = []
miLista = os.listdir(path)
print(miLista)
for nf in miLista:
    ftactual = cv2.imread(f'{path}/{nf}')
    imagenes.append(ftactual)
    nombrefotos.append(os.path.splitext(nf)[0])
print(nombrefotos)
 
def encontrarCodigo(imagenes):
    ListaCodigo =[]
    for img in imagenes:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        codificar = face_recognition.face_encodings(img)[0]
        ListaCodigo.append(codificar)
    return ListaCodigo
 


def marcarAsistencia(nombre):
    with open('listaAsistencia.csv','r+') as f:
        listaDatos = f.readlines()
        listaNombre = []
        for linea in listaDatos:
            entrada = linea.split(',')
            listaNombre.append(entrada[0])
        if nombre not in listaNombre:
            ahora = datetime.now()
            fechaString = ahora.strftime('%H:%M:%S')
            f.writelines(f'/n{nombre},{fechaString}')

ListaCodigoConocida = encontrarCodigo(imagenes)
print('Codificado completo')

cap = cv2.VideoCapture(0)
 
while True:
    success, img = cap.read()
    TmImg = cv2.resize(img,(0,0),None,0.25,0.25)
    TmImg = cv2.cvtColor(TmImg, cv2.COLOR_BGR2RGB)
 
CarasEnPantalla = face_recognition.face_locations(TmImg)
CodigoEnPantalla = face_recognition.face_encodings(TmImg,CarasEnPantalla)
 
for codigoCara,lugarCara in zip(CodigoEnPantalla,CarasEnPantalla):
    coincidencia = face_recognition.compare_faces(ListaCodigoConocida,codigoCara)
    disCara = face_recognition.face_distance(ListaCodigoConocida,codigoCara)
    indiceCoincidencia = np.argmin(disCara)
 
if coincidencia[indiceCoincidencia]:
    nombre = nombrefotos[indiceCoincidencia].upper()
    y1,x2,y2,x1 = lugarCara
    y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
    cv2.putText(img,nombre,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    marcarAsistencia(nombre)
 
cv2.imshow('Webcam',img)
cv2.waitKey(1)