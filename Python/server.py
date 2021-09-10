from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import json

from mesa import Agent, Model 
from mesa.space import MultiGrid
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector

import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams["animation.html"] = "jshtml"
matplotlib.rcParams['animation.embed_limit'] = 2**128


import numpy as np
import pandas as pd

import random
import math

import time
import datetime

#Alto
m = 15
#Ancho
n = 15

size = m*n
#Tiempo maximo de ejecucion
tiempoMax = 0.1

#Grid para crear los carros/banquetas/calles, etc.
roads = [
         0,0,0,0,0,3,5,1,0,3,0,0,0,0,0,
         0,0,0,0,0,3,0,1,0,3,0,0,0,0,0,
         0,0,0,0,0,3,0,1,0,3,0,0,0,0,0,
         0,0,0,0,0,3,0,1,0,3,0,0,0,0,0,
         0,0,0,0,0,3,0,1,0,3,0,0,0,0,0,
         3,3,3,3,3,3,2,4,0,3,3,3,3,3,3,
         0,0,0,0,0,0,0,0,0,2,0,0,0,0,5,
         1,1,1,1,1,4,0,0,0,4,1,1,1,1,1,
         0,0,5,0,0,2,0,0,0,0,5,0,0,0,0,
         3,3,3,3,3,3,0,4,2,3,3,3,3,3,3,
         0,0,0,0,0,3,0,1,5,3,0,0,0,0,0,
         0,0,0,0,0,3,0,1,0,3,0,0,0,0,0,
         0,0,0,0,0,3,0,1,0,3,0,0,0,0,0,
         0,0,0,0,0,3,0,1,0,3,0,0,0,0,0,
         0,0,0,0,0,3,0,1,0,3,0,0,0,0,0,
        ]

'''
GLOBAL LIVE - Todas las clases tienen el atributo live para poder compararlas 
y buscarlas

ROAD = 0
ROAD A LA QUE NO TE MUEVES = 1
ROAD DONDE TE PARAS AL SEMAFORO = 2
BANQUETA = 3
SEMAFORO = 4
CAR = 5
'''
#Funcion auxiliar para crear el grid apartir de nuestros valores
def getFloor():
    global roads
    val = roads.pop(0)
    return val

#Diccionario con las direcciones para facilitarnos 
way = {
    "up" : (-1,0),
    "left" : (0,-1),
    "down" : (1,0),
    "right" : (0,1),
}

#Sirve para asignarles las sensorCells a los semaforos
dirSens = [(9,8),(6, 9),(8,5),(5, 6)]


#Sirve para asginarles las direcciones iniciales a los carros cuando aparecen
dirCars = [way['down'],way['left'],way['right'],way['right'],way['up']]


#Funciones auxiliares para ir asignandoles a los semaforos/carros sus valores iniciales
def dirS():
    global dirSens
    val = dirSens.pop(0)
    return val

def dirC():
    global dirCars
    val = dirCars.pop(0)
    return val


#Checamos si esta vacia (algo ademas de una calle normal)
def checkEmpty(tupla):
    #Si hay robot no nos podemos mover ahi
    for agentt in model.grid.get_cell_list_contents(tupla):
        if agentt.live > 0:
            return False
    return True


#Checamos la celda (tupla) para ver si se encuentra un agente de tipo val (parametro entrada)
def checkCell(tupla,val):
    #Si hay robot no nos podemos mover ahi
    for agentt in model.grid.get_cell_list_contents(tupla):
        if agentt.live == val:
            return True
    return False


#Funcion auxiliar para la visualizacion/simulacion, aqui podemos cambiar los colores de los agentes
def get_grid(model):
    grid = np.zeros((model.grid.width, model.grid.height))

    #Por todas las celdas del grid
    for cell in model.grid.coord_iter():
        cell_content, x, y = cell

        #Por todos los agentes de la celda
        for agentss in cell_content:
            grid[x][y] = agentss.live

            if agentss.live == 4:
                if agentss.active:
                    grid[x][y] = 50
                else:
                    grid[x][y] = 90

            if agentss.live == 3:
                grid[x][y] = 100

            if agentss.live == 1:
                grid[x][y] = 70

            if agentss.live == 2:
                grid[x][y] = 75

            if agentss.live == 5:
                grid[x][y] = 25
    return grid

class Car(Agent):
    def __init__(self, unique_id, model,x,y,direction):
        super().__init__(unique_id, model)
        self.live = 5
        self.direction = direction
        self.cord = (x,y)

        #Para escoger que va a hacer en la interseccion
        self.choice = random.randrange(1,4)

        #Para las coordenadas para dar la vuelta
        self.cordTurns = []
 
    #Sirve para moverse en la direccion recta
    def move(self):
        goBack = self.cord
        newx = self.cord[0]+self.direction[0]
        newy = self.cord[1]+self.direction[1]

        self.cord = (newx,newy)

        #Adjust for toroidal space, se resetea al otro carril y escoge de nuevo que tipo de vuelta va a realizar
        if (self.cord[0]>14):
            self.cord = (0,self.cord[1])
            self.choice = random.randrange(1,4)

        if (self.cord[1]>14):
            self.cord = (self.cord[0],0)
            self.choice = random.randrange(1,4)


        if (self.cord[0]<0):
            self.cord = (14,self.cord[1])
            self.choice = random.randrange(1,4)


        if (self.cord[1]<0):
            self.cord = (self.cord[0],14)
            self.choice = random.randrange(1,4)


        #Si hay algo que no sea 0 o 2 no se mueve (hay banqueta, carro o semaforo)
        if not checkCell(self.cord,5) and not checkCell(self.cord,4) and not checkCell(self.cord,3) and not checkCell(self.cord,1):
            model.grid.move_agent(self, self.cord)
        else:
            self.cord = goBack


    #Sirve para checar el semaforo, ( para checar en que direccion es la izquierda)
    def checkNextDir(self):
        global way
        index = 0
        for name, tupla in way.items(): 
            if tupla == self.direction:
                index = name

        temp = list(way)

        try:
            return way[temp[temp.index(index) + 1]]

        except (ValueError, IndexError):
            return way[temp[0]]


    #Se va a la celda del semaforo relevante (de enfrente a la izquierda) para checar si esta prendido o no
    #El carro es el que checa porque asi se simula en vida real
    def semaforoActive(self):
        semx = self.cord[0]+self.direction[0]*4+self.checkNextDir()[0]
        semy = self.cord[1]+self.direction[1]*4+self.checkNextDir()[1]
    
        for agentt in model.grid.get_cell_list_contents((semx,semy)):
            if agentt.live == 4:

                if agentt.active == True:
                    #print("El semaforo esta activo",semx,semy)
                    return True
                else:
                    #print("El semaforo no esta activo",semx,semy)
                    return False

        return False


    def intersection(self):
        global way
        '''
        Dependiendo de lo que se escoge al azar cada vez que se "sale" del grid,
        escoge una opcion nueva 
        1 = Go straight (Necesita Semaforo)
        2 = Short Turn (No Necesita Semaforo)
        3 = Long Turn (Necesita Semaforo)
        4 = U-Turn (Necesita Semaforo)        
        '''
        if self.semaforoActive():

            #GO STRAIGHT
            if self.choice == 1:
                for x in range(4):
                    newx = self.cord[0]+(self.direction[0]*(x+1))
                    newy = self.cord[1]+(self.direction[1]*(x+1))
                    self.cordTurns.append((newx,newy))

            #SHORT TURN - NORMALMENTE NO SE NECESITA SEMAFORO PERO ESTABAN CHOCANDO MUCHO
            if self.choice == 2:
                newx = self.cord[0]+(self.direction[0])
                newy = self.cord[1]+(self.direction[1])
                dir = (newx,newy)
                
                self.cordTurns.append((dir))

                if self.direction == way['down']:
                    self.direction = way['left']

                elif self.direction == way['right']:
                    self.direction = way['down']

                elif self.direction == way['left']:
                    self.direction = way['up']

                elif self.direction == way['up']:
                    self.direction = way['right']

                self.cord = self.cordTurns[-1]

                newx = self.cord[0]+(self.direction[0])
                newy = self.cord[1]+(self.direction[1])
                self.cordTurns.append((newx,newy))

            #LONG TURN
            elif self.choice == 3:
                for x in range(3):
                    newx = self.cord[0]+(self.direction[0]*(x+1))
                    newy = self.cord[1]+(self.direction[1]*(x+1))
                    self.cordTurns.append((newx,newy))

                #Se puede hacer con next value pero se me hizo mas facil asi
                if self.direction == way['down']:
                    self.direction = way['right']

                elif self.direction == way['right']:
                    self.direction = way['up']

                elif self.direction == way['left']:
                    self.direction = way['down']

                elif self.direction == way['up']:
                    self.direction = way['left']

                self.cord = self.cordTurns[-1]
                

                for x in range(4):
                    newx = self.cord[0]+(self.direction[0]*(x+1))
                    newy = self.cord[1]+(self.direction[1]*(x+1))
                    self.cordTurns.append((newx,newy))

            #U TURN
            elif self.choice == 4:

                temp = self.direction
               

                newx = self.cord[0]+(self.direction[0])
                newy = self.cord[1]+(self.direction[1])
                self.cordTurns.append((newx,newy))

                #Se puede hacer con next value pero se me hizo mas facil asi
                if self.direction == way['down']:
                    self.direction = way['right']

                elif self.direction == way['right']:
                    self.direction = way['up']

                elif self.direction == way['left']:
                    self.direction = way['down']

                elif self.direction == way['up']:
                    self.direction = way['left']

                self.cord = self.cordTurns[-1]
                
                for x in range(3): 
                    newx = self.cord[0]+(self.direction[0]*(x+1))
                    newy = self.cord[1]+(self.direction[1]*(x+1))
                    self.cordTurns.append((newx,newy))

                if self.direction == way['down']:
                    self.direction = way['right']

                elif self.direction == way['right']:
                    self.direction = way['up']

                elif self.direction == way['left']:
                    self.direction = way['down']

                elif self.direction == way['up']:
                    self.direction = way['left']

                nex = self.cord[0]+(self.direction[0])
                ney = self.cord[1]+(self.direction[1])
                self.cordTurns.append((nex,ney))
            
            self.cord = self.cordTurns[-1]

    #Se esta moviendo a traves de la interseccion
    def moveInter(self):
        if checkEmpty(self.cordTurns[0]):
                nextCordy = self.cordTurns.pop(0)
                model.grid.move_agent(self, nextCordy)

    def step(self):
        #Si esta haciendo una vuelta/movimientos en la interseccion
        if self.cordTurns:
            self.moveInter()
            return


        #Checamos si esta en un "Stop"
        for agentt in model.grid.get_cell_list_contents(self.cord):
            if agentt.live == 2:
                #Se crean las coordenadas para la vuelta      
                self.intersection()
                return

        #Si no esta en un stop o moviendose en la vuelta se mueve en su direccion
        self.move()
        
    def advance(self):
        ''''''

class Banqueta(Agent):
    def __init__(self, unique_id, model,x,y):
        super().__init__(unique_id, model)
        self.live = 3
    
    def step(self):
        ''''''
    
    def advance(self):
        ''''''

class Road(Agent):
    def __init__(self, unique_id, model,x,y,tipo):
        super().__init__(unique_id, model)
        self.live = tipo
    
    def step(self):
        ''''''
    
    def advance(self):
        ''''''

queue = []
class Semaforo(Agent):
    def __init__(self, unique_id, model,x,y,sensorCell):      
        super().__init__(unique_id, model)
        self.live = 4
        self.sensorCell = sensorCell
        self.nextQ = None
        self.active = False
        self.wait = 7
    
    #Checa la "computadora" para ver si hay algun otro semaforo prendido
    def someoneActive(self):
        global arrSemaforos
        for sem in arrSemaforos:
            if sem.active == True:
                return True
        return False

    #Setter para apagar o prender
    def turnOff(self):
        self.active = False
    
    def turnOn(self):
        self.active = True

    def step(self):
        global arrSemaforos, queue

        #Si esta activo solo dura 7 turnos activo
        if self.active:
            if self.wait == 0:
                self.wait = 7
                self.active = False
                queue.pop(0)
                return

            #Si ya no hay nadie en el sensor, se quita para la wait, 
            #Si si hay alguien se queda donde mismo (+1 turno)
            if not checkCell(self.sensorCell,5):
                self.wait -= 1
            return

        #Si hay carro en su sensorCell
        if checkCell(self.sensorCell,5):
            if self not in queue:
                queue.append(self)

            #No hay nadie en la casilla
            if queue[0] == self:
                #Just in case turn off all the others
                for sem in arrSemaforos:
                    sem.turnOff()

                self.active = True

        #Si ya no hay nadie y esta en la queue, se quita asimismo de la queue
        else:
            if self in queue:
                queue.remove(self)

    
    def advance(self):
        ''''''

arrSemaforos = []

class FloorGrid(Model):
    #Define el modelo del piso/habitacion
    def __init__(self, width, height):
        self.num_agents = width * height
        self.grid = MultiGrid(width, height, True)
        self.schedule = SimultaneousActivation(self)
        
        for (content, x, y) in self.grid.coord_iter():
            value = getFloor()

            if value == 3:
                tempB = Banqueta((x, y), self, x,y)
                self.grid.place_agent(tempB, (x, y))
                self.schedule.add(tempB)

            elif value == 4:
                tempS = Semaforo((x, y), self, x, y, dirS())
                self.grid.place_agent(tempS, (x, y))
                self.schedule.add(tempS)

                           
                global arrSemaforos
                arrSemaforos.append(tempS)
     

            elif value == 5:
                tempC = Car((x, y), self, x,y,dirC())
                self.grid.place_agent(tempC, (x, y))
                self.schedule.add(tempC)

            else:
                tempR = Road((x, y), self, x,y,value)
                self.grid.place_agent(tempR, (x, y))
                self.schedule.add(tempR)
        
        # Aquí definimos con colector para obtener el grid completo.
        self.datacollector = DataCollector( model_reporters={"Grid": get_grid} )
    
    def step(self):
        #En cada paso el colector tomará la información que se definió y almacenará el grid para luego graficarlo.
        self.datacollector.collect(self)
        self.schedule.step()

        gridInfo = []
        tempInfo = []

        for cell in self.grid.coord_iter():
            cell_content, x, y = cell
            for content in cell_content:
                if isinstance(content, Car):
                    temp = ''
                    if content.direction == way['down']:
                        temp = 'down'
                    elif content.direction == way['right']:
                        temp = 'right'
                    elif content.direction == way['left']:
                        temp ='left'
                    elif content.direction == way['up']:
                        temp = 'up'
                    gridInfo.append([0, str(content.unique_id[0])+str(content.unique_id[1]), y, x, True, temp])
                elif isinstance(content, Semaforo):
                    tempInfo.append([1, str(content.unique_id[0])+str(content.unique_id[1]), y, x, content.active, 'NULL'])
        gridInfo += tempInfo
        return gridInfo

# Registramos el tiempo de inicio y corremos el modelo
model = FloorGrid(m, n)

#                                                                               Server


def gridToJSON(grid):
    gridDICT = []
    for coor in grid:
        coor_ = {
            "type" : coor[0],
            "id" : coor[1],
            "x" : coor[2],
            "y" : coor[3],
            "active" : coor[4],
            "direction" : coor[5]
        }
        gridDICT.append(coor_)
    return json.dumps(gridDICT)


class Server(BaseHTTPRequestHandler):
    
    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
    def do_GET(self):
        logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))
        self._set_response()
        self.wfile.write("GET request for {}".format(self.path).encode('utf-8'))

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        #post_data = self.rfile.read(content_length)
        post_data = json.loads(self.rfile.read(content_length))
        #logging.info("POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n",
                     #str(self.path), str(self.headers), post_data.decode('utf-8'))
        logging.info("POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n",
                     str(self.path), str(self.headers), json.dumps(post_data))
        
        gridInfo = model.step()
        #print(gridInfo)
        self._set_response()
        resp = "{\"Items\":" + gridToJSON(gridInfo) + "}"
        #print(resp)
        self.wfile.write(resp.encode('utf-8'))


def run(server_class=HTTPServer, handler_class=Server, port=8585):
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info("Starting httpd...\n") # HTTPD is HTTP Daemon!

    
    # f = open("output.txt", "a")
    # print(gridToJSON(model.step()), file=f)
    # f.close()
    try:
        httpd.serve_forever()

    except KeyboardInterrupt:   # CTRL+C stops the server
        pass
    httpd.server_close()
    logging.info("Stopping httpd...\n")

if __name__ == '__main__':
    from sys import argv
    
    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()

