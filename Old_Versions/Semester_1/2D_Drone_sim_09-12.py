import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def signal(k, r, a, b, St, Ht, Hr, lamba):

    #signal recived (desired and interferance)
    "k = constant"
    "r = distance between transmitter and reciver"
    "a = basic path loss exponent"
    "b = additional path loss exponent"
    "st = transmitted power"
    "g = break point of pathloss"
    "ht = hight of transmitter"
    "hr = hight of reciver"
    "lamba = wavelength of carrier"

    g = (4 * Ht * Hr)/lamba

    return (k/(r^a(1+r/g)^b))*St

#increase no. of UAVs to find best no.
#need to smart select the best transmitter

#need to have uav in rand pos
#need to increase no. uavs
#need to select closes transmitter
#work out SINR with signal function
#

# hard coded
## defining drone plane, only 9 spots a drone can be (including above the BS) ##
drone_location = [
    [0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 1, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 1, 1, 1, 1]]

