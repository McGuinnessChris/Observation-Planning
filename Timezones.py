#!/usr/bin/env python3

import numpy as np
import astroplan
import astropy
import matplotlib.pyplot as plt
import datetime as dt
import pytz
from astropy.time import Time

time = dt.datetime.now()
print('Current time is' , time)

Dublin = pytz.timezone('Europe/Dublin')
print('Time in Dublin =' , Time(time))

Los_Angeles = pytz.timezone('America/Los_Angeles')
Los_Angeles_Time = time.astimezone(Los_Angeles)
print('Time in Los Angeles =' , Time(Los_Angeles_Time))

Hong_Kong = pytz.timezone('Hongkong')
Hong_Kong_Time = time.astimezone(Hong_Kong)
print('Time in Hong Kong =' , Time(Hong_Kong_Time))

Hawaii = pytz.timezone('US/Hawaii')
Hawaii_Time = time.astimezone(Hawaii)
print('Time in Hawaii =' , Time(Hawaii_Time))
