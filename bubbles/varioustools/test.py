#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 19:59:25 2020

@author: alienor
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as C

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
print(len(colors))
print(colors[1])
f = np.linspace(0, 1, 100)
#f2 = np.linspace(1, 0, 50)
#f = np.array(list(f1) + list(f2))
x = np.linspace(0, 1, 100)

f = np.abs(f-1/2)
testcolor = np.zeros_like(x, dtype=tuple)

a, b, c = C.colorConverter.to_rgb(colors[0])
for i in range(len(testcolor)):
    testcolor[i] = (a + (1-a)*f[i], b + (1-b)*f[i], c + (1-c)*f[i])


print(testcolor[0])
print(testcolor[10])

plt.figure()
for i in range(len(f)):
#    col = C.colorConverter.to_rgb(colors[i])
#    print(len(col))
#    a, b, c = col
#    print(type(a))
#    col = tuple(col, dtype = float)
#    print(type(col))
#    print(col)
    plt.plot(x[i], x[i], 'o', color = testcolor[i])
    
plt.show()
