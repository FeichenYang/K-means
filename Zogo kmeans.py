#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 20:14:44 2020

@author: feichang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import datetime 

dataset = pd.read_csv("zogo.csv")
#retrieving 3 factors: number of points, referrals, and time created
#First 96 doesn't have a date created, clearly from inside
#X = dataset.iloc[96:,[4,7,11]].values
X = dataset.iloc[96:,[7,11]].values
"""
#calculate the age, leave the nan alone
for i in range(15849):
    if type(X[i,0]) == float:
        pass
    else:
        X[i,0] = 2020 - int(X[i,0][:4])
"""        

from datetime import datetime
from datetime import date

#calculate the create time, leave the nan alone, number of days
for i in range(15849):
    date_format = "%Y-%m-%d"
    date_created = datetime.strptime(X[i,0][:10],date_format)
    delta = datetime.today()-date_created
    X[i,0] = delta.days

#impute the nan in the age, with the mean
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=float('nan'), strategy='mean')
X = imp.fit_transform(X)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit_transform(X)

from sklearn.cluster import KMeans
"""
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++')
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)

elbow point is either n = 3 or 2
"""
kmeans = KMeans(n_clusters = 2, init = 'k-means++')
kmeans.fit(X)

y_k = kmeans.predict(X)

plt.scatter(X[y_k==0,0],X[y_k==0,1], s=30, color = "red")
plt.scatter(X[y_k==1,0],X[y_k==1,1], s=30, color = "blue")


