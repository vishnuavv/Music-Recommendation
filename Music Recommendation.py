#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pylab as pl
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#Reading playlist data from .txt file
PlaylistsData = [line.strip().split(" ") for line in open('playlists.txt').readlines()]

Playlists = []
for p in PlaylistsData:
    q = [int(x) for x in p]
    Playlists.append(q)

#Reading song names from .txt file 
songnames = np.genfromtxt('song_names.txt', delimiter='\t', dtype=str, usecols=(1))

#Making Dictionary of Song number with Song Title & Song Artist
SongTitleArtist = {}
for i in range(len(songnames)):
    SongTitleArtist[i] = songnames[i]
	
L_playlist, sup_playlist = apriori(Playlists, 0.002)


#Generating rules from the frequent itemsets discovered in the previous stepusing the 'lift' metric with a minimum lift value of 20.0
ruleslist = generateRules(L_playlist, sup_playlist, metric='lift', minMetric=20.0)

pntRules(ruleslist, SongTitleArtist)




