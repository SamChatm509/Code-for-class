import pandas as pd
import numpy as np

import csv
import os
import openpyxl
import re

import subprocess
import xml.etree.ElementTree as ET

from tkinter import filedialog
from tkinter import *

root = Tk()
#root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
#print (root.filename)

filename = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("CellsPCA1PCA2TMM","*.csv"),("all files","*.*")))
path =filename

genes = pd.read_csv(path, index_col = "PCA_Reduced_Genes")

geneList = genes.index.tolist()

for i in range (len(geneList)):
    print (geneList[i],  i)
pca_list = []

#tmm = pd.read_csv("TMMnorm2.csv", index_col = 'GeneName')
tmm = pd.read_csv("TMMnorm4.csv")
#'Melanoma1', 'Melanoma2', 'Melanoma3', 'Melanoma4', 'Melanoma5',
#'COLON1', 'COLON2','COLON3','COLON4','COLON5',
#'BREAST1', 'BREAST2', 'BREAST3', 'BREAST4', 'BREAST5'], columns = labels)

for rows in tmm.itertuples():
    row_list = [rows.GeneID,rows.Melanoma1,rows.Melanoma3,rows.Melanoma4, rows.Melanoma5,
                rows.COLON1,rows.COLON2,rows.COLON4,rows.COLON5,
                rows.BREAST1,rows.BREAST2,rows.BREAST3,rows.BREAST4]

    
    #obtain GeneName
    sample1 = row_list[0]
    sample2 = str(sample1).strip()
    if sample2 in geneList:
        pca_list.append(row_list)
    
    


# Build a DataFrame cars from my_dict: cars
#tmm_pca = pd.DataFrame(pca_dict)

#print(tmm_pca.head(15))

#Get the number of rows and columns
#print (tmm_pca.shape)
print (len(pca_list))
df = pd.DataFrame(pca_list, columns = ['GeneID','Melanoma1','Melanoma3', 'Melanoma4', 'Melanoma5',\
'COLON1', 'COLON2','COLON4','COLON5','BREAST1', 'BREAST2', 'BREAST3', 'BREAST4'],dtype = float)
print(df.head(15))
print(df.shape)
df.to_csv('pca_reduced_data4.csv')
