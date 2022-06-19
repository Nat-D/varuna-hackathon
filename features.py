
import numpy as np

# Normalized Difference Vegetation Index (NDVI)
def ndvi(raw_spectrum):
    b4 = raw_spectrum['b4'].astype(np.float16)
    b8 = raw_spectrum['b8'].astype(np.float16)
    
    ndvi = (b8-b4) / (b8+b4 + 1e-5) 

    return ndvi

# Green Normalized Difference Vegetation Index (GNDVI)
def gndvi(raw_spectrum):
    b3 = raw_spectrum['b3']
    b8 = raw_spectrum['b8']
    
    ndvi = (b8-b3)/(b8+b3+1e-8)

    return gndvi

# Enhanced Vegetation Index (EVI)
def evi(raw_spectrum):
    b2 = raw_spectrum['b2']
    b4 = raw_spectrum['b4']
    b8 = raw_spectrum['b8']

    evi = 2.5*((b8-b4)/(b8+6*b4-7.5*b2+1))
    return evi

# Advanced Vegetation Index (AVI)
def avi(raw_spectrum):
    b4 = raw_spectrum['b4']
    b8 = raw_spectrum['b8']

    avi = (b8*(1.-b4)*(b8-b4))**(1/3)
    return avi

# Soil Adjusted Vegetation Index (SAVI)
def savi(raw_spectrum):
    b4 = raw_spectrum['b4']
    b8 = raw_spectrum['b8']

    savi = (b8-b4)/(b8+b4+0.428)*(1.428)
    return savi

# Normalized Difference Moisture Index (NDMI)
def ndmi(raw_spectrum):
    b8 = raw_spectrum['b8']
    b11 = raw_spectrum['b11']

    ndmi = (b8-b11)/(b8+b11+1e-8)
    return ndmi

# Moisture Stress Index (MSI)
def msi(raw_spectrum):
    b8 = raw_spectrum['b8']
    b11 = raw_spectrum['b11']

    msi = b11/(b8+1e-8)
    return msi

# Green Coverage Index (GCI)
def gci(raw_spectrum):
    b3 = raw_spectrum['b3']
    b8 = raw_spectrum['b8']

    gci = (b8/(b3+1e-8))-1
    return gci

# Normalized Burned Ratio Index (NBRI)
def nbri(raw_spectrum):
    b8 = raw_spectrum['b8']
    b12 = raw_spectrum['b12']

    nbri = (b8-b12)/(b8+b12+1e-8)
    return nbri

# Bare Soil Index (BSI)
def bsi(raw_spectrum):
    b2 = raw_spectrum['b2']
    b4 = raw_spectrum['b4']
    b8 = raw_spectrum['b8']
    b11 = raw_spectrum['b11']

    bsi = ((b11+b4)-(b8+b2))/((b11+b4)+(b8+b2)+1e-8)
    return bsi

# Normalized Difference Water Index (NDWI)
def ndwi(raw_spectrum):
    b3 = raw_spectrum['b3']
    b8 = raw_spectrum['b8']

    ndwi = (b3-b8)/(b3+b8+1e-8)
    return ndwi

# Normalized Difference Snow Index (NDSI)
def ndsi(raw_spectrum):
    b3 = raw_spectrum['b3']
    b11 = raw_spectrum['b11']

    ndsi = (b3-b11)/(b3+b11+1e-8)
    return ndsi

# Normalized Difference Glacier Index (NDGI)
def ndgi(raw_spectrum):
    b3 = raw_spectrum['b3']
    b4 = raw_spectrum['b4']

    ndgi = (b3-b4)/(b3+b4+1e-8)
    return ndgi

# Atmospherically Resistant Vegetation Index (ARVI)
def arvi(raw_spectrum):
    b2 = raw_spectrum['b2']
    b4 = raw_spectrum['b4']
    b8 = raw_spectrum['b8']

    arvi = (b8-(2*b4)+b2)/(b8+(2*b4)+b2+1e-8)
    return arvi

# Structure Insensitive Pigment Index (SIPI)
def sipi(raw_spectrum):
    b2 = raw_spectrum['b2']
    b4 = raw_spectrum['b4']
    b8 = raw_spectrum['b8']

    sipi = (b8-b2)/(b8-b4+1e-8)
    return sipi