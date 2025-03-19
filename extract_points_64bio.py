"""
To divide the tidal flats into small patches based on sediment

"""
from osgeo import gdal_array as gdarr
from osgeo import gdal, osr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point
import math
import os
import glob

	
def ExtractData(img_path, vec_path):
        patches = []
        vals=[]
        img_fol=sorted(glob.glob(img_path+'*'))
        #print(img_fol)
        vec_fol=glob.glob(vec_path+'*')
        #print(vec_fol)
        for s in range(len(img_fol)):
                fol_img=os.path.basename(img_fol[s])
                fol_vec=os.path.basename(vec_fol[s])
                #print(fol_img)
                #print(fol_vec)
                if fol_vec==fol_img:
                        image_list= glob.glob(img_fol[s]+'/*.tif') #assuming gif
                        #print(image_list)
                        for j in range(len(image_list)):
                                sentin_band = image_list[j]
                                fie_vec=glob.glob(vec_fol[s]+'/*.shp')
                                vec1=gpd.read_file(fie_vec[0])
                                vec=vec1.loc[(vec1.iloc[:,7]!=0)]
                                vec=vec.reset_index(drop=True)
                                ds = gdal.Open(sentin_band, gdal.GA_ReadOnly)
                                gt = ds.GetGeoTransform()  # Geotransforms allow conversion of pixel to map coordinates
                                crs = ds.GetProjection()  
                                lon=vec.loc[:,vec.columns[3]]
                                lat=vec.loc[:,vec.columns[4]]
                                med=vec.loc[:,vec.columns[7]]#biomass
                                im_name=os.path.basename(sentin_band)
                                s1=im_name.split('.')
                                s2=s1[0].split('_')
                                for i in range(len(lon)):
                                        if s2[1]=='T32ULE':
                                                source = osr.SpatialReference()
                                                source.ImportFromEPSG(32631)  # WGS84 4326
                                                target = osr.SpatialReference()
                                                target.ImportFromEPSG(32632)
                                                transform = osr.CoordinateTransformation(source, target)
                                                mx, my, z = transform.TransformPoint(lon[i], lat[i])
                                                inv_gt = gdal.InvGeoTransform(gt)  
                                                px,py=gdal.ApplyGeoTransform(inv_gt,mx, my)
		        			# Apply the inverse GT and truncate the decimal places.
                                                px, py = (math.floor(f) for f in gdal.ApplyGeoTransform(inv_gt, mx, my))
						#print(px,py)
                                                xoff=px-32#32
                                                yoff=py-32#32
                                                win_x=64#64
                                                win_y=64#64
                                                px1=gdarr.DatasetReadAsArray(ds,xoff,yoff,win_x,win_y)
                                                patches.append(px1)
                                                vals.append(med[i])
                                        else:
                                                inv_gt = gdal.InvGeoTransform(gt)  
                                                px,py=gdal.ApplyGeoTransform(inv_gt,lon[i], lat[i])
						# Apply the inverse GT and truncate the decimal places.
                                                px, py = (math.floor(f) for f in gdal.ApplyGeoTransform(inv_gt, lon[i], lat[i]))
						#print(px,py)
                                                xoff=px-32
                                                yoff=py-32
                                                win_x=64
                                                win_y=64
                                                px1=gdarr.DatasetReadAsArray(ds,xoff,yoff,win_x,win_y)
                                                patches.append(px1)
                                                vals.append(med[i])
        return patches,vals
	
		
		
		
		
