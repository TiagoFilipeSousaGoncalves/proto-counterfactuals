# Imports
import os
import tqdm
import pandas as pd
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
import skimage.draw as drw
import cv2



# Utilities and Helpers
# Function: Prepare the Data Frame to be readable
def _fix_df(df):
    """
    Prepare the Data Frame to be readable
    """
    df_new = df.drop(['ID'], axis=0)
    df_new.columns = df_new.iloc[0,:]
    df_new.drop([np.nan], axis=0, inplace=True)
    df_new.columns.name = 'ID'
    return df_new 



# Function: Return mask given a contour and the shape of image
# contour_to_mask() and apply_mask() functions taken from https://github.com/matterport/Mask_RCNN
def contour_to_mask(contour_txt_path, img_shape):
    """
    Return mask given a contour and the shape of image
    """
    
    # Get coordinates
    c = np.loadtxt(contour_txt_path)
    
    # Generate mask
    mask = np.zeros(img_shape[:-1], dtype=np.uint8)
    rr, cc = drw.polygon(c[:,1], c[:,0])
    mask[rr, cc] = 1
    
    return mask



# Function: Apply the given mask to the image
def apply_mask(image, mask, color, alpha=0.5):
    """
    Apply the given mask to the image.
    """


    for c in range(3):
        image[:, :, c] = np.where(mask == 1, image[:, :, c] * (1 - alpha) + alpha * color[c] * 255, image[:, :, c])


    return image



# Function: Read clinical data
def read_clinical_data(patient_data_od_path, patient_data_os_path):
    """
    Return excel data as pandas Data Frame
    """
    
    df_od = pd.read_excel(patient_data_od_path, index_col=[0])
    df_os = pd.read_excel(patient_data_os_path, index_col=[0])


    return _fix_df(df=df_od), _fix_df(df=df_os)



# Function: Return three arrays of shape 488 with the diagnosis tag, eye ID (od, os) and patient ID
def get_diagnosis(patient_data_od_path, patient_data_os_path):
    """
    Return three arrays of shape 488 with the diagnosis tag, eye ID (od, os)
    and patient ID
    """
    df_od, df_os = read_clinical_data(patient_data_od_path, patient_data_os_path)
        
    index_od = np.ones(df_od.iloc[:,2].values.shape, dtype=np.int8)
    index_os = np.zeros(df_os.iloc[:,2].values.shape, dtype=np.int8)

    eyeID = np.array(list(zip(index_od, index_os))).reshape(-1)
    tag = np.array(list(zip(df_od.iloc[:,2].values, df_os.iloc[:,2].values))).reshape(-1)
    patID = np.array([[int(i.replace('#', ''))] * 2 for i in df_od.index]).reshape(-1)


    return tag, eyeID, patID