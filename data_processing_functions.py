import pandas as pd
import bz2
from datetime import datetime,timedelta
import tarfile
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from math import pi, sin
import tarfile
import cv2


def breakdown_dates(df,column): 
    df['year'] = pd.DatetimeIndex(df[column]).year
    df['month'] = pd.DatetimeIndex(df[column]).month
    df['day'] = pd.DatetimeIndex(df[column]).day
    df['hour'] = pd.DatetimeIndex(df[column]).hour
    df['minute'] = pd.DatetimeIndex(df[column]).minute
    return

seasons = {2:[3,4,5],
           3:[6,7,8],
           4:[9,10,11],
           1:[12,1,2]}

def replace_m_w_season(row):
    for s in seasons.keys():
#         print(s,seasons[s],row)
        if row in seasons[s]:
            return s
        else:
            pass

def datetime_blank_min_before(df,dt):
    df[f'{dt}_min_before'] = pd.DatetimeIndex(df['timestamp']) - timedelta(minutes=dt)
    return 

def numberOfDays(y, m):
    leap = 0
    if y% 400 == 0:
        leap = 1
    elif y % 100 == 0:
        leap = 0
    elif y% 4 == 0:
        leap = 1
    if m==2:
        return 28 + leap
    list = [1,3,5,7,8,10,12]
    if m in list:
        return 31
    return 30

def make_image_path(row):
    files = {}
    files['higher_file_path'] = make_higher_image_path(row)
    files['lower_file_path'] = make_lower_image_path(row)
    return files

def get_all_file_names(li_file_names,file_name_dict):
    for i in li_file_names:
        if '.DS_Store' in i: 
            pass
        elif len(i) > 4 and len(i) <= 7:
            yrmn = i 
            file_name_dict[i] = {}
        elif len(i) > 7 and len(i) <= 10:
            yrmnd = i
            file_name_dict[yrmn][i] = []
        elif len(i) > 10:
            try:
                file_name_dict[yrmn][yrmnd].append(i)
            except:
                print(i)
                break
        else:
            pass


# 2014/12/29/20141229_170300.jpg
def make_higher_image_path(row):
    mn,day,hour,mi,yr = (int(row["month"]),int(row["day"]),
                      int(row["hour"]),int(row["min"]),int(row["year"]))
    
    return datetime(yr,mn,day,hour,mi,46)

def make_lower_image_path(row):
    mn,day,hour,mi,yr = (row["month"],row["day"],
                      row["hour"],row["min"],row["year"])
    mn = f"{mn:02}"
    day = f"{day:02}"
    hour = f"{hour:02}"
    mi = f"{mi:02}"
    yr = str(yr)
    
    if mi == "00":
        if hour == "00":
            if (day == "01") and (mn == "01"):
                y = int(yr) - 1 
                if y in [2014,2015,2016]:
                    yr_l = y
                    mn_l = 12
                    day_l = numberOfDays(y, mn_l)
                    hour_1 = 23
                    mi_l = 59
                    s_l = 45
                else:
                    yr_1 = int(yr)
                    mn_l = int(mn)
                    hour_1 = int(hour)
                    day_l = int(day)
                    mi_l = 0
                    s_l = 0

            elif (day == "01") and (mn != "01"):
                yr_l = int(yr)
                mn_l = int(mn) - 1
                day_l = numberOfDays(int(yr), mn_l)
                hour_1 = 23
                mi_l = 59
                s_l = 45
            else:
                yr_l = int(yr)
                mn_l = int(mn)
                day_l = int(day) - 1
                hour_1 = 23
                mi_l = 59
                s_l = 45
        else:
            yr_l = int(yr)
            day_l = int(day)
            mn_l = int(mn)
            hour_1 = int(hour) - 1
            mi_l = 59
            s_l = 45
    else:
        hour_1 = int(hour)
        day_l = int(day)
        mn_l = int(mn)
        yr_l = int(yr)
        mi_l = int(mi) - 1
        s_l = 45
        
    return datetime(yr_l,mn_l,day_l,hour_1,mi_l,s_l)


def get_correct_file(row,file_dict):
    mn,day,hour,mi = (row["month"],row["day"],
                      row["hour"],row["min"])
    higher, lower = row["higher_file"],row["lower_file"]
    
    mn = f"{mn:02}"
    day = f"{day:02}"
    hour = f"{hour:02}"
    mi = f"{mi:02}"
    
    yrmn,date = f"{str(row['year'])}/{mn}", f"{str(row['year'])}/{mn}/{day}"
    if date in file_dict[str(row['year'])][yrmn].keys():
        for file in file_dict[str(row['year'])][yrmn][date]:
            dt_f = datetime(int(file[:4]),int(file[5:7]),int(file[8:10]),int(file[20:22]),int(file[22:24]),int(file[24:26]))
            
            if (dt_f >= lower) and (dt_f <= higher):
                return file
            else:
                pass
    else:
        return 0
    

def save_pickle(file_name,obj):
    with open(file_name, 'wb') as fout:
        pickle.dump(obj, fout)

def open_pickle(file_name):
    with open(file_name, 'rb') as handle:
        obj = pickle.load(handle)
    return obj

def update_df_for_model(df,column):
    col = column + '_i'
#     df['Y'] = df.apply(lambda row: [row['ghi_x'],row['dni_x'],row['dhi_x']],axis=1)
#     df[col] = df.apply(lambda row: [row['ghi_y'],row['dni_y'],row['dhi_y']],axis=1)
    df = df[['ghi_x','timestamp_x',column,'air_temp','relhum', 'press', 'windsp', 
             'winddir', 'max_windsp', 'precipitation','file','ghi_y']]

    return df.rename(columns={'timestamp_x':'timestamp','ghi_x':'Y','ghi_y':col})

def preview_df(df):
    df_dtypes = pd.DataFrame(df.dtypes,columns=['dtypes'])
    df_dtypes = df_dtypes.reset_index()
    df_dtypes['name'] = df_dtypes['index']
    df_dtypes = df_dtypes[['name','dtypes']]
    df_dtypes['first value'] = df.loc[0].values
    data_dictionary = pd.DataFrame(df.columns).rename(columns={0:"name"})
    preview = df_dtypes.merge(data_dictionary, on='name',how='left')
    
    return preview

time_to_period = {'month':12,'day':31,'hour':24,'minute':60}

def process_time_to_sin(df,cols,time_to_period):
    for col in cols:
        p = time_to_period[col]
        df[col] = df[col].apply(lambda row: sin((2*pi*row)/p))
    
    return df[cols]
# time = [-5:-1] (['month', 'day', 'hour', 'minute'])
# cont = cols[:-5]
# cat = cols[-1]

def process_timeahead_attributes(df_name,train,test,time_to_period):
    cols = train.columns.to_list()
    cat_cols = cols[-1]
    con_cols = cols[:-5]
    time_cols = ['month', 'day', 'hour', 'minute']

    # performin min-max scaling each continuous feature column to
    # the range [0, 1]
    cs = MinMaxScaler()
    trainContinuous = cs.fit_transform(train[con_cols])
    testContinuous = cs.transform(test[con_cols])
    
    # one-hot encode the categorical data (by definition of
    # one-hot encoding, all output features are now in the range [0, 1])
                                
    trainCategorical = pd.get_dummies(train[cat_cols],drop_first=True)
    testCategorical = pd.get_dummies(test[cat_cols],drop_first=True)
    
    #preform a sin transformation on our time columns: 
    #The sin function will output all the features in the range [-1,1]
    trainTimeCols = process_time_to_sin(train,time_cols,time_to_period)
    testTimeCols = process_time_to_sin(test,time_cols,time_to_period)
    
#     # construct our training and testing data points by concatenating
#     # the categorical features with the continuous features
    trainX = np.hstack([trainContinuous,trainTimeCols,trainCategorical])
    testX = np.hstack([testContinuous,testTimeCols,testCategorical])
#     # return the concatenated training and testing data
    return (trainX, testX)

def save_pickle(file_name,obj):
    with open(file_name, 'wb') as fout:
        pickle.dump(obj, fout)

def open_pickle(file_name):
    with open(file_name, 'rb') as handle:
        obj = pickle.load(handle)
    return obj

def get_np_array_from_tar_object(tar_extractfl):
    '''converts a buffer from a tar file in np.array'''
    return np.asarray(bytearray(tar_extractfl.read())
                      , dtype=np.uint8)


def load_sky_images(df,yr=None,tar=None):
    # initialize our images array (i.e., the house images themselves)
    images = []
    
    yr = '0'
    for img in df.values:
#         print(img[0][:4])
        yr_file = img[0][:4]
#         print(yr_file)
        if yr == yr_file:
            pass
        else:
            if tar:
                print("deleted tar")
                del tar
            
            yr = img[0][:4]
            file = f'data/Folsom_sky_images_{yr}.tar.bz2'
            print(yr,file)
            tar = tarfile.open(file)
        
        image = cv2.imdecode(get_np_array_from_tar_object(tar.extractfile(img[0])), 0)
        image = cv2.resize(image, (64, 64))
        images.append(image)
    
    return np.array(images)

def get_index_of_img(train,test,sorted_img_dict):
    
    train_img,test_img = [],[]
    
    for i in train:
        train_img.append(sorted_img_dict[sorted_img_dict['index'] == i].index[0])
    
    for i in test:
        test_img.append(sorted_img_dict[sorted_img_dict['index'] == i].index[0])

    return train_img,test_img

def get_img_from_index(train_img_ind,test_img_ind,img_matrices):
    
    train_img,test_img = [],[]
    
    for i in train_img_ind:
        train_img.append(img_matrices[i])
    
    for i in test_img_ind:
        test_img.append(img_matrices[i])
    
    return np.array(train_img),np.array(test_img)