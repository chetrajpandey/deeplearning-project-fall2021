#In this python program, the original goes_flare_integrated file is used as the label source.
#To create the label, the maximum intensity of flare between midnight to midnight
#and noon to noon with respective date is used.

from datetime import datetime as dt
from datetime import date,timedelta
import pandas as pd
pd.options.mode.chained_assignment = None 
import numpy as np

#In this function, to create the label, the maximum intensity of flare between midnight to midnight
#and noon to noon with respective date is used.
def bi_daily_obs(df, pws, pwe, stop):
    #Datetime 
    df['start'] = pd.to_datetime(df['start_time'], format='%Y-%m-%d %H:%M:%S')

    #New Empty DF
    emp = pd.DataFrame()

    #List to store intermediate results
    lis = []
    cols = ['label', 'goes_class']

    #Loop to check max from midnight to midnight and noon to noon
    for i in range(len(df)):
        #Date with max intensity of flare with in the 24 hour window
        emp = df[ (df.start > pws) & (df.start <= pwe) ].sort_values('goes_class', ascending=False).head(1).squeeze(axis=0)
        if pd.Series(emp.goes_class).empty:
            ins = ''
        else:
            ins = emp.goes_class
        lis.append([pws, ins])
        pws = pws + pd.Timedelta(hours=12)
        pwe = pwe + pd.Timedelta(hours=12)
        if pwe >= stop:
            break

    df_result = pd.DataFrame(lis, columns=cols)
    print('Completed!')
    return df_result

def count_class(dataf):
    df = dataf.copy()
    df.replace(np.nan, str(0), inplace=True)
    df.replace('', str(0), inplace=True)
    df['class'] = df['goes_class'].astype('str').apply (lambda row: row[0:1])
    search_list = [['A', 'B', '0'], ['C'], ['M'], ['X']]
    for i in range(4):
        search_for = search_list[i]
        mask = df['class'].apply(lambda row: row).str.contains('|'.join(search_for))
        present = df[mask]
        print(present['class'].value_counts())

#This is to binarize the labels into 0 and 1 class where 0 is Non-Flare and 1 is Flare
#modes='M' is used for greater than or equal to M1.0 class flares
#modes= 'C' is used for greater than or equal to C1.0 class flares
def binarize(df, modes):
    #Empty space and nan values are filled with 0 in the goes_class column
    df.replace(np.nan, str(0), inplace=True)
    df.replace('', str(0), inplace=True)

    #Replacing X and M class flare with 1 and rest with 0 in goes_class column
    if(modes=='M'):
        for i in range(len(df)):
            if (df.goes_class[i][0] == 'X' or  df.goes_class[i][0] == 'M'):
                df.goes_class[i] = 1
            else:
                df.goes_class[i] = 0
    else:
        for i in range(len(df)):
            if (df.goes_class[i][0] == 'X' or  df.goes_class[i][0] == 'M' or df.goes_class[i][0] == 'C'):
                df.goes_class[i] = 1
            else:
                df.goes_class[i] = 0
    return df

#This function is used to convert the timestamps to name of images that we use in this research
def date_to_filename(df):
    df['label'] = pd.to_datetime(df['label'], format='%Y-%m-%d %H:%M:%S')

    #Renaming label(Date) to this format of file HMI.m2010.05.21_12.00.00 
    df['label'] = 'AIA.m'+ df.label.dt.year.astype(str) + '.' \
        + df.label.dt.month.map("{:02}".format).astype(str) + '.'\
        + df.label.dt.day.map("{:02}".format).astype(str) + '_' \
        + df.label.dt.hour.map("{:02}".format).astype(str) + '.'\
        + df.label.dt.minute.map("{:02}".format).astype(str) + '.'\
        + df.label.dt.second.map("{:02}".format).astype(str) + '.png'
    
    return df

#Due to some reasons, Helioviewer has missing images for some instances, so we use this function to 
#filter our labels with actual images available
def filter_labels(df1):
    data = pd.read_csv (r'aia_filenames.csv')   
    df2 = pd.DataFrame(data, columns= ['label'])

    #Empty Dataframe to store the merged result
    df3 = pd.DataFrame(columns=['label', 'goes_class'])

    #Merging created binary label with filenames 
    df3 = df1.merge(df2, on=['label'])

    return df3

#Creating time-segmented 4 tri-monthly partitions
def create_partitions(df):
    search_list = [['01', '02', '03'], ['04', '05', '06'], ['07', '08', '09'], ['10', '11', '12']]
    for i in range(4):
        search_for = search_list[i]
        mask = df['label'].apply(lambda row: row[10:12]).str.contains('|'.join(search_for))
        partition = df[mask]
        print(partition['goes_class'].value_counts())
        # Dumping the dataframe into CSV with label as Date and goes_class as intensity
        # partition.to_csv(r'data_labels/gt_{mode}/class_gt_{mode}_Partition{i}.csv'.format(mode=mode, i=i+1), index=False, header=True, columns=['label', 'goes_class'])

#Creating time-segmented 4-Fold CV Dataset, where 9 months of data is used for training and rest 3 for validation
def create_CVDataset(df):
    search_list = [['01', '02', '03'], ['04', '05', '06'], ['07', '08', '09'], ['10', '11', '12']]
    for i in range(4):
        search_for = search_list[i]
        mask = df['label'].apply(lambda row: row[10:12]).str.contains('|'.join(search_for))
        train = df[~mask]
        val = df[mask]
        print(train['goes_class'].value_counts())
        print(val['goes_class'].value_counts())
        # Dumping the dataframe into CSV with label as Date and goes_class as intensity
        #train.to_csv(r'data_labels/gt_{mode}/class_gt_{mode}_Fold{i}_train.csv'.format(mode=mode, i=i+1), index=False, header=True, columns=['label', 'goes_class'])
        #val.to_csv(r'data_labels/gt_{mode}/class_gt_{mode}_Fold{i}_val.csv'.format(mode=mode, i=i+1), index=False, header=True, columns=['label', 'goes_class'])
        # train.to_csv(r'aia_data_labels/C_Bin_Fold{i}_train.csv'.format(i=i+1), index=False, header=True, columns=['label', 'goes_class'])
        # val.to_csv(r'aia_data_labels/C_Bin_Fold{i}_val.csv'.format(i=i+1), index=False, header=True, columns=['label', 'goes_class'])

def thresholding(df, mode):
    df.replace(np.nan, str(0), inplace=True)
    df.replace('', str(0), inplace=True)
    if(mode == 'multi-class'):
        print(mode)
        for i in range(len(df)):
                if (df.goes_class[i] < 'C4.0'):
                    df.goes_class[i] = 0       
                elif(df.goes_class[i] >= 'C4.0' and df.goes_class[i] < 'M1.0'):
                    df.goes_class[i] = 1
                else:
                    df.goes_class[i] = 2
    else:
        print(mode)
        for i in range(len(df)):
                if (df.goes_class[i] < 'M1.0'):
                    df.goes_class[i] = 0
                else: 
                    df.goes_class[i] = 1

    return df

#Load Original source for Goes Flare X-ray Flux 
data = pd.read_csv (r'label_source/goes_flares_integrated.csv')   

#Convert to DataFrame
dataframe = pd.DataFrame(data, columns= ['start_time','goes_class'])

#Prediction window Start
pws = pd.to_datetime('2010-05-01 00:00:00',format='%Y-%m-%d %H:%M:%S')

#Prediction Window Stop
pwe = pd.to_datetime('2010-05-01 23:59:59',format='%Y-%m-%d %H:%M:%S')

#Data available till 2018-12-30
stop = pd.to_datetime('2018-12-30 23:59:59',format='%Y-%m-%d %H:%M:%S')

#modes='M' is used to create labels for greater than or equal to M1.0 class flares
#modes= 'C' is used to create labels for greater than or equal to C1.0 class flares
mode = 'M'

#Calling functions in order
df_res = bi_daily_obs(dataframe, pws, pwe, stop)
df_res2 = date_to_filename(df_res)
df_res3 = filter_labels(df_res2)
# count_class(df_res3)
# df_res4 = thresholding(df_res3, 'binary')

# print("Total: ", df_res4['goes_class'].value_counts())
#df_res4.to_csv(r'test.csv', index=False, header=True, columns=['label', 'goes_class'])
df_res4 = binarize(df_res3, mode)
print("Total: ", df_res4['goes_class'].value_counts())
# print('Partitions')
# create_partitions(df_res4)
create_CVDataset(df_res4)
