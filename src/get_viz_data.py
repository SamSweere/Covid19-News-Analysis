import pandas as pd
import os

source_path = "data/" 

def load_data(folder_name):
    arr = os.listdir(source_path+folder_name)
    print(arr)

    # filter out the scv files
    files = []
    for file in arr:
        file[-3:] == ".scv"
        files.append(file)

    df = None

    for file in files:
        file_path = source_path + folder_name + "/" + file

        if(df is None):
            df =  pd.read_csv(file_path, index_col=0)
        else:
            df = pd.concat([df, pd.read_csv(file_path, index_col=0)])


    return df

