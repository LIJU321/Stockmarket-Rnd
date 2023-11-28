
import pandas as pd
import json
import os

"""
# Get the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))
print(current_directory)
"""
def Recommend():
           
    file_path = os.path.join("D:/PROJECT_DJANGO/RND/RND_APP", "ratings_Beauty.csv") 
    df1 = pd.read_csv(os.path.join(file_path))
    # Product pupularity based system targetted at new customers
    df_filled = df1.fillna(0)
    productid = df_filled.loc["0":][["ProductId","Rating"]] # df.loc["row"]of["column"]
    productid = productid.head(100)


    Popular_Product3 = productid
    Popular_Product4 = Popular_Product3.to_numpy()
    Popular_Product4 = list(Popular_Product4)
    #print()
    Popular_Product4 = list(Popular_Product4)

    Popular_Product5 = []
    for i in Popular_Product4:
        if i[1]>=3:
            #print("FILTER",i)
            Popular_Product5.append(i)

    #################     ################

    DICT3 = {}
    for i in range(len(Popular_Product5)):
        # print("item2",i)
        DICT2 = {Popular_Product5[i][0]:Popular_Product5[i][1]}
        DICT3.update(DICT2)
     
    DICTJSOn = json.dumps(DICT3)

    with open("DICTFILTERED.json", "w") as file:
        # Convert the dictionary to JSON and write it to the file
        json.dump(DICTJSOn,file)
    print(DICTJSOn)

    return DICTJSOn













