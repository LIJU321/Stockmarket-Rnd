
import pandas as pd
# import numpy as np
import os
# import sklearn
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import LabelEncoder
import json
from scipy.sparse import csr_matrix
from fuzzywuzzy import process
#%matplotlib inline

from sklearn.neighbors import NearestNeighbors

#df1 = pd.read_csv(r"Evolvier_rating_products.csv")


file_path = os.path.join("D:/PROJECT_DJANGO/RND/RND_APP","Evolvier_rating.products.csv") 
df1 = pd.read_csv(os.path.join(file_path),usecols=["userId","productId","rating"])

#df1 = pd.read_csv(r"Evolvier_rating_products.csv",usecols=["userId","productId","rating"])


"""" WE HAVE TO REDUCE THE DIMENTIONS TO FIX THE MEMORY ISSUE"""

# Product pupularity based system targetted at new customers

df2 = df1[["productId","userId","rating"]]
user_and_product = df1[["productId","userId"]]
dataframe = df2.pivot(index="productId",columns="userId",values="rating").fillna(0) # PIVOT TABLE  and FILL ALL  NAN WITH 0
sparsed_matrix1 = csr_matrix(dataframe.values)
#print(sparsed_matrix1) # ROW AND COLUMN & VALUE
knn = NearestNeighbors(metric = "cosine",algorithm="brute",n_neighbors=20)
knn.fit(sparsed_matrix1)




""" SIMILAR USER RECOMMEND  FROM INPUT USERID """

def Recommend_User(USERID,neighbors):
    idx2 = process.extractOne(USERID,user_and_product["userId"]) # 1st Argument is USERID 
    userid_index = idx2[2] #Index
    print("SECLECTED_USER:",user_and_product["userId"][userid_index],"Index:",userid_index)
    print()
    Cosine_Similarity,index = knn.kneighbors(sparsed_matrix1[userid_index],n_neighbors=neighbors)
    print(Cosine_Similarity,index)
    print()
      # Create a DataFrame with similarity scores and indices
    df = pd.DataFrame({'similarity': Cosine_Similarity.flatten(), 'index': index.flatten()})
    threshold = 0.5
    # Filter the DataFrame based on the threshold
    filtered_df = df[df['similarity'] >= threshold]

    # Retrieve the recommended products from the filtered indices
    recommended_products = user_and_product.loc[filtered_df['index'], 'productId']

  
    DICT = {}
    # Convert the user Series to a JSON serializable format
    user_json =  recommended_products.to_json(orient='values')
    DICT['user'] = json.loads(user_json)

    # print(json.dumps(DICT))

    return DICT
    
        
# Recommend_User("2fAcGgtB5oAoSfxku",sparsed_matrix1,10)
# Recommend_User("2fAcGgtB5oAoSfxku",10)







""" SIMILAR PRODCUTS RECOMMEND FROM INPUT AS PRODUCTID """

# def Recommend_Product(PRODUCTID,neighbors):
#     idx1 = process.extractOne(PRODUCTID,user_and_product["productId"])
#     productid_index = idx1[2] #Index
#     productid_index
#     print("SECLECTED_PRODUCT:",user_and_product["productId"][productid_index],"Index:",productid_index)
#     print()
#     Cosine_Similarity,index = knn.kneighbors(sparsed_matrix1[productid_index],n_neighbors=neighbors)
#     print(Cosine_Similarity,index)
#     print()
#     for i in index:
#         print(user_and_product["productId"][i].where(i!=productid_index))
          # print(user_and_product["productId"][i])
        
# Recommend_Product("zzC29i4ZZfAv5fpLe",sparsed_matrix1,10)




