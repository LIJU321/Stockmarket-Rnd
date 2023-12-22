
import pandas as pd
#%matplotlib inline
from sklearn.neighbors import NearestNeighbors
import json
from scipy.sparse import csr_matrix
from fuzzywuzzy import process


# df1 = pd.read_csv(r"Evolvier_rating_products.csv")

df1 = pd.read_csv("RND_APP\Evolvier_rating.products.csv",usecols=["userId","productId","rating"])

productId_category = pd.read_csv("RND_APP\Evolvier_productId_category.csv",usecols=["category","productId"])


"""" WE HAVE TO REDUCE THE DIMENTIONS TO FIX THE MEMORY ISSUE"""

# Product pupularity based system targetted at new customers


productId_category.dropna(inplace=True)

Ratings_df = df1[["productId","userId","rating"]]

Ratings = Ratings_df["rating"]

userId_and_productId_df = df1[["productId","userId"]]

# Concatenate df2 to df1 along the next column axis=0
category_df = pd.concat([productId_category,Ratings], axis=1)

Productid_and_category_df = category_df.pivot(index="category",columns="productId",values="rating").fillna(0) # PIVOT TABLE  and FILL ALL  NAN WITH 0
Productid_and_category_df = Productid_and_category_df.T


dataframe = Ratings_df.pivot(index="userId",columns="productId",values="rating").fillna(0) # PIVOT TABLE  and FILL ALL  NAN WITH 0
dataframe = dataframe.T

sparsed_matrix1 = csr_matrix(dataframe.values)
# print(sparsed_matrix1) # ROW AND COLUMN & VALUE

category_sparsed_matrix = csr_matrix(Productid_and_category_df.values)
# print(category_sparsed_matrix)




################################  MODEL TRAINING #######################################

# CAtegory  &  PRODUCT SIMILARITY
knn2 = NearestNeighbors(metric ='cosine',algorithm="brute",n_neighbors=50)
knn2.fit(category_sparsed_matrix)

##############  MODEL TRAINING ###########################################################

# USER & PRODUCT
knn = NearestNeighbors(metric ='cosine',algorithm="brute",n_neighbors=50)
knn.fit(sparsed_matrix1)

##################################### MODEL TRAINING ####################################



""" SIMILAR USER RECOMMEND  FROM INPUT AS USERID """

""" SIMILAR PRODUCT RECOMMEND  FROM INPUT USERID or User-based collaborative filtering """

def Recommend_User(USERID,neighbors):
    idx2 = process.extractOne(USERID,userId_and_productId_df["userId"]) # 1st Argument is USERID 
    userid_index = idx2[2] #Index
    print("SECLECTED_USER:",userId_and_productId_df["userId"][userid_index],"Index:",userid_index)
    print()
    
    Cosine_Similarity,index = knn.kneighbors(sparsed_matrix1[userid_index],n_neighbors=neighbors)
    print(Cosine_Similarity,index)
    print()
    
      # Create a DataFrame with similarity scores and indices
    df = pd.DataFrame({'similarity': Cosine_Similarity.flatten(), 'index': index.flatten()})
    THRESHOLD = 0.60
    
    # Filter the DataFrame based on the threshold
    filtered_df = df[df['similarity'] >= THRESHOLD]

    # Retrieve the recommended products from the filtered indices
    recommended_products = userId_and_productId_df.loc[filtered_df['index'], 'productId']
    
    print(userId_and_productId_df.loc[filtered_df['index'] ,'productId'])
    
    DICT = {}
    # Convert the user Series to a JSON serializable format
    user_json =  recommended_products.to_json(orient='values')
    DICT['products'] = json.loads(user_json)

    print(json.dumps(DICT))

    return DICT
    
# Recommend_User("29E6kDag8g6FG6brn",10)
        


""" SIMILAR PRODCUTS RECOMMEND FROM INPUT PRODUCTID  or Item-based collaborative filtering"""

def Recommend_Product(PRODUCTID,neighbors):
    idx1 = process.extractOne(PRODUCTID,userId_and_productId_df["productId"])
    productid_index = idx1[2] #Index
    productid_index
    print("SECLECTED_PRODUCT:",userId_and_productId_df["productId"][productid_index],"Index:",productid_index)
    print()
    Cosine_Similarity,index = knn.kneighbors(sparsed_matrix1[productid_index],n_neighbors=neighbors)
    print(Cosine_Similarity,index)
    print()
    
    DICT = {}

    for i in index:
        print(userId_and_productId_df["productId"][i].where(i!=productid_index))
        recommended_items = userId_and_productId_df["productId"][i].where(i!=productid_index)
    # Convert the user Series to a JSON serializable format
    user_json =  recommended_items.to_json(orient='values')
    DICT['items'] = json.loads(user_json)
    return DICT


# Recommend_Product("i9x394QDjXJc6aMAH" ,10")


""" SIMILAR CATEGORY  RECOMMEND  FROM INPUT AS CATEGROY """

############## CATEGROY ################

def Recommend_Category(CATEGORYID,neighbors):
    idx2 = process.extractOne(CATEGORYID,productId_category["category"]) # 1st Argument is USERID 
    category_id_index = idx2[2] # Index
    print(idx2)
    print("SELECTED_CATEGORY:",productId_category["category"][category_id_index], "Index:",category_id_index)
    print()
    cosine_similarity1, index1 = knn2.kneighbors(category_sparsed_matrix[100], n_neighbors=neighbors)
    print(cosine_similarity1, index1)
    print()
    DICT = {}
    for i in index1:
        print(productId_category["category"][i].where(i!= category_id_index))
        recommended_Category = productId_category["category"][i].where(i!= category_id_index)

    # Convert the user Series to a JSON serializable format
    user_json =  recommended_Category.to_json(orient='values')
    DICT['Category'] = json.loads(user_json)
    return DICT




# Assuming 'category_sparsed_matrix' is the user-product matrix
# Recommend_Category("22LgEjLjvhCrzaQcT",2)


# productId_category[productId_category["category"] == "i9gtnx6AmejAJ6mzK"]
# productId_category[productId_category["category"] == "GaaG495CesFGRr3Lj"]
# productId_category[productId_category["category"] == "N5kXtazLjiBoisoMW"]
# productId_category[productId_category["category"] =="BsCK5EMCffnjAkRSM"]

######################################## MY KNN RECOMNEDANTIONS ON ################

# df1[df1["userId"] == "29E6kDag8g6FG6brn"]

# df1[df1["userId"] == "c9ZvowN9tGyaEbFse"]

# df1[df1["userId"] == "9WL3RyMTrqyGnJydy"]

# df1[df1["userId"] == "i2ocQ8dHohvbwZNKd"]

# df1[df1["userId"] == "SHxdGHFb2jQhKKk24"]

# df1[df1["userId"] == "BqHB5bLEGJzSH839t"]


# df1[df1["productId"] == "oxNCJrkboE3ZfsCpo"]

# df1[df1["productId"] == "R3wx4r6wBWLZmmF2F"]

# df1[df1["productId"] == "zy5D43EokzptJbLsp"]

# df1[df1["productId"] == "euprna8hgDFSaWcmG"]

# df1[df1["productId"] == "wf82v7egZMXiqZ3fp"]

# df1[df1["productId"] == "oW92vBFZHXEvpstCD"]

# df1[df1["productId"] == "YSYWYYKaiuaHEvjLP"]

# df1[df1["productId"] == "iYeZTzcpMsmWpFYCx"]




