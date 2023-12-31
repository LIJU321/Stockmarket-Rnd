from django.shortcuts import render
from django.http  import HttpResponse
from django.http import JsonResponse
import json
from .  import RNDRUN
# from . import Evolvier_RND
from . import Category_Evolvier_RND
from . import STOCK_FROM_ONLY_OPEN
from . import STOCK_MARKET_LOW
import requests
import numpy as np

# Create your views here.

"""
from django.http import JsonResponse
import subprocess

def home(request):
    # Run the script as a subprocess
   result = subprocess.run(['python', 'D:\PROJECT_DJANGO\RND\scripts\RND_FILTER_FOR_TOP_RATED.py'], capture_output=True, text=True)
   output = result.stdout  # Capture the output from the script
   error = result.stderr  # Capture the error from the script

# Format the output and error as an API response
   response_data = {'output': output, 'error': error}
   return JsonResponse(response_data)

"""

def RND_INIT(request):
    return HttpResponse("RND_PROJECT")
  


def API(request):
    #import RND
    output = RNDRUN.Recommend()
    #return HttpResponse("API"))
    response_data = {'ProductID':output}
    # c = response_data["ProductID"]["0733001998"]
    # HttpResponse(c)
    return JsonResponse(response_data)



def UI(request): # UI
    return render(request,'index.html')



####################    STOCK CLOSE   ########################
def Linear_Regressor(request, price):
    try:
        # Validate and convert the 'price' parameter to an integer
        price = int(price)

        # Make a prediction using your machine learning model
        output = STOCK_FROM_ONLY_OPEN.Regressor.predict([[price]])

        # Convert the NumPy array to a Python list
        output_list = output.tolist()

        # Prepare the JSON response
        response_data = {'Close_price': output_list}
        s = np.asarray(response_data)
 
        # Log the prediction (you can replace 'print' with a proper logger)
        print(f'Predicted Close_price: {output_list}')

        # Return the JSON response
        return JsonResponse(response_data)
    except ValueError:
        # Handle invalid 'price' parameter
        return JsonResponse({'error': 'Invalid price parameter'}, status=400)
    except Exception as e:
        # Handle other exceptions (e.g., model prediction errors)
        return JsonResponse({'error': str(e)}, status=500)
        

def KNN_Model(request, price):
    try:
        # Validate and convert the 'price' parameter to an integer
        price = int(price)

        # Make a prediction using your machine learning model
        output = STOCK_FROM_ONLY_OPEN.knn.predict([[price]])

        # Convert the NumPy array to a Python list
        output_list = output.tolist()

        # Prepare the JSON response
        response_data = {'Close_price': output_list}
        s = np.asarray(response_data)
 
        # Log the prediction (you can replace 'print' with a proper logger)
        print(f'Predicted Close_price: {output_list}')

        # Return the JSON response
        return JsonResponse(response_data)
    except ValueError:
        # Handle invalid 'price' parameter
        return JsonResponse({'error': 'Invalid price parameter'}, status=400)
    except Exception as e:
        # Handle other exceptions (e.g., model prediction errors)
        return JsonResponse({'error': str(e)}, status=500)


def RNN_Model(request, price):
    try:
        # Validate and convert the 'price' parameter to an integer
        price = int(price)

        # Make a prediction using your machine learning model
        output = STOCK_FROM_ONLY_OPEN.Rnnmodel.predict([[price]])

        # Convert the NumPy array to a Python list
        output_list = output.tolist()

        # Prepare the JSON response
        response_data = {'Close_price': output_list}
        s = np.asarray(response_data)
 
        # Log the prediction (you can replace 'print' with a proper logger)
        print(f'Predicted Close_price: {output_list}')

        # Return the JSON response
        return JsonResponse(response_data)
    except ValueError:
        # Handle invalid 'price' parameter
        return JsonResponse({'error': 'Invalid price parameter'}, status=400)
    except Exception as e:
        # Handle other exceptions (e.g., model prediction errors)
        return JsonResponse({'error': str(e)}, status=500)
    

def Neural_Network(request, price):
    try:
        # Validate and convert the 'price' parameter to an integer
        price = int(price)

        # Make a prediction using your machine learning model
        output = STOCK_FROM_ONLY_OPEN.model.predict([[price]])

        # Convert the NumPy array to a Python list
        output_list = output.tolist()

        # Prepare the JSON response
        response_data = {'Close_price': output_list}
        s = np.asarray(response_data)
 
        # Log the prediction (you can replace 'print' with a proper logger)
        print(f'Predicted Close_price: {output_list}')

        # Return the JSON response
        return JsonResponse(response_data)
    except ValueError:
        # Handle invalid 'price' parameter
        return JsonResponse({'error': 'Invalid price parameter'}, status=400)
    except Exception as e:
        # Handle other exceptions (e.g., model prediction errors)
        return JsonResponse({'error': str(e)}, status=500)
    


        
def SVM_Model(request, price):
    try:
        # Validate and convert the 'price' parameter to an integer
        price = int(price)

        # Make a prediction using your machine learning model
        output = STOCK_FROM_ONLY_OPEN.svmclassifier.predict(np.array([price]).reshape(1,-1))

        # Convert the NumPy array to a Python list
        output_list = output.tolist()

        # Prepare the JSON response
        response_data = {'Close_price': output_list}
        s = np.asarray(response_data)
 
        # Log the prediction (you can replace 'print' with a proper logger)
        print(f'Predicted Close_price: {output_list}')

        # Return the JSON response
        return JsonResponse(response_data)
    except ValueError:
        # Handle invalid 'price' parameter
        return JsonResponse({'error': 'Invalid price parameter'}, status=400)
    except Exception as e:
        # Handle other exceptions (e.g., model prediction errors)
        return JsonResponse({'error': str(e)}, status=500)


def Decision_Tree(request, price):
    try:
        # Validate and convert the 'price' parameter to an integer
        price = int(price)

        # Make a prediction using your machine learning model
        output = STOCK_FROM_ONLY_OPEN.DecisionTRee.predict(np.array([price]).reshape(1,-1))

        # Convert the NumPy array to a Python list
        output_list = output.tolist()

        # Prepare the JSON response
        response_data = {'Close_price': output_list}
        s = np.asarray(response_data)
 
        # Log the prediction (you can replace 'print' with a proper logger)
        print(f'Predicted Close_price: {output_list}')

        # Return the JSON response
        return JsonResponse(response_data)
    except ValueError:
        # Handle invalid 'price' parameter
        return JsonResponse({'error': 'Invalid price parameter'}, status=400)
    except Exception as e:
        # Handle other exceptions (e.g., model prediction errors)
        return JsonResponse({'error': str(e)}, status=500)

#################### HIGH PRICES #########################################
        
def HIgh_price_on_Linear_regression(request, price):
    try:
        # Validate and convert the 'price' parameter to an integer
        price = int(price)

        # Make a prediction using your machine learning model
        output = STOCK_FROM_ONLY_OPEN.RegressorHigh.predict(np.array([price]).reshape(1,-1))

        # Convert the NumPy array to a Python list
        output_list = output.tolist()

        # Prepare the JSON response
        response_data = {'Close_price': output_list}
        s = np.asarray(response_data)
 
        # Log the prediction (you can replace 'print' with a proper logger)
        print(f'Predicted Close_price: {output_list}')

        # Return the JSON response
        return JsonResponse(response_data)
    except ValueError:
        # Handle invalid 'price' parameter
        return JsonResponse({'error': 'Invalid price parameter'}, status=400)
    except Exception as e:
        # Handle other exceptions (e.g., model prediction errors)
        return JsonResponse({'error': str(e)}, status=500)



def HIgh_price_on_svmclassifier(request, price):
    try:
        # Validate and convert the 'price' parameter to an integer
        price = int(price)

        # Make a prediction using your machine learning model
        output = STOCK_FROM_ONLY_OPEN.svmclassifierHigh.predict(np.array([price]).reshape(1,-1))

        # Convert the NumPy array to a Python list
        output_list = output.tolist()

        # Prepare the JSON response
        response_data = {'Close_price': output_list}
        s = np.asarray(response_data)
 
        # Log the prediction (you can replace 'print' with a proper logger)
        print(f'Predicted Close_price: {output_list}')

        # Return the JSON response
        return JsonResponse(response_data)
    except ValueError:
        # Handle invalid 'price' parameter
        return JsonResponse({'error': 'Invalid price parameter'}, status=400)
    except Exception as e:
        # Handle other exceptions (e.g., model prediction errors)
        return JsonResponse({'error': str(e)}, status=500)



def HIgh_price_on_knn(request, price):
    try:
        # Validate and convert the 'price' parameter to an integer
        price = int(price)

        # Make a prediction using your machine learning model
        output = STOCK_FROM_ONLY_OPEN.Highknn.predict(np.array([price]).reshape(1,-1))

        # Convert the NumPy array to a Python list
        output_list = output.tolist()

        # Prepare the JSON response
        response_data = {'Close_price': output_list}
        s = np.asarray(response_data)
 
        # Log the prediction (you can replace 'print' with a proper logger)
        print(f'Predicted Close_price: {output_list}')

        # Return the JSON response
        return JsonResponse(response_data)
    except ValueError:
        # Handle invalid 'price' parameter
        return JsonResponse({'error': 'Invalid price parameter'}, status=400)
    except Exception as e:
        # Handle other exceptions (e.g., model prediction errors)
        return JsonResponse({'error': str(e)}, status=500)

def HIgh_price_on_Nerual_Network(request, price):
    try:
        # Validate and convert the 'price' parameter to an integer
        price = int(price)

        # Make a prediction using your machine learning model
        output = STOCK_FROM_ONLY_OPEN.Highmodel.predict(np.array([price]).reshape(1,-1))

        # Convert the NumPy array to a Python list
        output_list = output.tolist()

        # Prepare the JSON response
        response_data = {'Close_price': output_list}
        s = np.asarray(response_data)
 
        # Log the prediction (you can replace 'print' with a proper logger)
        print(f'Predicted Close_price: {output_list}')

        # Return the JSON response
        return JsonResponse(response_data)
    except ValueError:
        # Handle invalid 'price' parameter
        return JsonResponse({'error': 'Invalid price parameter'}, status=400)
    except Exception as e:
        # Handle other exceptions (e.g., model prediction errors)
        return JsonResponse({'error': str(e)}, status=500)


def HIgh_price_on_Feed_forward_Neuaral_Network(request, price):
    try:
        # Validate and convert the 'price' parameter to an integer
        price = int(price)

        # Make a prediction using your machine learning model
        output = STOCK_FROM_ONLY_OPEN.ffnn_model_High.predict(np.array([price]).reshape(1,-1))

        # Convert the NumPy array to a Python list
        output_list = output.tolist()

        # Prepare the JSON response
        response_data = {'Close_price': output_list}
        s = np.asarray(response_data)
 
        # Log the prediction (you can replace 'print' with a proper logger)
        print(f'Predicted Close_price: {output_list}')

        # Return the JSON response
        return JsonResponse(response_data)
    except ValueError:
        # Handle invalid 'price' parameter
        return JsonResponse({'error': 'Invalid price parameter'}, status=400)
    except Exception as e:
        # Handle other exceptions (e.g., model prediction errors)
        return JsonResponse({'error': str(e)}, status=500)
 
       
def Decision_Tree_on_High(request, price):
    try:
        # Validate and convert the 'price' parameter to an integer
        price = int(price)

        # Make a prediction using your machine learning model
        output = STOCK_FROM_ONLY_OPEN.DecisionTReeHigh.predict(np.array([price]).reshape(1,-1))

        # Convert the NumPy array to a Python list
        output_list = output.tolist()

        # Prepare the JSON response
        response_data = {'Close_price': output_list}
        s = np.asarray(response_data)
 
        # Log the prediction (you can replace 'print' with a proper logger)
        print(f'Predicted Close_price: {output_list}')

        # Return the JSON response
        return JsonResponse(response_data)
    except ValueError:
        # Handle invalid 'price' parameter
        return JsonResponse({'error': 'Invalid price parameter'}, status=400)
    except Exception as e:
        # Handle other exceptions (e.g., model prediction errors)
        return JsonResponse({'error': str(e)}, status=500)


 ################################### LOW PRICES ####################################
def Low_on_Linear_regression(request, price):
    try:
        # Validate and convert the 'price' parameter to an integer
        price = int(price)

        # Make a prediction using your machine learning model
        output = STOCK_MARKET_LOW.RegressorLow.predict(np.array([price]).reshape(1,-1))

        # Convert the NumPy array to a Python list
        output_list = output.tolist()

        # Prepare the JSON response
        response_data = {'Close_price': output_list}
        s = np.asarray(response_data)
 
        # Log the prediction (you can replace 'print' with a proper logger)
        print(f'Predicted Close_price: {output_list}')

        # Return the JSON response
        return JsonResponse(response_data)
    except ValueError:
        # Handle invalid 'price' parameter
        return JsonResponse({'error': 'Invalid price parameter'}, status=400)
    except Exception as e:
        # Handle other exceptions (e.g., model prediction errors)
        return JsonResponse({'error': str(e)}, status=500)



def Low_price_svmclassifier(request, price):
    try:
        # Validate and convert the 'price' parameter to an integer
        price = int(price)

        # Make a prediction using your machine learning model
        output = STOCK_MARKET_LOW.svmclassifierLow.predict(np.array([price]).reshape(1,-1))

        # Convert the NumPy array to a Python list
        output_list = output.tolist()

        # Prepare the JSON response
        response_data = {'Close_price': output_list}
        s = np.asarray(response_data)
 
        # Log the prediction (you can replace 'print' with a proper logger)
        print(f'Predicted Close_price: {output_list}')

        # Return the JSON response
        return JsonResponse(response_data)
    except ValueError:
        # Handle invalid 'price' parameter
        return JsonResponse({'error': 'Invalid price parameter'}, status=400)
    except Exception as e:
        # Handle other exceptions (e.g., model prediction errors)
        return JsonResponse({'error': str(e)}, status=500)



def Low_price_on_knn(request, price):
    try:
        # Validate and convert the 'price' parameter to an integer
        price = int(price)

        # Make a prediction using your machine learning model
        output = STOCK_MARKET_LOW.Lowknn.predict(np.array([price]).reshape(1,-1))

        # Convert the NumPy array to a Python list
        output_list = output.tolist()

        # Prepare the JSON response
        response_data = {'Close_price': output_list}
        s = np.asarray(response_data)
 
        # Log the prediction (you can replace 'print' with a proper logger)
        print(f'Predicted Close_price: {output_list}')

        # Return the JSON response
        return JsonResponse(response_data)
    except ValueError:
        # Handle invalid 'price' parameter
        return JsonResponse({'error': 'Invalid price parameter'}, status=400)
    except Exception as e:
        # Handle other exceptions (e.g., model prediction errors)
        return JsonResponse({'error': str(e)}, status=500)




def Low_price_on_Nerual_Network(request, price):
    try:
        # Validate and convert the 'price' parameter to an integer
        price = int(price)

        # Make a prediction using your machine learning model
        output = STOCK_MARKET_LOW.Lowmodel_neural_network.predict(np.array([price]).reshape(1,-1))

        # Convert the NumPy array to a Python list
        output_list = output.tolist()

        # Prepare the JSON response
        response_data = {'Close_price': output_list}
        s = np.asarray(response_data)
 
        # Log the prediction (you can replace 'print' with a proper logger)
        print(f'Predicted Close_price: {output_list}')

        # Return the JSON response
        return JsonResponse(response_data)
    except ValueError:
        # Handle invalid 'price' parameter
        return JsonResponse({'error': 'Invalid price parameter'}, status=400)
    except Exception as e:
        # Handle other exceptions (e.g., model prediction errors)
        return JsonResponse({'error': str(e)}, status=500)
    


    
def Low_price_on_Feed_forward_Neuaral_Network(request, price):
    try:
        # Validate and convert the 'price' parameter to an integer
        price = int(price)

        # Make a prediction using your machine learning model
        output = STOCK_MARKET_LOW.ffnn_model_low.predict(np.array([price]).reshape(1,-1))

        # Convert the NumPy array to a Python list
        output_list = output.tolist()

        # Prepare the JSON response
        response_data = {'Close_price': output_list}
        s = np.asarray(response_data)
 
        # Log the prediction (you can replace 'print' with a proper logger)
        print(f'Predicted Close_price: {output_list}')

        # Return the JSON response
        return JsonResponse(response_data)
    except ValueError:
        # Handle invalid 'price' parameter
        return JsonResponse({'error': 'Invalid price parameter'}, status=400)
    except Exception as e:
        # Handle other exceptions (e.g., model prediction errors)
        return JsonResponse({'error': str(e)}, status=500)



def Decision_Tree_on_Low(request, price):
    try:
        # Validate and convert the 'price' parameter to an integer
        price = int(price)

        # Make a prediction using your machine learning model
        output = STOCK_MARKET_LOW.DecisionTReeLow.predict(np.array([price]).reshape(1,-1))

        # Convert the NumPy array to a Python list
        output_list = output.tolist()

        # Prepare the JSON response
        response_data = {'Close_price': output_list}
        s = np.asarray(response_data)
 
        # Log the prediction (you can replace 'print' with a proper logger)
        print(f'Predicted Close_price: {output_list}')

        # Return the JSON response
        return JsonResponse(response_data)
    except ValueError:
        # Handle invalid 'price' parameter
        return JsonResponse({'error': 'Invalid price parameter'}, status=400)
    except Exception as e:
        # Handle other exceptions (e.g., model prediction errors)
        return JsonResponse({'error': str(e)}, status=500)


#############################    STCOK   ########################################






######## Request #######
def Request (request):
    # Access request information
    method = request.method
    path = request.path
    headers = request.headers
    GET_params = request.GET
    POST_params = request.POST

    # Print the request information
    output = f"Request Method: {method}\n"
    output += f"Request Path: {path}\n"
    output += f"Request Headers: {headers}\n"
    output += f"GET Parameters: {GET_params}\n"
    output += f"POST Parameters: {POST_params}\n"
    # Return the output as an HTTP response
    return HttpResponse(output,"RND_PROJECT")


def Recommend_user(request,userid,neighbors):
    # output = Evolvier_RND.Recommend_User("kh29kXskRCYhufooS",10)
    output = Category_Evolvier_RND.Recommend_User(userid,neighbors)
    response_data = {'similar_Products': output}
    return JsonResponse(response_data)


def Recommend_Products(request,productid,neighbors):
    # output = Evolvier_RND.Recommend_User("kh29kXskRCYhufooS",10)
    output = Category_Evolvier_RND.Recommend_Product(productid,neighbors)
    response_data = {'Similar_Items':output}
    return JsonResponse(response_data)


def Recommend_Categories(request,categoryid,neighbors):
    output = Category_Evolvier_RND.Recommend_Category(categoryid,neighbors)
    response_data = {'Similar_Categrory':output}
    return JsonResponse(response_data)


