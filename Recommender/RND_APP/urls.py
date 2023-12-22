from django.urls import path
from . import views

# path("/",module.function,name =)

urlpatterns = [
    # path("",views.RND_INIT, name="RND"), 
    path("request/",views.Request, name="Request"),
    path("Api/",views.API, name="API"),
    path("",views.UI,name="UI"), # UI OF REACT SAME PAGE ui
    path("Regression/<str:price>/",views.Linear_Regressor,name="Regression"), ## CLOSE PRICE ###
    path("KNN/<str:price>/",views.KNN_Model,name="KNN"),
    path("RNN/<str:price>/",views.RNN_Model,name="RNN"),
    path("Neural-Network/<str:price>/",views.Neural_Network,name="Neural-Network"),
    path("SVM/<str:price>/",views.SVM_Model,name="SVM"),
    path("Dicision_Tree/<str:price>/",views.Decision_Tree,name="Dicision_Tree"),
    path("HIgh_on_regression/<str:price>/",views.HIgh_price_on_Linear_regression,name="HIgh_on_regression"), ## HIGH PRICE ##
    path("HIgh_svmclassifier/<str:price>/",views.HIgh_price_on_svmclassifier,name="HIgh_svmclassifier"), 
    path("HIgh_price_on_knn/<str:price>/",views.HIgh_price_on_knn,name="HIgh_price_on_knn"),
    path("HIgh_price_on_Nerual_Network/<str:price>/",views.HIgh_price_on_Nerual_Network,name="HIgh_price_on_Nerual_Network"),
    path("HIgh_price_on_Feed_forward_Neuaral_Network/<str:price>/",views.HIgh_price_on_Feed_forward_Neuaral_Network,name="HIgh_price_on_Feed_forward_Neuaral_Network"),
    path("Dicision_Tree_on_High/<str:price>/",views.Decision_Tree_on_High,name="Dicision_Tree_on_High"),
    path("Low_on_regression/<str:price>/",views.Low_on_Linear_regression,name="Low_on_regression"), ## LOW price ##
    path("Low_svmclassifier/<str:price>/",views.Low_price_svmclassifier,name="Low_svmclassifier"), 
    path("Low_price_on_knn/<str:price>/",views.Low_price_on_knn,name="Low_price_on_knn"),
    path("Low_price_on_Nerual_Network/<str:price>/",views.Low_price_on_Nerual_Network,name="Low_price_on_Nerual_Network"),
    path("Low_on_Feed_forward_Neuaral_Network/<str:price>/",views.Low_price_on_Feed_forward_Neuaral_Network,name="Low_on_Feed_forward_Neuaral_Network"),
    path("Dicision_Tree_on_Low/<str:price>/",views.Decision_Tree_on_Low,name="Dicision_Tree_on_Low"),
    path('User/<str:userid>/<int:neighbors>/',views.Recommend_user, name='recommend_user'),
    path("Prod/<str:productid>/<int:neighbors>/",views.Recommend_Products,name="Product"),
    path("Cate/<str:categoryid>/<int:neighbors>/",views.Recommend_Categories,name="Cate"),
   
]
