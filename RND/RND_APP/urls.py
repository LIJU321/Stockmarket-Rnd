from django.urls import path
from . import views


# path("/",module.function,name =)

urlpatterns = [
    # path("",views.RND_INIT, name="RND"), 
    path("request/",views.Request, name="Request"),
    path("Api/",views.API, name="API"),
    path("",views.UI,name="UI"), # UI OF REACT SAME PAGE ui
    path("UI2/<str:price>/",views.UI2,name="UI2"), # UI OF REACT SAME PAGE ui
    path("UI3/<str:price>/",views.UI3,name="UI3"),
    path("UI4/<str:price>/",views.UI4,name="UI4"),
    path("UI5/<str:price>/",views.UI5,name="UI5"),
    path("UI6/<str:price>/",views.UI6,name="UI6"), # SVM 
    path('User/<str:userid>/<int:neighbors>/', views.Recommend_user, name='recommend_user'),
    path("Prod/<str:productid>/<int:neighbors>/",views.Recommend_Products,name="Product"),
    path("Cate/<str:categoryid>/<int:neighbors>/",views.Recommend_Categories,name="Cate"),
   
] 
