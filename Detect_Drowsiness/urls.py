from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('Detect_Drowsiness', views.index, name='index'),
    path(' ', views.close_all, name='close_all')
]
