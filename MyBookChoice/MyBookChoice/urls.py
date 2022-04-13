"""
MyBookChoice URL Configuration

The `urlpatterns` list routes URLs to views. 
"""
from django.contrib import admin
from django.urls import path, include
from base import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('api/', include('api.urls')),
    path('admin/', admin.site.urls),
    path('', views.home, name="home"),
    path('home/', views.home, name="home"),
    path('signup/', views.signup, name="signup"),
    path('accounts/login/', views.user_login, name="login"),
    path("accounts/logout/", views.user_logout, name="logout"),
    path('recommend-by-title/', views.title_recommend_page, name="title_recommend"),
    path('plot/', views.plot_recommend, name="plot_recommend"),
    path("profile/", views.profile, name="profile"),
    path('categories/', views.all_categories, name="all_categories"),
    path('categories/<str:c>', views.topincategories, name='topincategories'),
    path('readlist/<str:bookid>', views.add_to_readlist, name='add_to_readlist'),
    path('readlist/<str:bookid>/remove',
         views.remove_from_readlist, name='remove_from_readlist'),
    path('readlist/<str:bookid>/removed',
         views.removed, name='removed'),
    path('', include('pwa.urls')),

]+static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
