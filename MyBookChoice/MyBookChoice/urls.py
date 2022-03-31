"""MyBookChoice URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from base import views
from base.views import BookList, BookDetail
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('api/', include('api.urls')),
    path('admin/', admin.site.urls),
    path('', views.home, name="home"),
    path('home/', views.home, name="home"),
    path('test/', views.testpage, name="test"),
    path('recommend-by-title/', views.title_recommend_page, name="title_recommend"),
    #path('descriptive/', views.descriptive, name="descriptive"),
    #path('search', views.search, name="search"),
    path('signup/', views.signup, name="signup"),
    path('login/', views.user_login, name="login"),
    path('categories/', views.all_categories, name="all_categories"),
    path('plot/', views.plot_recommend, name="plot_recommend"),
    path("logout/", views.user_logout, name="logout"),
    path("profile/", views.profile, name="profile"),
    path('book/', BookList.as_view()),
    path('book/<int:pk>/', BookDetail, name='BookDetail'),
    path('categories/<str:c>', views.topincategories, name='topincategories'),

]+static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
