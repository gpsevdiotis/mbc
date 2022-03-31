from django.urls import path, include
from rest_framework.urlpatterns import format_suffix_patterns
from . import views

urlpatterns = [
    path('', views.apiOverview, name='api-overview'),
    path('book/', views.BookList.as_view()),
    path('book-add/', views.BookAdd.as_view()),
    path('book/<int:pk>/', views.BookDetail.as_view()),
    path('api-auth/', include('rest_framework.urls')),
]

urlpatterns = format_suffix_patterns(urlpatterns)
