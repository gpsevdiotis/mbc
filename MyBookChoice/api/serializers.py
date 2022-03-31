from rest_framework import serializers
from .models import Book
from django.contrib.auth.models import User


class BookSerializer(serializers.ModelSerializer):

    class Meta:
        model = Book
        fields = ('isbn10', 'title', 'authors', 'categories', 'thumbnail', 'description',
                  'published_year', 'num_pages')
