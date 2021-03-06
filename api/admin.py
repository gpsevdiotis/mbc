from django.contrib import admin
from .models import Book, ReadingList  # Rating


@admin.register(Book)
class bookAdmin(admin.ModelAdmin):
    list_display = ('isbn10', 'title', 'authors', 'categories', 'thumbnail', 'description',
                    'published_year', 'num_pages')


'''
@admin.register(Rating)
class ratingAdmin(admin.ModelAdmin):
    list_display = ('user', 'book', 'rating', 'rated_date')
'''

admin.site.register(ReadingList)
