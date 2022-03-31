from django.db import models
from django.contrib.auth.models import User


class Book(models.Model):
    isbn13 = models.TextField()
    isbn10 = models.TextField()
    title = models.TextField()
    subtitle = models.TextField()
    authors = models.TextField()
    categories = models.TextField()
    thumbnail = models.TextField()
    description = models.TextField()
    published_year = models.IntegerField()
    average_rating = models.IntegerField()
    num_pages = models.IntegerField()
    ratings_count = models.IntegerField()

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)

    def __str__(self):
        return str(self.pk)


'''
class Rating(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, default=None)
    book = models.ForeignKey(Book, on_delete=models.CASCADE, default=None)
    rating = models.CharField(max_length=70)
    rated_date = models.DateTimeField(auto_now_add=True)
'''
