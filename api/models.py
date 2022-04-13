from django.db import models
from django.contrib.auth.models import User


class Book(models.Model):
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


class ReadingList(models.Model):
    user = models.CharField(max_length=300, blank=True, default=None)
    readlist = models.CharField(max_length=3000, blank=True, default=None)
    added_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.user
