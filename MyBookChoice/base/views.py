
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from django.shortcuts import render, HttpResponseRedirect, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from api.forms import SignUpForm,  LoginForm  # , AddRatingForm
from api.models import Book, ReadingList  # , Rating
from django.contrib import messages
from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from django.contrib.auth.decorators import login_required

'''
Dataset Read
'''
BASE_DIR = os.getcwd()
df = pd.read_csv(
    BASE_DIR+"/templates/BookRecommender/books.csv", low_memory=False)

''''''


# print(categories)

''' 
Home View
'''


def home(request):
    page_title = "MyBookChoice"
    dictionary = {'page_title': page_title}
    return render(request, 'BookRecommender/home.html', context=dictionary)


''' 
Title Recommendation Algorithm
'''


def get_recommendations(title):
    # Get the index of the book by its title
    # Convert it into Series
    allindices = pd.Series(df.index, index=df['title'])

    # Convert all the titles into different vectors
    # TF-IDF Vectorizer to remove all english stop words
    vector = TfidfVectorizer(analyzer='word', ngram_range=(
        2, 2), min_df=1, stop_words='english')

    # TF-IDF matrix construction by fitting and transforming the data
    tfidf_matrix = vector.fit_transform(df['title'])

    # Find the index of the book which is correspondant to the title
    idx = allindices[title]

    # Cosine Similarity
    cos = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Get the pairwsie similarity scores of all books according to that book
    pss = list(enumerate(cos[idx]))

    # Sort the the pss values
    pss = sorted(pss, key=lambda x: x[1], reverse=True)

    # Get the books indices
    book_indices = [i[0] for i in pss]

    # Return top 5 most similar book
    return df.iloc[book_indices].head(6)


''' 
Title Recommendation Page
'''


def title_recommend_page(request):
    results = Book.objects.all()
    enteredtitle = (request.POST.get('booktitle'))
    page_title = "MyBookChoice | Title Recommendation"
    if request.method == 'POST':
        try:
            page_heading = "Recommended books based on " + enteredtitle
            recommendations = get_recommendations(enteredtitle)
            try:
                user = request.user.username
                saved_readlist = ReadingList.objects.get(user=user)
                reading_list = saved_readlist.readlist.split(',')
            except:
                reading_list = []
                pass

        except Exception as error:
            page_title = "MyBookChoice | Error"
            error_heading = "We dont have any recommendations based on " + \
                enteredtitle + " right now. Please try again"
            dictionary = {'page_title': page_title, 'enteredtitle': enteredtitle,
                          'error_heading': error_heading}
            return render(request, 'BookRecommender/error.html', context=dictionary)

        dictionary = {'page_title': page_title, 'page_heading': page_heading, 'bookresults': results,
                      'recommendations': recommendations, 'reading_list': reading_list}

        return render(request, 'BookRecommender/recommendations/recommendations.html', context=dictionary)

    return render(request, 'BookRecommender/recommendations/title_recommend.html', {"showcity": results, "page_title": page_title})


''' 
Categories Recommendation Algorithm & Page
'''
categories_list = df['categories'].tolist()
categories_set = set(categories_list)
list_res = (list(categories_set))
categories = []
for item in list_res:
    categories.append(item)


def topincategories(request, c):
    try:
        new_df = df[df['categories'].str.contains(c)]
        # Get mean average vote rating across the whole dataset.
        C = new_df['average_rating'].mean()
        # Cutoff the 25th percentile from m
        # m = minimum votes required to be listed
        m = new_df['ratings_count'].quantile(0.25)
        new_books = new_df.copy().loc[new_df['ratings_count'] >= m]

        def weighted_rating(x, m=m, C=C):
            # Number of votes for each book
            v = x['ratings_count']
            # Average rating for each book
            R = x['average_rating']
            # Calculation based on the IMDB formula
            return (v/(v+m) * R) + (m/(m+v) * C)
        # New feature 'score' calculate value with weighted_rating() function
        new_books['score'] = new_books.apply(weighted_rating, axis=1)
        # Sorted books based on calculated score
        new_books = new_books.sort_values('score', ascending=False)
        # Return top 10 books according to collaborative filter
        recommendations = new_books.head(10)

    except Exception as error:
        page_title = "MyBookChoice | Error"
        error_heading = "This category does not pass the minimum number of votes required to be included in the charts. Please select a different one."
        dictionary = {'page_title': page_title, 'error_heading': error_heading}
        return render(request, 'BookRecommender/error.html', context=dictionary)
    try:
        current_user = request.user.username
        saved_readlist = ReadingList.objects.get(user=current_user)
        reading_list = saved_readlist.readlist.split(',')
    except:
        reading_list = []
        pass

    page_title = "MyBookChoice | Categories Recommendation"
    page_heading = "Top Books based on " + c
    dictionary = {'page_title': page_title, 'page_heading': page_heading,
                  'recommendations': recommendations, 'reading_list': reading_list}

    return render(request, 'BookRecommender/recommendations/recommendations.html', context=dictionary)


''' 
Categories Recommendation Page
'''


def all_categories(request):
    page_title = "MyBookChoice | Categories"
    try:
        return render(request, 'BookRecommender/recommendations/all_categories.html', {'page_title': page_title, 'categories': categories})
    except Exception as error:
        page_title = "MyBookChoice | Error"
        error_heading = "We dont have any recommendations based on  right now. Please try again"
        dictionary = {'page_title': page_title, 'error_heading': error_heading}
        return render(request, 'BookRecommender/error.html', context=dictionary)


''' 
Plot Recommendation Algorithm
'''


def plot_recommend(request):
    page_title = "MyBookChoice | Plot Recommendation"

    def clean_text(text):
        text = text.lower()  # lowercase text
        # replace the matched string with ' '
        text = re.sub(re.compile("\'s"), ' ', text)  # matches `'s` from text
        text = re.sub(re.compile("\\r\\n"), ' ', text)  # matches `\r` and `\n`
        # matches all non 0-9 A-z whitespace
        text = re.sub(re.compile(r"[^\w\s]"), ' ', text)
        return text

    def tokenizer(sentence, min_words=4, max_words=1000, stopwords=set(stopwords.words('english')), lemmatize=True):
        # Lemmatize, tokenize, crop and remove stop words.
        if lemmatize:
            stemmer = WordNetLemmatizer()
            tokens = [stemmer.lemmatize(w) for w in word_tokenize(sentence)]
        else:
            tokens = [w for w in word_tokenize(sentence)]
        token = [w for w in tokens if (len(w) > min_words and len(w) < max_words
                                       and w not in stopwords)]
        return tokens

    def extract_best_indices(m, topk, mask=None):
        # return the sum on all tokens of cosinus for each sentence
        if len(m.shape) > 1:
            cos_sim = np.mean(m, axis=0)
        else:
            cos_sim = m
        index = np.argsort(cos_sim)[::-1]  # from highest idx to smallest score
        if mask is not None:
            assert mask.shape == m.shape
            mask = mask[index]
        else:
            mask = np.ones(len(cos_sim))
        # eliminate 0 cosine distance
        mask = np.logical_or(cos_sim[index] != 0, mask)
        best_index = index[mask][:topk]
        return best_index

    def get_recommendations_tfidf(sentence, tfidf_mat):
        # Embed the query sentence
        tokens_query = [str(tok) for tok in tokenizer(sentence)]
        embed_query = vectorizer.transform(tokens_query)
        # Create list with similarity between query and dataset
        mat = cosine_similarity(embed_query, tfidf_mat)
        # Best cosine distance for each token independantly
        best_index = extract_best_indices(mat, topk=10)
        return best_index

    stop_words = set(stopwords.words('english'))
    token_stop = tokenizer(' '.join(stop_words), lemmatize=False)
    vectorizer = TfidfVectorizer(stop_words=token_stop, tokenizer=tokenizer)
    tfidf_mat = vectorizer.fit_transform(df['description'].values)

    ''' 
    Plot Recommendation Page
    '''

    enteredplot = (request.POST.get('booktitle'))
    if request.method == 'POST':
        try:
            recommendation = get_recommendations_tfidf(enteredplot, tfidf_mat)
            recommendations = (df.iloc[recommendation])

            try:
                user = request.user.username
                saved_readlist = ReadingList.objects.get(user=user)
                reading_list = saved_readlist.readlist.split(',')
            except:
                reading_list = []
                pass

            page_title = "MyBookChoice | Results"
            page_heading = "Recommendations based on " + enteredplot + ":"
            dictionary = {'recommendations': recommendations,
                          'enteredplot': enteredplot, 'page_title': page_title, 'page_heading': page_heading, 'reading_list': reading_list}

            return render(request, 'BookRecommender/recommendations/recommendations.html', context=dictionary)
        except Exception as error:
            # print(error)
            page_title = "MyBookChoice | Error"
            error_heading = "We dont have any recommendations based on " + \
                enteredplot + " right now. Please try again"
            dictionary = {'page_title': page_title,
                          'error_heading': error_heading}
            return render(request, 'BookRecommender/error.html', context=dictionary)

    dictionary = {'page_title': page_title}
    return render(request, 'BookRecommender/recommendations/plot_recommend.html', context=dictionary)


''' 
Sign Up Page
'''


def signup(request):
    if not request.user.is_authenticated:
        if request.method == 'POST':
            fm = SignUpForm(request.POST)
            if fm.is_valid():
                user = fm.save()
                messages.success(request, 'Account Created Successfully!!!')
                return HttpResponseRedirect('/login/')
        else:
            if not request.user.is_authenticated:
                fm = SignUpForm()
        return render(request, 'BookRecommender/user/signup.html', {'form': fm})
    else:
        return HttpResponseRedirect('/home/')


''' 
Login Page
'''


def user_login(request):
    if not request.user.is_authenticated:
        if request.method == 'POST':
            fm = LoginForm(request=request, data=request.POST)
            if fm.is_valid():
                uname = fm.cleaned_data['username']
                upass = fm.cleaned_data['password']
                user = authenticate(username=uname, password=upass)
                if user is not None:
                    login(request, user)
                    messages.success(request, 'Logged in Successfully!!')
                    return HttpResponseRedirect('/home/')
        else:
            fm = LoginForm()
        return render(request, 'BookRecommender/user/login.html', {'form': fm})
    else:
        return HttpResponseRedirect('/home/')


''' 
Logout Page
'''


def user_logout(request):
    if request.user.is_authenticated:
        logout(request)
        return HttpResponseRedirect('/home/')


''' 
Profile Page
'''


@login_required
def profile(request):
    try:
        current_user = request.user.username
        saved_readlist = ReadingList.objects.get(user=current_user)
        reading_list = saved_readlist.readlist.split(',')
        reading_list = [i.strip() for i in reading_list if len(i.strip()) > 0]
        readlist = df[df['isbn10'].isin(reading_list)]
        # print(reading_list)
        #print('Total favourite books : '+str(len(readlist)))
    except:
        reading_list = []
        readlist = ''
        pass

    page_title = 'MyBookChoice | Profile'

    my_dict = {'readlist': readlist,
               'page_title': page_title, 'reading_list': reading_list}

    return render(request, 'BookRecommender/user/profile.html', context=my_dict)


''' 
Reading List Adding & Page
'''


@login_required
def add_to_readlist(request, bookid):
    try:
        user = request.user.username
    except Exception as error:
        page_title = "MyBookChoice | Error"
        error_heading = "Invalid action, please try again"
        dictionary = {'page_title': page_title, 'error_heading': error_heading}
        return render(request, 'BookRecommender/error.html', context=dictionary)

    try:
        saved_readlist = ReadingList.objects.get(user=user)

        if bookid not in saved_readlist.readlist:
            saved_readlist.readlist += bookid+','
            saved_readlist.save()

    except Exception as error:
        try:
            m = ReadingList(user=user, readlist=str(bookid)+',')
            m.save()
        except Exception as error:
            # print(error)
            pass

    page_title = "MyBookChoice | Add Book"
    title = list(df[df['isbn10'] == bookid]['title'])[0]
    image = list(df[df['isbn10'] == bookid]['thumbnail'])[0]
    detail_dict = {
        'title': title,
        'image': image,
        'bookid': bookid,
    }
    dictionary = {'page_title': page_title, 'detail_dict': detail_dict}
    return render(request, 'BookRecommender/readinglist/readinglist_add.html', context=dictionary)


''' 
Reading List Removing & Page
'''


@login_required
def remove_from_readlist(request, bookid):
    # current_user = request.user.username
    page_title = "MyBookChoice | Remove Book"
    title = list(df[df['isbn10'] == bookid]['title'])[0]
    image = list(df[df['isbn10'] == bookid]['thumbnail'])[0]

    detail_dict = {
        'title': title,
        'image': image,
        'bookid': bookid,
    }
    dictionary = {'page_title': page_title, 'detail_dict': detail_dict}

    return render(request, 'BookRecommender/readinglist/readinglist_remove.html', context=dictionary)


''' 
Reading List Removing
'''


@login_required
def removed(request, bookid):
    current_user = request.user.username
    saved_readlist = ReadingList.objects.get(user=current_user)
    replace_bookid = str(bookid)+','
    saved_readlist.readlist = saved_readlist.readlist.replace(
        replace_bookid, '')
    saved_readlist.save()

    if 'next' in request.GET:
        return redirect(request.GET['next'])

    return redirect('profile')
