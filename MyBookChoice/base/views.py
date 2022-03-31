
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import torch
from sentence_transformers import SentenceTransformer, util
from django.shortcuts import render, HttpResponseRedirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from api.forms import SignUpForm,  LoginForm  # , AddRatingForm
from api.models import Book  # , Rating
from django.contrib import messages
from math import ceil
# Create your views here.
from django.views.generic import ListView
from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity


def BookGenreFilter():
    # filtering by genres
    allBooks = []
    categoriesBook = Book.objects.values('categories', 'id')
    genres = {item["categories"] for item in categoriesBook}
    for genre in genres:
        book = Book.objects.filter(categories=genre)
        n = len(book)
        nSlides = n // 4 + ceil((n / 4) - (n // 4))
        print(n)
        allBooks.append([book, range(1, nSlides), nSlides])
    params = {'allBooks': allBooks}
    return params


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
        return render(request, 'BookRecommender/signup.html', {'form': fm})
    else:
        return HttpResponseRedirect('/home/')


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
        return render(request, 'BookRecommender/login.html', {'form': fm})
    else:
        return HttpResponseRedirect('/home/')


def user_logout(request):
    if request.user.is_authenticated:
        logout(request)
        return HttpResponseRedirect('/login/')


def profile(request):
    if request.user.is_authenticated:

        return render(request, 'BookRecommender/profile.html')
    else:
        return HttpResponseRedirect('/login/')


class BookList(ListView):
    model = Book


def BookDetail(request, pk):
    book = Book.objects.get(id=pk)
    return render(request, 'BookRecommender/book.html', {'book': book})


def home(request):
    return render(request, 'BookRecommender/home.html')


BASE_DIR = os.getcwd()
df = pd.read_csv(
    BASE_DIR+"/templates/BookRecommender/books.csv", low_memory=False)

categories_list = df['categories'].tolist()
categories_set = set(categories_list)
list_res = (list(categories_set))
categories = []
for item in list_res:
    categories.append(item)
# print(categories)

'''
categories = ['Detective and mystery stories', 'Africa, East',
              'Hyland, Morn (Fictitious character)',
              'Detective and mystery stories, English', 'Ireland',
              "Children's stories, English", 'Literary Collections',
              'Imaginary wars and battles',
              'Characters and characteristics in motion pictures',
              'Fantasy fiction', 'Hallucinogenic drugs', 'Fiction',
              'Baggins, Frodo (Fictitious character)', 'Authors',
              'Conduct of life', 'Alienation (Social psychology)', 'Cowboys',
              'History', 'Juvenile Fiction', 'Literary Criticism', 'Science',
              'Biography & Autobiography', 'Family & Relationships',
              'Juvenile Nonfiction', 'Business & Economics', 'Poetry',
              'Self-Help', 'Sports & Recreation', 'True Crime', 'Psychology',
              'Religion', 'Travel', 'Social Science', 'Health & Fitness',
              'Music', 'Political science', 'Medical', 'Philosophy',
              'Body, Mind & Spirit', 'Language Arts & Disciplines', 'Education',
              'Political Science', 'Antiques & Collectibles', 'Reference',
              'Humor', 'Physicists', 'American fiction', 'American literature',
              'Anger', 'Comedy', 'Gangs', 'Short stories, American', 'Cults',
              'Computers', 'Mathematics', 'Art', 'Existential psychotherapy',
              'Drama', 'BIOGRAPHY & AUTOBIOGRAPHY', 'Humorous stories, English',
              'High schools', 'Dead', 'Families', 'American wit and humor',
              'Novelists, American', 'Men', 'French drama', 'Classical fiction',
              'Authors, English', 'Design', 'Adult children', 'Pets',
              'Authors, American', 'Performing Arts', 'Cancer',
              'Erinyes (Greek mythology)', 'Greek drama (Tragedy)',
              'English fiction', 'Beowulf', 'Zero (The number)', 'Photography',
              'Art museum curators', 'Cooking', 'Bibles', 'Adultery', 'Nature',
              'Literary Criticism & Collections', 'Young Adult Fiction',
              'Diary fiction', 'British', 'Bail bond agents', 'Catholics',
              'Political leadership', 'Bosnia and Hercegovina',
              'Mormon fundamentalism', 'India', 'Paris (France)', 'Books',
              'FICTION', 'Antisemitism', 'Popular culture', 'Great Britain',
              'Apartheid', 'Gardening', 'Cooking, French',
              'Comics & Graphic Novels', 'Bible', 'Short stories',
              'Foreign Language Study', 'Horror stories', 'Trials (Witchcraft)',
              'Ghost stories', 'Law', 'Architecture', 'Fairy tales',
              'Cider house rules. (Motion picture)', 'Apprentices',
              'Manuscripts', 'Amazon River Region', 'Latin poetry',
              'Polish poetry', 'Poets, American',
              'Englisch - Geschichte - Lyrik - Aufsatzsammlung',
              'Black humor (Literature)', 'Discworld (Imaginary place)',
              'Folklore', 'English language', 'Sexual behavior surveys',
              'Espionage', "Children's stories", 'Electronic books',
              'Alcestis (Greek mythology)', 'Sex customs',
              'Technology & Engineering', 'Essentialism (Philosophy)',
              'Music trade', 'Computer science', 'Democracy', 'Americans',
              'Reducing diets', 'Fantasy fiction, American', 'Adventure stories',
              'Explorers', 'Love', 'Games & Activities', 'Games',
              'Language and languages', 'Film producers and directors',
              'American wit and humor, Pictorial',
              'Amyotrophic lateral sclerosis', 'Cats', 'Magic', 'Transportation',
              'Finance, Personal', 'JUVENILE FICTION', 'Canada',
              'Insane, Criminal and dangerous', 'Study Aids',
              'Detective and mystery comic books, strips, etc',
              'Theology, Doctrinal', 'Bumppo, Natty (Fictitious character)',
              'Illinois', 'Humorous stories, American', 'Europe', 'Black market',
              'Shipwrecks', 'Vampires', 'Cerebrovascular disease',
              'Cities and towns', 'Botanique', 'American poetry', 'House & Home',
              'Crafts & Hobbies', 'Inventions', 'Building laws',
              'LITERARY CRITICISM', 'Human-animal relationships',
              'Church work with the poor',
              'Death (Fictitious character : Gaiman)', 'Astronomers', 'Girls',
              'Conan (Fictitious character)', 'Otherland (Imaginary place)',
              'Disasters', 'Lisbon (Portugal)', 'Consumer behavior',
              'Authors, Arab', 'Detective and mystery stories, American',
              'Everest, Mount (China and Nepal)', 'Boats and boating',
              'Minimal brain dysfunction in children', 'Spiritual life',
              'Meditation', 'Indic fiction (English)']
'''


def all_categories(request):
    return render(request, 'BookRecommender/all_categories.html', {'categories': categories})


def topincategories(request, c):
    data_return = df[df['categories'].str.contains(c)]
    data_return = data_return.sort_values(
        by=['average_rating'], ascending=False)[:5]
    return render(request, 'BookRecommender/topincategories.html', {'category': c, 'data_return': data_return})


def title_recommend_page(request):
    results = Book.objects.all()
    enteredtitle = (request.POST.get('booktitle'))
    # enteredtitle = (request.POST.get('fullname'))
    if request.method == 'POST':

        print('Entered text :'+str(enteredtitle))
        data_return = get_recommendations(enteredtitle)

        print(data_return)
        # return render(request, 'BookRecommender/book.html', {'bookresults': results, 'data_return': data_return, 'title': enteredtitle})
        return render(request, 'BookRecommender/title_recommend_results.html', {'bookresults': results, 'data_return': data_return, 'title': enteredtitle})
    return render(request, 'BookRecommender/title_recommend.html', {"showcity": results})


def get_recommendations(title):

    # Convert  index into series
    indices = pd.Series(df.index, index=df['title'])

    # Converting the book title into vectors
    tf = TfidfVectorizer(analyzer='word', ngram_range=(
        2, 2), min_df=1, stop_words='english')
    tfidf_matrix = tf.fit_transform(df['title'])

    # Cosine Similarity
    sg = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Get the index corresponding to original_title
    idx = indices[title]
    # Get the pairwsie similarity scores
    sig = list(enumerate(sg[idx]))
    # Sort the books
    sig = sorted(sig, key=lambda x: x[1], reverse=True)
    # scores of Top 10 Books
    # Book indicies
    book_indices = [i[0] for i in sig]
    # Top 5 book recommendation
    return df.iloc[book_indices].head(6)


def testpage(request):
    results = Book.objects.all()
    if request.method == 'POST':
        enteredtitle = (request.POST.get('fullname'))
        print('Entered text :'+str(enteredtitle))

        return render(request, 'BookRecommender/titlerecommend-results.html', {'bookresults': results, 'title': enteredtitle})
    return render(request, 'BookRecommender/test.html', {"bookresults": results})


def top_categories():
    C = df['average_rating']
    m = df['ratings_count'].quantile(0.25)
    q_books = df.copy(
    ).loc[df['ratings_count'] >= m]

    def weighted_rating(x, m=m, C=C):
        v = x['ratings_count']
        R = x['average_rating']
        # Calculation based on the IMDB formula
        return (v/(v+m) * R) + (m/(m+v) * C)

    q_books['score'] = q_books.apply(weighted_rating, axis=1)
    q_books = q_books.sort_values('score', ascending=False)

    # return q_books[['recommended_books', 'totalratings', 'rating', 'score']].head(5)
    return q_books.head(10)


def plot_recommend(request):

    def clean_text(text):
        """
        Series of cleaning. String to lower case, remove non words characters and numbers (punctuation, curly brackets etc).
            text (str): input text
        return (str): modified initial text
        """

        text = text.lower()  # lowercase text
        # replace the matched string with ' '
        text = re.sub(re.compile("\'s"), ' ', text)  # matches `'s` from text
        text = re.sub(re.compile("\\r\\n"), ' ', text)  # matches `\r` and `\n`
        # matches all non 0-9 A-z whitespace
        text = re.sub(re.compile(r"[^\w\s]"), ' ', text)
        return text

    def tokenizer(sentence, min_words=4, max_words=1000, stopwords=set(stopwords.words('english')), lemmatize=True):
        """
        Lemmatize, tokenize, crop and remove stop words.
        Args:
        sentence (str)
        min_words (int)
        max_words (int)
        stopwords (set of string)
        lemmatize (boolean)
        returns:
        list of string
        """
        if lemmatize:
            stemmer = WordNetLemmatizer()
            tokens = [stemmer.lemmatize(w) for w in word_tokenize(sentence)]
        else:
            tokens = [w for w in word_tokenize(sentence)]
        token = [w for w in tokens if (len(w) > min_words and len(w) < max_words
                                       and w not in stopwords)]
        return tokens

    def clean_sentences(df):
        """
        Remove irrelavant characters (in new column clean_sentence).
        Lemmatize, tokenize words into list of words (in new column tok_lem_sentence).
        Args: 
        df (dataframe)
        returns:
        df
        """
        df['clean_sentence'] = df['description'].apply(clean_text)
        df['tok_lem_sentence'] = df['clean_sentence'].apply(
            lambda x: tokenizer(x, min_words=4, max_words=1000, stopwords=set(stopwords.words('english'))))
        return df

    def extract_best_indices(m, topk, mask=None):
        """
        Use sum of the cosine distance over all tokens ans return best mathes.
        m (np.array): cos matrix of shape (nb_in_tokens, nb_dict_tokens)
        topk (int): number of indices to return (from high to lowest in order)
        """
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
        """
        Return the database sentences in order of highest cosine similarity relatively to each 
        token of the target sentence. 
        """
        # Embed the query sentence
        tokens_query = [str(tok) for tok in tokenizer(sentence)]
        embed_query = vectorizer.transform(tokens_query)
        # Create list with similarity between query and dataset
        mat = cosine_similarity(embed_query, tfidf_mat)
        # Best cosine distance for each token independantly
        best_index = extract_best_indices(mat, topk=10)
        return best_index

    stop_words = set(stopwords.words('english'))
    # Adapt stop words
    token_stop = tokenizer(' '.join(stop_words), lemmatize=False)

    # Fit TFIDF
    vectorizer = TfidfVectorizer(stop_words=token_stop, tokenizer=tokenizer)
    # -> (num_sentences, num_vocabulary)
    tfidf_mat = vectorizer.fit_transform(df['description'].values)

    # Return best threee matches between query and dataset
    #test_sentence = 'test'
    #recommendation = get_recommendations_tfidf(test_sentence, tfidf_mat)

    #print(df[['title', 'categories', 'description']].iloc[recommendation])
    enteredplot = (request.POST.get('booktitle'))
    if request.method == 'POST':
        print('Entered text :'+str(enteredplot))
        recommendation = get_recommendations_tfidf(enteredplot, tfidf_mat)
        data_return = (df.iloc[recommendation])

        #data_return = get_recommendations(enteredtitle)
        # print(data_return)
        # return render(request, 'BookRecommender/book.html', {'bookresults': results, 'data_return': data_return, 'title': enteredtitle})
        return render(request, 'BookRecommender/plot_recommend_results.html', {'data_return': data_return, 'enteredplot': enteredplot})
    return render(request, 'BookRecommender/plot_recommend.html')
