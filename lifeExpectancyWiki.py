#!/usr/bin/env python3
# coding: utf-8
from bs4 import BeautifulSoup
from contextlib import closing
from requests import get
from requests.exceptions import RequestException
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from time import time
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import random
import re
import seaborn as sns
import unicodedata
import wikipedia as wiki


path = os.path.dirname(os.path.abspath(__file__)) + '/objects/'
###############################################################################
###############################################################################
# MISCELLANEOUS FUNCTIONS
# =============================================================================
def save_obj(obj, name):
    """
    Enregistre au format pickle l'objet en entrée, avec le nom en entrée,
    dans le dossier visé par get_path(3)
    Args:
        obj (obj): python object to save
        name (str): name (without pkl extension) of the file
    """
    with open(path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    """Charge l'objet qui a été enregistré via la fonction save_obj, en utilisant
    le même nom que celui utilisé pour enregistrer"""
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def wanted(character):
    # return character.isalnum() or character in '\\/'
    return character.isalnum()

def fast_replace(string, replacement=' '):
    ascii_characters = [chr(ordinal) for ordinal in range(128)]
    ascii_code_point_filter = [c if wanted(c) else ' ' for c in ascii_characters]
    # Remove all non-ASCII characters. Heavily optimised.
    string = string.encode('ascii', errors='ignore').decode('ascii')

    # Remove unwanted ASCII characters
    return string.translate(ascii_code_point_filter)

def simple_get(url):
    """
    Attempts to get the content at `url` by making an HTTP GET request.
    If the content-type of response is some kind of HTML/XML, return the
    text content, otherwise return None.
    """

    def is_good_response(resp):
        """
        Returns True if the response seems to be HTML, False otherwise.
        """
        content_type = resp.headers['Content-Type'].lower()
        return (resp.status_code == 200 
                and content_type is not None 
                and content_type.find('html') > -1)

    def log_error(e):
        """
        It is always a good idea to log errors. 
        This function just prints them, but you can
        make it do anything.
        """
        print(e)

    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None

    except RequestException as e:
        log_error('Error during requests to {0} : {1}'.format(url, str(e)))
        return None

def dl_several_dates(list_year, size_batch=1000):
    """
    Get data ('import_wiki_pages_batch') for size_batch people dead in each
    year of list_years_death.
    """

    def get_list_indiv_event(year, event, size_batch=100):
        """ Gets the names of wikipedia pages of persons who died/were born in the
        given year """
        if year < 0:
            str_year = f"BC {abs(year)}"
        else:
            str_year = str(year)
        page = wiki.page(f"Category:{str_year} {event}s")
        url = page.url
        names = []
        # Get names of individuals through "{event}s of the year" category
        while len(names) < size_batch:
            raw_html = simple_get(url)
            html = BeautifulSoup(raw_html, 'html.parser')

            # Extract the names of the individuals of a page "year {event}s"
            start = 0
            for li in html.select('li'):
                if (li.text == str(year) or
                    li.text == f"AD {year}" or
                    li.text == f"{abs(year)} BC") and start == 1:
                    break
                if start:
                    names += [li.text]
                if 'A B C'in li.text or year < 0:
                    start = 1
            # Change page
            p_next = False
            for a in html.select('a'):
                try:
                    if a['title'] == f'Category:{str_year} {event}s' and a.text == 'next page':
                        url = 'https://en.wikipedia.org' + a['href']
                        p_next = True
                        # ipdb.set_trace()
                        break
                except KeyError:
                    continue
            # If no next page (we reached the end of that year's list of {event})
            if not p_next:
                break
        return names[0:size_batch]

    l_indiv_birth = []
    l_indiv_death = []
    for year in list_year:
        if year == 0:
            continue
        print(f"Download of wikipedia name lists of people born/dead in {year}")
        births = get_list_indiv_event(year=year, event='birth',
                                      size_batch=size_batch)
        deaths = get_list_indiv_event(year=year, event='death',
                                      size_batch=size_batch)
        l_indiv_birth += births
        l_indiv_death += deaths
        print(f"{len(births)} births, {len(deaths)} deaths")
    # We only keep names having both birth and death dates
    l_indiv_birth = np.unique(l_indiv_birth)
    l_indiv_death = np.unique(l_indiv_death)
    l_indiv, nb_occ = np.unique(list(l_indiv_birth) + list(l_indiv_death), return_counts=True)
    l_indiv = list(l_indiv[nb_occ == 2])
    try:
        l_indiv_old = load_obj(f"l_indiv_{list_year[0]}_{list_year[-1]}")
        save_obj(l_indiv_old, f"l_indiv_{list_year[0]}_{list_year[-1]}_save")
    except FileNotFoundError:
        pass
    save_obj(l_indiv, f"l_indiv_{list_year[0]}_{list_year[-1]}")

def import_wiki_pages_batch(list_year_death, duration_h=1, nb_indiv_batch=100):
    """ Download the categories and summary (and content?) of wikipedia pages
    and store them locally to an easier use later. Use ind_n to know where to
    resume downloading """
    t1 = time()
    timer = 3600*duration_h  # Length of the time period of downloading (in minutes)
    l_indiv = load_obj(f"l_indiv_{list_year_death[0]}_{list_year_death[-1]}")
    while (time() - t1) < timer:
        t_start_loop = time()
        time_left = timer - (t_start_loop - t1)
        print(f"Temps restant : {time_left//3600:.0f}h{time_left%3600//60:.0f}min")

        try:
            d_data = load_obj(f"d_data_{list_year_death[0]}_{list_year_death[-1]}")
        except FileNotFoundError:
            d_data = {}
        nb_d_init = len(d_data)
        save_obj(d_data, f"save_d_data_{list_year_death[0]}_{list_year_death[-1]}")
        try:
            ind_n = load_obj(f"ind_n_{list_year_death[0]}_{list_year_death[-1]}")
        except FileNotFoundError:
            ind_n = 0
        ind_n1 = ind_n + nb_indiv_batch
        for i in l_indiv[ind_n:ind_n1]:
            if i not in d_data:
                try:
                    data = wiki.page(i)
                    d_data[i] = {'cat': data.categories,
                                 'sum': data.summary}
                except:
                    # print("This term is too vague : ", i)
                    continue
        save_obj(ind_n1, f"ind_n_{list_year_death[0]}_{list_year_death[-1]}")
        save_obj(d_data, f"d_data_{list_year_death[0]}_{list_year_death[-1]}")
        nb_d_final = len(d_data)
        print(f"{(nb_d_final - nb_d_init)/((time() - t_start_loop)/3600):.0f} people's pages imported per hour")

def dates_to_df(d_data):
    """ Extract dates of birth/deaths from wiki pages locally stored """

    def cat_to_date(cat, bir_dea):
        date = cat.split(' ' + bir_dea)[0].lower()
        if 'bc' in date:
            return -int(date.split(' bc')[0].split('s')[0])
        elif 'ad' in date:
            return int(date.split('ad ')[1].split('s')[0])
        else:
            return int(date.split('s')[0])

    # Get their date of birth and death when available
    d_birth = {}
    for indiv in d_data:
        for cat in d_data[indiv]['cat']:
            if 'birth' in cat and 'century' not in cat:
                try:
                    d_birth[indiv] = cat_to_date(cat, 'birth')
                    break
                except:
                    print("Problem with : ", indiv, cat.split(' birth')[0])

    d_death = dict.fromkeys(d_birth, np.nan)
    for indiv in d_birth.keys():
        for cat in d_data[indiv]['cat']:
            if 'death' in cat and 'century' not in cat:
                try:
                    d_death[indiv] = cat_to_date(cat, 'death')
                    break
                except:
                    print("Problem (not an int date) with : ", indiv, cat.split(' death')[0])

    # Put in a dataframe the details about each individual : name, birth,
    # death, lifetime...
    names = list(d_birth)
    df = pd.DataFrame(data=[[d_birth[n], d_death[n]] for n in names],
                      index=names, columns=["Birth", "Death"])
    # ipdb.set_trace()
    df['Lifetime'] = df['Death'] - df['Birth']
    df['Centuries'] = np.floor(df['Birth']/100)*100
    # Clear negative lifetime
    df.drop(df[df.Lifetime < 1].index, inplace=True)

    return names, df

def a_voir():
    print ("TF-IDF on text data ... ")
    tfidf = TfidfVectorizer(binary=True)
    def tfidf_features(txt, flag):
        if flag == "train":
            x = tfidf.fit_transform(txt)
        else:
            x = tfidf.transform(txt)
        x = x.astype('float16')
        return x 
    X = tfidf_features(train_text, flag="train")
    X_test = tfidf_features(test_text, flag="test")

# About the names
def about_names():
    l_indiv = []
    list_year_death = range(1000, 1100)
    l_indiv += load_obj(f'l_indiv_{list_year_death[0]}_{list_year_death[-1]}')
    list_year_death = range(1000, 1800)
    l_indiv += load_obj(f'l_indiv_{list_year_death[0]}_{list_year_death[-1]}')
    list_year_death = range(1800, 2000)
    l_indiv += load_obj(f'l_indiv_{list_year_death[0]}_{list_year_death[-1]}')

    concat_indivs = ''
    for n in l_indiv:
        concat_indivs += ' ' + n
    l_words = []
    for i in concat_indivs.split(','):
        l_words += i.split(' ')
    l_words = [i for i in l_words if len(i) > 3]
    l_words, nb_occ = np.unique(l_words, return_counts=True)
    df_names = pd.DataFrame()
    df_names['Names'] = l_words
    df_names['Occurrences'] = nb_occ
    df_names.sort_values(by='Occurrences', ascending=False, inplace=True)
    print(df_names)

# About the categories
def about_categories():
    cats = []
    # for p in d_data:
    for p in names:
        cats += list(d_data[p]['cat'])
    cat, nb = np.unique(cats, return_counts=True)
    df_cat = pd.DataFrame()
    df_cat['Categories'] = cat
    df_cat['Occurrences'] = nb
    df_cat.sort_values(by='Occurrences', ascending=False, inplace=True)
    print(df_cat.iloc[0:100, :])

# Extract dates from dictionnary of wiki pages
def plot_lifespan_by_period(list_year_death, date_min=1800, date_max=1960):
    d_data = load_obj(f'd_data_{list_year_death[0]}_{list_year_death[-1]}')
    names, df = dates_to_df(d_data)

    # Miscellaneous plots
    # df.hist(column='Lifetime', bins=25);plt.show()
    # df.hist(column='Birth', bins=25);plt.show()

    # Boxplots by century
    ax = sns.boxplot(x='Centuries',
                     y='Lifetime',
                     data=df[df.Centuries.isin(range(date_min, date_max, 100))])
    ax = sns.stripplot(x='Centuries',
                       y='Lifetime',
                       data=df[df.Centuries.isin(range(date_min, date_max, 100))],
                       color="grey", jitter=0.1, size=3)
    plt.title("Lifetime by century", weight='demibold')
    plt.xticks(weight='demibold', size='small', rotation=75)
    plt.xlabel('', weight='demibold', size='medium')
    plt.ylabel('Lifetime in years', weight='demibold', size='medium')
    plt.show()

    # Regression lin, Lifetime vs Birth
    indiv_filter = np.logical_and(df['Death'].notna(),
                                  np.logical_and(df['Birth'] > date_min,
                                                 df['Birth'] < date_max))
    x = np.array(df.loc[indiv_filter, 'Birth'])
    y = np.array(df.loc[indiv_filter, 'Lifetime'])
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    reg_text = ("y = {:.2f}x + {:.2f} , avec r² = {:.2f}".
                format(slope, intercept, r_value**2))
    print("Résultats de la régression linéaire y = ax + b : a = {}, b = {}, "
          "pvalue = {}, std_err = {}" .format(slope, intercept, r_value, p_value,
                                              std_err))
    # Scatter plot
    plt.clf()
    plt.plot(x, intercept + slope*x, 'r', label='fitted line')
    plt.scatter(x, y, 1)
    plt.title('Lifetime against date of bith', weight='medium')
    plt.xlabel('Year of birth', weight='demibold')
    plt.ylabel('Lifetime (years)', weight='demibold')
    plt.text(np.min(x), np.max(y), reg_text, color='red', weight='demibold', size='medium')
    plt.show()

# Focus on the summary of each personality, clean the str to just ascii char
def word_probability(d_data):
    """
    Go through each summary and calculate for each unique word, its proportion
    among all the word appearances, and the proportion of pages it appears out
    of all the pages.
    """
    l_summary = [fast_replace(
        unicodedata.normalize('NFKD', d_data[p]['sum'].
                              lower()).encode('ASCII', 'ignore').
        decode('utf-8')) for p in d_data]
    nb_tot_page = len(l_summary)
    # List of all the words of all summaries of all personality wiki pages
    l_tot_words = ' '.join(l_summary).split()
    nb_tot_words = len(l_tot_words)
    l_unique_words, tot_occu = np.unique(l_tot_words, return_counts=True)
    with open('unique_words.txt', 'w') as f:
        for word in l_unique_words:
            print(word, file=f)
    # List of all the words, same as above, but counted only once per summary
    l_l_words_per_page = [string.split() for string in l_summary]
    l_tot_unique_words = [word for str_sum in l_summary for word in np.unique(str_sum.split())]
    l_unique_words, tot_unique_occu = np.unique(l_tot_unique_words, return_counts=True)

    # dic of words and their characteristics
    d_words = {word: {'nb_absol': tot_occ,
                      'nb_page': tot_page_occ,
                      'P_absol_presence': tot_occ/nb_tot_words,
                      'P_per_page_presence': tot_page_occ/nb_tot_page}
               for word, tot_occ, tot_page_occ in zip(l_unique_words,
                                                      tot_occu,
                                                      tot_unique_occu)}
    # DataFrame version from dict
    df_words = pd.DataFrame.from_dict(d_words, orient='index')
    df_words.sort_values(by='P_absol_presence', ascending=False, inplace=True)
    with open('unique_words_charac_abs.txt', 'w') as f:
        print(df_words.to_string(), file=f)

    df_words.sort_values(by='P_per_page_presence', ascending=False, inplace=True)
    with open('unique_words_charac_page.txt', 'w') as f:
        print(df_words.to_string(), file=f)

    # Save the final dict and df
    save_obj(l_summary, 'l_summary')
    save_obj(d_words, 'd_words')
    save_obj(df_words, 'df_words')

    return l_summary, d_words, df_words


###############################################################################
###############################################################################
# MAIN
# =============================================================================
if __name__ == '__main__':
    # Set variables
    # list_year_death = range(1000, 1100)
    # list_year_death = range(1000, 1800)
    list_year_death = range(1800, 2000)

    # Load data
    l_indiv = load_obj(f'l_indiv_{list_year_death[0]}_{list_year_death[-1]}')
    d_data = load_obj(f'd_data_{list_year_death[0]}_{list_year_death[-1]}')
    # l_summary, d_words, df_words = word_probability(d_data)
    l_summary = load_obj('l_summary')
    d_words = load_obj('d_words')
    df_words = load_obj('df_words')

    # Import more wiki pages about individuals and store them locally
    if 0:
        # dl_several_dates(list_year_death, size_batch=500)
        import_wiki_pages_batch(list_year_death, duration_h=6)


    # Do stuff
    # plot_lifespan_by_period(list_year_death, date_min=1400, date_max=1500)

    # P's of the unique words of a couple of random summaries
    # for i in np.random.choice(len(l_summary), size=10, replace=False):
    for i in np.arange(10):
        summ = l_summary[i]
        name = list(d_data.keys())[i]
        W = []
        P = []
        for w in np.unique(summ.split()):
            P.append(d_words[w]['P_per_page_presence'])
            W.append(w)
        pos = np.arange(len(P))/len(P)
        ind_sort = np.argsort(P)
        print('++++++++++++++++++++++++')
        print(name, np.array(W)[ind_sort][np.logical_and(pos > 0.1, pos < 0.6)])
        # plt.scatter(pos, np.sort(P))
