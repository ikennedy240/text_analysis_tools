# import some sklearn stuff
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib
import gensim
import pandas as pd
import numpy as np
import nltk
import regex as re

class word_prevalence_by_group():
    def __init__(self, feature_names):
        self.feature_names = feature_names
        
    def fit(self, dtm, group_list):
        full_df = pd.DataFrame()
        for group in np.unique(group_list):
            group_means = X_vectorized[y==group].mean(0)
            group_frame = pd.DataFrame(group_means)
            group_frame.columns = self.feature_names
            group_frame['group'] = group
            full_df = full_df.append(group_frame)
        full_df.set_index('group', inplace = True)
        self.prevalence = full_df
        self.coef_ = full_df.to_numpy()
        return self
    
    def score(self, *args):
        return("No score on this one")

def plot_word_differences(data, identifier, group_col, text_col, index_col = 0, group_levels = None, use_na = False, stopwords = None, 
                          word_len = 3, prevalence_function = LogisticRegression, prevalence_options = {}, max_iter = 100, n_features = 20, cmap = 'Set1', group_labels = None, model_stats = 'print', tts = True):
    if type(data) is str:
        if identifier != 'index':
            data = pd.read_csv(data, dtype = {identifier:str}, index_col = index_col)
        else:
            data = pd.read_csv(data, dtype = {identifier:str}, index_col = index_col)
    elif type(data) is not pd.core.frame.DataFrame:
        raise ValueError("Data must be either a pandas dataframe or a valid path to a csv file")


    if type(model_stats) == str:
        model_state = [model_stats]

    if use_na:
        if any(data[group_col].isna()):
            data.fillna({group_col:'no cluster'}, inplace=True)
    else:
        data.dropna(subset = [group_col], inplace = True)

    if group_levels is None:
        groups = data[group_col].astype(str).unique()
    else:
        groups = group_levels
        data = data[data[group_col].astype(str).isin(groups)]

    if group_labels is None:
        group_labels = groups
    else:
        print("Using {} groups".format(', '.join(groups)))
        print("Using {} group labels".format(', '.join(group_labels)))

    group_range = range(len(groups))



    if len(group_labels) != len(groups):
        raise ValueError("there must be one group_label for each group, got {} groups and {} group_labels".format(len(groups), len(group_labels)))

    # get stop words and compile into regex

    if stopwords is None:
        stopwords = stop_words.ENGLISH_STOP_WORDS

    pattern = re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*', flags=re.IGNORECASE)


    X = [[word for word in pattern.sub('', document.lower()).split() if len(word)>word_len] for document in data[text_col].values]
    X = [' '.join(x) for x in X]
    y = data[group_col].astype(str)

    if tts:
        from sklearn.model_selection import train_test_split
        # this is important because it's where we set our outcome variable: above median white proportion
        X, X_test, y, y_test = train_test_split(X, y, random_state=0)

    # Fit the CountVectorizer to the training data
    vect = CountVectorizer(stop_words=stopwords).fit(X)
    print("Total Features: ", len(vect.get_feature_names()))
    X_vectorized =  vect.transform(X)

    # fit the model
    model = prevalence_function(**prevalence_options).fit(X_vectorized, y)
    # get the feature names as numpy array
    feature_names = np.array(vect.get_feature_names())
    # Sort the coefficients from the model
    coefs = model.coef_

    if 'print' in model_stats:
        print('In-sample accuracy is {}'.format(model.score(X_vectorized,y)))
        if tts:
            X_test_vectorized = vect.transform(X_test)
            print('Out-of-sample accuracy is {}'.format(model.score(X_test_vectorized,y_test)))

    coef_df = pd.DataFrame()
    for i in group_range:
        group_coefs = coefs[i]
        sorted_coefs = group_coefs[group_coefs.argsort()]
        sorted_features = feature_names[group_coefs.argsort()]
        group_df = pd.DataFrame({'coefs':sorted_coefs, 'features':sorted_features})
        group_df['group'] = groups[i]
        group_df['top_features'] = ','.join(sorted_features[-n_features:])
        coef_df = coef_df.append(group_df)
        coef_df.head()

    plot_df = pd.DataFrame()
    for i in group_range:
        group_df=coef_df[coef_df['group']==groups[i]]
        top_features = group_df.top_features.iloc[0].split(',')
        tmp_df = coef_df[[x in top_features for x in coef_df.features.values]].copy()
        tmp_df['plot_group'] = groups[i]
        plot_df = plot_df.append(tmp_df)



    fig, axs = plt.subplots(nrows =1, ncols = len(groups), figsize=(15,8))
    plt.xlabel("Model Effect",size=15)
    plt.ylabel("Term",size=15)

    for i in group_range:
        group = groups[i]
        plt_df = plot_df[plot_df.plot_group == group].copy()
        plt_df['sort'] = plt_df.coefs*((plt_df.group == group)*-1)
        plt_df = plt_df.sort_values(['sort'], ascending = True)

        ax = axs[i]

        c = plt_df.group.replace(groups, group_range).values

        scatter = ax.scatter(plt_df.coefs,plt_df.features,
                             c = c,
                    cmap = cmap)
        ax.set_ylim(n_features*1.05, n_features-n_features*1.03)
        ax.set_title(group_labels[i])
        if i == 0:
            # produce a legend with the unique colors from the scatter
            legend1 = ax.legend(*scatter.legend_elements(),
                                loc="lower left", title="Groups")
            ax.add_artist(legend1)
    plt.tight_layout()