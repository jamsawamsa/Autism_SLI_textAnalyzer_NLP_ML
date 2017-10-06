# imports for machine learners
from sklearn import datasets, tree, neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.ensemble import VotingClassifier


# Pre-processing imports
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

# Post-processing imports
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from pylab import *
import matplotlib.pyplot as plt

# General
import numpy as np
from itertools import product
import itertools
import time
import sys

__author__ = 'James'
"""
This file contains the engine for the machine learning portion of the project, including reporting
"""


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def get_proportions(list):
    """
    Gets the counts of the 3 values supplied in the list
    :param list: list of 3 integer elements, which can be either 0,1 or 2
    :return: total counts of each category
    """
    zer = 0
    one = 0
    two = 0
    for i in range(len(list)):
        if list[i] == 0.:
            zer += 1
        elif list[i] == 1.:
            one +=1
        elif list[i] == 2.:
            two +=1
    return zer, one, two


def print_scores_clf(clfr, data, target, clf_title, cv_, cmap_=plt.cm.Greys):
    """
    Prints classification scores
    """
    print(clf_title + ': ')
    score_cv = cross_val_score(clfr, data, target, cv=cv_)  # cross validation scores
    score = clfr.score(data, target)  # mean accuracy
    print("Mean accuracy: %0.2f" % score)
    print("Accuracy from CV scoring: %0.2f (+/- %0.2f)" % (score_cv.mean(), score_cv.std()))
    prediction = clfr.predict(data)

    # Classification report
    print(metrics.classification_report(target, prediction))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(target, prediction)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['SLI', 'ASD', 'TYP'], title='Confusion Matrix: ' + clf_title, cmap=cmap_)
    print('--------------------------\n')


def tabulate_scores(X_train, y_train, X_test, y_test ,flag=False):
    """
    Tabulates the scores given where:
    [ ]_train represents the training set and [ ]_test represents the testing set
    X denotes the feature set and y denotes the target label tags
    """
    splits = 4
    skf = StratifiedKFold(splits) # Stratified k-fold with 3 splits for use in cross validation scoring

    # Decision tree classifier
    clf_decision_tree = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_features='sqrt', max_depth=10)
    clf_decision_tree = clf_decision_tree.fit(X_train, y_train)

    # k-nearest neighbor classifier
    k = 8 # larger k suppresses noise but will have less distinct boundaries
    clf_nearest_neighbor = neighbors.KNeighborsClassifier(k, weights='uniform', algorithm='auto')
    clf_nearest_neighbor.fit(X_train, y_train)

    # Gaussian Naive Bayes
    gnb = GaussianNB()
    clf_gaussian_naive_bayes = gnb.fit(X_train, y_train)

    # Neural Network, Multi Layer Perceptron
    clf_neural_network = MLPClassifier(solver='lbfgs', alpha=5e-5, hidden_layer_sizes=(50,), random_state=1, tol=1e-5)
    clf_neural_network.fit(X_train, y_train)
    # Support vector machine with RBF kernel
    clf_svm = svm.SVC(C=2.0, kernel='linear', class_weight='balanced', decision_function_shape='ovr',probability=True)
    clf_svm.fit(X_train, y_train)
    # Soft voting estimator
    eclf = VotingClassifier(estimators=[('dt', clf_decision_tree), ('knn', clf_nearest_neighbor),
                                        ('mlp', clf_neural_network), ('gnb', clf_gaussian_naive_bayes),
                                        ('svc', clf_svm)], voting='soft', weights=[1,1,2,2,3])
    # Hard voting estimator
    # eclf = VotingClassifier(estimators=[('dt', clf_decision_tree), ('knn', clf_nearest_neighbor),
    #                                    ('mlp', clf_neural_network), ('gnb', clf_gaussian_naive_bayes),
    #                                    ('svc', clf_svm)], voting='hard')
    eclf = eclf.fit(X_train, y_train)

    print('Estimator scoring including: cross validation with stratified K-fold sampling (k = %d), mean accuracy, '
          'accuracy precision, recall and f-measure' % splits)
    print('--------------------------')
    print_scores_clf(clf_decision_tree, X_test, y_test, 'Decision Tree', skf)
    print_scores_clf(clf_nearest_neighbor, X_test, y_test, 'K-Nearest Neighbor', skf)
    print_scores_clf(clf_gaussian_naive_bayes, X_test, y_test, 'Gaussian Naive Bayes', skf)
    print_scores_clf(clf_neural_network, X_test, y_test, 'Neural Network, Multilayer Perceptron', skf)
    print_scores_clf(clf_svm, X_test, y_test, 'Support Vector Machine', skf)
    print_scores_clf(eclf, X_test, y_test, 'Ensemble method, Soft Voting', skf)

    # # Plotting decision regions
    # Taking the two most significant features
    if flag:
        X_presentation = X[:, [5, 14]]

        clf1 = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_features='sqrt', max_depth=10)
        clf2 = neighbors.KNeighborsClassifier(k, weights='uniform', algorithm='auto')
        clf3 = gnb = GaussianNB()
        clf4 = MLPClassifier(solver='lbfgs', alpha=5e-5, hidden_layer_sizes=(10,), random_state=1, tol=1e-5)
        clf5 = svm.SVC(C=2.0, kernel='linear', class_weight='balanced', decision_function_shape='ovr',probability=True)
        eclf2 = VotingClassifier(estimators=[('dt', clf_decision_tree), ('knn', clf_nearest_neighbor),
                                             ('mlp', clf_neural_network), ('gnb', clf_gaussian_naive_bayes),
                                             ('svc', clf_svm)], voting='soft', weights=[1,2,1,2,3])

        clf1.fit(X_presentation, y)
        clf2.fit(X_presentation, y)
        clf3.fit(X_presentation, y)
        clf4.fit(X_presentation, y)
        clf5.fit(X_presentation, y)
        eclf2.fit(X_presentation, y)

        x_min, x_max = X_presentation[:, 0].min() - 1, X_presentation[:, 0].max() + 1
        y_min, y_max = X_presentation[:, 1].min() - 1, X_presentation[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))

        f, axarr = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(10, 8))
        for idx, clf, tt in zip(product([0, 1], [0, 1, 2]),
                                [clf1, clf2, clf3, clf4, clf5, eclf2],
                                ['Decision Tree (depth=10)', 'KNN (k=8)', 'GNB', 'Neural Network',
                                 'Linear Kernel SVM', 'Soft Voting']):

            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
            axarr[idx[0], idx[1]].scatter(X_presentation[:, 0], X_presentation[:, 1], c=y, alpha=0.8)
            axarr[idx[0], idx[1]].set_title(tt)

        p1 = Circle((0, 0), fc="lightcoral")
        p2 = Circle((0.2, 0), fc="cornflowerblue")
        p3 = Circle((0, 0), fc="lightgreen")
        f.legend([p1, p2, p3], ['TYP', 'SLI', 'ASD'])


def print_baselines(y, y_train, y_test, feature_names):
    """
    Shows the baselines
    """
    sli, asd, typ = get_proportions(y)
    total = sli + asd + typ
    print('total labels: %d' % total)
    print('number of TYP: %d, ratio over total: (%f)' % (typ, typ/total))
    print('number of ASD: %d, ratio over total: (%f)' % (asd, asd/total))
    print('number of SLI: %d, ratio over total: (%f)' % (sli, sli/total))
    print('largest population: %d counts' % max(sli, asd, typ))
    print('therefore the baseline accuracy expected for all estimators is (%f) which is equivalent to randomly labeling'
          'everything as the largest population' % (max(sli, asd, typ)/total))
    print('training set proportions: ', get_proportions(y_train))
    print('test set proportions: ', get_proportions(y_test))
    print('--------------------------\n')

    print("Features: ")
    for i in range(len(features_names)):
        print("%d : %s" % (i, features_names[i]))
    print('--------------------------\n')


def feature_extractor(X_, y_):
    """
    Extracts features given a data set and the labels
    """
    # tree based feature selection
    clf_extra_trees = ExtraTreesClassifier(random_state=0)
    clf_extra_trees = clf_extra_trees.fit(X_, y_)
    importances = clf_extra_trees.feature_importances_

    model = SelectFromModel(clf_extra_trees, prefit=True)
    X_new = model.transform(X_)

    # Retrieving important features
    std = np.std([t.feature_importances_ for t in clf_extra_trees.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X_.shape[1]):
        print("%d. feature %d: %s (%f)" % (f + 1, indices[f], features_names[indices[f]], importances[indices[f]]))

    print('--------------------------\n')
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_.shape[1]), importances[indices], color="g", yerr=std[indices], align="center")
    plt.xticks(range(X_.shape[1]), indices)
    plt.xlim([-1, X_.shape[1]])

    return X_new

if __name__ == "__main__": # main
    # init vars
    t = time.time()
    orig_stdout = sys.stdout

    # file writing
    w = open('report.txt', 'w')
    sys.stdout = w

    features_names = ['total words', 'number of different words', 'total utterances', 'mean length of utterances',
                      'average syllables per word', 'flesch kincaid score', 'raw verbs vs total verbs',
                      'number of different pos tags', 'number of repeated words/phrases', 'number of partial words',
                      'number of filler words', 'degree of conversational support', 'prosody',
                      'average clauses per sentence', 'average left branching depth', 'max parse tree height',
                      'language model average uni-gram probability', 'language model average bi-gram probability',
                      'language model average tri-gram probability', 'language model average 4-gram probability']
    # Data set initialization
    # read file
    f = open('output_file', 'r+')
    X_ = []  # data set
    y_ = []  # target labels
    for i in f:
        s = i.replace('\t', ' ').strip('\n')
        s = [float(j) for j in s.split(' ')]
        y_.append(s[-1])
        X_.append(s[:-1])

    # Splitting the data set
    # Labels: 2 - TYP, 1 - ASD, 0 - SLI
    X = np.array(X_)
    y = np.array(y_)

    X_train_, X_test_, y_train_, y_test_ = train_test_split(X, y, test_size=0.3, random_state=0) # Regular splitting
    print_baselines(y, y_train_, y_test_, features_names)
    tabulate_scores(X_train_, y_train_, X_test_, y_test_)
    X_new = feature_extractor(X, y)
    print('Retraining estimators with extracted features only: ')
    X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y, test_size=0.3, random_state=0)
    tabulate_scores(X_train_new, y_train_new, X_test_new, y_test_new, True)
    t_ = time.time() - t
    print('Classification report generated in %0.4f seconds' % t_)
    plt.show()
