import nltk, re, pprint
from subject_object_extraction import findSVOs
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk import word_tokenize

import itertools
from collections import Counter
import time
__author__ = 'James'
"""
"""


def total_syl(d, words_list):
    """
    Calculates the total syllables of all words for a text

    :param d: dictionary object
    :param words_list: list of words
    :return: the total value
    :rtype: int
    """
    # Average number of syllables per word
    n_syllables = 0
    for i in range(len(words_list)):
        try:
            n_syllables += int(n_syl_cmu(d, words_list[i])[0])
        except KeyError:
            n_syllables += n_syl(words_list[i])
    return n_syllables
    # avg = n_syllables/len(words_list)
    # print("The average number of syllables per word is {}".format(avg))
    # return avg


def deg_conv_support(inv_sents):
    """
    Degree of conversational support/ no. of utterances made by investigator
    :param inv_sents: list of sentences spoken by the investigator
    :return: length of the list
    :rtype: int
    """
    # inv_sents = conti4_xml.sents(text, speaker=['INV'])
    return len(inv_sents)


def flesch_kincaid_score(t_words, t_sents, t_syl):
    """
    returns the Flesch-Kincaid score

    :param t_words: total number of words of a text
    :param t_sents: total number of sentences of a text
    :param t_syl: total number of syllables of a text
    :return: the Flesch-Kincaid score of a text
    :rtype: float
    """
    return 0.39*(t_words/t_sents) + 11.8 * (t_syl/t_words) - 15.59


def n_syl_cmu(d, word):
    """
    returns the number of syllables in a word if the word is in the carnegie mellon university dictionary

    :param d: dictionary object
    :param word: word to extract syllables from
    :return: number of syllables in word based on data from the dictionary
    """
    return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]]


def n_syl(word):
    # counts the number of syllables in a word
    count = 0
    vowels = 'aeiouy'
    word = word.lower().strip(".:;?!")
    try:
        if word[0] in vowels:
            count += 1
        for index in range(1,len(word)):
            if word[index] in vowels and word[index-1] not in vowels:
                count += 1
        if word.endswith('e'):
            count -= 1
        if word.endswith('le'):
            count += 1
        if count == 0:
            count += 1
    except IndexError:
        pass
    return count


def n_pos_tags(word_pos_list):
    """
    Number of different POS tags

    :param word_pos_list: list of tuples with words paired with it's POS tag
    :return: number of unique POS tags
    :rtype: int
    """
    # pos_tag_list = []
    # for i in range(len(word_pos_list)):
    #     pos_tag_list.append(str(word_pos_list[i][1]))
    # print(len(set(pos_tag_list)))
    return len(set(pos[1] for pos in word_pos_list))


def n_filler_words(word_list):
    """
    Calculates the number of English filler words used in the list

    :param word_list: list of words, lower cased
    :return: number of English filler words found in the list
     :rtype: int
    """
    # Returns total number of filler words
    return len([w for w in word_list if re.search('^[u]+[m]+$', w)]) + \
           len([w for w in word_list if re.search('^[e]+[r]+$', w)]) + \
           len([w for w in word_list if re.search('^[u]+[h]+$', w)]) + \
           len([w for w in word_list if re.search("^[h]+[a]+$", w)]) + word_list.count('xxx') + word_list.count('yyy')


def n_diff_words(word_list):
    # NDW = 1  # total number of different words
    # child_words_replaced = conti4.words(text, speaker=['CHI'], replace=True)
    # for i in range(1, len(child_words_replaced)):
    #     if child_words_replaced[i] not in child_words_replaced[:i-1]:
    #         NDW += 1
    # child_words_replaced = set(word_list)
    # print_content(child_words_replaced)
    """
    Finds the total number of different words

    :param word_list: list of words spoken by the child, all words must be lower cased, xml format
    :return: number of distinct words in the given list
    :rtype: int
    """
    return len(set(word_list))


def verbal_fluency(sents_list_plain, child_words_xml):
    """
    Returns the number of partial words and phrase/word repetitions spoken by the child

    :param sents_list_plain: list of sentences from a text corpus, specifically from the plain text reader
    :return: the number of repetition instances and the number of partial words
    """
    n_partial_words = 0
    n_repetitions = 0
    for i in range(len(sents_list_plain)):
        for j in range(len(sents_list_plain[i])):
            # Considers repetitions, retracings a hesitation marker from SALT
            if sents_list_plain[i][j] == '[/]' or sents_list_plain[i][j] == '[//]' or sents_list_plain[i][j] == '[/?]':
                n_repetitions += 1
            if sents_list_plain[i][j] == '(':
                try:
                    if sents_list_plain[i][j + 2] == ')' and len(sents_list_plain[i][j + 1]) == 1:
                        n_partial_words += 1
                        continue
                    # to accommodate the concatenation of a period at the end of the sentence just after the incomplete
                    # word marker
                    if sents_list_plain[i][j + 2] == ').' and len(sents_list_plain[i][j + 1]) == 1:
                        n_partial_words += 1
                        continue
                except IndexError:
                    continue
            if sents_list_plain[i][j].startswith('&'):
                n_partial_words += 1

    stops = set(stopwords.words('english'))
    child_words = [w.lower() for w in child_words_xml if w.lower() not in stops]
    # if there are no recorded/transcribed instances of [/], [//] and [/?], we measure the repetitions manually with
    # words not in the stop list
    if n_repetitions == 0:
        counter = dict(Counter(child_words))
        n_repetitions = max(counter.values())
    # if there are no recorded/transcribed partial words, we consider missing words
    if n_partial_words == 0:
        for sent in sents_list_plain:
            for w in sent:
                if w.startswith('0'):
                    n_partial_words += 1
    return n_repetitions, n_partial_words


def n_words(word_list):
    """
    Number of words spoken by the child

    :param word_list: list of words from the text, xml format
    :return: length of the given list
    :rtype: int
    """
    return len(word_list)


def n_utterances(sents_list):
    """
    Number of utterances spoken by child

    :param sents_list: list of utterances spoken by th child from the text, xml format
    :return: length of the given list
    :rtype: int
    """
    return len(sents_list)


def print_content(sentence):
    print(len(sentence))
    print(sentence)


def print_items(a_list):
    for item in a_list:
        print(item)


def prosody_measurement(corpus_ic, child_words):
    """
    Measures the similarity between nouns in the text

    :param corpus_ic: Corpus information content
    :param child_words: list of words from the text
    :return: returns the total similarity score between the first 10 nouns of the text
    :rtype: float
    """
    # corpus_ic = wn.ic(corpus, True, 1.0) # Should be generated outside of the function
    tagged_tokens = nltk.pos_tag(nltk.word_tokenize(' '.join(child_words)), tagset='universal')
    wl = []
    for i in tagged_tokens:
        # take nouns that are legitimately transcribed
        if i[1] == 'NOUN' and i[0] != 'xxx' and i[0] != 'www' and i[0] !='yyy' and i[0] not in wl:
            wl.append(i[0])
    if len(wl) > 10:
        wl = wl[:10]  # take the first 10 nouns if there are more than 10 nouns
    pos = []
    i = 0
    # Get appropriate synsets
    while i < len(wl):
        w = wn.synsets(wl[i])
        for j in range(len(w)):
            if w[j].pos() == 'n':
                pos.append((wl[i], j))
                break
        if len(pos) != len(wl[:i+1]):
            del wl[i]
            i -= 1
        i += 1
    sim_score = 0
    list1 = wl
    list2 = wl
    for word1, word2 in itertools.product(list1, list2):
        # print(word1, word2)
        if word1 == word2:
            continue
        word_from_list1 = wn.synsets(word1)[pos[wl.index(word1)][1]]
        word_from_list2 = wn.synsets(word2)[pos[wl.index(word2)][1]]
        # print('{w1}, {w2}: {s} '.format(
        #     w1=word_from_list1.name()[:word_from_list1.name().index('.')],
        #     w2=word_from_list2.name()[:word_from_list2.name().index('.')],
        #     s=word_from_list1.res_similarity(word_from_list2, corpus_ic)))
        sim_score += word_from_list1.res_similarity(word_from_list2, corpus_ic)
    return sim_score


def raw_verbs_vs_verbs(child_tagged_word_list, child_words_stemmed_xml):
    """
    Calculates the ratio of raw verbs to total verbs

    :param child_tagged_word_list: list of tuples with a word and it's POS tag
    :param child_words_stemmed_xml: list of words, stemmed
    :return: ratio of the raw verbs over the total verbs
    :rtype: float
    """
    # ratio of raw verbs to verbs
    raw_verbs = 0
    verbs = 0
    for i in range(len(child_tagged_word_list)):
        if child_tagged_word_list[i][1] == 'v':
            if child_tagged_word_list[i][0] == child_words_stemmed_xml[i]:
                raw_verbs += 1
            else:
                verbs += 1
    verbs += raw_verbs
    # print(raw_verbs, verbs)
    return (raw_verbs + 1)/(verbs + 1)


def rec_traverse(parser, sent):
    """
    recursively traverses down the parse tree of a sentence to get the height of the tree

    :param parser: parser object from spacy
    :param sent: sentence to be analyzed, string
    :return: height of the parse tree
    :rtype: int
    """
    parsed = parser(sent)
    root = ''
    height = 0
    for token in parsed:
        if token.dep_ == 'ROOT':
            root = token
            break
    if root != '':
        height = rec_traverse_aux(root)
    return height


def rec_traverse_aux(token, h=0, max_h=0):
    """
    Auxiliary function for recursive traversal of a parse tree
    :param token: current token that is evaluated
    :param h: height
    :param max_h: max height
    :return: max height found
    """
    if h > max_h:
        max_h = h
    left = [t for t in token.lefts if t.dep_ != 'punct']
    for i in range(len(left)):
        new_h = rec_traverse_aux(left[i], h+1, max_h)
        if new_h > max_h:
            max_h = new_h
    right = [t for t in token.rights]
    for i in range(len(right)):
        new_h = rec_traverse_aux(right[i], h+1, max_h)
        if new_h > max_h:
            max_h = new_h
    return max_h


def avg_clauses_per_sent(parser, child_sents):
    """
    Returns the average number of clauses per sentence in a text

    :param parser: English parser object from spacy
    :param child_sents: list of sentences
    :return: returns the average number of clausses per sentence from the list
    :rtype: float
    """
    sum = 0
    for i in child_sents:
        parse = parser(' '.join(i))
        sum += len(findSVOs(parse))
    return sum/len(child_sents)


def time_execution(f, n=1):
    """
    Timer function used for testing

    :param f: lambda expression containing the function to test
    :param n: number of times to run the function
    """
    s = time.clock()
    for i in range(n):
        f()
    print('Function executed '+ str(n) + ' time(s) in ' + str(time.clock() - s ) + ' seconds.')


def average_left_dependency(parser, sents):
    """
    Gets the average depth of left dependencies for every sentence based on Yngve's left branching sentence complexity
    hypothesis

    :param parser: spacy parser object
    :param sents: list of sentences
    :return: average depth of left dependencies
    """
    total_left_dependencies = 0
    for i in range(len(sents)):
        parsed_sent = parser(' '.join(sents[i]))
        for token in parsed_sent:
            t_left = [t for t in token.lefts]
            total_left_dependencies += len(t_left)
    return total_left_dependencies/len(sents)

