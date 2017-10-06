from features import *
from nltk.corpus.reader import PlaintextCorpusReader
from nltk.corpus.reader import CHILDESCorpusReader
from nltk.corpus import cmudict
from LanguageModel import LanguageModel
from spacy.en import English

__author__ = 'James'
"""
The main driver function for data processing, and collecting features.
"""
if __name__ == '__main__':
    t = time.time() # Initialization
    output = []
    d = cmudict.dict()
    parser = English()

    # get corpus directories
    corpus_root_xml = nltk.data.find('C:\\Users\\James\\PycharmProjects\\FIT3036\\xml')
    corpus_root_plain = 'C:\\Users\\James\\PycharmProjects\\FIT3036\\plain_text'

    # get all xml and plain text files from specified directories
    corpus_xml = CHILDESCorpusReader(corpus_root_xml, '.*.xml')
    corpus_plain = PlaintextCorpusReader(corpus_root_plain, '.*.cha')

    # get all the words spoken by a child
    all_words = [w.lower() for w in corpus_xml.words(speaker=['CHI'])]

    # init wordnet and language model
    corpus_ic = wn.ic(corpus_xml, True, 1.0)
    lm = LanguageModel(all_words)

    # collect all the features for each corpus
    for j in range(len(corpus_xml.fileids())):
        current_features = [] # init empty array to store features
        # Text initialization
        text_xml = corpus_xml.fileids()[j]
        text_plain = corpus_plain.fileids()[j]

        # list of words spoken by the child in lowercase
        child_words_xml = [w.lower() for w in corpus_xml.words(text_xml, speaker=['CHI'])]

        # list of words spoken by the child in lowercase with replaced words
        child_words_replaced_xml = [w.lower() for w in corpus_xml.words(text_xml, speaker=['CHI'], replace=True)]

        # list of words spoken by the child in lowercase with the stemmed words
        child_words_stemmed_xml = [w.lower() for w in corpus_xml.words(text_xml, speaker=['CHI'], stem=True)]

        # list of words spoken by the child in lowercase with stemmed and replaced words
        # child_words_stemmed_replaced_xml =[w.lower() for w in conti4_xml.words(text_xml, speaker=['CHI'],
        # replace=True, stem=True)]

        # list of words spoken by the child with POS tags
        child_words_tagged_xml = corpus_xml.tagged_words(text_xml, speaker=['CHI'])

        # List of sentences/utterances spoken by the child
        child_sents_xml = corpus_xml.sents(text_xml, speaker=['CHI'])

        # List of sentences/utterances spoken by the investigator
        inv_sents_xml = corpus_xml.sents(text_xml, speaker=['INV', 'CLN', 'MOT', 'CLI'])

        # List of sentences spoken by the child in plain text with all annotations included
        child_sents_plain = []
        s = corpus_plain.sents(text_plain)
        for k in range(len(s)):
            for w in range(len(s[k])):
                try:
                    if s[k][w] == '*' and s[k][w+1] == 'CHI':
                        child_sents_plain.append(s[k][w:])
                except IndexError:
                    continue

        # current_text = Text()

        """
        Extracts features for some text
        features names: total number of words, number of different words, total number of utterances, mean length of
        utterance, average number of syllables per word, Flesch-Kincaid score, ratio of raw-verbs to total number of
        verbs, number of different POS tags, number of repeated words/phrases, number of partial words,
        number of filler words
        """
        total_words = n_words(child_words_xml)
        total_sents = n_utterances(child_sents_xml)
        total_syallables = total_syl(d, child_words_xml)

        # append collected features to the array
        current_features.append(total_words)
        current_features.append(n_diff_words(child_words_xml))
        current_features.append(total_sents)
        current_features.append(corpus_xml.MLU(text_xml)[0])
        current_features.append(total_syallables/total_words)
        current_features.append(flesch_kincaid_score(total_words, total_sents, total_syallables))
        current_features.append(raw_verbs_vs_verbs(child_words_tagged_xml, child_words_stemmed_xml))
        current_features.append(n_pos_tags(child_words_tagged_xml))

        n_repetitions, n_partial_words = verbal_fluency(child_sents_plain, child_words_xml)

        current_features.append(n_repetitions)
        current_features.append(n_partial_words)
        current_features.append(n_filler_words(child_words_xml))
        current_features.append(deg_conv_support(inv_sents_xml))
        current_features.append(prosody_measurement(corpus_ic, child_words_xml))
        current_features.append(avg_clauses_per_sent(parser, child_sents_xml))
        current_features.append(average_left_dependency(parser, child_sents_xml))

        # get parsed sentence tree height
        h = 0
        for i in range(len(child_sents_xml)):
            sentence = ' '.join(child_sents_xml[i])
            h = max(rec_traverse(parser, sentence), h)
        current_features.append(h)

        # language model features
        prob = lm.get_probabilities(child_words_xml, child_words_replaced_xml)
        prob = lm.get_probabilities(child_words_xml, child_sents_xml)
        for i in range(len(prob)):
            current_features.append(prob[i])

        # add tag
        if corpus_xml.fileids()[j][:3].lower() == 'sli':
            current_features.append(0)
        elif corpus_xml.fileids()[j][:3].lower() == 'asd':
            current_features.append(1)
        elif corpus_xml.fileids()[j][:3].lower() == 'typ':
            current_features.append(2)

        output.append(current_features)

    # Writing file
    f = open('output_file', 'w+')
    for i in range(len(output)):
        string = ['%.3g' % elem for elem in output[i]]
        f.write(' '.join(string) + '\n')

    # show time taken
    t_ = time.time() - t
    print('Classification report generated in %0.4f seconds' % t_)