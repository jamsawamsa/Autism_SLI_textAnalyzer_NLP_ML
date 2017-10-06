__author__ = 'James'
"""
Class definition for the language model and word node classes
"""


class LanguageModel:
    def __init__(self, words, n=4):
        self.words = words
        self.n = n + 1
        self.dictionary, self.total = self.get_n_grams(words)

    def get_n_grams(self, words):
        """
        Gets the n_gram dictionary for the Language Model which includes the occurrences of each n_gram

        :param words: list of all words that make up the dictionary
        :return counts: the dictionary of the text
        :rtype: dictionary
        """
        counts = {}
        total = 0
        for i in range(len(words)):
            counts[words[i].lower()] = counts.get(words[i].lower(), 0) + 1
            total += 1
            if words[i] not in words[:i]:
                total += 1
            for j in range(2, self.n):
                if i+j > len(words):
                    break
                n_gram = ' '.join(words[i:i+j]).lower()
                counts[n_gram] = counts.get(n_gram, 0) + 1

        return counts, total

    def get_probabilities(self, words, sents):
        """
        Gets the probabilities of each sentence based on the range of n-grams and the dictionary of words defined in
        the LanguageModel class. Each sentence's probability is measured by the product of the probabilities of the
        n-grams that make up the sentence. Cross validation is used to measure the probabilities by first removing the
        occurrences of words of the current set from the dictionary.

        :param words: list of words to be analysed
        :param sents: list of sentences to be analysed
        :return p: list of tuples with the shape number_of_sentences * n-gram_range containing the probabilties of each
         sentence in each row
        :rtype: list of tuples
        """
        # Copying dictionary
        temp_dict = dict(self.dictionary)
        total_mod = 0

        # Cross validation phase - removing the occurrences of n-grams from the dictionary of the current set of text
        for i in range(len(words)):
            temp_dict[words[i].lower()] = temp_dict.get(words[i].lower(), 0) - 1
            total_mod += 1
            if temp_dict[words[i].lower()] == 0:
                total_mod += 1
            for j in range(2, self.n):
                if i+j > len(words):
                    break
                n_gram = ' '.join(words[i:i+j]).lower()
                temp_dict[n_gram] = temp_dict.get(n_gram, 0) - 1

        # Calculate the probabilities by getting their products. For uni-grams the probability is no. occurrences of the
        # word over the total no. of words. For n-grams > 2 the probability is measured by the no. of occurrences of the
        # n-gram over the no. of occurrences of the n-minus-1-gram
        p = []
        for i in range(len(sents)):
            prob = [1.0]*(self.n-1)
            s = ' '.join(sents[i]).strip().lower().split()
            for k in range(len(s)):
                prob[0] *= (float(temp_dict.get(s[k], 0) + 1)/(abs(self.total - total_mod) + 1))
                for j in range(2, self.n):
                    if k+j > len(s):
                        break
                    n_gram_numerator = ' '.join(s[k:k+j]).lower()
                    n_gram_denominator = ' '.join(s[k:k+j-1]).lower()
                    # Laplace smoothing
                    if (float(temp_dict.get(n_gram_denominator, 0))) == 0:
                        prob[j-1] *= 0
                    elif j == 2:
                        prob[j-1] *= (float(temp_dict.get(n_gram_numerator, 0) + 1) /
                                      (float(temp_dict.get(n_gram_denominator, 0 ) + 1)))
                    else:
                        prob[j-1] *= (float(temp_dict.get(n_gram_numerator, 0) + 1) /
                                      (float(temp_dict.get(n_gram_denominator, 0) + 1)))
            p.append(tuple(prob))

        p_output = [0. for i in range(self.n-1)]
        for i in range(len(p)):
            for j in range(len(p[i])):
                p_output[j] += p[i][j]
        p_output = [i/len(p) for i in p_output]
        return p_output


class WordNode:
    """
    Unused
    """
    def __init__(self, prev, next):
        self.prev = prev
        self.next = next


# if __name__ == "__main__":
