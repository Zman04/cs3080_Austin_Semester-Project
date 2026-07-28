from base_summarizer import BaseSummarizer
from nltk import sent_tokenize, word_tokenize

class FrequencySummarizer(BaseSummarizer):

    def summarize(self, text):
        sentence_scores = {}
        sentences = sent_tokenize(text) # Sentences is a list of sentences
        words = word_tokenize(text) # Words is a list of words including punctuation
        words_clean = [word.lower() for word in words if word.isalnum()]

        word_freq = {}
        for word in words_clean:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

        for i, sentence in enumerate(sentences):
            sentence_words = word_tokenize(sentence)
            sentence_words_clean = [sentence_word.lower() for sentence_word in sentence_words if sentence_word.isalnum()]

            score = sum(word_freq[sentence_word] for sentence_word in sentence_words_clean if sentence_word in word_freq)

            sentence_scores[i] = score
        """
        for idx, sentence in enumerate(sentences):
            words = [word.lower() for word in word_tokenize(sentence)]

            score = sum(word_freq[word] for word in words if word in word_freq)
            sentence_scores[idx] = score
        highest_scores = []
        sentence_scores_sorted = sorted(sentence_scores.items(), key=lambda x: x[1])[::-1] #sorted by values
        highest_scores = sentence_scores_sorted[:self.summary_length]
        highest_scores_sorted = sorted(highest_scores) # numerical order by keys
        #highest_scores = [highest_scores.append(value) for value in sentence_scores_sorted if [value.enumerate()] <= value in range(int(self.summary_length))]
        return " ".join([sentences[idx] for idx, score in highest_scores_sorted])"""