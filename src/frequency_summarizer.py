from base_summarizer import BaseSummarizer
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords


class FrequencySummarizer(BaseSummarizer): # Create subclass

    def summarize(self, text): # 
        sentence_scores = {}
        sentences = sent_tokenize(text) # Sentences is a list of sentences

        stop_words = set(stopwords.words('english'))

        words = word_tokenize(text) # Words is a list of words including punctuation

        # Clean words so that trash/dupes don't skew scores
        words_clean = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]

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

        ranked_sentences = sorted(sentence_scores.items(), key=lambda pair: pair[1], reverse=True)

        top_ranked_sentences = ranked_sentences[:self.summary_length]

        top_ranked_sentences_sorted = sorted(top_ranked_sentences)

        summarized_sentences = " ".join([sentences[i] for i, score in top_ranked_sentences_sorted])

        return summarized_sentences