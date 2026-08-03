from base_summarizer import BaseSummarizer # Import base blueprint
from nltk import sent_tokenize, word_tokenize # Import slicing tools from nltk
from nltk.corpus import stopwords # Import stop words


class FrequencySummarizer(BaseSummarizer): # Create subclass inheriting from BaseSummarizer

    def summarize(self, text):
        sentence_scores = {} # Prepare a dictionary to tally each sentence's worth
        # sent_tokenize splits the entire text into sentences
        sentences = sent_tokenize(text) # Sentences is a list of sentences
        stop_words = set(stopwords.words('english')) # Gather a list of stop words like "the", "is", or "at"

        # Chop full text into individual words and punctuation tokens
        words = word_tokenize(text) # Words is a list of words including punctuation

        # Clean words so that trash/dupes don't skew scores
        words_clean = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]

        # Count popularity of each clean word. If a word appears, increment its tally; if new, we set it to one.
        word_freq = {}
        for word in words_clean:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

        # Loop over each original sentence with its index i.
        for i, sentence in enumerate(sentences):
            sentence_words = word_tokenize(sentence) # Tokenize the sentence into words
            sentence_words_clean = [sentence_word.lower() for sentence_word in sentence_words if sentence_word.isalnum()] # and keep only the alphanumeric ones in lowercase

            # Calculate the sentence's total score by summing up the frequency scores of all its constituent words
            score = sum(word_freq[sentence_word] for sentence_word in sentence_words_clean if sentence_word in word_freq)

            sentence_scores[i] = score # Records the score in our dictionary mapped to sentence's original index

        # Sort sentences by calculated score in descending order.
        ranked_sentences = sorted(sentence_scores.items(), key=lambda pair: pair[1], reverse=True)
    
        # Slice list to keep only the top sentences
        top_ranked_sentences = ranked_sentences[:self.summary_length]

        # Sort top sentences back into chronological order
        top_ranked_sentences_sorted = sorted(top_ranked_sentences)

        # Join together the new sorted sentences
        summarized_sentences = " ".join([sentences[i] for i, score in top_ranked_sentences_sorted])

        return summarized_sentences