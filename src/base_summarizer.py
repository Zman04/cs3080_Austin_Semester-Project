'''
Defines a clean, unified function signature so both the frequency and the transformer
algorithms can be swapped between
'''

class BaseSummarizer:
    """
    The parent blueprint for all summarization engines.
    
    summary_length: The summary length is required for the frequency_summarizer
    """
    def __init__(self, summary_length: int):
        self.summary_length = summary_length

    def summarize(self, text: str):
        """Process raw text and return a condensed summary."""
        raise NotImplementedError("Each specific summarizer must implement its own logic.")