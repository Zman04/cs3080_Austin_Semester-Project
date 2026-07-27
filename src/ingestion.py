# We call this file ingestion.py because it stands for data ingestion: Taking raw data from outside sources

class BaseSummarizer:
    """The parent blueprint for all summarization engines."""
    def __init__(self, summary_length: int):
        self.summary_length = summary_length

    def summarize(self, text: str) -> str:
        """Process raw text and return a condensed summary."""
        raise NotImplementedError("Each specific summarizer must implement its own logic.")