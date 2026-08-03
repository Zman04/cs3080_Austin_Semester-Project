from base_summarizer import BaseSummarizer
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class TransformerSummarizer(BaseSummarizer):
    def __init__(
            self,
            model_name: str= "facebook/bart-large-cnn", # Model configuration
            max_length: int = 150, # 150 Tokens (110 - 150 words)
            min_length: int = 60, # 30 Tokens (20 - 30 words)
            num_beams: int = 4, # Beam search width (Number of alternative word sequences tracked)
            device: str = None, # Default will cause us to run through from fastest to slowest
            length_penalty=2.0
    ):
        super().__init__(summary_length=max_length)
        self.model_name = model_name
        self.max_length = max_length
        self.min_length = min_length
        self.num_beams = num_beams

        # Hardware seletion routine: CUDA -> MPS -> CPU
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name) # Load tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name) # Load sequence-to-sequence architecture

        self.model = self.model.to(self.device) # Transfer model parameters to computing device
        self.model.eval() # Set to evaluation mode (disable training-specific behaviors)

    def summarize(self, text: str):

        # Convert input text into PyTorch tensors
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024 # Truncate text beyond 1024 tokens
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()} # Shift input tensors onto the chosen hardware device

        # The model uses beam search to explore candidate summary paths
        with torch.no_grad(): # Disable gradiant since we are not training
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=self.max_length,
                min_length=self.min_length,
                num_beams=self.num_beams,
                no_repeat_ngram_size=3, # Avoid repeating 3-word phrases
                early_stopping=True
            )
        # Convert generated output token IDs back into human readable text
        summary_text = self.tokenizer.decode(
            summary_ids[0], 
            skip_special_tokens=True
        )

        return summary_text