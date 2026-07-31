from base_summarizer import BaseSummarizer
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class TransformerSummarizer(BaseSummarizer):
    def __init__(
            self,
            model_name: str= "facebook/bart-base",
            max_length: int = 150,
            min_length: int = 30,
            num_beams: int = 4,
            device: str = None
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.min_length = min_length
        self.num_beams = num_beams

        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)

        self.model = self.model.to(self.device)
        self.model.eval()

    def summarize(self, text: str):
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=self.max_length,
                min_length=self.min_length,
                num_beams=self.num_beams,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
        summary_text = self.tokenizer.decode(
            summary_ids[0], 
            skip_special_tokens=True
        )
        
        return summary_text