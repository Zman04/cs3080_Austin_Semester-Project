from frequency_summarizer import FrequencySummarizer
from transformer_summarizer import TransformerSummarizer

sample_text = """It is always a pleasure to see a familiar face wander up the 
mountain path to the shrine. The breeze up here at the 
Grand Narukami Shrine is particularly refreshing today, isn't it? 
Though I must say, your blue tail seems to be swaying with a certain... 
focused intent. 
You always bring the most intriguing little puzzles with you when you come to visit."""

freq_sum = FrequencySummarizer(2)
print(freq_sum.summarize(sample_text))

trans_sum = TransformerSummarizer(device="cpu")
print(trans_sum.summarize(sample_text))