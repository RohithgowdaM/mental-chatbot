from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Define directories for saving models
eng_swa_model_dir = "./model/eng_swa_model/"
swa_eng_model_dir = "./model/swa_eng_model/"

# Download English to Swahili model and tokenizer
print("Downloading English to Swahili model...")
eng_swa_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-swc", cache_dir=eng_swa_model_dir)
eng_swa_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-swc", cache_dir=eng_swa_model_dir)
print("English to Swahili model downloaded and saved in:", eng_swa_model_dir)

# Download Swahili to English model and tokenizer
print("Downloading Swahili to English model...")
swa_eng_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-swc-en", cache_dir=swa_eng_model_dir)
swa_eng_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-swc-en", cache_dir=swa_eng_model_dir)
print("Swahili to English model downloaded and saved in:", swa_eng_model_dir)
