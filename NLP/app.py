import streamlit as st
from transformers import pipeline
from transformers import AutoTokenizer, T5Tokenizer
from transformers import T5ForConditionalGeneration

## Sentiment-Analysis/Classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

## Translation from English to German
model_name = "google/flan-t5-large"
flan_tokenizer = T5Tokenizer.from_pretrained(model_name)
flan_model = T5ForConditionalGeneration.from_pretrained(model_name)

## Summarization
bart_model_name = "facebook/bart-large-cnn"
summarizer = pipeline("summarization", model=bart_model_name)


# Function to classify sequence
def classify_text(sequence_to_classify, candidate_labels):
    results = classifier(sequence_to_classify, candidate_labels)
    max_value, max_index = max(
        (value, index) for index, value in enumerate(results["scores"])
    )
    st.write(((results["labels"])[max_index]).upper())


# Function to Translate text
def generate_text(prompt_text):
    text_embeddings = flan_tokenizer(
        "Translate text from English to German: " + prompt_text, return_tensors="pt"
    ).input_ids
    model_output = flan_model.generate(text_embeddings, max_new_tokens=50)
    # Decode and print response
    del_response = flan_tokenizer.decode(model_output[0], skip_special_tokens=True)
    st.write(del_response)


# Function to summarize text
def summarize_text(prompt_text):
    st.write(
        summarizer(prompt_text, max_length=130, min_length=30, do_sample=False)[0][
            "summary_text"
        ]
    )


## add in casual language modeling include prompt engineering + finetuning


# Function to start chatbot
def start_chatbot(prompt_text):
    pass
    st.write(" ")


if __name__ == "__main__":
    st.title("Generative AI Use Cases")
    prompt_text = st.text_input("Translate English to German:", value=" ")
    if prompt_text != " ":
        generate_text(prompt_text)

    seq2classify = st.text_input("Sequence to classify", value=" ")
    if seq2classify != " ":
        candidate_labels = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
        classify_text(seq2classify, candidate_labels)

    prompt_text_sum = st.text_input("Enter text to summarize:", value=" ")
    if prompt_text_sum != " ":
        summarize_text(prompt_text_sum)
