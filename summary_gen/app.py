from flask import Flask, render_template, request, make_response
import io
import os
import openai
from reportlab.pdfgen import canvas
from transformers import T5ForConditionalGeneration, GPT2LMHeadModel, PegasusForConditionalGeneration, T5Tokenizer, GPT2Tokenizer, PegasusTokenizer
import time
import nltk
from nltk.tokenize import word_tokenize
import openai
import requests
from simplet5 import SimpleT5
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
import re
from nltk.tokenize import word_tokenize
import transformers
from youtube_transcript_api import YouTubeTranscriptApi as yta
from pipelines import pipeline

# load Q&A

nlp = pipeline("multitask-qa-qg")


# load Google Pegasus

model_name = "google/pegasus-xsum"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
modelPega = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

# load Trained T5 model
# The trained model can be found at this drive link https://drive.google.com/drive/folders/1ikwBy0SKgMuUE8tjfnzxPB1xsanD9TRT?usp=sharing

model = SimpleT5()
model.from_pretrained(model_type="t5", model_name="t5-base")
modelt5 = model.load_model("t5","/Users/syedmohdgulambaquer/PycharmProjects/flaskProject1/simplet5-epoch-1-train-loss-2.844-val-loss-3.2345", use_gpu=False)

# Download the punkt
nltk.download('punkt')

# Relace rhe key with your own key

openai.api_key = 'OPEN_API_KEY'


# FUnction to extract the questions and answers from the dictionary

def extract_questions_and_answers(qa_list):
    questions = [qa['question'] for qa in qa_list]
    answers = [qa['answer'] for qa in qa_list]
    return questions, answers


# Preprocess Function
def preprocess_sentence(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)
    sentence = ' '.join(sentence.split())
    tokens = word_tokenize(sentence)
    preprocessed_sentence = ' '.join(tokens)
    return preprocessed_sentence


# Generate Summary using the GPT, for human readability
def generate_summary_gpt(input_text):

    stripped_lines = [line.strip() for line in input_text.split('\n')]
    non_empty_lines = [line for line in stripped_lines if line]
    input_text = ' '.join(non_empty_lines)
    print(input_text)
    input_text = preprocess_sentence(input_text)
    tokens = word_tokenize(input_text)

    if len(tokens) > 10000:
        print(f"Input text has too many tokens: {len(tokens)}")
        return "Input text has too many tokens: {len(tokens)}"

    if len(tokens) < 3300:
        try:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt="Give me detailed summary of "+input_text,
                temperature=0.7,
                max_tokens=120,
                top_p=0.9,
                frequency_penalty=0.0,
                presence_penalty=1
            )

            print(response)
            print(response["choices"][0]["text"])
            return response["choices"][0]["text"]
        except Exception as e:
            print(f"Error occurred while generating summary: {str(e)}")
            return "Error occurred while generating summary"

    chunk_size = 3300
    text_chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]

    chunk_summaries = []

    requests_made = 0

    for chunk in text_chunks:
        prompt = "Give me detailed summary of ".join(chunk)

        if requests_made == 3:
            requests_made = 0
            time.sleep(60)

        try:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                temperature=0.7,
                max_tokens=120,
                top_p=0.9,
                frequency_penalty=0.0,
                presence_penalty=1
            )

            summary = response["choices"][0]["text"]
            chunk_summaries.append(summary)
        except Exception as e:
            print(f"Error occurred while generating summary: {str(e)}")
            chunk_summaries.append("Error occurred while generating summary")

        requests_made += 1

    combined_summary = " ".join(chunk_summaries)
    return combined_summary

# Generate Summary Using the Google Pegasus Model

def generate_combined_summary(input_text, model, tokenizer, device):

    def tokenize_and_remove_quotes(text):
        tokens = word_tokenize(text)
        tokens_without_quotes = [token.replace('"', '') for token in tokens]
        return tokens_without_quotes

    stripped_lines = [line.strip() for line in input_text.split('\n')]
    non_empty_lines = [line for line in stripped_lines if line]
    input_text = ' '.join(non_empty_lines)
    input_text = preprocess_sentence(input_text)
    print(input_text)
    max_words_per_chunk = 900
    words = tokenize_and_remove_quotes(input_text)
    print(len(words))
    chunks = []
    chunk = []
    current_word_count = 0

    for word in words:
        if current_word_count + len(word.split()) <= max_words_per_chunk:
            chunk.append(word)
            current_word_count += len(word.split())
        else:
            chunks.append(" ".join(chunk))
            chunk = [word]
            current_word_count = len(word.split())

    if chunk:
        chunks.append(" ".join(chunk))
    print(len(chunks))
    generated_summaries = []

    for chunk in chunks:
        batch = tokenizer(chunk, truncation=True, padding="longest", return_tensors="pt").to(device)
        translated = model.generate(**batch)
        tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        generated_summaries.append(tgt_text[0])

    final_combined_summary = "\n".join(generated_summaries)
    return final_combined_summary

# Generate Summary Using the Trained Model
def generate_summary_t5(input_text, model):

    def tokenize_and_remove_quotes(text):
        tokens = word_tokenize(text)
        tokens_without_quotes = [token.replace('"', '') for token in tokens]
        return tokens_without_quotes

    stripped_lines = [line.strip() for line in input_text.split('\n')]
    non_empty_lines = [line for line in stripped_lines if line]
    input_text = ' '.join(non_empty_lines)
    print(input_text)
    input_text = preprocess_sentence(input_text)
    max_words_per_chunk = 400
    words = tokenize_and_remove_quotes(input_text)
    chunks = []
    chunk = []
    current_word_count = 0
    print(len(words))
    for word in words:
        if current_word_count + len(word.split()) <= max_words_per_chunk:
            chunk.append(word)
            current_word_count += len(word.split())
        else:
            chunks.append(" ".join(chunk))
            chunk = [word]
            current_word_count = len(word.split())

    if chunk:
        chunks.append(" ".join(chunk))

    generated_summaries = []
    print(len(chunks))
    for chunk in chunks:
        model.load_model("t5",
                   "/Users/syedmohdgulambaquer/PycharmProjects/flaskProject1/simplet5-epoch-1-train-loss-2.844-val-loss-3.2345",
                   use_gpu=False)
        tgt_text = model.predict(chunk)
        generated_summaries.append(tgt_text)

    final_combined_summary = ""
    for summ in generated_summaries:
        for inside in summ:
            final_combined_summary += inside
    return final_combined_summary


# Routes to the answers page
@app.route('/qa_generator', methods=['GET'])
def qa_generator():
    return render_template('qa_generator.html')


# Routes to the question and answers model
@app.route('/generate_qa', methods=['POST'])
def generate_qa():
    input_text = request.form['input_text']
    if input_text.startswith("https://www.youtube.com/"):
        try:
            print(True)
            video_id = input_text.split("v=")[1]
            data = yta.get_transcript(video_id)
            transcript = ''
            for value in data:
                for key, value1 in value.items():
                    if key == 'text':
                        transcript = transcript + value1 + " "
            input_text = transcript
        except Exception as e:
            print("Error extracting captions:", str(e))
            return render_template('index.html', qa_result="Error extracting captions")
    qa_result = nlp(input_text)
    print(input_text)
    print(type(qa_result))
    question, answer = extract_questions_and_answers(qa_result)
    qa_pairs = zip(question, answer)
    return render_template('qa_result.html', qa_pairs=qa_pairs)


# Routes to the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    summary = ""
    isGPT = False
    isT5 = False
    isPega = False
    if request.method == 'POST':
        input_text = request.form['input_text']
        model_name = request.form['model_name']

        if model_name == 't5':
            isT5 = True
        elif model_name == 'Da-Vinci':
            isGPT = True
        elif model_name == 'pegasus':
            isPega = True

        if input_text.startswith("https://www.youtube.com/"):
            try:
                video_id = input_text.split("v=")[1]
                data = yta.get_transcript(video_id)
                transcript = ''
                for value in data:
                    for key,value1 in value.items():
                        if key == 'text':
                            transcript = transcript + value1 + " "
                input_text = transcript
            except Exception as e:
                print("Error extracting captions:", str(e))
                return render_template('index.html', summary="Error extracting captions")
        # Generate summary using the selected model
        if isGPT:
            summary = generate_summary_gpt(input_text)
        elif isT5:
            summary = generate_summary_t5(input_text, model)
        elif isPega:
            summary = generate_combined_summary(input_text, modelPega, tokenizer, device)
    # Render the home page template with the generated summary
    return render_template('index.html', summary=summary)
# Create a Flask application instance
app = Flask(__name__)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
