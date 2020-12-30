import os
import re
import pandas as pd
import telebot
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2")
model = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2", output_attentions=True)
print("Загрузка GPT3")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

TELE_API = os.environ.get('TELEGRAM_API')
bot = telebot.TeleBot(TELE_API)

with open('data_qa_.pickle', 'rb') as f:
    df = pickle.load(f)
print("Загрузка базы вопросов-ответов")

tfidf = TfidfVectorizer(min_df=5, max_df=.3, max_features=5000)
tfidf.fit(df.qp)
X = tfidf.transform(df.qp)
X = pd.DataFrame(X.todense(), columns=tfidf.get_feature_names())
print("Подготовка базы вопросов-ответов")

def gen_text(model, device, prompt_text):
    MODEL_LENGTH = 20

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
    encoded_prompt = encoded_prompt.to(device)

    output_sequences = model.generate(
        encoded_prompt,
        max_length=30,
        num_beams=10,
        early_stopping=True,
        top_p=0.95,
        no_repeat_ngram_size=5,
    )
    text = tokenizer.decode(output_sequences.cpu().flatten().numpy(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[&].+[;\n]', '', text)
    return text

@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    if (message.text != "Пока"):

        x = tfidf.transform([message.text])
        cosine_similarities = x.dot(X.T)
        arg_max = cosine_similarities.argmax()
        if arg_max:
            reply = df.loc[arg_max]
            answer = reply.answer
        else:
            answer = gen_text(model, device, message.text)

        bot.send_message(message.from_user.id, answer)

    else:
        return 0

bot.polling()