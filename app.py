from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer
from qa_model import IslamicQAModel  # Asumsikan Anda menyimpan definisi model di file model.py
import sqlite3
import pandas as pd

app = Flask(__name__)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = IslamicQAModel.from_pretrained('path/to/your/saved/model')
model.to(device)
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")

# Load dataset
def load_hadist_data():
  hadith_db_path = './database/hadist_database.db'
  conn = sqlite3.connect(hadith_db_path)
  hadist_tables = ['musnad_ahmad', 'musnad_darimi', 'musnad_syafii', 'muwatho_malik', 'shahih_bukhari', 
                 'shahih_muslim', 'sunan_abu_daud', 'sunan_ibnu_majah', 'sunan_nasai', 'sunan_tirmidzi']
  
  all_hadist_df = pd.concat([pd.read_sql_query(f"SELECT * FROM {table}", conn) for table in hadist_tables])
  conn.close()
  return all_hadist_df

def load_quran_data():
  conn = sqlite3.connect('./database/quran_database.db')
  ayat_df = pd.read_sql_query("SELECT * FROM table_ayat", conn)
  surah_df = pd.read_sql_query("SELECT * FROM table_surah", conn)
  conn.close()
  return ayat_df, surah_df

hadist_df = load_hadist_data()
quran_ayat_df, quran_surah_df = load_quran_data()

import re

def clean_text(text):
  text = re.sub(r'\[(.*?)\]', r'\1', text) # buat clean [] dalam sanad
  text = re.sub(r'_', ' ', text) # clean _ 
  text = re.sub(r'\s+', ' ', text) # ilangin spasi berlebih
  return text.strip().lower()

def preprocess_hadist(df):
  df['arab_clean'] = df['arab'].apply(clean_text)
  df['terjemah_clean'] = df['terjemah'].apply(clean_text)
  df['kitab_clean'] = df['kitab'].apply(clean_text)
  return df

def preprocess_quran(ayat_df, surah_df):
  ayat_df['Arab_clean'] = ayat_df['Arab'].apply(clean_text)
  ayat_df['Terjemahan_clean'] = ayat_df['Terjemahan'].apply(clean_text)
  
  # Gabungkan informasi surah ke dalam ayat
  ayat_df = pd.merge(ayat_df, surah_df[['Surah', 'Ayat']], on='Surah', how='left')
  ayat_df['Surah_Ayat'] = ayat_df['Ayat_y'] + ' (' + ayat_df['Surah'].astype(str) + ':' + ayat_df['Ayat_x'].astype(str) + ')'
    
  return ayat_df

hadist_df = preprocess_hadist(hadist_df)
quran_df = preprocess_quran(quran_ayat_df, quran_surah_df)

# Gabungkan data hadist dan quran
combined_df = pd.concat([
    hadist_df[['terjemah_clean', 'kitab_clean']].rename(columns={'terjemah_clean': 'text', 'kitab_clean': 'source'}),
    quran_df[['Terjemahan_clean', 'Surah_Ayat']].rename(columns={'Terjemahan_clean': 'text', 'Surah_Ayat': 'source'})
])
combined_df = combined_df.reset_index(drop=True)

import torch
from torch.utils.data import Dataset, DataLoader

class IslamicDataset(Dataset):
	def __init__(self, texts, sources, tokenizer, max_length=512):
		self.texts = texts
		self.sources = sources
		self.tokenizer = tokenizer
		self.max_length = max_length

	def __len__(self):
		return len(self.texts)

	def __getitem__(self, idx):
		text = self.texts[idx]
		source = self.sources[idx]

		encoding = self.tokenizer.encode_plus(
				text,
				add_special_tokens=True,
				max_length=self.max_length,
				return_token_type_ids=False,
				padding='max_length',
				truncation=True,
				return_attention_mask=True,
				return_tensors='pt'
		)
			
		return {
				'input_ids': encoding['input_ids'].flatten(),
				'attention_mask': encoding['attention_mask'].flatten(),
				'text': text,
				'source': source
		}

# Buat dataset
dataset = IslamicDataset(combined_df['text'].tolist(), combined_df['source'].tolist(), tokenizer)

def answer_question(question, qa_model, tokenizer, dataset, max_length=512, top_k=5):
	qa_model.eval()
	
	# Lakukan retrieval untuk menemukan passages yang paling relevan
	retrieval_scores = []
	with torch.no_grad():
		for item in dataset:
			input_ids = item['input_ids'].unsqueeze(0).to(device)
			attention_mask = item['attention_mask'].unsqueeze(0).to(device)
			_, _, retrieval_score = qa_model(input_ids, attention_mask)
			retrieval_scores.append(retrieval_score.item())
	
	# Ambil top-k passages
	top_k_indices = sorted(range(len(retrieval_scores)), key=lambda i: retrieval_scores[i], reverse=True)[:top_k]
	top_k_passages = [dataset[i]['text'] for i in top_k_indices]
	
	# Gabungkan passages menjadi satu context
	context = " ".join(top_k_passages)
	
	# Tokenisasi input (pertanyaan + context)
	inputs = tokenizer.encode_plus(
		question,
		context,
		add_special_tokens=True,
		max_length=max_length,
		truncation=True,
		padding='max_length',
		return_tensors='pt'
	)
	
	input_ids = inputs['input_ids'].to(device)
	attention_mask = inputs['attention_mask'].to(device)
	
	# Dapatkan prediksi
	with torch.no_grad():
		start_logits, end_logits, _ = qa_model(input_ids, attention_mask)
	
	# Dapatkan indeks awal dan akhir jawaban
	start_index = torch.argmax(start_logits)
	end_index = torch.argmax(end_logits)
	
	# Decode jawaban
	answer_tokens = input_ids[0][start_index:end_index+1]
	answer = tokenizer.decode(answer_tokens)
	
	return answer


@app.route('/answer', methods=['POST'])
def get_answer():
    data = request.json
    question = data['question']
    answer = answer_question(question, model, tokenizer, dataset)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)