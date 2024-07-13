import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
