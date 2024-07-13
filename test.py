from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn.functional as F

def train_epoch(model, dataloader, optimizer, device, accumulation_steps):
    model.train()
    total_loss = 0

    for i, batch in enumerate(tqdm(dataloader, desc="Training")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        start_logits, end_logits, retrieval_score = model(input_ids, attention_mask)

        # Hitung loss
        start_positions = torch.zeros(start_logits.size(0), dtype=torch.long).to(device)
        end_positions = torch.zeros(end_logits.size(0), dtype=torch.long).to(device)
        retrieval_labels = torch.ones(retrieval_score.size(0), dtype=torch.float).to(device)
            
        loss = F.cross_entropy(start_logits, start_positions) + \
            F.cross_entropy(end_logits, end_positions) + \
            F.binary_cross_entropy_with_logits(retrieval_score, retrieval_labels)

        loss = loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    return total_loss / len(dataloader)  # Mengembalikan rata-rata loss

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_retrieval_preds = []
    all_retrieval_labels = []
    all_start_preds = []
    all_end_preds = []
    all_start_labels = []
    all_end_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            start_logits, end_logits, retrieval_score = model(input_ids, attention_mask)
            
            start_positions = torch.zeros(start_logits.size(0), dtype=torch.long).to(device)
            end_positions = torch.zeros(end_logits.size(0), dtype=torch.long).to(device)
            retrieval_labels = torch.ones(retrieval_score.size(0), dtype=torch.float).to(device)
            
            loss = F.cross_entropy(start_logits, start_positions) + \
                F.cross_entropy(end_logits, end_positions) + \
                F.binary_cross_entropy_with_logits(retrieval_score, retrieval_labels)
            
            total_loss += loss.item()
            
            # Simpan prediksi dan label untuk metrik
            all_retrieval_preds.extend(torch.sigmoid(retrieval_score).cpu().numpy() > 0.5)
            all_retrieval_labels.extend(retrieval_labels.cpu().numpy())

            all_start_preds.extend(torch.argmax(start_logits, dim=1).cpu().numpy())
            all_end_preds.extend(torch.argmax(end_logits, dim=1).cpu().numpy())
            all_start_labels.extend(start_positions.cpu().numpy())
            all_end_labels.extend(end_positions.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    retrieval_accuracy = accuracy_score(all_retrieval_labels, all_retrieval_preds)

    # Hitung F1-score untuk posisi start dan end
    start_f1 = f1_score(all_start_labels, all_start_preds, average='macro')
    end_f1 = f1_score(all_end_labels, all_end_preds, average='macro')

    # Hitung rata-rata F1-score untuk QA
    qa_f1 = (start_f1 + end_f1) / 2
    
    return {
        'loss': avg_loss,
        'retrieval_accuracy': retrieval_accuracy,
        'qa_f1': qa_f1
    }

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn.functional as F

def train_epoch(model, dataloader, optimizer, device, accumulation_steps):
	model.train()
	total_loss = 0
	
	for i, batch in enumerate(tqdm(dataloader, desc="Training")):
		input_ids = batch['input_ids'].to(device)
		attention_mask = batch['attention_mask'].to(device)
	
		start_logits, end_logits, retrieval_score = model(input_ids, attention_mask)
		
		# Hitung loss
		start_positions = torch.zeros(start_logits.size(0), dtype=torch.long).to(device)
		end_positions = torch.zeros(end_logits.size(0), dtype=torch.long).to(device)
		retrieval_labels = torch.ones(retrieval_score.size(0), dtype=torch.float).to(device)
		
		loss = F.cross_entropy(start_logits, start_positions) + \
						F.cross_entropy(end_logits, end_positions) + \
						F.binary_cross_entropy_with_logits(retrieval_score, retrieval_labels)
		
		loss = loss / accumulation_steps
		loss.backward()
		
		if (i + 1) % accumulation_steps == 0:
			optimizer.step()
			optimizer.zero_grad()
     
		total_loss += loss.item() * accumulation_steps

	return total_loss / len(dataloader)

# evaluasi
def evaluate(model, dataloader, device):
	model.eval()
	total_loss = 0
	all_retrieval_preds = []
	all_retrieval_labels = []
	all_start_preds = []
	all_end_preds = []
	all_start_labels = []
	all_end_labels = []
    
	with torch.no_grad():
		for batch in tqdm(dataloader, desc="Evaluating"):
			input_ids = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			
			start_logits, end_logits, retrieval_score = model(input_ids, attention_mask)
			
			# Hitung loss
			start_positions = torch.zeros(start_logits.size(0), dtype=torch.long).to(device)
			end_positions = torch.zeros(end_logits.size(0), dtype=torch.long).to(device)
			retrieval_labels = torch.ones(retrieval_score.size(0), dtype=torch.float).to(device)
			
			loss = F.cross_entropy(start_logits, start_positions) + \
							F.cross_entropy(end_logits, end_positions) + \
							F.binary_cross_entropy_with_logits(retrieval_score, retrieval_labels)
			
			total_loss += loss.item()
			
			# Simpan prediksi dan label untuk metrik
			all_retrieval_preds.extend(torch.sigmoid(retrieval_score).cpu().numpy() > 0.5)
			all_retrieval_labels.extend(retrieval_labels.cpu().numpy())
			
			all_start_preds.extend(torch.argmax(start_logits, dim=1).cpu().numpy())
			all_end_preds.extend(torch.argmax(end_logits, dim=1).cpu().numpy())
			all_start_labels.extend(start_positions.cpu().numpy())
			all_end_labels.extend(end_positions.cpu().numpy())
	
	# Hitung metrik
	avg_loss = total_loss / len(dataloader)
	retrieval_accuracy = accuracy_score(all_retrieval_labels, all_retrieval_preds)
	
	# Hitung F1-score untuk posisi start dan end
	start_f1 = f1_score(all_start_labels, all_start_preds, average='macro')
	end_f1 = f1_score(all_end_labels, all_end_preds, average='macro')
	
	# Hitung rata-rata F1-score untuk QA
	qa_f1 = (start_f1 + end_f1) / 2
	
	return {
   'loss': avg_loss,
   'retrieval_accuracy': retrieval_accuracy,
   'qa_f1': qa_f1
	}