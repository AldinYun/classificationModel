#gdrive에서 데이터셋 파일을 가져오기 위해
from google.colab import drive
drive.mount('/gdrive',force_remount=True)
#데이터셋 임포트를 위해
import pandas as pd

import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel

class InternalDocumentClassifier(nn.Module):
    def __init__(self, num_classes):
        super(InternalDocumentClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.distilbert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        pooled_output = last_hidden_state[:, 0]  # CLS 토큰에 대한 출력만 사용
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

# 데이터셋과 데이터로더 정의
class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs["input_ids"])  # 'input_ids' 키를 이용하여 길이를 반환

    def __getitem__(self, idx):
        input_ids = self.inputs["input_ids"][idx]
        attention_mask = self.inputs["attention_mask"][idx]
        label = self.labels[idx]
        return {"input_ids": input_ids, "attention_mask": attention_mask}, label

# 데이터셋 준비
# 엑셀 파일 읽기
data = pd.read_excel("/gdrive/My Drive/데이터셋.xlsx")

# 열 선택 및 데이터 추출
inputs = data[data.columns[1]].tolist()  # B열을 리스트로 변환하여 inputs에 할당
labels = data[data.columns[0]].tolist()  # A열을 리스트로 변환하여 labels에 할당

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
tokenized_inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
dataset = FakeDataset(tokenized_inputs, labels)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# 모델 생성 및 학습
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InternalDocumentClassifier(num_classes=3).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for inputs_batch, labels_batch in data_loader:
        inputs_batch = {k: v.to(device) for k, v in inputs_batch.items()}
        labels_batch = labels_batch.to(device)

        optimizer.zero_grad()
        outputs = model(**inputs_batch)
        loss = nn.CrossEntropyLoss()(outputs, labels_batch)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

#모델을 활용해 분류 결과를 확인
def classify_sentence(sentence, model, tokenizer, device):
    # 입력 문장을 토크나이징하고 모델에 맞는 형식으로 변환
    tokenized_input = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")
    input_ids = tokenized_input['input_ids'].to(device)
    attention_mask = tokenized_input['attention_mask'].to(device)

    # 모델을 사용하여 분류 결과 예측
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    predicted_probabilities = nn.functional.softmax(outputs, dim=1)
    _, predicted_labels = torch.max(predicted_probabilities, 1)

    return predicted_labels.item()

# 분류 해 볼 문장
example_sentence = ""

# 모델을 CPU나 GPU로 이동
model.to(device)

# 문장 분류
predicted_label = classify_sentence(example_sentence, model, tokenizer, device)

# 결과 출력
print("Predicted Label:", predicted_label)
