# General Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# PyTorch Imports
import torch
import torch.optim as optim
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

# Huggingface Imports
from datasets import load_dataset
#|%%--%%| <ErGWh2AQVX|Y8hnsrHafW>
r"""°°°
# Dataset Used
- Found on huggingface
- https://huggingface.co/datasets/Simezu/brain-tumour-MRI-scan
- Use command `!pip install datasets` to install the datasets library
°°°"""
#|%%--%%| <Y8hnsrHafW|iWWe8Skd2l>

# Load the dataset
dataset = load_dataset("Simezu/brain-tumour-MRI-scan")

# Convert to pandas DF
df_train = dataset["train"].to_pandas()
df_test = dataset["test"].to_pandas()

print(len(dataset['train']))
print(len(dataset['test']))
#|%%--%%| <iWWe8Skd2l|Z0UPfrDb4C>
r"""°°°
# Exploratory Data Analysis (EDA)
- Check Data Type : ["image", "label"]
- Plot Bar Graphs of Unique Values : Balanced
- Sample and Check Images : Grayscale
°°°"""
#|%%--%%| <Z0UPfrDb4C|e8hdcoSN5J>

def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary_data = []

    for col_name in df.columns:

        col_dtype = df[col_name].dtype
        n_nulls = df[col_name].isnull().sum()
        n_not_nulls = df[col_name].notnull().sum()
        if (df[col_name].dtype == "object"):
            n_distinct = None
            distinct_value_counts = None

        else:
            n_distinct = df[col_name].nunique()
            if n_distinct <= 10:
                distinct_value_counts = df[col_name].value_counts().to_dict()
            else:
                distinct_value_counts = df[col_name].value_counts().head(10).to_dict()
            distinct_value_counts = {k: v for k, v in sorted(distinct_value_counts.items(), key=lambda item: item[1], reverse=True)}

        summary = {
            "dtype": col_dtype,
            "n_null": n_nulls,
            "n_not_null": n_not_nulls,
            "n_distinct": n_distinct,
            "distinct": distinct_value_counts
        }
        summary_data.append(summary)

    summary_df = pd.DataFrame(summary_data, index=df.columns)
    return summary_df

print(column_summary(df_train))
print(column_summary(df_test))
#|%%--%%| <e8hdcoSN5J|69CN9HScqA>

value_counts = df_train["label"].value_counts().head(10).to_dict()
k = [str(s) for s in list(value_counts.keys())]
k = sorted(k, key=lambda item: int(item))
v = list(value_counts.values())

plt.figure(figsize=(10, 6))
plt.bar(k, v)
plt.xlabel("Labels")
plt.ylabel("Counts")
plt.title("Top 10 Label Counts")
plt.show()
#|%%--%%| <69CN9HScqA|ib15CGkWON>

sample_images = df_train.sample(9)
fig, axes = plt.subplots(3, 3, figsize=(10,10))
for ax, (img_path, label) in zip(axes.flatten(), sample_images.itertuples(index=False)):
    img = Image.open(img_path["path"])
    ax.imshow(img)
    ax.set_title(label)
    ax.axis("off")
plt.tight_layout()
plt.show()
#|%%--%%| <ib15CGkWON|CivImgAjhx>
r"""°°°
# Custom Dataset Definition
- Define a Custom Dataset Class
- Create DataLoaders
°°°"""
#|%%--%%| <CivImgAjhx|5FPpvsGp5n>

class TumorDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: bool = None) -> None:
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]["path"]
        label = self.df.iloc[idx, 1]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = TumorDataset(df_train, transform)
test_dataset = TumorDataset(df_test, transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#|%%--%%| <5FPpvsGp5n|QkBz7CmOJV>
r"""°°°
# Model Finetuning
- Use Resnet50 and Modify input and output feature count in fc
°°°"""
#|%%--%%| <QkBz7CmOJV|nYWFgvNujH>

model = models.resnet50(pretrained=True)

n_classes = len(df_train["label"].unique())
model.fc = torch.nn.Linear(model.fc.in_features, n_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
#|%%--%%| <nYWFgvNujH|I8FMtYGYll>
r"""°°°
Training Loop
°°°"""
#|%%--%%| <I8FMtYGYll|LkttzNCOHZ>

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 10
for epoch in range(n_epochs):
    model.train()
    train_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    print(f"Epoch: {epoch+1}")
    print(f"\tTrain Loss: {train_loss:.4f}")

    model.eval()
    test_loss = 0.0
    test_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    print(f"\tTest Loss: {test_loss:.4f}")
    print(f"\tTest Accuracy: {100*test_correct/total:.2f}%\n")

torch.save(model.state_dict(), f'model-{test_loss}.pth')
