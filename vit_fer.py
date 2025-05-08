import os
import time
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import requests

from datasets import Dataset, Features, ClassLabel, Array3D
from transformers import ViTImageProcessor, ViTModel, TrainingArguments, Trainer
from transformers.modeling_outputs import SequenceClassifierOutput
from evaluate import load


# Detect device: mps (Apple GPU) → cuda (NVIDIA) → cpu (fallback)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Constants
DATA_PATH = "./fer2013/fer2013.csv"
STRING_LABELS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
MODEL_NAME = 'google/vit-base-patch16-224-in21k'
NUM_LABELS = len(STRING_LABELS)
METRIC_NAME = "accuracy"


def prepare_fer_data(data):
    image_list = []
    image_labels = list(map(int, data['emotion']))
    for row in data.index:
        image = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ').reshape(48, 48)
        image = np.repeat(image[..., np.newaxis], 3, axis=2).astype(int).tolist()
        image_list.append(image)
    return pd.DataFrame(list(zip(image_list, image_labels)), columns=['img', 'label'])


def preprocess_images(examples, feature_extractor):
    images = [np.moveaxis(np.array(img, dtype=np.uint8), -1, 0) for img in examples['img']]
    inputs = feature_extractor(images=images)
    examples['pixel_values'] = inputs['pixel_values']
    return examples


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch, channels, _, _ = x.size()
        weights = self.fc(self.global_avg_pool(x).view(batch, channels)).view(batch, channels, 1, 1)
        return x * weights


class CNNFeatureExtractorImproved(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.se = SEBlock(128)
        self.conv3 = nn.Conv2d(128, output_channels, 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.resize = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.se(self.relu(self.conv2(x)))
        x = self.conv3(x)
        return self.resize(x)


class ViTForImageClassificationWithCNN(nn.Module):
    def __init__(self, num_labels=NUM_LABELS):
        super().__init__()
        self.cnn = CNNFeatureExtractorImproved(input_channels=3, output_channels=3)
        self.vit = ViTModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values=None, labels=None, **kwargs):
        if pixel_values is None:
            raise ValueError("pixel_values must be provided")
        pixel_values = pixel_values.to(DEVICE)
        cnn_features = self.cnn(pixel_values)
        outputs = self.vit(pixel_values=cnn_features)
        logits = self.classifier(self.dropout(outputs.last_hidden_state[:, 0]))
        loss = None
        if labels is not None:
            labels = labels.to(DEVICE)
            loss = nn.CrossEntropyLoss()(logits.view(-1, self.classifier.out_features), labels.view(-1))
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


def check_hf_hub_access():
    try:
        response = requests.get("https://huggingface.co", timeout=5)
        return response.status_code == 200
    except:
        return False


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    if check_hf_hub_access():
        try:
            return load(METRIC_NAME).compute(predictions=predictions, references=labels)
        except:
            print("Warning: Failed to load metric from Hugging Face Hub, falling back to sklearn.metrics")
            return {"accuracy": accuracy_score(labels, predictions)}
    else:
        print("Warning: Hugging Face Hub is not accessible, using sklearn.metrics")
        return {"accuracy": accuracy_score(labels, predictions)}


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, xticklabels=STRING_LABELS, yticklabels=STRING_LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    timestamp = time.strftime("%Y%m&d%H%M%S")
    plt.savefig(f"confusion_matrix_{timestamp}.png")
    plt.close()


def main():
    feature_extractor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    df = pd.read_csv(DATA_PATH)

    train_df = prepare_fer_data(df[df['Usage'] == 'Training'])
    val_df = prepare_fer_data(df[df['Usage'] == 'PublicTest'])
    test_df = prepare_fer_data(df[df['Usage'] == 'PrivateTest'])

    train_ds = Dataset.from_pandas(train_df).train_test_split(test_size=0.15)['train']
    val_ds = Dataset.from_pandas(val_df)
    test_ds = Dataset.from_pandas(test_df)

    features = Features({
        'label': ClassLabel(names=STRING_LABELS),
        'img': Array3D(dtype="int64", shape=(3, 48, 48)),
        'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
    })

    # Preprocess and save datasets (only needed once)
    for name, ds in [('train', train_ds), ('val', val_ds), ('test', test_ds)]:
        if not os.path.exists(f'preprocessed_{name}_ds.pickle'):
            print(f"Preprocessing {name} dataset...")
            preprocessed_ds = ds.map(lambda x: preprocess_images(x, feature_extractor), batched=True, features=features)
            with open(f'preprocessed_{name}_ds.pickle', 'wb') as handle:
                pickle.dump(preprocessed_ds, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Load preprocessed datasets from pickle
    with open('preprocessed_train_ds.pickle', 'rb') as handle:
        train_ds = pickle.load(handle)
    with open('preprocessed_val_ds.pickle', 'rb') as handle:
        val_ds = pickle.load(handle)
    with open('preprocessed_test_ds.pickle', 'rb') as handle:
        test_ds = pickle.load(handle)

    model = ViTForImageClassificationWithCNN().to(DEVICE)

    training_args = TrainingArguments(
        "vit-fer",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=6,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=METRIC_NAME,
        logging_dir='logs',
        dataloader_pin_memory=False,  # Add this to suppress MPS warning
    )

    os.environ["WANDB_DISABLED"] = "true"
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    outputs = trainer.predict(test_ds)
    y_true = outputs.label_ids
    y_pred = outputs.predictions.argmax(1)
    plot_confusion_matrix(y_true, y_pred)
    trainer.save_model("model")


if __name__ == "__main__":
    start_time = time.perf_counter()

    main()

    end_time = time.perf_counter()
    print(f"Total Time Used: {end_time - start_time:.2f} Seconds")

