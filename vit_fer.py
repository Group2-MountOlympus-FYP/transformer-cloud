import os
import numpy as np
import pandas as pd
import pickle
import torch.nn as nn
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from datasets import Dataset, Features, ClassLabel, Array3D
from transformers import (
    ViTFeatureExtractor, ViTModel, TrainingArguments, Trainer, 
    SequenceClassifierOutput
)
from evaluate import load


# Constants
DATA_PATH = "/fer2013/fer2013.csv"
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
        self.cnn = CNNFeatureExtractorImproved()
        self.vit = ViTModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)

    def forward(self, pixel_values, labels=None):
        cnn_features = self.cnn(pixel_values)
        outputs = self.vit(pixel_values=cnn_features)
        logits = self.classifier(self.dropout(outputs.last_hidden_state[:, 0]))
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, self.classifier.out_features), labels.view(-1))
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return load(METRIC_NAME).compute(predictions=predictions, references=labels)


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, xticklabels=STRING_LABELS, yticklabels=STRING_LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


def main():
    feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_NAME)
    df = pd.read_csv(DATA_PATH)

    train_df = prepare_fer_data(df[df['Usage'] == 'Training'])
    val_df = prepare_fer_data(df[df['Usage'] == 'PublicTest'])
    test_df = prepare_fer_data(df[df['Usage'] == 'PrivateTest'])

    train_ds = Dataset.from_pandas(train_df).train_test_split(test_size=0.15)['train']
    val_ds = Dataset.from_pandas(val_df)
    test_ds = Dataset.from_pandas(test_df)

    features = Features({
        'label': ClassLabel(names=STRING_LABELS),
        'img': Array3D("int64", (3, 48, 48)),
        'pixel_values': Array3D("float32", (3, 224, 224)),
    })

    for name, ds in [('train', train_ds), ('val', val_ds), ('test', test_ds)]:
        preprocessed_ds = ds.map(lambda x: preprocess_images(x, feature_extractor), batched=True, features=features)
        with open(f'preprocessed_{name}_ds.pickle', 'wb') as handle:
            pickle.dump(preprocessed_ds, handle, protocol=pickle.HIGHEST_PROTOCOL)

    model = ViTForImageClassificationWithCNN()
    training_args = TrainingArguments(
        "vit-fer",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=6,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=METRIC_NAME,
        logging_dir='logs',
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
    plot_confusion_matrix(outputs.label_ids, outputs.predictions.argmax(1))
    trainer.save_model("model")


if __name__ == "__main__":
    main()
