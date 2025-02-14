from flask import Flask, request
from google.cloud import storage
import re
import pandas as pd
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


SRC_MODEL_URI = os.environ.get("SRC_MODEL_URI","")
MODEL_NAME = "model.pth"
print(f"SRC_MODEL_URI: {SRC_MODEL_URI}")
print(f"MODEL_NAME: {MODEL_NAME}")



# GCSパスからバケット名とオブジェクト名を抽出
def extract_bucket_and_object(gcs_path):
    match = re.match(
        r"gs://(?P<bucket_name>[^\/]+)/(?P<object_path>.*)", gcs_path)
    if match:
        return match.group("bucket_name"), match.group("object_path")
    else:
        raise ValueError("Invalid GCS path: {}".format(gcs_path))


# バケット名とオブジェクト名を抽出
bucket_name, object_name = extract_bucket_and_object(SRC_MODEL_URI
                                                     + "/"
                                                     + MODEL_NAME)
storage_client = storage.Client()


# バケットとオブジェクトを取得
bucket = storage_client.get_bucket(bucket_name)
blob = bucket.blob(object_name)


# オブジェクトをローカルにダウンロード
blob.download_to_filename(MODEL_NAME)


# CPU または GPU のデバイスの利用を決める
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


# NN を定義する
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)


# モデルファイルをロード
model.load_state_dict(torch.load("model.pth", map_location=torch.device(device), weights_only=True))



app = Flask(__name__)


@app.route("/")
def health():
    return "Health is good"


@app.route("/predict", methods=["POST"])
def predict():
    # https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements#image

    request_data = request.get_json()
    print(f"request_data: {request_data}")

#     predictions = []
    
#     for instance in request_data["instances"]:
#         print(f"[instance: {instance}]")
#         X = pd.DataFrame(instance).T

#         # 予測
#         pred = model.predict(X)
#         predictions.append(pred[0])
#         print(f"predictions: {predictions}")

#     response = {"predictions": predictions}
#     print(f"response{response}")

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
        response = {"predictions": [predicted]}


    return response


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
