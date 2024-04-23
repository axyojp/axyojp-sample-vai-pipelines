from flask import Flask, request
from google.cloud import storage
import re
import lightgbm as lgb
import pandas as pd
import os

SRC_MODEL_URI = os.environ.get("SRC_MODEL_URI","")
MODEL_NAME = "model.lgb"
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


# モデルファイルをロード
model = lgb.Booster(model_file=MODEL_NAME)

app = Flask(__name__)


@app.route("/")
def health():
    return "Health is good"


@app.route("/predict", methods=["POST"])
def predict():
    # https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements#image

    request_data = request.get_json()
    print(f"request_data: {request_data}")

    predictions = []
    
    for instance in request_data["instances"]:
        print(f"[instance: {instance}]")
        X = pd.DataFrame(instance).T

        # 予測
        pred = model.predict(X)
        predictions.append(pred[0])
        print(f"predictions: {predictions}")

    response = {"predictions": predictions}
    print(f"response{response}")

    return response


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
