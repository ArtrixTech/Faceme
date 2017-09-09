import requests
import numpy
from PIL import Image
from components import functions
import json

API_KEY = 'az6lgs_zGjW1xhWK__ru6CMx_brdNHpK'
API_SECRET = 'PvVHpOw0cCu4NoAJr-fT0kCklu5hprcX'

detect_url = "https://api-cn.faceplusplus.com/facepp/v3/detect"
face_set_url = "https://api-cn.faceplusplus.com/facepp/v3/faceset/create"
compare_url = "https://api-cn.faceplusplus.com/facepp/v3/compare"

model_token="a7edb56f8386d7f826be350a2f55d44e"


def get_face(numpy_img_arr):

    data = {
        "api_key": API_KEY,
        "api_secret": API_SECRET,
        "return_landmark": "0",
        "image_base64": functions.img_to_base64(
            numpy_img_arr,
            is_numpy_img_array=True),
        "return_attributes": "gender,age"}
    response = requests.post(detect_url, data=data).json()["faces"][0]
    token = response["face_token"]
    age = response["attributes"]["age"]["value"]
    return token, age


def create_face(numpy_img_arr):

    token, age = get_face(numpy_img_arr)

    data = {
        "api_key": API_KEY,
        "api_secret": API_SECRET,
        "display_name": "detection",
        "face_tokens": token}
    response = requests.post(face_set_url, data=data).json()
    faceset_token = response["faceset_token"]
    return token, faceset_token


def compare_face(numpy_img_arr,compare_face_token):
    data = {
        "api_key": API_KEY,
        "api_secret": API_SECRET,
        "image_base64_2": functions.img_to_base64(numpy_img_arr,is_numpy_img_array=True),
        "face_token1": compare_face_token}
    response = requests.post(compare_url, data=data).json()
    print(response["confidence"])
