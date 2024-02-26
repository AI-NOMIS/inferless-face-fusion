import cv2
from PIL import Image
from modelscope.pipelines import pipeline
import base64
from io import BytesIO
import requests


class InferlessPythonModel:
    def initialize(self):
        self.image_face_fusion = pipeline(
            "image-face-fusion",
            "/var/nfs-mount/weigts-volume/face-fusion/cv_unet-image-face-fusion_damo",
        )

    def infer(self, inputs):
        usr_img = Image.open(BytesIO(requests.get(inputs["user_img"]).content)).convert("RGB")
        temp_img = Image.open(BytesIO(requests.get(inputs["template_img"]).content)).convert("RGB")

        input_data = {"template": usr_img, "user": temp_img}

        result = self.image_face_fusion(input_data)
        output_path = "output.png"
        cv2.imwrite(output_path, result["output_img"])

        with open(output_path, "rb") as img_file:
            img_data = img_file.read()

        return {"output_image": base64.b64encode(img_data).decode("utf-8")}

    def finalize(self):
        pass