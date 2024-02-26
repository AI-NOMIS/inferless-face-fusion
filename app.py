import shutil
import os
import cv2
from PIL import Image
from cog import Path
from modelscope.pipelines import pipeline


class Predictor:
    def setup(self):
        self.image_face_fusion = pipeline(
            "image-face-fusion",
            "/Users/intizar/MyWorld/ai-nomis/gcloud/k8s-services/cog-face-fusion/weights/cv_unet-image-face-fusion_damo",
        )

    def predict(self, user_img: str, template_img: str) -> str:
        usr_img = Image.open(str(user_img)).convert("RGB")
        temp_img = Image.open(str(template_img)).convert("RGB")

        output_dir = "./results"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        Path(output_dir).mkdir(exist_ok=True)

        input_data = {"template": usr_img, "user": temp_img}

        result = self.image_face_fusion(input_data)
        # result = {'outputs': output}

        output_path = os.path.join(output_dir, "output.png")
        cv2.imwrite(output_path, result["output_img"])

        return output_path


if __name__ == "__main__":
    predictor = Predictor()
    predictor.setup()
