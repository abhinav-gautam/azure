import os
import sys
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from azure.ai.vision.face import FaceClient
from azure.ai.vision.face.models import (
    FaceDetectionModel,
    FaceRecognitionModel,
    FaceAttributeTypeDetection01,
)
from azure.core.credentials import AzureKeyCredential


def main():
    load_dotenv()

    ai_key = os.getenv("FACE_KEY")
    ai_endpoint = os.getenv("FACE_ENDPOINT")

    image_file = "./images/face1.jpg"

    if len(sys.argv) > 1:
        image_file = sys.argv[1]

    with open(image_file, "rb") as file:
        image_data = file.read()

    # Create face client
    face_client = FaceClient(ai_endpoint, AzureKeyCredential(ai_key))

    # Specify facial features to be retrieved
    features = [
        FaceAttributeTypeDetection01.HEAD_POSE,
        FaceAttributeTypeDetection01.OCCLUSION,
        FaceAttributeTypeDetection01.ACCESSORIES,
    ]

    # Get faces
    detected_faces = face_client.detect(
        image_content=image_data,
        detection_model=FaceDetectionModel.DETECTION01,
        recognition_model=FaceRecognitionModel.RECOGNITION01,
        return_face_attributes=features,
        return_face_id=False,
    )

    face_count = 0
    if len(detected_faces) > 0:
        print(len(detected_faces), "faces detected.")
        for face in detected_faces:

            # Get face properties
            face_count += 1
            print(f"\nFace number {face_count}")
            print(f" - Head Pose (Yaw): {face.face_attributes.head_pose.yaw}")
            print(f" - Head Pose (Pitch): {face.face_attributes.head_pose.pitch}")
            print(f" - Head Pose (Roll): {face.face_attributes.head_pose.roll}")
            print(
                f" - Forehead occluded?: {face.face_attributes.occlusion['foreheadOccluded']}"
            )
            print(f" - Eye occluded?: {face.face_attributes.occlusion['eyeOccluded']}")
            print(
                f" - Mouth occluded?: {face.face_attributes.occlusion['mouthOccluded']}"
            )
            print(" - Accessories:")
            for accessory in face.face_attributes.accessories:
                print(f"   - {accessory.type}")
            # Annotate faces in the image
            annotate_faces(image_file, detected_faces)


def annotate_faces(image_file, detected_faces):
    print("\nAnnotating faces in image...")

    # Prepare image for drawing
    fig = plt.figure(figsize=(8, 6))
    plt.axis("off")
    image = Image.open(image_file)
    draw = ImageDraw.Draw(image)
    color = "lightgreen"

    # Annotate each face in the image
    face_count = 0
    for face in detected_faces:
        face_count += 1
        r = face.face_rectangle
        bounding_box = ((r.left, r.top), (r.left + r.width, r.top + r.height))
        draw = ImageDraw.Draw(image)
        draw.rectangle(bounding_box, outline=color, width=5)
        annotation = "Face number {}".format(face_count)
        plt.annotate(annotation, (r.left, r.top), backgroundcolor=color)

    # Save annotated image
    plt.imshow(image)
    outputfile = "./output/detected_faces.jpg"
    fig.savefig(outputfile)
    print(f"  Results saved in {outputfile}\n")


if __name__ == "__main__":
    main()
