import os
import argparse
from roboflow import Roboflow
import supervision as sv
import cv2

# List of products to check for in the image
products_to_check = ["shoe", "sneaker", "bottle", "cup", "sandal", "perfume", "toy", "sunglasses",
                     "car", "water bottle", "chair", "office chair", "can", "cap", "hat", "couch",
                     "wristwatch", "glass", "bag", "handbag", "baggage", "suitcase", "headphones",
                     "jar", "vase"]

def main(input_dir, product_output_dir, no_product_output_dir):
    # Initialize Roboflow client
    rf = Roboflow(api_key="W9VikDr39oibIVgg5UOS")

    # Access project and model
    project = rf.workspace().project("coco-dataset-vdnr1")
    model = project.version(11).model

    # Create output directories if they don't exist
    os.makedirs(product_output_dir, exist_ok=True)
    os.makedirs(no_product_output_dir, exist_ok=True)

    # List input images
    input_images = os.listdir(input_dir)

    # Process each input image
    for image_name in input_images:
        # Perform prediction on an image
        result = model.predict(os.path.join(input_dir, image_name), confidence=40).json()

        # Extract labels from the predictions
        labels = [item["class"] for item in result["predictions"]]

        # Check if any of the labels match the products_to_check list
        found_product = any(label in products_to_check for label in labels)

        # Convert Roboflow predictions to Supervisely Detections
        detections = sv.Detections.from_roboflow(result)

        # Initialize annotators
        label_annotator = sv.LabelAnnotator()
        mask_annotator = sv.MaskAnnotator()

        # Read the original image
        image = cv2.imread(os.path.join(input_dir, image_name))

        # Annotate the image with masks
        annotated_image = mask_annotator.annotate(scene=image, detections=detections)

        # Annotate the image with labels
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        # Determine the output directory based on whether a product is found or not
        if found_product:
            output_dir = product_output_dir
        else:
            output_dir = no_product_output_dir

        # Save the annotated image to the appropriate output directory
        cv2.imwrite(os.path.join(output_dir, image_name), annotated_image)

        print(f"Processed and saved annotated image: {image_name}")

    print("Processing completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate images based on detected products.")
    parser.add_argument("--input_dir", type=str, help="Path to the directory containing input images.")
    parser.add_argument("--product_output_dir", type=str, default="/content/images_with_products/", 
                        help="Path to the directory for images with detected products.")
    parser.add_argument("--no_product_output_dir", type=str, default="/content/images_without_products/", 
                        help="Path to the directory for images without detected products.")
    args = parser.parse_args()
    main(args.input_dir, args.product_output_dir, args.no_product_output_dir)
