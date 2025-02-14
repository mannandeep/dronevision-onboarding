from ultralytics import YOLO
import os

def train_model(model, data_config, epochs,imgsz):
    """
    Trains the model using the specified dataset configuration and epochs.
    """
    print(f"Starting training for {epochs} epoch(s) on data config: {data_config}")
    results = model.train(data=data_config, epochs=epochs, imgsz=imgsz)
    return results

def predict_image(model, image_path):
    """
    Runs inference on the given image and displays the results.
    """
    print(f"Running inference on {image_path}...")
    results = model(image_path)
    print(results)

def main():
    retrain_choice = input("Do you want to retrain the model? (yes/no):").strip().lower()

    if retrain_choice == 'yes':
        # Load a pre-trained YOLO model for fine-tuning
        print("Loading a pre-trained model for training")
        model = YOLO("yolo11n.pt")  # This can also be made user defined
        #making the yaml file configurable too because this makes it a generic script which can be used for various yaml files
        data_config = input("Enter path to your dataset YAML configuration file: ").strip()
        #customizable epochs
        try:
            epochs = int(input("Enter number of training epochs: "))
        except ValueError:
            print("Invalid input for epochs. Using default of 1 epoch.")
            epochs = 1
        # Train the model
        train_model(model, data_config, epochs, imgsz=640)
        # Saving the trained model to reuse later
        trained_model_path = "trained_model.pt"
        model.save(trained_model_path)
        print("Model trained and saved to {trained_model_path}")
    else:
        # Try to load an existing trained model; fall back to pre-trained if not found.
        trained_model_path = "trained_model.pt"
        if os.path.exists(trained_model_path):
            print("Loading the previously trained model")
            model = YOLO(trained_model_path)
        else:
            print("No trained model found. Loading pre-trained model instead")
            model = YOLO("yolo11n.pt")

    #Taking image path from the user
    image_path = input("Enter the path to the image for prediction:").strip()
    if not os.path.exists(image_path):
        print(f"Image file '{image_path}' not found. Exiting.")
        return

    predict_image(model, image_path)

if __name__ == '__main__':
    main()
