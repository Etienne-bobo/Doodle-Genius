import onnx
import onnxruntime as ort
import torch
import torchvision.transforms as transforms
from PIL import Image

# Load the saved model from the ONNX file
onnx_model = onnx.load("version4_model.onnx")
input_name = onnx_model.graph.input[0].name

sess = ort.InferenceSession("version4_model.onnx")

# Define the data transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load the new hand-drawing
hand_drawing = Image.open("./test_score.jpg").convert("L")

# Apply the data transformation to the hand-drawing
input_tensor = transform(hand_drawing).unsqueeze(0)

# Convert the PyTorch tensor to a NumPy array
input_array = input_tensor.numpy()

# Run the inference using the ONNX model
output = sess.run(None, {input_name: input_array})[0]

# Print the similarity score
print("The MSE value between the hand-drawing and the reference image is:", output[0])
