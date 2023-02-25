import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import onnx

# Set the device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the data transformations
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load the reference image and convert it to black and white
reference_image = Image.open("./apple/apple.jpg").convert("L")

# Define the dataset and data loaders
train_dataset = datasets.ImageFolder("./apple/apple_train/", transform=transform)
val_dataset = datasets.ImageFolder("./apple/apple_validate/", transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

# Define the ResNet-18 architecture for a grayscale input image
model = models.resnet18(pretrained=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)

# Move the model to the device
model.to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Define the training loop
def train(model, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        # Train the model for one epoch
        running_loss = 0.0
        for i, (inputs, _) in enumerate(train_loader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, torch.ones(inputs.shape[0], 1).to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)

        # Evaluate the model on the validation set
        val_loss = 0.0
        with torch.no_grad():
            for i, (inputs, _) in enumerate(val_loader):
                inputs = inputs.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, torch.ones(inputs.shape[0], 1).to(device))
                val_loss += loss.item()
            val_loss = val_loss / len(val_loader)

        print("Epoch [{}/{}], train loss: {:.4f}, val loss: {:.4f}".format(
            epoch+1, num_epochs, epoch_loss, val_loss))

# Train the model
train(model, criterion, optimizer, num_epochs=10)

# Save the model as ONNX
dummy_input = torch.randn(1, 1, 300, 300).to(device)
onnx_model_path = "version4_model.onnx"
torch.onnx.export(model, dummy_input, onnx_model_path)

# Load the ONNX model and update its IR version to 3
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)
onnx_model.ir_version = 3

# Save the updated model
onnx.save(onnx_model, onnx_model_path)
