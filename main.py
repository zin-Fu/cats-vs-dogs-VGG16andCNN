print("🚀 Importing required files...")
from model import *
from val import *
from train import *
print("Loading down!\n")

print("Calculating dataset size ...")
calc_PicNum()
print("Calculation completed!\n")

show_Onepic(picName='cat.100.jpg')

print("Moving 2500 images from the training folder into a test set & val folder...")
move_pic()
print("Moving completed!\n")

device = torch.device(DEVICE)
torch.manual_seed(RANDOM_SEED)

print("Building model...")
# model = VGG16(num_classes=NUM_CLASSES)
# model = model.to(device)

model2 = CNN(num_classes=NUM_CLASSES)
model2 = model2.to(device)
print("Successfully build!\n")

# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimizer = torch.optim.Adam(model2.parameters(), lr=LEARNING_RATE)

print("🚀Training on ", device)
# train(model=model, optimizer=optimizer)
train(model=model2, optimizer=optimizer)
print("Training completed!\n")

print("Evaluating...")
# evaluation_and_show(model=model, test_loader=test_loader)
evaluation_and_show(model=model2, test_loader=test_loader)

