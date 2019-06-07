import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

import net

root = '/input_dir/datasets/Chinese_calligraphy/train/'

data_transform = transforms.Compose([
        transforms.Resize(80),
        transforms.Pad(padding=100, padding_mode="edge"),
        transforms.CenterCrop(100),
        transforms.Resize(64),
        transforms.Grayscale(),
        transforms.RandomRotation(degrees = 15),
        transforms.RandomPerspective(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
calli_dataset = datasets.ImageFolder(root=root, transform=data_transform)

batch_size = 4
N = len(calli_dataset)
val_rate = 0.1
trainset, valset = torch.utils.data.random_split(calli_dataset,[int(N*(1-val_rate)),int(N*val_rate)])

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=True)
# Cannot set num_workers larger than 0 because of the operating system.

learning_rate = 1e-3
max_iter = 1  # Maximum Iteration

# Import net
# model = net.CalliNet(num_classes=100, batch_size=4)  # Instance
model = net.resnet101()
if torch.cuda.is_available():
    model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005, nesterov=True)

# Training
accuracy = []
loss_trace = []

for epoch in range(max_iter):
    correct = 0
    total = 0

    if epoch > 0 and epoch%3 == 0:
        learning_rate = learning_rate / 2
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005, nesterov=True)

    for i_batch, sample_batch in enumerate(trainloader):
        imgs, labels = sample_batch

        # To GPU
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()

        # optimizer reset before calculation
        optimizer.zero_grad()

        # Forward
        labels_pred = model(imgs)
        
        loss = criterion(labels_pred, labels)
        loss_trace.append(loss)

        # Backward
        loss.backward()
        # Renew params with optimizer
        optimizer.step()

        # The output label is the one corresponding to the largest possibility
        labels_pred_choice = labels_pred.data.max(1)[1]

        # Count correct prediction
        correct += labels_pred_choice.eq(labels.data).cpu().sum()
        total += len(labels)
        accuracy.append(float(correct)/float(total))

        # Print results in specific epochs:
        if i_batch % 100 == 0:
            print("batch_index: [%d/%d]" % (i_batch, len(trainloader)))
            print("Train epoch: [%d]" % epoch)
            print("Correct/Sum: %d/%d, %.4f" % (correct, total, float(correct)/float(total)))

# Print the training process
# Accuracy
plt.plot(accuracy)
plt.title('Accuracy')
plt.show()
# Loss
plt.plot(loss_trace)
plt.title('Loss')
plt.show()

def val():    
    model.eval()
    correct = 0
    total = 0
    for i_batch, sample_batch in enumerate(valloader):
        imgs, labels = sample_batch

        # To GPU
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()
        
        labels_pred = model(imgs)
        labels_pred_choice = labels_pred.data.max(1)[1]
        
        correct += labels_pred_choice.eq(labels.data).cpu().sum()
        total += len(labels)
        accuracy.append(float(correct)/float(total))

        # Print results in specific epochs:
        if i_batch % 100 == 0:
            print("batch_index: [%d/%d]" % (i_batch, len(valloader)))
            print("Train epoch: [%d]" % epoch)
            print("Correct/Sum: %d/%d, %.4f" % (correct, total, float(correct)/float(total)))
            
val()
