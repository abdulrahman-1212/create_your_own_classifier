import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import json
from PIL import Image
from torch.autograd import Variable
import torchvision.models as models
import torch
from torch import nn, optim

def load_data(root = "./flowers"):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    data_dir = root
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    return train_dataloader, valid_dataloader, test_dataloader, train_data

arch = {"vgg16":25088,
        "densenet121":1024}

def setup_network(structure='vgg16',dropout=0.1,hidden_units=4096, lr=0.001, device='gpu'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    
    # Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    for param in model.parameters():
        param.requires_grad = False

    from collections import OrderedDict    

    model.classifier = nn.Sequential(
         nn.Linear(arch['vgg16'] , hidden_units),
         nn.ReLU(),
         nn.Dropout(dropout),
         nn.Linear(hidden_units, 102),
         nn.LogSoftmax(dim=1)
     )
    print(model)
    model = model.to('cuda')
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    if torch.cuda.is_available() and device == 'gpu':
        model.cuda()

    return model, criterion

def save_checkpoint(train_data,model=0,path='checkpoint.pth',structure ='vgg16', hidden_units = 4096,dropout=0.3,lr=0.001,epochs=1):
    model.class_to_idx =  train_data.class_to_idx
    torch.save({'structure' :structure,
                'hidden_units':hidden_units,
                'dropout':dropout,
                'learning_rate':lr,
                'no_of_epochs':epochs,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                path)
    
def load_checkpoint(path='checkpoint.pth'):
    checkpoint = torch.load(path)
    lr=checkpoint['learning_rate']
    hidden_units = checkpoint['hidden_units']
    dropout = checkpoint['dropout']
    epochs = checkpoint['no_of_epochs']
    structure = checkpoint['structure']

    model,_ = setup_network(structure , dropout,hidden_units,lr)
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def predict(image_path, model, topk=5, device='gpu'):   
    model.to('cuda')
    model.eval()
    img = process_image(image_path)
    img = img.numpy()
    img = torch.from_numpy(np.array([img])).float()

    with torch.no_grad():
        output = model.forward(img.cuda())
        
    probability = torch.exp(output).data
    
    return probability.topk(topk)


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img_pil = Image.open(image)
    img_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    
    image = img_transforms(img_pil)
    
    return image

parser = argparse.ArgumentParser(
    description = 'Parser for train.py'
)
parser.add_argument('data_dir', action="store", default="./flowers/")
parser.add_argument('--save_dir', action="store", default="./checkpoint.pth")
parser.add_argument('--arch', action="store", default="vgg16")
parser.add_argument('--learning_rate', action="store", type=float,default=0.01)
parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default=512)
parser.add_argument('--epochs', action="store", default=3, type=int)
parser.add_argument('--dropout', action="store", type=float, default=0.5)
parser.add_argument('--gpu', action="store", default="gpu")

args = parser.parse_args()
where = args.data_dir
path = args.save_dir
lr = args.learning_rate
struct = args.arch
hidden_units = args.hidden_units
power = args.gpu
epochs = args.epochs
dropout = args.dropout

if power == 'gpu':
    device = 'cuda'
else:
    device = 'cpu'

def main():
    train_dataloader, valid_dataloader, test_dataloader, train_data = load_data(where)
    model, criterion = setup_network(struct,dropout,hidden_units,lr,power)
    optimizer = optim.Adam(model.classifier.parameters(), lr= 0.001)
    
    # Train Model
    steps = 0
    running_loss = 0
    print("--Training starting--")
    for epoch in range(epochs):
        for images, labels in train_dataloader:
            steps += 1
            # Move input and label tensors to the default device
            if torch.cuda.is_available() and power =='gpu':
                images, labels = images.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            #Forward pass
            logps = model.forward(images)
            loss = criterion(logps, labels)
            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % 10 == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in valid_dataloader:
                        images, labels = images.to('cuda'), labels.to('cuda')

                        logps = model.forward(images)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Loss: {running_loss/10:.3f}.. "
                      f"Validation Loss: {valid_loss/len(valid_dataloader):.3f}.. "
                      f"Accuracy: {accuracy/len(valid_dataloader):.3f}")
                running_loss = 0
                model.train()
    
    #fmodel.save_checkpoint(traindata,model,path,struct,hidden_units,dropout,lr)
    model.class_to_idx =  train_data.class_to_idx
    torch.save({'structure' :struct,
                'hidden_units':hidden_units,
                'dropout':dropout,
                'learning_rate':lr,
                'no_of_epochs':epochs,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                path)
    print("Saved checkpoint!")
if __name__== "__main__":
    main()