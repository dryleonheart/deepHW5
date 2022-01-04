import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import time

########################################
# You can define whatever classes if needed
########################################

class IdentityResNet(nn.Module):
    
    # __init__ takes 4 parameters
    # nblk_stage1: number of blocks in stage 1, nblk_stage2.. similar
    def __init__(self, nblk_stage1, nblk_stage2, nblk_stage3, nblk_stage4):
        super(IdentityResNet, self).__init__()
        hw = 64
        self.nblk_stage1 =nblk_stage1
        self.nblk_stage2 =nblk_stage2
        self.nblk_stage3 =nblk_stage3
        self.nblk_stage4 =nblk_stage4
        self.conv1 = nn.Conv2d(3,64,kernel_size = 3, stride = 1, padding = 1)
        self.bn = nn.ModuleList()
        self.convdir = nn.ModuleList()
        self.stage_block = nn.ModuleList()
        self.stage_double = nn.ModuleList()
        
        for i in range(4):
            self.bn.append(nn.BatchNorm2d(hw*(2**i)))
        
        for i in range(4):
            self.convdir.append(nn.Conv2d(hw*(2**i),hw*(2**(i+1)),kernel_size = 1, stride = 2))
        
        for i in range(4):
            self.stage_double.append( nn.Sequential(
                nn.Conv2d(hw*(2**i),hw*(2**(i+1)),kernel_size = 3, stride = 2, padding = 1),
                nn.BatchNorm2d(hw*(2**(i+1))),
                nn.ReLU(),
                nn.Conv2d(hw*(2**(i+1)),hw*(2**(i+1)),kernel_size = 3, stride = 1, padding = 1),
            ))
        

        for i in range(4):
            self.stage_block.append(nn.Sequential(
                nn.BatchNorm2d(hw*(2**i)),
                nn.ReLU(),
                nn.Conv2d(hw*(2**i),hw*(2**i),kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm2d(hw*(2**i)),
                nn.ReLU(),
                nn.Conv2d(hw*(2**i),hw*(2**i),kernel_size = 3, stride = 1, padding = 1),
            ))
        self.fc = nn.Linear(512,10)
    ########################################
    # Implement the network
    # You can declare whatever variables
    ########################################

    ########################################
    # You can define whatever methods
    ########################################
    
    def forward(self, x):
        #conv
        out = self.conv1(x)
        #stage1
        for i in range(self.nblk_stage1):
            out = F.relu(self.stage_block[0](out)+out)
        
        for stage in range(1,4):
            #stage2 block1
            out = self.bn[stage-1](out)
            out = F.relu(out)
            out = F.relu(self.stage_double[stage-1](out) + self.convdir[stage-1](out))
            #stage2 after block1
            for i in range(self.nblk_stage2-1):
                out = F.relu(self.stage_block[stage](out) + out)
        
        out = F.avg_pool2d(out,kernel_size = 4, stride = 4)
        out = out.view(-1,out.shape[1])
        out = self.fc(out)
        ########################################
        # Implement the network
        # You can declare or define whatever variables or methods
        ########################################
        return out

########################################
# Q1. set device
# First, check availability of GPU.
# If available, set dev to "cuda:0";
# otherwise set dev to "cpu"
########################################
if torch.cuda.is_available() :
    dev = torch.device("cuda:0")
else :
    dev = torch.device("cpu")
print('current device: ', dev)


########################################
# data preparation: CIFAR10
########################################

########################################
# Q2. set batch size
# set batch size for training data
########################################
batch_size = 40

# preprocessing
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

# load test data
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# define network
net = IdentityResNet(nblk_stage1=2, nblk_stage2=2,
                     nblk_stage3=2, nblk_stage4=2)

########################################
# Q3. load model to GPU
# Complete below to load model to GPU
########################################
#net.load_state_dict(torch.load('./', map_location=dev))
net.to(dev)


# set loss function
criterion = nn.CrossEntropyLoss()

########################################
# Q4. optimizer
# Complete below to use SGD with momentum (alpha= 0.9)
# set proper learning rate
########################################
optimizer = torch.optim.SGD(net.parameters(), lr = 0.01, momentum = 0.9)

# start training
t_start = time.time()

for epoch in range(5):  # loop over the dataset multiple times
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(dev), data[1].to(dev)
        
        ########################################
        # Q5. make sure gradients are zero!
        # zero the parameter gradients
        ########################################
        optimizer.zero_grad()
        
        ########################################
        # Q6. perform forward pass
        ########################################
        outputs = net(inputs)
        
        # set loss
        loss = criterion(outputs, labels)
        
        ########################################
        # Q7. perform backprop
        ########################################
        loss.backward()
        
        ########################################
        # Q8. take a SGD step
        ########################################
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i % 1250 == 1249:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            t_end = time.time()
            print('elapsed:', t_end-t_start, ' sec')
            t_start = t_end

print('Finished Training')


# now testing
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

########################################
# Q9. complete below
# when testing, computation is done without building graphs
########################################
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(dev), data[1].to(dev)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# per-class accuracy
for i in range(10):
    print('Accuracy of %5s' %(classes[i]), ': ',
          100 * class_correct[i] / class_total[i],'%')

# overall accuracy
print('Overall Accurracy: ', (sum(class_correct)/sum(class_total))*100, '%')


with torch.no_grad():
      num_plot = 3
      images, labels = data[0].to(dev), data[1].to(dev)
      sample_index = np.random.randint(0, images.shape[0], (num_plot,))
      outputs = net(images)
      _, predicted = torch.max(outputs, 1)
      c = (predicted == labels).squeeze()
      plt.figure(figsize=(12, 4))

      for i in range(num_plot):
        idx = sample_index[i]
        img = np.squeeze(images[idx]).cpu()
        ax = plt.subplot(1, num_plot, i + 1)
        plt.imshow(np.transpose(img, (1,2,0)), interpolation='nearest')

        # if our prediction is correct, the title will be in black color
        # otherwise, for incorrect predictions, the title will be in red
        if labels[idx] == predicted[idx]:
            title_color = 'k'
        else:
            title_color = 'r'

        ax.set_title('GT:' + classes[labels[idx]] + '\n Pred:' + classes[predicted[idx]], color=title_color)


