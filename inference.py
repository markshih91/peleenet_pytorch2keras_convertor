from keras_peleenet import peleenet_model
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)


model = peleenet_model(input_shape=(224, 224, 3))
model.load_weights('peleenet_keras_weights.h5')

file_name = 'synset_words.txt'
classes = {}
for line in open(file_name):
    line = line.rstrip().split(':')
    classes[int(line[0])] = line[1]

print(classes)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

img = 'images/pig.jpeg'
img = Image.open(img)
np_img = np.asarray(img)
img = transform(img)
img.unsqueeze_(dim=0)
print(img.shape)

img = img.cpu().numpy()
img = img.transpose((0, 2, 3, 1))

output = model.predict(img)[0]

print(output)

output = softmax(output)

print(classes[np.argmax(output)])