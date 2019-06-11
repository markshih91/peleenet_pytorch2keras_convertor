import numpy as np
import torch
import argparse
from keras_peleenet import peleenet_model

parser = argparse.ArgumentParser(description='Pytorch model to Keras.')
parser.add_argument('--pytorch-file', type=str, default='peleenet.pth')
args = parser.parse_args()

model = torch.load(args.pytorch_file)
pytorch_model_dict = {}

for k, v in model.state_dict().items():
    if len(v.shape) > 0:
        pytorch_model_dict[k] = v
        print(k, v.shape)

model = peleenet_model(input_shape=(224, 224, 3))

for layer in model.layers:
    lname = layer.name
    weights = np.asarray(model.get_layer(lname).get_weights())
    if len(weights) > 0:
        print(lname, weights.shape)

        pytorch_lname = lname.replace('_', '.')
        pytorch_lname = pytorch_lname.replace('bbn', 'module')

        if pytorch_lname.endswith('conv'):
            weight = pytorch_model_dict[pytorch_lname + '.weight'].cpu().numpy()
            weight = weight.transpose((2, 3, 1, 0))
            model.get_layer(lname).set_weights([weight])
            print('get weights of: ', lname)
        elif pytorch_lname.endswith('norm'):
            weight = pytorch_model_dict[pytorch_lname + '.weight'].cpu().numpy()
            bias = pytorch_model_dict[pytorch_lname + '.bias'].cpu().numpy()
            running_mean = pytorch_model_dict[pytorch_lname + '.running_mean'].cpu().numpy()
            running_var = pytorch_model_dict[pytorch_lname + '.running_var'].cpu().numpy()
            model.get_layer(lname).set_weights([weight, bias, running_mean, running_var])
            print('get weights of: ', lname)
        elif pytorch_lname.endswith('classifier'):
            weight = pytorch_model_dict[pytorch_lname + '.weight'].cpu().numpy()
            bias = pytorch_model_dict[pytorch_lname + '.bias'].cpu().numpy()
            weight = weight.transpose((1, 0))
            model.get_layer(lname).set_weights([weight, bias])
            print('get weights of: ', lname)

model.save_weights('peleenet_keras_weights.h5')