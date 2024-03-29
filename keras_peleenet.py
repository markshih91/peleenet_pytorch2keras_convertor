from keras.models import Model
from keras import backend as K
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, \
    MaxPooling2D, Concatenate, AveragePooling2D, Flatten, Dropout, Dense


def dense_graph(x, growth_rate, bottleneck_width, name=''):
    growth_rate = int(growth_rate / 2)
    inter_channel = int(growth_rate * bottleneck_width / 4) * 4

    num_input_features = K.int_shape(x)[-1]

    if inter_channel > num_input_features / 2:
        inter_channel = int(num_input_features / 8) * 4
        print('adjust inter_channel to ', inter_channel)

    branch1 = basic_conv2d_graph(
        x, inter_channel, kernel_size=1, strides=1, padding='valid', name=name + '_branch1a')
    branch1 = basic_conv2d_graph(
        branch1, growth_rate, kernel_size=3, strides=1, padding='same', name=name + '_branch1b')

    branch2 = basic_conv2d_graph(
        x, inter_channel, kernel_size=1, strides=1, padding='valid', name=name + '_branch2a')
    branch2 = basic_conv2d_graph(
        branch2, growth_rate, kernel_size=3, strides=1, padding='same', name=name + '_branch2b')
    branch2 = basic_conv2d_graph(
        branch2, growth_rate, kernel_size=3, strides=1, padding='same', name=name + '_branch2c')

    out = Concatenate(axis=-1)([x, branch1, branch2])

    return out


def dense_block_graph(x, num_layers, bn_size, growth_rate, name=''):
    for i in range(num_layers):
        x = dense_graph(x, growth_rate, bn_size, name=name + '_denselayer{}'.format(i + 1))

    return x


def stem_block_graph(x, num_init_features, name=''):
    num_stem_features = int(num_init_features / 2)

    out = basic_conv2d_graph(x, num_init_features, kernel_size=3, strides=2, padding='same', name=name + '_stem1')

    branch2 = basic_conv2d_graph(
        out, num_stem_features, kernel_size=1, strides=1, padding='valid', name=name + '_stem2a')
    branch2 = basic_conv2d_graph(
        branch2, num_init_features, kernel_size=3, strides=2, padding='same', name=name + '_stem2b')

    branch1 = MaxPooling2D(pool_size=2, strides=2)(out)

    out = Concatenate(axis=-1)([branch1, branch2])

    out = basic_conv2d_graph(out, num_init_features, kernel_size=1, strides=1, padding='valid', name=name + '_stem3')

    return out


def basic_conv2d_graph(x, out_channels, kernel_size, strides, padding, activation=True, name=''):
    x = Conv2D(
        out_channels, kernel_size=kernel_size, strides=strides,
        padding=padding, use_bias=False, name=name + '_conv')(x)
    x = BatchNormalization(name=name + '_norm')(x)
    if activation:
        x = ReLU()(x)

    return x


def peleenet_model(input_shape, growth_rate=32, block_config=[3, 4, 8, 6],
                 num_init_features=32, bottleneck_width=[1, 2, 4, 4], drop_rate=0.05, num_classes=1000):

    x = Input(shape=input_shape)

    if type(growth_rate) is list:
        growth_rates = growth_rate
        assert len(growth_rates) == 4, 'The growth rate must be the list and the size must be 4'
    else:
        growth_rates = [growth_rate] * 4

    if type(bottleneck_width) is list:
        bottleneck_widths = bottleneck_width
        assert len(bottleneck_widths) == 4, 'The bottleneck width must be the list and the size must be 4'
    else:
        bottleneck_widths = [bottleneck_width] * 4

    features = stem_block_graph(x, num_init_features, name='bbn_features_stemblock')
    num_features = num_init_features
    for i, num_layers in enumerate(block_config):
        features = dense_block_graph(
            features, num_layers=num_layers, bn_size=bottleneck_widths[i],
            growth_rate=growth_rates[i], name='bbn_features_denseblock{}'.format(i + 1))

        num_features = num_features + num_layers * growth_rates[i]
        features = basic_conv2d_graph(
            features, num_features, kernel_size=1, strides=1,
            padding='valid', name='bbn_features_transition{}'.format(i + 1))

        if i != len(block_config) - 1:
            features = AveragePooling2D(pool_size=2, strides=2)(features)

    features_shape = K.int_shape(features)
    out = AveragePooling2D(pool_size=(features_shape[-3], features_shape[-2]))(features)
    out = Flatten()(out)
    if drop_rate > 0:
        out = Dropout(drop_rate)(out)
    out = Dense(num_classes, name='bbn_classifier')(out)

    model = Model(x, out, name='peleenet')
    return model


if __name__ == '__main__':
    model = peleenet_model(input_shape=(224, 224, 3))
    print(model.summary())
