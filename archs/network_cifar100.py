"""dense net in pytorch
[1] Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger.
    Densely Connected Convolutional Networks
    https://arxiv.org/abs/1608.06993v5
"""

import torch
import torch.nn as nn


#"""Bottleneck layers. Although each layer only produces k
#output feature-maps, it typically has many more inputs. It
#has been noted in [37, 11] that a 1×1 convolution can be in-
#troduced as bottleneck layer before each 3×3 convolution
#to reduce the number of input feature-maps, and thus to
#improve computational efficiency."""
class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        #"""In  our experiments, we let each 1×1 convolution
        #produce 4k feature-maps."""
        inner_channel = 4 *growth_rate

        #"""We find this design especially effective for DenseNet and
        #we refer to our network with such a bottleneck layer, i.e.,
        #to the BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3) version of H ` ,
        #as DenseNet-B."""
        self.bottle_neck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return torch.cat([x, self.bottle_neck(x)], 1)

#"""We refer to layers between blocks as transition
#layers, which do convolution and pooling."""
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #"""The transition layers used in our experiments
        #consist of a batch normalization layer and an 1×1
        #convolutional layer followed by a 2×2 average pooling
        #layer""".
        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.down_sample(x)

#DesneNet-BC
#B stands for bottleneck layer(BN-RELU-CONV(1x1)-BN-RELU-CONV(3x3))
#C stands for compression factor(0<=theta<=1)
class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_class=100, eps=None):
        super().__init__()
        self.growth_rate = growth_rate
        self.eps = eps
        self.current_max = 0

        #"""Before entering the first dense block, a convolution
        #with 16 (or twice the growth rate for DenseNet-BC)
        #output channels is performed on the input images."""
        inner_channels = 2 * growth_rate

        #For convolutional layers with kernel size 3×3, each
        #side of the inputs is zero-padded by one pixel to keep
        #the feature-map size fixed.
        self.conv1 = nn.Conv2d(3, inner_channels, kernel_size=3, padding=1, bias=False)

        self.features = nn.Sequential()

        for index in range(len(nblocks) - 1):
            self.features.add_module("dense_block_layer_{}".format(index), self._make_dense_layers(block, inner_channels, nblocks[index]))
            inner_channels += growth_rate * nblocks[index]

            #"""If a dense block contains m feature-maps, we let the
            #following transition layer generate θm output feature-
            #maps, where 0 < θ ≤ 1 is referred to as the compression
            #fac-tor.
            out_channels = int(reduction * inner_channels) # int() will automatic floor the value
            self.features.add_module("transition_layer_{}".format(index), Transition(inner_channels, out_channels))
            inner_channels = out_channels

        self.features.add_module("dense_block{}".format(len(nblocks) - 1), self._make_dense_layers(block, inner_channels, nblocks[len(nblocks)-1]))
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]
        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('relu', nn.ReLU(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(inner_channels, num_class)

        self.MLP = Network(query_size = inner_channels, output_size=num_class)

        self.softmax = nn.Softmax()

    def forward(self, x, mask = None, contextual=False):
        output = self.conv1(x)
        output = self.features(output)
        output = self.avgpool(output)
        
        if contextual == True:
            return output
    
        output = output.view(output.size()[0], -1)
    
        if self.eps == None:
            # output = self.linear(output)
            output = self.MLP(output)
        else:
            query_logits = self.linear(output)
            query_mask = torch.where(mask == 1, -1e9, 0.)
            query_mask = query_mask.type_as(query_logits)

            query_logits = query_logits + query_mask

            query = self.softmax(query_logits / self.eps)

            query = (self.softmax(query_logits / 1e-9) - query).detach() + query
            return query

        return output

    def get_logits(self, x, mask = None):
        output = self.conv1(x)
        output = self.features(output)
        output = self.avgpool(output)
        output = output.view(output.size()[0], -1)

        if self.eps == None:
            output = self.MLP(output)
        else:
            query_logits = self.linear(output)
            query_mask = torch.where(mask == 1, -1e9, 0.)
            query_mask = query_mask.type_as(query_logits)

            query_logits = query_logits + query_mask

            return query_logits

        return output

    def _make_dense_layers(self, block, in_channels, nblocks):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module('bottle_neck_layer_{}'.format(index), block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return dense_block

    def change_eps(self, eps):
        self.eps = eps

def densenet121(num_classes=100, eps=None, share=False, num_class_querier=625):
    if share == False:
        return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32, num_class=num_classes, eps=eps)
    return DenseNet_Shared(Bottleneck, [6,12,24,16], growth_rate=32, num_class=num_classes, eps=eps, num_class_querier=num_class_querier)

def densenet169(num_classes=100, eps=None, share=False, num_class_querier=625):
    if share == False:
        return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32, num_class=num_classes, eps=eps)
    return DenseNet_Shared(Bottleneck, [6, 12, 32, 32], growth_rate=32, num_class=num_classes, eps=eps, num_class_querier=num_class_querier)

def densenet201(num_classes=100, eps=None, share=False, num_class_querier=625):
    if share == False:
        return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32, num_class=num_classes, eps=eps)
    return DenseNet_Shared(Bottleneck, [6, 12, 48, 32], growth_rate=32, num_class=num_classes, eps=eps, num_class_querier=num_class_querier)

def densenet161(num_classes=100, eps=None, share=False, num_class_querier=625):
    if share == False:
        return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48, num_class=num_classes, eps=eps)
    return DenseNet_Shared(Bottleneck, [6, 12, 36, 24], growth_rate=48, num_class=num_classes, eps=eps, num_class_querier=num_class_querier)

class DenseNet_Shared(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_class=100, num_class_querier=625, eps=None):
        super().__init__()
        self.growth_rate = growth_rate
        self.eps = eps
        self.current_max = 0

        #"""Before entering the first dense block, a convolution
        #with 16 (or twice the growth rate for DenseNet-BC)
        #output channels is performed on the input images."""
        inner_channels = 2 * growth_rate

        #For convolutional layers with kernel size 3×3, each
        #side of the inputs is zero-padded by one pixel to keep
        #the feature-map size fixed.
        self.conv1 = nn.Conv2d(3, inner_channels, kernel_size=3, padding=1, bias=False)

        self.features = nn.Sequential()

        for index in range(len(nblocks) - 1):
            self.features.add_module("dense_block_layer_{}".format(index), self._make_dense_layers(block, inner_channels, nblocks[index]))
            inner_channels += growth_rate * nblocks[index]

            #"""If a dense block contains m feature-maps, we let the
            #following transition layer generate θm output feature-
            #maps, where 0 < θ ≤ 1 is referred to as the compression
            #fac-tor.
            out_channels = int(reduction * inner_channels) # int() will automatic floor the value
            self.features.add_module("transition_layer_{}".format(index), Transition(inner_channels, out_channels))
            inner_channels = out_channels

        self.features.add_module("dense_block{}".format(len(nblocks) - 1), self._make_dense_layers(block, inner_channels, nblocks[len(nblocks)-1]))
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]
        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('relu', nn.ReLU(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear_classifier = nn.Linear(inner_channels, num_class)
        self.linear_querier = nn.Linear(inner_channels, num_class_querier)
        self.softmax = nn.Softmax()

    def forward(self, x, mask = None):
        output = self.conv1(x)
        output = self.features(output)
        output = self.avgpool(output)
        output = output.view(output.size()[0], -1)


        cls_logits = self.linear_classifier(output)
        query = None
        if mask is not None:
            query_logits = self.linear_querier(output)
            query_mask = torch.where(mask == 1, -1e9, 0.)
            query_mask = query_mask.type_as(query_logits)

            query_logits = query_logits + query_mask

            query = self.softmax(query_logits / self.eps)

            query = (self.softmax(query_logits / 1e-9) - query).detach() + query

        return query, cls_logits

    def get_logits(self, x, mask = None):
        output = self.conv1(x)
        output = self.features(output)
        output = self.avgpool(output)
        output = output.view(output.size()[0], -1)

        if self.eps == None:
            output = self.linear_classifier(output)
        else:
            query_logits = self.linear_querier(output)
            query_mask = torch.where(mask == 1, -1e9, 0.)
            query_mask = query_mask.type_as(query_logits)

            query_logits = query_logits + query_mask

            return query_logits

        return output

    def _make_dense_layers(self, block, in_channels, nblocks):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module('bottle_neck_layer_{}'.format(index), block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return dense_block

    def change_eps(self, eps):
        self.eps = eps


class Network(nn.Module):
    def __init__(self, query_size = 312, output_size=312, eps=None):
        super().__init__()
        self.query_size = query_size
        self.output_dim = output_size
        self.layer1 = nn.Linear(self.query_size, 2000)
        self.layer2 = nn.Linear(2000, 500)
        self.classifier = nn.Linear(500, self.output_dim)

        self.eps = eps
        self.current_max = 0

        self.norm1 = torch.nn.LayerNorm(2000)
        self.norm2 = torch.nn.LayerNorm(500)
        # activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x, mask=None):
        x = self.relu(self.norm1(self.layer1(x)))
        x = self.relu(self.norm2(self.layer2(x)))

        if self.eps == None:
         return self.classifier(x)

        else:
            query_logits = self.classifier(x)
            query_mask = torch.where(mask == 1, -1e9, 0.)
            query_logits = query_logits + query_mask.cuda()

            query = self.softmax(query_logits / self.eps)

            query = (self.softmax(query_logits / 1e-9) - query).detach() + query
            return query

    def change_eps(self, eps):
        self.eps = eps