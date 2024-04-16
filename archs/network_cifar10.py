'''DLA in PyTorch.

Reference:
    Deep Layer Aggregation. https://arxiv.org/abs/1707.06484
'''
import torch, pdb
import torch.nn as nn
import torch.nn.functional as F
import sys


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=1, padding=(kernel_size - 1) // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, xs):
        x = torch.cat(xs, 1)
        out = F.relu(self.bn(self.conv(x)))
        return out


class Tree(nn.Module):
    def __init__(self, block, in_channels, out_channels, level=1, stride=1):
        super(Tree, self).__init__()
        self.level = level
        if level == 1:
            self.root = Root(2*out_channels, out_channels)
            self.left_node = block(in_channels, out_channels, stride=stride)
            self.right_node = block(out_channels, out_channels, stride=1)
        else:
            self.root = Root((level+2)*out_channels, out_channels)
            for i in reversed(range(1, level)):
                subtree = Tree(block, in_channels, out_channels,
                               level=i, stride=stride)
                self.__setattr__('level_%d' % i, subtree)
            self.prev_root = block(in_channels, out_channels, stride=stride)
            self.left_node = block(out_channels, out_channels, stride=1)
            self.right_node = block(out_channels, out_channels, stride=1)

    def forward(self, x):
        xs = [self.prev_root(x)] if self.level > 1 else []
        for i in reversed(range(1, self.level)):
            level_i = self.__getattr__('level_%d' % i)
            x = level_i(x)
            xs.append(x)
        x = self.left_node(x)
        xs.append(x)
        x = self.right_node(x)
        xs.append(x)
        out = self.root(xs)
        return out

class DLA(nn.Module):
    def __init__(self, block=BasicBlock, num_classes=10, eps=None, resize_conv=False):
        super(DLA, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        self.layer3 = Tree(block,  32,  64, level=1, stride=1)
        self.layer4 = Tree(block,  64, 128, level=2, stride=2)
        self.layer5 = Tree(block, 128, 256, level=2, stride=2)
        self.layer6 = Tree(block, 256, 512, level=1, stride=2)

        if resize_conv:
            self.deconv1 = nn.Sequential(*[
                nn.Upsample(size=(5, 5), mode='nearest'),
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
            ])
            self.dnorm1 = nn.BatchNorm2d(256)
            self.deconv2 = nn.Sequential(*[
                nn.Upsample(size=(6, 6), mode='nearest'),
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
            ])
            self.dnorm2 = nn.BatchNorm2d(128)
            self.deconv3 = nn.Sequential(*[
                nn.Upsample(size=(7, 7), mode='nearest'),
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
            ])
            self.dnorm3 = nn.BatchNorm2d(64)

            self.deconv4 = nn.Sequential(*[
                nn.Conv2d(64, 1, 1)
            ])

        else:
            self.linear = nn.Linear(512, num_classes)
            self.MLP = Network(query_size=512, output_size=num_classes)
        self.eps = eps
        self.current_max = 0
        self.resize_conv = resize_conv
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()

    def forward(self, x, mask=None):
        out = self.base(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)

        if self.resize_conv:
            out = self.relu(self.dnorm1(self.deconv1(out)))
            out = self.relu(self.dnorm2(self.deconv2(out)))
            out = self.relu(self.dnorm3(self.deconv3(out)))
            out = self.deconv4(out)
            if self.eps == None:
                sys.exit("ERROR, invalid input")
            else:
                query_logits = out.flatten(1)
                query_mask = torch.where(mask == 1, -1e9, 0.)

                query_logits = query_logits + query_mask.cuda()

                query = self.softmax(query_logits / self.eps)

                query = (self.softmax(query_logits / 1e-9) - query).detach() + query
                return query
        else:
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)

            if self.eps == None:
                out = self.MLP(out)
            else:
                query_logits = self.linear(out)
                query_mask = torch.where(mask == 1, -1e9, 0.)

                query_logits = query_logits + query_mask.cuda()

                query = self.softmax(query_logits / self.eps)

                query = (self.softmax(query_logits / 1e-9) - query).detach() + query
                return query
            return out

    def change_eps(self, eps):
        self.eps = eps

def test():
    net = DLA()
    print(net)
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    print(y.size())


if __name__ == '__main__':
    test()

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