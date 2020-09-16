import torch.nn as nn
import torch.nn.functional as F


class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class ConvBlock(nn.Module):
    leaky = False

    def __init__(
        self, in_channel, out_channel, k=3, stride=1, padding=1,
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channel, out_channel, kernel_size=k, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(out_channel),
        )
        if self.leaky:
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.conv(x))


class CRNN(nn.Module):
    def __init__(self, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        ConvBlock.leaky = leakyRelu
        self.net = nn.Sequential(
            ConvBlock(nc, 64),  # 32
            nn.MaxPool2d(2, 2),  # 16
            ConvBlock(64, 128),
            nn.MaxPool2d(2, 2),  # 8
            ConvBlock(128, 256),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # 4
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # 2
            ConvBlock(512, 512, k=2, padding=0, stride=1),  # 1
        )
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh), BidirectionalLSTM(nh, nh, nclass)
        )
        for ele in self.modules():
            if isinstance(ele, nn.Conv2d):
                ele.weight.data.normal_(0.0, 0.02)
            elif isinstance(ele, nn.BatchNorm2d):
                ele.weight.data.normal_(1.0, 0.02)
                ele.bias.data.fill_(0)

    def forward(self, input):
        # conv features
        conv = self.net(input)
        b, c, h, w = conv.size()
        print(b, c, h, w)
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        # add log_softmax to converge output
        output = F.log_softmax(output, dim=2)

        return output


if __name__ == '__main__':
    crnn = CRNN(nc=1, nclass=26, nh=256)
    import torch
    img = torch.rand((64, 1, 32, 20))
    output = crnn(img)
