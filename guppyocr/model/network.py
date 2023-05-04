import torch
from torch import nn

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)
    def forward(self, input):
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output

class CalamariNetv2(nn.Module):
    def __init__(self, Channel, feature_height, feature_width, embedding_dim, output_classes, hidden_units, layers, keep_prob, seq_len, device, leakyRelu=False):
        super(CalamariNetv2, self).__init__()

        self.channel = Channel
        self.feature_height = feature_height
        self.feature_width = feature_width
        self.output_classes = output_classes
        self.seq_len = seq_len

        #assert opt['imgH'] % 16 == 0, 'imgH has to be a multiple of 16'
        assert self.feature_height % 16 == 0


        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = self.channel if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))
        #convRelu(2, True)
        """convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)

        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16
        """
        self.cnn = cnn

        #self.rnn = nn.Sequential(
        # BidirectionalLSTM(512, 128, 128),
        #BidirectionalLSTM(2048, 128, self.output_classes))
        self.rnn = nn.LSTM(1536,200,proj_size=self.output_classes)
        #print(self.cnn)

    def forward(self, input, _):
        # conv features

        conv = self.cnn(input)
        b, c, h, w = conv.size()
        #print(f"Conv Size: {b} {c} {h} {w}")
        #assert h == 1, "height of conv must be 1"
        conv = conv.reshape((b,c*h,w))
        #conv = conv.squeeze(2)
        #print(conv.size())
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        #print(conv.size())
        # rnn features
        #self.rnn.flatten_parameters()
        output = self.rnn(conv)
        if isinstance(output,tuple):
            output=output[0]
        #print(output.size())
        #print("Pre Transpose", output.size())
        output = output.transpose(1,0) #Tbh to bth
        #print("Output size", output.size())
        return output, None, None, None

class CRNNTiny(nn.Module):

    def __init__(self, Channel, feature_height, feature_width, embedding_dim, output_classes, hidden_units, layers, keep_prob, seq_len, device, leakyRelu=False):
        super(CRNN, self).__init__()

        self.channel = Channel
        self.feature_height = feature_height
        self.feature_width = feature_width
        self.output_classes = output_classes
        self.seq_len = seq_len

        #assert opt['imgH'] % 16 == 0, 'imgH has to be a multiple of 16'
        assert self.feature_height % 16 == 0


        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = self.channel if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        """convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)

        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16
        """
        self.cnn = cnn

        self.rnn = nn.Sequential(
            # BidirectionalLSTM(512, 128, 128),
            BidirectionalLSTM(1024, 200, self.output_classes))
        #self.rnn = nn.LSTM(512,self.output_classes)
        #print(self.cnn)

    def forward(self, input, _):
        # conv features

        conv = self.cnn(input)
        b, c, h, w = conv.size()
        #print(f"Conv Size: {b} {c} {h} {w}")
        #assert h == 1, "the height of conv must be 1"
        conv = conv.reshape((b,c*h,w))
        #conv = conv.squeeze(2)
        #print(conv.size())
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        #print(conv.size())
        # rnn features
        #self.rnn.flatten_parameters()
        output = self.rnn(conv)
        #print(output.size())
        #print("Pre Transpose", output.size())
        output = output.transpose(1,0) #Tbh to bth
        #print("Output size", output.size())
        return output, None, None, None


class CRNN(nn.Module):

    def __init__(self, Channel, feature_height, feature_width, embedding_dim, output_classes, hidden_units, layers, keep_prob, seq_len, device, leakyRelu=False):
        super(CRNN, self).__init__()

        self.channel = Channel
        self.feature_height = feature_height
        self.feature_width = feature_width
        self.output_classes = output_classes
        self.seq_len = seq_len

        #assert opt['imgH'] % 16 == 0, 'imgH has to be a multiple of 16'
        assert self.feature_height % 16 == 0


        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = self.channel if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((4, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16
        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, 128, 128),
            BidirectionalLSTM(128, 128, self.output_classes))
        #print(self.cnn)


    def forward(self, input, _):
        # conv features

        conv = self.cnn(input)
        b, c, h, w = conv.size()
        #print(f"Conv Size: {b} {c} {h} {w}")
        assert h == 1, f"the height of conv must be 1, is: {h}"
        conv = conv.squeeze(2)
        #print(conv.size())
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        #print(conv.size())
        # rnn features
        output = self.rnn(conv)
        #print(output.size())
        output = output.transpose(1,0) #Tbh to bth
        #print(output.size())
        return output, None, None, None