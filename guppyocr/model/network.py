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

    def __init__(self, Channel, feature_height, feature_width, embedding_dim, output_classes, hidden_units, layers, keep_prob, seq_len, device, leakyRelu=False, pre_lstm_dropout: float = 0.0, inter_lstm_dropout: float = 0.0):
        super(CRNN, self).__init__()

        self.channel = Channel
        self.feature_height = feature_height
        self.feature_width = feature_width
        self.output_classes = output_classes
        self.seq_len = seq_len
        #self.dropout = dropout

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
        if pre_lstm_dropout > 0:
            cnn.add_module('dropout0', nn.Dropout(pre_lstm_dropout))
        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, 128, 128),
            nn.Dropout(inter_lstm_dropout) if inter_lstm_dropout > 0 else nn.Identity(),
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

class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(10, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(1520, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        #print("x", x.shape)
        xs = self.localization(x)
        #print("xs", xs.shape)
        xs = xs.view(-1, 1520)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = nn.functional.affine_grid(theta, x.size())
        x = nn.functional.grid_sample(x, grid)

        return x

    def forward(self, input, _):
        return self.stn(input)



class TrCRNN(nn.Module):

    def __init__(self, Channel, feature_height, feature_width, embedding_dim, output_classes, hidden_units, layers, keep_prob, seq_len, device, leakyRelu=False, dropout: float = 0.0):
        super(TrCRNN, self).__init__()

        self.channel = Channel
        self.feature_height = feature_height
        self.feature_width = feature_width
        self.output_classes = output_classes
        self.seq_len = seq_len
        self.dropout = dropout

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
        cnn.add_module('dropout0', nn.Dropout(0.3))
        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, 256, 256),
            nn.Dropout(0.3),
            BidirectionalLSTM(256, 128, self.output_classes))
        #print(self.cnn)
        self.transformer = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=512, nhead=8), num_layers=4)
        self.compression = nn.Linear(512,self.output_classes)



    def forward(self, input, _):
        # conv features

        conv = self.cnn(input)
        b, c, h, w = conv.size()
        #print(f"Conv Size: {b} {c} {h} {w}")
        assert h == 1, f"the height of conv must be 1, is: {h}"
        conv = conv.squeeze(2)
        #print(conv.size())
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        output = self.transformer(conv)

        #print(conv.size())
        # rnn features
        #output = self.rnn(conv)

        #print(output.size())
        output = output.transpose(1,0) #Tbh to bth
        output = self.compression(output)
        #print(output.size())
        return output, None, None, None

class STNCRNN(nn.Module):

    def __init__(self, Channel, feature_height, feature_width, embedding_dim, output_classes, hidden_units, layers, keep_prob, seq_len, device, leakyRelu=False, dropout: float = 0.0):
        super(STNCRNN, self).__init__()
        self.channel = Channel
        self.feature_height = feature_height
        self.feature_width = feature_width
        self.output_classes = output_classes
        self.seq_len = seq_len
        self.dropout = dropout

        #assert opt['imgH'] % 16 == 0, 'imgH has to be a multiple of 16'
        assert self.feature_height % 16 == 0


        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        self.stn = SpatialTransformer()
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
        cnn.add_module('dropout0', nn.Dropout(0.3))
        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, 256, 256),
            nn.Dropout(0.3),
            BidirectionalLSTM(256, 128, self.output_classes))
        #print(self.cnn)


    def forward(self, input, _):
        # conv features
        input = self.stn(input, None)
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