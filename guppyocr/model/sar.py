'''
This code is to construct encoder for SAR - two layer LSTMs
'''
import torch
import torch.nn as nn



'''
This code is to construct backbone network for SAR - with 13 layers of customized ResNet.
'''
import torch
import torch.nn as nn

__all__ = ['basicblock','backbone']
    
class basicblock(nn.Module):
    def __init__(self, depth_in, output_dim, kernel_size, stride):
        super(basicblock, self).__init__()
        self.identity = nn.Identity()
        self.conv_res = nn.Conv2d(depth_in, output_dim, kernel_size=1, stride=1)
        self.batchnorm_res = nn.BatchNorm2d(output_dim)
        self.conv1 = nn.Conv2d(depth_in, output_dim, kernel_size=kernel_size, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(output_dim)
        self.batchnorm2 = nn.BatchNorm2d(output_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.depth_in = depth_in
        self.output_dim = output_dim

    def forward(self, x):
        # create shortcut path
        if self.depth_in == self.output_dim:
            residual = self.identity(x)
        else:
            residual = self.conv_res(x)
            residual = self.batchnorm_res(residual)
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.batchnorm2(out)

        out += residual
        out = self.relu2(out)

        return out

class backbone(nn.Module):

    def __init__(self, input_dim):
        super(backbone, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Block 1 starts
        self.basicblock1 = basicblock(128, 256, kernel_size=3, stride=1)
        # Block 1 ends
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Block 2 starts
        self.basicblock2 = basicblock(256, 256, kernel_size=3, stride=1)
        self.basicblock3 = basicblock(256, 256, kernel_size=3, stride=1)
        # Block 2 ends
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        # Block 5 starts
        self.basicblock4 = basicblock(256, 512, kernel_size=3, stride=1)
        self.basicblock5 = basicblock(512, 512, kernel_size=3, stride=1)
        self.basicblock6 = basicblock(512, 512, kernel_size=3, stride=1)
        self.basicblock7 = basicblock(512, 512, kernel_size=3, stride=1)
        self.basicblock8 = basicblock(512, 512, kernel_size=3, stride=1)
        # Block 5 ends
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.batchnorm5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU(inplace=True)
        # Block 3 starts
        self.basicblock9 = basicblock(512, 512, kernel_size=3, stride=1)
        self.basicblock10 = basicblock(512, 512, kernel_size=3, stride=1)
        self.basicblock11 = basicblock(512, 512, kernel_size=3, stride=1)
        # Block 3 ends
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.batchnorm6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.maxpool1(x)
        x = self.basicblock1(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.relu3(x)
        x = self.maxpool2(x)
        x = self.basicblock2(x)
        x = self.basicblock3(x)
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.relu4(x)
        x = self.maxpool3(x)
        x = self.basicblock4(x)
        x = self.basicblock5(x)
        x = self.basicblock6(x)
        x = self.basicblock7(x)
        x = self.basicblock8(x)
        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = self.relu5(x)
        x = self.basicblock9(x)
        x = self.basicblock10(x)
        x = self.basicblock11(x)
        x = self.conv6(x)
        x = self.batchnorm6(x)
        x = self.relu6(x)

        return x

# unit test
if __name__ == '__main__':

    batch_size = 32
    Height = 48
    Width = 160
    Channel = 3

    input_images = torch.randn(batch_size,Channel,Height,Width)
    model = backbone(Channel)
    output_features = model(input_images)

    print("Input size is:",input_images.shape)
    print("Output feature map size is:",output_features.shape)


class encoder(nn.Module):
    def __init__(self, H, C, hidden_units=512, layers=2, keep_prob=1.0, device='cpu'):
        super(encoder, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=(H,1), stride=1)
        self.lstm = nn.LSTM(input_size=C, hidden_size=hidden_units, num_layers=layers, batch_first=True, dropout=keep_prob)
        self.layers = layers
        self.hidden_units = hidden_units
        self.device = device

    def forward(self, x):
        self.lstm.flatten_parameters()
        # x is feature map in [batch, C, H, W]
        # Initialize hidden state with zeros
        h_0 = torch.zeros(self.layers*1, x.size(0), self.hidden_units).to(self.device)
        # Initialize cell state
        c_0 = torch.zeros(self.layers*1, x.size(0), self.hidden_units).to(self.device)
        x = self.maxpool(x) # [batch, C, 1, W]
        x = torch.squeeze(x) # [batch, C, W]
        if len(x.size()) == 2: # [C, W]
            x = x.unsqueeze(0) # [batch, C, W]
        x = x.permute(0,2,1) # [batch, W, C]
        _, (h, _) = self.lstm(x, (h_0, c_0)) # h with shape [layers*1, batch, hidden_uints]

        return h[-1] # shape [batch, hidden_units]

'''
This code is to construct decoder for SAR - two layer LSTMs combined with feature map with attention mechanism 
'''
import torch
import torch.nn as nn

class word_embedding(nn.Module):
    def __init__(self, output_classes, embedding_dim):
        super(word_embedding, self).__init__()
        '''
        output_classes: number of output classes for the one hot encoding of a word
        embedding_dim: embedding dimension for a word
        '''
        self.linear = nn.Linear(output_classes, embedding_dim) # linear transformation

    def forward(self,x):
        x = self.linear(x)

        return x

class attention(nn.Module):
    def __init__(self, hidden_units, H, W, D):
        super(attention, self).__init__()
        '''
        hidden_units: hidden units of decoder
        H: height of feature map
        W: width of feature map
        D: depth of feature map
        '''
        self.conv1 = nn.Conv2d(hidden_units, D, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(D, D, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(D, 1, kernel_size=1, stride=1)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=-1)
        self.H = H
        self.W = W
        self.D = D

    def forward(self, h, feature_map):
        '''
        h: hidden state from decoder output, with size [batch, hidden_units]
        feature_map: feature map from backbone network, with size [batch, channel, H, W]
        '''
        # reshape hidden state [batch, hidden_units] to [batch, hidden_units, 1, 1]
        h = h.unsqueeze(2)
        h = h.unsqueeze(3)
        h = self.conv1(h) # [batch, D, 1, 1]
        h = h.repeat(1, 1, self.H, self.W) # tiling to [batch, D, H, W]
        feature_map_origin = feature_map
        feature_map = self.conv2(feature_map) # [batch, D, H, W]
        combine = self.conv3(self.dropout(torch.tanh(feature_map + h))) # [batch, 1, H, W]
        combine_flat = combine.view(combine.size(0), -1) # resize to [batch, H*W]
        attention_weights = self.softmax(combine_flat) # [batch, H*W]
        attention_weights = attention_weights.view(combine.size()) # [batch, 1, H, W]
        glimpse = feature_map_origin * attention_weights.repeat(1, self.D, 1, 1) # [batch, D, H, W]
        glimpse = torch.sum(glimpse, dim=(2,3)) # [batch, D]

        return glimpse, attention_weights

class decoder(nn.Module):
    def __init__(self, output_classes, H, W, D=512, hidden_units=512, seq_len=40, device='cpu'):
        super(decoder, self).__init__()
        '''
        output_classes: number of output classes for the one hot encoding of a word
        H: feature map height
        W: feature map width
        D: glimpse depth
        hidden_units: hidden units of encoder/decoder for LSTM
        seq_len: output sequence length T
        '''
        self.linear1 = nn.Linear(output_classes, hidden_units)
        self.lstmcell1 = [nn.LSTMCell(hidden_units, hidden_units) for i in range(seq_len+1)]
        self.lstmcell2 = [nn.LSTMCell(hidden_units, hidden_units) for i in range(seq_len+1)]
        self.attention = attention(hidden_units, H, W, D)
        self.linear2 = nn.Linear(hidden_units+D, output_classes)
        self.softmax = nn.LogSoftmax(dim=1)
        self.seq_len = seq_len
        self.START_TOKEN = output_classes - 3 # Same as END TOKEN
        self.output_classes = output_classes
        self.hidden_units = hidden_units
        self.device = device

        self.lstmcell1 = torch.nn.ModuleList(self.lstmcell1)
        self.lstmcell2 = torch.nn.ModuleList(self.lstmcell2)

    def forward(self,hw,y,V):
        '''
        hw: embedded feature from encoder [batch, hidden_units]
        y: ground truth label one hot encoder [batch, seq, output_classes]
        V: feature map for backbone network [batch, D, H, W]
        '''
        outputs = []
        attention_weights = []
        batch_size = hw.shape[0]
        y_onehot = torch.zeros(batch_size, self.output_classes).to(self.device)
        for t in range(self.seq_len + 1):
            if t == 0:
                inputs_y = hw # size [batch, hidden_units]
                # LSTM layer 1 initialization:
                hx_1 = torch.zeros(batch_size, self.hidden_units).to(self.device) # initial h0_1
                cx_1 = torch.zeros(batch_size, self.hidden_units).to(self.device) # initial c0_1
                # LSTM layer 2 initialization:
                hx_2 = torch.zeros(batch_size, self.hidden_units).to(self.device) # initial h0_2
                cx_2 = torch.zeros(batch_size, self.hidden_units).to(self.device) # initial c0_2
            elif t == 1:
                y_onehot.zero_()
                y_onehot[:,self.START_TOKEN] = 1.0
                inputs_y = y_onehot
                inputs_y = self.linear1(inputs_y) # [batch, hidden_units]
            else:
                if self.training:
                    inputs_y = y[:,t-2,:] # [batch, output_classes]
                else:
                    # greedy search for now - beam search to be implemented!
                    index = torch.argmax(outputs[t-1], dim=-1) # [batch]
                    index = index.unsqueeze(1) # [batch, 1]
                    y_onehot.zero_()
                    inputs_y = y_onehot.scatter_(1, index, 1) # [batch, output_classes]

                inputs_y = self.linear1(inputs_y) # [batch, hidden_units_encoder]

            # LSTM cells combined with attention and fusion layer
            hx_1, cx_1 = self.lstmcell1[t](inputs_y, (hx_1,cx_1))
            hx_2, cx_2 = self.lstmcell2[t](hx_1, (hx_2,cx_2))
            glimpse, att_weights = self.attention(hx_2, V) # [batch, D], [batch, 1, H, W]
            combine = torch.cat((hx_2,glimpse), dim=1) # [batch, hidden_units_decoder+D]
            out = self.linear2(combine) # [batch, output_classes]
            out = self.softmax(out) # [batch, output_classes]
            outputs.append(out)
            attention_weights.append(att_weights)

        outputs = outputs[1:] # [seq_len, batch, output_classes]
        attention_weights = attention_weights[1:] # [seq_len, batch, 1, H, W]
        outputs = torch.stack(outputs) # [seq_len, batch, output_classes]
        outputs = outputs.permute(1,0,2) # [batch, seq_len, output_classes]
        attention_weights = torch.stack(attention_weights) # [seq_len, batch, 1, H, W]
        attention_weights = attention_weights.permute(1,0,2,3,4) # [batch, seq_len, 1, H, W]

        return outputs, attention_weights



'''
This code is to construct the complete SAR model by combining backbone+encoder+decoder as one integrated model.
'''
import torch
import torch.nn as nn

class sar(nn.Module):
    def __init__(self, channel, feature_height, feature_width, embedding_dim, output_classes, hidden_units=512, layers=2, keep_prob=1.0, seq_len=40, device='cpu'):
        super(sar, self).__init__()
        '''
        channel: channel of input image
        feature_height: feature height of backbone feature map
        embedding_dim: embedding dimension for a word
        output_classes: number of output classes for the one hot encoding of a word
        hidden_units: hidden units for both LSTM encoder and decoder
        layers: layers for both LSTM encoder and decoder, should be set to 2
        keep_prob: keep_prob probability dropout for LSTM encoder
        seq_len: decoding sequence length
        '''
        self.backbone = backbone(channel)
        self.encoder_model = encoder(feature_height, 512, hidden_units, layers, keep_prob, device)
        self.decoder_model = decoder(output_classes, feature_height, feature_width, 512, hidden_units, seq_len, device)
        self.embedding_dim = embedding_dim
        self.output_classes = output_classes
        self.hidden_units = hidden_units
        self.layers = layers
        self.keep_prob = keep_prob
        self.seq_len = seq_len
        self.device = device

    def forward(self,x,y):
        '''
        x: input images [batch, channel, height, width]
        y: output labels [batch, seq_len, output_classes]
        '''
        V = self.backbone(x) # (batch, feature_depth, feature_height, feature_width)
        hw = self.encoder_model(V) # (batch, hidden_units)
        outputs, attention_weights = self.decoder_model(hw, y, V) # [batch, seq_len, output_classes], [batch, seq_len, 1, feature_height, feature_width]

        return outputs, attention_weights, V, hw

# unit test
if __name__ == '__main__':
    '''
    Need to change the import to do unit test:
    from backbone import backbone
    from encoder import encoder
    from decoder import decoder
    '''
    torch.manual_seed(0)

    batch_size = 2
    Height = 12
    Width = 24
    Channel = 3
    output_classes = 94
    embedding_dim = 512
    hidden_units = 512
    layers = 2
    keep_prob = 1.0
    seq_len = 40

    feature_height = Height // 4
    feature_width = Width // 8

    y = torch.randn(batch_size, seq_len, output_classes)
    x = torch.randn(batch_size, Channel, Height, Width)
    print("Input image size is:", x.shape)
    print("Input label size is:", y.shape)

    model = sar(Channel, feature_height, feature_width, embedding_dim, output_classes, hidden_units, layers, keep_prob, seq_len)

    predict1, att_weights1, V1, hw1 = model.train()(x,y)
    print("Prediction size is:", predict1.shape)
    print("Attention weight size is:", att_weights1.shape)

    predict2, att_weights2, V2, hw2 = model.train()(x,y)
    print("Prediction size is:", predict2.shape)
    print("Attention weight size is:", att_weights2.shape)

    print("Difference:", torch.sum(predict1-predict2))