import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with batch normalization for better gradient flow"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out)
        return out


class SpatialAttention(nn.Module):
    """Spatial attention mechanism to focus on relevant image regions"""
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        
    def forward(self, x):
        attention = torch.sigmoid(self.conv(x))
        return x * attention


class ImprovedCNN(nn.Module):
    """Enhanced CNN backbone with residual connections and attention"""
    def __init__(self, in_channels, leaky_relu=True):
        super(ImprovedCNN, self).__init__()
        
        self.activation = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
        
        # Initial conv layer
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            self.activation
        )
        
        # Residual blocks with pooling
        self.res1 = ResidualBlock(64, 128)
        self.pool1 = nn.MaxPool2d(2, 2)  # H/2, W/2
        
        self.res2 = ResidualBlock(128, 256)
        self.pool2 = nn.MaxPool2d(2, 2)  # H/4, W/4
        
        self.res3 = ResidualBlock(256, 256)
        self.attention1 = SpatialAttention(256)
        self.pool3 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))  # H/8, W/4
        
        self.res4 = ResidualBlock(256, 512)
        self.pool4 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))  # H/16, W/4
        
        self.res5 = ResidualBlock(512, 512)
        self.attention2 = SpatialAttention(512)
        
        # Final 1x1 conv to reduce height to 1
        self.final_pool = nn.AdaptiveMaxPool2d((1, None))  # Ensures H=1 for any input
        
    def forward(self, x):
        x = self.conv0(x)
        
        x = self.res1(x)
        x = self.pool1(x)
        
        x = self.res2(x)
        x = self.pool2(x)
        
        x = self.res3(x)
        x = self.attention1(x)
        x = self.pool3(x)
        
        x = self.res4(x)
        x = self.pool4(x)
        
        x = self.res5(x)
        x = self.attention2(x)
        
        x = self.final_pool(x)
        
        return x


class BidirectionalGRU(nn.Module):
    """GRU-based recurrent layer (faster than LSTM for similar performance)"""
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super(BidirectionalGRU, self).__init__()
        self.rnn = nn.GRU(
            input_size, 
            hidden_size, 
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=False
        )
        self.embedding = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.reshape(T * b, h)
        output = self.embedding(t_rec)
        output = output.reshape(T, b, -1)
        return output


class PositionalEncoding(nn.Module):
    """Add positional information to sequence features"""
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [seq_len, batch, features]
        x = x + self.pe[:x.size(0), :]
        return x


class ImprovedCRNN(nn.Module):
    """
    Improved CRNN with:
    - Residual connections for better gradient flow
    - Spatial attention mechanisms
    - GRU instead of LSTM (faster, similar accuracy)
    - Deeper recurrent layers
    - Positional encoding
    - Layer normalization
    """
    def __init__(
        self, 
        channel, 
        feature_height, 
        feature_width, 
        embedding_dim, 
        output_classes, 
        hidden_units=256, 
        rnn_layers=2, 
        keep_prob=1.0, 
        seq_len=None, 
        device='cuda', 
        leaky_relu=False, 
        pre_lstm_dropout=0.0, 
        inter_lstm_dropout=0.2,
        use_gru=True  # Use GRU instead of LSTM
    ):
        super(ImprovedCRNN, self).__init__()

        self.channel = channel
        self.feature_height = feature_height
        self.feature_width = feature_width
        self.output_classes = output_classes
        self.seq_len = seq_len
        self.hidden_units = hidden_units

        assert self.feature_height % 16 == 0, "Feature height must be divisible by 16"

        # Improved CNN backbone
        self.cnn = ImprovedCNN(channel, leaky_relu)
        
        # Dropout after CNN
        self.cnn_dropout = nn.Dropout(pre_lstm_dropout) if pre_lstm_dropout > 0 else nn.Identity()
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(512)
        
        # Layer normalization before RNN
        self.layer_norm = nn.LayerNorm(512)
        
        # Improved RNN with multiple layers
        if use_gru:
            self.rnn = nn.ModuleList([
                BidirectionalGRU(
                    512 if i == 0 else hidden_units,
                    hidden_units,
                    hidden_units if i < rnn_layers - 1 else output_classes,
                    num_layers=1
                )
                for i in range(rnn_layers)
            ])
        else:
            # Keep LSTM option for compatibility
            self.rnn = nn.ModuleList([
                BidirectionalLSTM(
                    512 if i == 0 else hidden_units,
                    hidden_units // 2,
                    hidden_units if i < rnn_layers - 1 else output_classes
                )
                for i in range(rnn_layers)
            ])
        
        # Dropout between RNN layers
        self.rnn_dropout = nn.Dropout(inter_lstm_dropout) if inter_lstm_dropout > 0 else nn.Identity()
        self.rnn_layers = rnn_layers

    def forward(self, input, _=None):
        # CNN feature extraction
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        
        assert h == 1, f"Height of conv must be 1, got {h}"
        
        # Reshape for RNN: [width, batch, channels]
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)
        
        # Add positional encoding
        conv = self.pos_encoder(conv)
        
        # Layer normalization
        conv = self.layer_norm(conv)
        
        # Apply dropout after CNN
        conv = self.cnn_dropout(conv)
        
        # RNN processing with multiple layers
        output = conv
        for i, rnn_layer in enumerate(self.rnn):
            output = rnn_layer(output)
            # Apply dropout between layers (but not after last layer)
            if i < self.rnn_layers - 1:
                output = self.rnn_dropout(output)
        
        # Transpose to [batch, time, features] for CTC loss
        output = output.transpose(1, 0)
        
        return output, None, None, None


class BidirectionalLSTM(nn.Module):
    """Original LSTM implementation for compatibility"""
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, batch_first=False)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.reshape(T * b, h)
        output = self.embedding(t_rec)
        output = output.reshape(T, b, -1)
        return output
