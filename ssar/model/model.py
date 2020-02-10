import torch
import torch.nn as nn
from torch.nn.init import orthogonal_, zeros_, xavier_normal_
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence


class SSAREncoder(nn.Module):
    def __init__(self, ResNet):
        super(SSAREncoder, self).__init__()
        self.conv1 = ResNet.conv1
        self.bn1 = ResNet.bn1
        self.relu = ResNet.relu
        self.maxpool = ResNet.maxpool
        self.layer1 = ResNet.layer1
        self.layer2 = ResNet.layer2
        self.conv2 = nn.Conv2d(128, 128, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1)
        self.bn3 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        return x


class SSARDecoder(nn.Module):

    def __init__(self):
        super(SSARDecoder, self).__init__()
        self.deconv0 = nn.ConvTranspose2d(256, 64, 4, 2, 1)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(16, 8, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(8, 2, 4, 2, (2, 1))

    def forward(self, x):

        x = self.deconv0(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        mask = self.deconv4(x)

        return mask


class SSAREmbeddingGenerator(nn.Module):

    def __init__(self):
        super(SSAREmbeddingGenerator, self).__init__()
        self.linear0 = nn.Linear(7168, 2048)
        self.bn = nn.BatchNorm1d(2048)
        self.linear1 = nn.Linear(2048, 83)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        batch_size, depth, width, height = x.size(0), x.size(1), x.size(2), x.size(3)
        encoded_features = x.view(batch_size, depth*width*height)
        encoded_features = self.linear0(encoded_features)
        encoded_features = self.bn(encoded_features)
        encoded_features = self.relu(encoded_features)
        encoded_features = self.linear1(encoded_features)
        return encoded_features


class SSARLSTM(nn.Module):

    def __init__(self, input_size, number_of_classes, batch_size, dropout):
        super(SSARLSTM, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.input_size = input_size
        self.hidden_size = int(input_size)
        self.number_of_classes = number_of_classes
        self.batch_size = batch_size
        self.lstm1 = nn.LSTM(input_size, self.hidden_size, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True, dropout=dropout)
        self.lstm3 = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True, dropout=dropout)
        self.lstm4 = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(self.hidden_size, self.number_of_classes)
        self.init_lstm_weights()

    def init_lstm_weights(self):
        # Xavier Normal for input weights
        orthogonal_(self.lstm1.all_weights[0][0])
        xavier_normal_(self.lstm2.all_weights[0][0])
        orthogonal_(self.lstm3.all_weights[0][0])
        xavier_normal_(self.lstm4.all_weights[0][0])
        # Orthogonal for recurrent weights
        orthogonal_(self.lstm1.all_weights[0][1])
        xavier_normal_(self.lstm2.all_weights[0][1])
        orthogonal_(self.lstm3.all_weights[0][1])
        xavier_normal_(self.lstm4.all_weights[0][1])
        # Zeros for biases
        zeros_(self.lstm1.all_weights[0][2])
        zeros_(self.lstm1.all_weights[0][3])
        zeros_(self.lstm2.all_weights[0][2])
        zeros_(self.lstm2.all_weights[0][3])
        zeros_(self.lstm3.all_weights[0][2])
        zeros_(self.lstm3.all_weights[0][3])
        zeros_(self.lstm4.all_weights[0][2])
        zeros_(self.lstm4.all_weights[0][3])

    def forward(self, x, lengths=None, hidden=None):
        if hidden is None:
            hidden = [None, None, None, None]

        sequence_labels, hidden[0] = self.lstm1(x, hidden[0])
        sequence_labels, hidden[1] = self.lstm2(sequence_labels, hidden[1])
        sequence_labels, hidden[2] = self.lstm3(sequence_labels, hidden[2])
        sequence_labels, hidden[3] = self.lstm4(sequence_labels, hidden[3])
        if type(sequence_labels) is nn.utils.rnn.PackedSequence:
            label_data = self.fc(sequence_labels.data)
            label = PackedSequence(label_data, sequence_labels.batch_sizes, sequence_labels.sorted_indices, sequence_labels.unsorted_indices)
            label, seq_lengths = pad_packed_sequence(sequence=label, batch_first=True)
        else:
            label = self.fc(sequence_labels)
        label = label.permute([0, 2, 1]) # [N, D, C] -> [N, C, D] format for nn.CrossEntropyLoss
        
        return label, hidden


class SSAR(nn.Module):

    def __init__(self, ResNet, input_size, number_of_classes, batch_size, dropout):
        super(SSAR, self).__init__()
        self.encoder = SSAREncoder(ResNet)
        self.decoder = SSARDecoder()
        self.embedding_generator = SSAREmbeddingGenerator()
        self.lstms = SSARLSTM(input_size, number_of_classes, batch_size, dropout)
        self.batch_size = batch_size

    def forward(self, x, lstm_hidden=None, lengths=None, get_mask=False, get_lstm_state=False):
        packed_seq = None
        if type(x) is nn.utils.rnn.PackedSequence:
            packed_seq = x
            x = packed_seq.data

        x = self.encoder(x)

        embeddings = self.embedding_generator(x)
        if lengths is None:
            lengths = [embeddings.shape[0]]
            embeddings = embeddings.unsqueeze(0)

        if packed_seq is not None:
            embeddings = PackedSequence(embeddings, packed_seq.batch_sizes, packed_seq.sorted_indices, packed_seq.unsorted_indices)

        label, lstm_hidden = self.lstms(embeddings, lengths, lstm_hidden)

        outputs = []
        if get_mask:
            mask = self.decoder(x)
            if packed_seq is not None:
                mask = PackedSequence(mask, packed_seq.batch_sizes, packed_seq.sorted_indices, packed_seq.unsorted_indices)
                mask, _ = pad_packed_sequence(sequence=mask, batch_first=True)
            outputs.append(mask)

        outputs.append(label)

        if get_lstm_state:
            outputs.append(lstm_hidden)

        # Handle single output case
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

