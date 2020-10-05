import torch
import torch.nn as nn
import librosa
import numpy as np

class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvNorm, self).__init__()
        self.conv_norm = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_norm(x)



class DeepCNN(nn.Module):
    def __init__(self, hparams):
        super(DeepCNN, self).__init__()
        padding_size = int((hparams.kernel_size-1)/2)
        self.conv_module = nn.Sequential(
            ConvNorm(48, hparams.conv_size, hparams.kernel_size, padding_size),
            nn.MaxPool1d(4),
            ConvNorm(hparams.conv_size, hparams.conv_size, hparams.kernel_size, padding_size),
            nn.MaxPool1d(4),
            ConvNorm(hparams.conv_size, hparams.conv_size, hparams.kernel_size, padding_size),
            nn.MaxPool1d(4),
            ConvNorm(hparams.conv_size, hparams.conv_size, hparams.kernel_size, padding_size),
            nn.MaxPool1d(4),
            ConvNorm(hparams.conv_size, hparams.conv_size, hparams.kernel_size, padding_size),
        )
        if hparams.average_pool:
            self.last_pool = nn.AvgPool1d(7)
        else:    
            self.last_pool = nn.MaxPool1d(7)
        self.linear = nn.Sequential(
            nn.Linear(hparams.conv_size, hparams.conv_size),
            nn.ReLU(),
            nn.Dropout(hparams.drop_out),
            nn.Linear(hparams.conv_size, hparams.out_size))

    def forward(self, mel):
        out = self.conv_module(mel)
        out = self.last_pool(out)
        out = out.squeeze(2)
        # out.transpose_(1,2)
        # out = self.attention(out)
        out = self.linear(out)
        return out
    
    def fwd_wo_pool(self, mel):
        out = self.conv_module(mel)
        out = out.permute(0,2,1)
        return self.linear(out)

class MelClassifier(nn.Module):
    def __init__(self, hparams):
        super(MelClassifier, self).__init__()
        self.conv_module = DeepCNN(hparams)
        # self.linear = nn.Linear(hparams.out_size, 30)

    def forward(self, mel):
        out = self.conv_module(mel)
        # out.transpose_(1,2)
        # out = self.attention(out)
        out = torch.sigmoid(out)
        return out

class SiameseNet(nn.Module):
    def __init__(self, hparams):
        super(SiameseNet, self).__init__()
        self.cnn = DeepCNN(hparams)
        # self.similarity_fn = torch.nn.CosineSimilarity(1, eps=1e-6)

    def forward(self, anchor, positive_sample, negative_sample):
        negative_sample = negative_sample.view(-1, 48, 1876)
        concatenated = torch.cat((anchor, positive_sample, negative_sample), 0)
        out = self.cnn(concatenated)
        anchor_rep = out[:anchor.shape[0],:]
        pos_rep = out[anchor.shape[0]:-negative_sample.shape[0],:]
        neg_rep = out[-negative_sample.shape[0]:,:]

        return anchor_rep, pos_rep, neg_rep

    def inference(self, input_mel):
        return self.cnn(input_mel)

    def inference_with_audio(self, input_audio, input_sr=44100, target_sr=16000):
        if input_sr != target_sr: 
            input_audio = np.copy(input_audio)
            input_audio = librosa.core.resample(input_audio, input_sr, target_sr)
        mel = librosa.feature.melspectrogram(y=input_audio, sr=16000, n_fft=512, hop_length=256, n_mels=48)
        return self.inference(torch.Tensor(mel).to('cuda'))

class TransferNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(TransferNet, self).__init__()
        self.transfer = nn.Sequential(
            nn.Linear(in_dim, out_dim)
        )
    def forward(self, audio_embedding):
        return self.transfer(audio_embedding) / 20