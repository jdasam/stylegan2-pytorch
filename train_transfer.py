import torch
import pickle
import argparse
import numpy as np
import librosa
from audio_model import SiameseNet, TransferNet


def load_audio_model(args):
    with open("hparams.dat", "rb") as f:
        hparams = pickle.load(f)
    audio_embedder = SiameseNet(hparams)
    audio_checkpoint = torch.load(args.ckpt)
    audio_embedder.load_state_dict(audio_checkpoint["state_dict"])
    return audio_embedder



def train(args, device):
    audio_embedder = load_audio_model(args).to(device)
    model = TransferNet(100, 512).to(device)
    learning_rate = args.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=args.weight_decay)
    loss_fn = torch.nn.MSELoss()
    data = np.load(args.data_path, allow_pickle=True)
    # embs = [audio_embedder.inference_with_audio(x['audio'])[0] for x in data]
    styles = [x['style'] for x in data]
    audios = [librosa.core.resample(x['audio'], 44100, 16000) for x in data]
    mels = [librosa.feature.melspectrogram(y=x, sr=16000, n_fft=512, hop_length=256, n_mels=48) for x in audios]
    mels = torch.Tensor(mels).to(device)
    
    audio_embedder.eval()
    with torch.no_grad():
        embs = audio_embedder.cnn.fwd_wo_pool(mels)
    # audios =torch.Tensor([librosa.core.resample(x['audio'], 44100, 16000) for x in data])
    # embs = torch.Tensor(embs).to(device)
    styles = torch.Tensor(styles).to(device).unsqueeze(1).repeat(1,embs.shape[1],1)

    model.train()
    for i in range(args.epoch):
        model.zero_grad()
        transfer_style = model(embs)
        loss = loss_fn(transfer_style, styles)
        loss.backward()
        print('Loss value is {}'.format(loss.item()))
        optimizer.step()

    torch.save({'iteration': i,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'learning_rate': learning_rate}, 'transfer.pt')
    
    


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")
    parser.add_argument("--style_dim",type=int, default=512,help="style_dim",)
    parser.add_argument("--epoch",type=int, default=300,help="num training epochs")
    parser.add_argument("--learning_rate",type=float, default=1e-3)
    parser.add_argument("--weight_decay",type=float, default=1e-6)

    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoint_best",
        help="path to the audio model checkpoint",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="label_data.npy",
        help="label pair path",
    )

    args = parser.parse_args()
    train(args, device)