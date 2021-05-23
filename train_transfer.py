import torch
import pickle
import argparse
import numpy as np
import librosa
from audio_model import SiameseNet, TransferNet, HParams
from music_embedder import extractor


def load_audio_model(args):
    # with open("hparams.dat", "rb") as f:
    #     hparams = pickle.load(f)
    hparams = HParams()
    audio_embedder = SiameseNet(hparams)
    audio_checkpoint = torch.load(args.ckpt)
    audio_embedder.load_state_dict(audio_checkpoint["state_dict"])
    return audio_embedder

def load_audio_model_and_get_embedding(audio, model_types):
    # audio_length = audio.shape[1]
    input_length, model, checkpoint_path = extractor.load_model(model_types)
    # audio = extractor.make_frames_of_batch(audio, input_length, target_fps=1/3)[:,1,:]
    audio = extractor.make_frames_of_batch(audio, input_length, target_fps=0.5).view(-1, input_length)
    # audio = extractor.make_audio_batch(audio, input_length)

    state_dict = torch.load("music_embedder/"+checkpoint_path, map_location=torch.device('cpu'))

    new_state_map = {model_key: model_key.split("model.")[1] for model_key in state_dict.get("state_dict").keys()}
    new_state_dict = {new_state_map[key]: value for (key, value) in state_dict.get("state_dict").items() if key in new_state_map.keys()}
    model.load_state_dict(new_state_dict)
    
    audio = audio.to('cuda')
    model = model.to('cuda')

    with torch.no_grad():
        model.eval()
        if "CPC" in model_types:
            embeddings = torch.zeros(audio.shape[0], 256)
            for i in range(0, audio.shape[0], 100):
                _, embeddings[i:i+100] =model.get_emb(audio[i:i+100])
        else:
            embeddings = model.get_emb(audio)
    return embeddings

def train(args, device):
    if "siamese" in args.model_code:
        audio_embedder = load_audio_model(args).to(device)
        embd_size = audio_embedder.conv_size
    elif "FCN037" in args.model_code:
        embd_size = 512
    elif "CPC" in args.model_code:
        embd_size = 256
    else:
        embd_size = 64
    model = TransferNet(embd_size, args.style_dim)
    style_stats = torch.load("style_latent_stat.pt")
    # style_stats = torch.load("ocean_image_stat_half_std.pt")
    model.bias = torch.nn.Parameter(style_stats['mean'].squeeze(0), requires_grad=False)
    model.std = torch.nn.Parameter(style_stats['std'].squeeze(0), requires_grad=False)
    model = model.to(device)
    learning_rate = args.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=args.weight_decay)
    # loss_fn = torch.nn.MSELoss()
    loss_fn = torch.nn.L1Loss()

    data = np.load(args.data_path, allow_pickle=True)
    # embs = [audio_embedder.inference_with_audio(x['audio'])[0] for x in data]
    styles = [x['style'] for x in data]
    audios = [x['audio'] for x in data]
    # audios = [librosa.core.resample(x['audio'], 44100, 16000) for x in data]


    # mels = [librosa.feature.melspectrogram(y=x, sr=16000, n_fft=512, hop_length=256, n_mels=48) for x in audios]
    # mels = torch.Tensor(mels).to(device)
    # audio_embedder = load_audio_model(args).to(device)
    # audio_embedder.eval()
    # with torch.no_grad():
    #     embs = audio_embedder.cnn.fwd_wo_pool(mels)
    # audios =torch.Tensor([librosa.core.resample(x['audio'], 44100, 16000) for x in data])
    # embs = torch.Tensor(embs).to(device)
    
    if "siamese" in args.model_code:
        # mel_basis = librosa.filters.mel(16000, n_fft=512, n_mels=48)
        # spec = np.asarray([librosa.stft(x, n_fft=512, hop_length=256, win_length=512, window='hann') for x in audios ])
        # mel_spec = np.dot(mel_basis, librosa.core.amplitude_to_db(np.abs(spec)).transpose(2,1,0)).transpose(2,0,1)
        # mel_spec = np.dot(mel_basis, np.abs(spec.transpose(2,1,0))).transpose(2,0,1)
        mel_spec = np.asarray([librosa.feature.melspectrogram(x, sr=16000,n_mels=48, n_fft=512, hop_length=256, win_length=512, window='hann') for x in audios ])
        mel_spec = mel_spec / 80 + 0.5

        mels = torch.Tensor(mel_spec).to(device)
        # audio_embedder = load_audio_model(args).to(device)
        audio_embedder.eval()
        with torch.no_grad():
            embs = audio_embedder.infer_mid_level(mels, max_pool=False).permute(0,2,1)
        # embs = torch.Tensor(embs).to(device)
    else:
        audios = torch.Tensor(audios)
        embs = load_audio_model_and_get_embedding(audios, model_types=args.model_code).to(device)
        embs = embs.view(audios.shape[0], -1, embs.shape[-1])

    styles = torch.Tensor(styles).to(device).unsqueeze(1).repeat(1,embs.shape[1],1)


    model.train()
    for i in range(args.epoch):
        model.zero_grad()
        transfer_style = model(embs)
        loss = loss_fn(transfer_style, styles)
        loss.backward()
        print('Step {}: Loss value is {}'.format(i, loss.item()))
        optimizer.step()

    torch.save({'iteration': i,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'learning_rate': learning_rate}, 'tf_tanh_L1_{}_it{}_lr{}.pt'.format(args.model_code, args.epoch, args.learning_rate))
    

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")
    parser.add_argument("--style_dim",type=int, default=512,help="style_dim",)
    parser.add_argument("--epoch",type=int, default=30000,help="num training epochs")
    parser.add_argument("--learning_rate",type=float, default=1e-5)
    parser.add_argument("--weight_decay",type=float, default=1e-6)
    parser.add_argument("--model_code",type=str,
                            # default="artist_siamese")
                             default="FCN037")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoint_best",
        help="path to the audio model checkpoint",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="label_data_with_16kHz_audio.npy",
        help="label pair path",
    )
    # torch.cuda.set_device(1)
    args = parser.parse_args()
    train(args, device)