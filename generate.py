import argparse
from os import write

import torch
from torchvision import utils
from model import Generator
from audio_model import SiameseNet, TransferNet
from tqdm import tqdm
import av
import pickle
from pydub import AudioSegment
import librosa
import numpy as np
import subprocess

# from mhmovie.code import movie, music


def interpolate(sample_z, num_interpol=10):
    sample_a = sample_z[0,:]
    sample_b = sample_z[-1,:]
    diff = sample_b - sample_a
    new_samples = torch.zeros((num_interpol+2, sample_a.shape[0]),device=sample_a.device)
    for i in range(1,num_interpol+1):
        new_samples[i,:] = sample_a + i * diff/num_interpol
    new_samples[0,:] = sample_a
    new_samples[-1,:] = sample_b

    return new_samples

def write_video(filename, video_array, fps, bitrate, video_codec='libx264', options=None):
    """
    Writes a 4d tensor in [T, H, W, C] format in a video file

    Parameters
    ----------
    filename : str
        path where the video will be saved
    video_array : Tensor[T, H, W, C]
        tensor containing the individual frames, as a uint8 tensor in [T, H, W, C] format
    fps : Number
        frames per second
    """
    video_array = torch.as_tensor(video_array, dtype=torch.uint8).numpy()

    container = av.open(filename, mode='w')

    stream = container.add_stream(video_codec, rate=fps)
    stream.width = video_array.shape[2]
    stream.height = video_array.shape[1]
    stream.pix_fmt = 'yuv420p' if video_codec != 'libx264rgb' else 'rgb24'
    stream.options = options or {}
    stream.bit_rate = bitrate

    for img in video_array:
        frame = av.VideoFrame.from_ndarray(img, format='rgb24')
        frame.pict_type = 'NONE'
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush stream
    for packet in stream.encode():
        container.mux(packet)

    # Close the file
    container.close()


def get_embedding_from_audio(path, model):
    song = AudioSegment.from_file(path, 'm4a').set_frame_rate(16000).set_channels(1)._data
    decoded = np.frombuffer(song, dtype=np.int16) / 32768
    mel = librosa.feature.melspectrogram(y=decoded, sr=16000, n_fft=512, hop_length=256, n_mels=48)
    with torch.no_grad():
        model.eval()
        embedd = model.cnn.fwd_wo_pool(torch.Tensor(mel).unsqueeze(0))
    return embedd


def generate(args, g_ema, device, mean_latent, audio_embd, audio_path):

    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
            # sample_z = torch.randn(args.sample, args.latent * 2, device=device)
            # sample_z = interpolate(sample_z, num_interpol=8)
            # sample, _ = g_ema(
            #     [sample_z], truncation=args.truncation, truncation_latent=mean_latent, randomize_noise=False,
            # )

            # utils.save_image(
            #     sample,
            #     f"sample/{str(i).zfill(6)}.png",
            #     nrow=1,
            #     normalize=True,
            #     range=(-1, 1),
            # )
            sample, _ = g_ema(
                [audio_embd.to(device)], truncation=args.truncation, truncation_latent=mean_latent, randomize_noise=False,
                input_is_latent=True, interpolate_styles=True
            )
            out_path = f"sample/{str(i).zfill(6)}.mp4"
            write_video(out_path, sample, fps=args.fps, bitrate=args.bitrate, video_codec='h264')
            
            del sample
            combined_out_path = f"sample/{str(i).zfill(6)}_combined.mp4"
            cmd = 'ffmpeg -i {} -i {} -c:v copy -c:a aac -strict -2 {}'.format(out_path, audio_path, combined_out_path)
            subprocess.call(cmd, shell=True)                                     # "Muxing Done


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=1024, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=1, help="number of images to be generated"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="network-snapshot-012052.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--ad_ckpt",
        type=str,
        default="checkpoint_best",
        help="path to the audio model checkpoint",
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        # default="/home/svcapp/userdata/musicai/flo_data/433/090/433090157.m4a",
        default="/home/svcapp/userdata/musicai/flo_data/433/081/433081542.m4a",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=21,
        help="frame per second for video",
    )
    parser.add_argument(
        "--bitrate",
        type=int,
        default=1e7,
        help="bitrate for video encoding",
    )


    args = parser.parse_args()


    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)

    checkpoint = torch.load(args.ckpt)
    g_ema.load_state_dict(checkpoint["g_ema"])
    g_ema.noises = torch.nn.Module()
    for layer_idx in range(g_ema.num_layers):
        res = (layer_idx + 5) // 2
        shape = [1, 1, 2 ** res, 2 ** res]
        g_ema.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None
    
    with open("hparams.dat", "rb") as f:
        hparams = pickle.load(f)
    audio_embedder = SiameseNet(hparams)

    audio_checkpoint = torch.load(args.ad_ckpt)
    checkpoint = torch.load(args.ckpt)
    audio_embedder.load_state_dict(audio_checkpoint["state_dict"])
    audio_embd = get_embedding_from_audio(args.audio_path, audio_embedder)

    transfer_net = TransferNet(100,512)
    transfer_checkpoint = torch.load('transfer.pt')
    transfer_net.load_state_dict(transfer_checkpoint['state_dict'])

    audio_embd = transfer_net(audio_embd)

    # audio_data = AudioSegment.from_file(args.audio_path, 'm4a').set_frame_rate(44100).set_channels(1)._data
    # audio_data = np.frombuffer(audio_data, dtype=np.int16) / 32768

    generate(args, g_ema, device, mean_latent, audio_embd[0], args.audio_path)
