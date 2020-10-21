import argparse
from os import write
from librosa.filters import mel

import torch
from model import Generator
from audio_model import SiameseNet, TransferNet, HParams
from tqdm import tqdm
import av
from pydub import AudioSegment
import librosa
import numpy as np
import subprocess
import math
from music_embedder import extractor
from pathlib import Path

# from mhmovie.code import movie, music

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


def get_embedding_from_audio(path, model, target_fps=3):
    file_format = path[-3:]
    if file_format == 'aac':
        file_format = 'm4a'
    song = AudioSegment.from_file(path, file_format).set_frame_rate(16000).set_channels(1)._data
    audio = np.frombuffer(song, dtype=np.int16) / 32768

    view_size = 130304
    audio_batch_size = 100
    dummy = torch.zeros((1, audio.shape[1] + view_size *2))
    dummy[:,view_size//2:view_size//2+audio.shape[1]] = audio
    audio = dummy
    num_frame = math.ceil(audio.shape[1]/ 16000 * target_fps)
    num_batch = math.ceil(num_frame / audio_batch_size)

    model = model.to('cuda')
    model.eval() 
    total_embeddings = []

    mel_basis = librosa.filters.mel(16000, n_fft=512, n_mels=48)

    with torch.no_grad():
    # _, embeddings = model(audio_input)
        for batch_i in range(num_batch):
            start_idx = int(batch_i * audio_batch_size * 16000/target_fps)
            if batch_i == num_batch -1:
                num_segments = num_frame % audio_batch_size
            else:
                num_segments = audio_batch_size
            batch_audio = torch.stack([audio[0,start_idx+int(i*16000/target_fps):start_idx+int(i*16000/target_fps)+view_size ] for i in range(num_segments) ])
            # mel = torchaudio.
            spec = librosa.stft(batch_audio, n_fft=512, hop_length=256, win_length=512, window='hann')
            mel_spec = np.dot(mel_basis, np.abs(spec))
            mel_spec = mel_spec / 80 + 0.5

            embeddings = model.cnn.fwd_wo_pool(torch.Tensor(mel).to('cuda'))
            total_embeddings.append(embeddings[:,0,:])


    with torch.no_grad():
        model.eval()
        embedd = model.cnn.fwd_wo_pool(torch.Tensor(mel).to('cuda').unsqueeze(0))
    return embedd


def generate(args, g_ema, device, mean_latent, total_audio_embd, audio_path):
    num_interpol = int(args.fps//args.audio_fps)
    c_window_length = int(args.cw_sec * args.fps)
    m_window_length = int(args.mw_sec * args.fps)
    f_window_length = int(args.fw_sec * args.fps)
    if c_window_length % 2 == 0:
        c_window_length += 1
    if m_window_length % 2 == 0:
        m_window_length += 1
    if f_window_length % 2 == 0:
        f_window_length += 1
    
    with torch.no_grad():
        g_ema.eval()
        # for i in tqdm(range(args.pics)):
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
        for i in tqdm(range(len(total_audio_embd))):
            audio_embd = total_audio_embd[i]
            path = Path(audio_path[i])
            sample, _ = g_ema(
                [audio_embd.to(device)], truncation=args.truncation, truncation_latent=mean_latent, randomize_noise=False,
                input_is_latent=True, interpolate_styles=True, num_interpol=num_interpol, smoothing=True, 
                coarse_window_length=c_window_length, middle_window_length=m_window_length, fine_window_length=f_window_length
            )
            out_path = f"sample/{path.stem}_{args.cw_sec}_{args.mw_sec}_{args.tf_ckpt}.mp4"
            write_video(out_path, sample, fps=args.fps, bitrate=args.bitrate, video_codec='h264')
            
            del sample
            torch.cuda.empty_cache()
            combined_out_path = f"sample/{path.stem}_{args.cw_sec}_{args.mw_sec}_{args.tf_ckpt}_combined.mp4"
            cmd = 'ffmpeg -i {} -i {} -c:v copy -c:a aac -strict -2 {}'.format(out_path, str(path), combined_out_path)
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
        "--audio_model",
        type=str,
        default="FCN037",
        help="model code for audio embedding model",
    )
    parser.add_argument(
        "--ad_ckpt",
        type=str,
        default="checkpoint_best",
        help="path to the audio model checkpoint",
    )
    parser.add_argument(
        "--tf_ckpt",
        type=str,
        default="tf_tanh_L1_FCN037_it30000_lr0.0001.pt",
        help="path to the transfer model checkpoint",
    )
    parser.add_argument(
        "--audio_path",
        nargs='*',
        type=str,
        default=[
            # "sample/Queen_bohemian.mp3"
            "sample/Queen_bohemian.mp3"
            # "/home/svcapp/userdata/musicai/flo_data/433/090/433090157.m4a",
        # default="/home/svcapp/userdata/musicai/flo_data/433/081/433081542.m4a",
        # default="/home/svcapp/userdata/musicai/MSD/1/2/1203820.clip.mp3",
        ],
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )
    parser.add_argument( "--cw_sec", type=float, default=5,
        help="smoothing window length for coarse layer styles",
    )
    parser.add_argument( "--mw_sec", type=float, default=2,
        help="smoothing window length for middle layer styles",
    )
    parser.add_argument( "--fw_sec", type=float, default=0.333,
        help="smoothing window length for fine layer styles",
    )
    parser.add_argument("--fps", type=int,default=15, help="frame per second for video",
    )
    parser.add_argument("--audio_fps", type=int,default=3, help="number of embeddings per second for audio embedding",
    )
    parser.add_argument(
        "--bitrate",
        type=float,
        default=1e7,
        help="bitrate for video encoding",
    )


    args = parser.parse_args()
    # if isinstance(args.audio_path, str):
    #     args.audio_path = [args.audio_path]
    torch.cuda.set_device(0)
    args.latent = 512
    args.n_mlp = 8
    assert (args.fps / args.audio_fps).is_integer()

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
    
    total_audio_embd = []
    transfer_net = TransferNet(512,512).to('cuda')
    transfer_checkpoint = torch.load(args.tf_ckpt, map_location='cpu')
    transfer_net.load_state_dict(transfer_checkpoint['state_dict'])

    if "siamese" in args.audio_model:
        # with open("hparams.dat", "rb") as f:
        #     hparams = pickle.load(f)
        hparams = HParams()
        audio_embedder = SiameseNet(hparams)
        audio_checkpoint = torch.load(args.ad_ckpt)
        checkpoint = torch.load(args.ckpt)
        audio_embedder.load_state_dict(audio_checkpoint["state_dict"])
        audio_embedder = audio_embedder.to(device)
        
    for audio_path in args.audio_path:
        if "siamese" in args.audio_model:
            audio_embd = get_embedding_from_audio(audio_path, audio_embedder)
        else:
            # audio_embd = extractor.embedding_extractor(audio_path, "FCN037")
            audio_embd = extractor.get_frame_embeddings(audio_path, "FCN037", args.audio_fps)

        
        audio_embd = transfer_net(audio_embd)
        total_audio_embd.append(audio_embd)
    torch.cuda.empty_cache()

    # audio_data = AudioSegment.from_file(args.audio_path, 'm4a').set_frame_rate(44100).set_channels(1)._data
    # audio_data = np.frombuffer(audio_data, dtype=np.int16) / 32768

    generate(args, g_ema, device, mean_latent, total_audio_embd, args.audio_path)
