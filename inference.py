import platform
import random
import numpy as np
import json
import torch
import torchaudio
import soundfile as sf
import librosa
import pandas as pd
from glob import glob
from aeiou.viz import audio_spectrogram_image
from einops import rearrange
from safetensors.torch import load_file
from torch.nn import functional as F
from torchaudio import transforms as T

from omniaudio.inference.generation import generate_diffusion_cond
from omniaudio.models.factory import create_model_from_config
from omniaudio.models.utils import load_ckpt_state_dict
from omniaudio.inference.utils import prepare_audio
from omniaudio.utils import copy_state_dict, generate_mask
from tqdm import tqdm
import os
from pathlib import Path
import shutil
from omniaudio import get_pretrained_model

from extract_video_features import extract_video_features
from huggingface_hub import hf_hub_download


model = None
sample_rate = 32000
sample_size = 1920000


def load_model(model_config=None, model_ckpt_path=None, pretrained_name=None, pretransform_ckpt_path=None,
               device="cuda", model_half=False, infer_type=None):
    global model, sample_rate, sample_size

    infer_type = "diffusion"
    prefix = infer_type + "."
    if pretrained_name is not None:
        print(f"Loading pretrained model {pretrained_name}")
        model, model_config = get_pretrained_model(pretrained_name)

    elif model_config is not None:
        print(f"Creating model from config")
        model = create_model_from_config(model_config)

        # 🟡 If model_ckpt_path is None, download from HuggingFace
        if model_ckpt_path is None:
            print(f"No checkpoint provided. Downloading from HuggingFace repo omniaudio/OmniAudio360V2SA...")
            model_ckpt_path = hf_hub_download(
                repo_id="omniaudio/OmniAudio360V2SA",
                filename="model.ckpt"
            )

        print(f"Loading model checkpoint from {model_ckpt_path}")
        copy_state_dict(model, load_ckpt_state_dict(model_ckpt_path, prefix=prefix))


    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    if pretransform_ckpt_path is not None:
        print(f"Loading pretransform checkpoint from {pretransform_ckpt_path}")
        model.pretransform.load_state_dict(load_ckpt_state_dict(pretransform_ckpt_path), strict=False)
        print(f"Done loading pretransform")

    model.to(device).eval().requires_grad_(False)

    if model_half:
        model.to(torch.float16)

    print(f"Done loading model")

    return model, model_config

def generate_cond_f(
        cond_key1,
        cond_tensor1,
        cond_key2,
        cond_tensor2,
        negative_prompt=None,
        seconds_start=0,
        seconds_total=30,
        cfg_scale=6.0,
        steps=250,
        preview_every=None,
        seed=-1,
        sampler_type="dpmpp-3m-sde",
        sigma_min=0.03,
        sigma_max=1000,
        cfg_rescale=0.0,
        use_init=False,
        init_audio=None,
        init_noise_level=1.0,
        mask_cropfrom=None,
        mask_pastefrom=None,
        mask_pasteto=None,
        mask_maskstart=None,
        mask_maskend=None,
        mask_softnessL=None,
        mask_softnessR=None,
        mask_marination=None,
        batch_size=1,
        filename='output',
        dirname='result'
):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    # print(f"Prompt: {prompt}")
    # import ipdb
    # ipdb.set_trace()
    global preview_images
    preview_images = []
    if preview_every == 0:
        preview_every = None

    # Return fake stereo audio
    conditioning = [{cond_key1: cond_tensor1, cond_key2: cond_tensor2}] * batch_size

    if negative_prompt:
        negative_conditioning = [{"prompt": negative_prompt}] * batch_size
    else:
        negative_conditioning = None

    # Get the device from the model
    device = next(model.parameters()).device

    seed = int(seed)

    if not use_init:
        init_audio = None

    input_sample_size = sample_size

    if init_audio is not None:
        in_sr, init_audio = init_audio
        # Turn into torch tensor, converting from int16 to float32
        init_audio = torch.from_numpy(init_audio).float().div(32767)

        if init_audio.dim() == 1:
            init_audio = init_audio.unsqueeze(0)  # [1, n]
        elif init_audio.dim() == 2:
            init_audio = init_audio.transpose(0, 1)  # [n, 2] -> [2, n]

        if in_sr != sample_rate:
            resample_tf = T.Resample(in_sr, sample_rate).to(init_audio.device)
            init_audio = resample_tf(init_audio)

        audio_length = init_audio.shape[-1]

        if audio_length > sample_size:
            input_sample_size = audio_length + (
                    model.min_input_length - (audio_length % model.min_input_length)) % model.min_input_length

        init_audio = (sample_rate, init_audio)
    def progress_callback(callback_info):
        global preview_images
        denoised = callback_info["denoised"]
        current_step = callback_info["i"]
        sigma = callback_info["sigma"]

        if (current_step - 1) % preview_every == 0:
            if model.pretransform is not None:
                denoised = model.pretransform.decode(denoised)
            denoised = rearrange(denoised, "b d n -> d (b n)")
            denoised = denoised.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            audio_spectrogram = audio_spectrogram_image(denoised, sample_rate=sample_rate)
            preview_images.append((audio_spectrogram, f"Step {current_step} sigma={sigma:.3f})"))

    # If inpainting, send mask args
    # This will definitely change in the future
    if mask_cropfrom is not None:
        mask_args = {
            "cropfrom": mask_cropfrom,
            "pastefrom": mask_pastefrom,
            "pasteto": mask_pasteto,
            "maskstart": mask_maskstart,
            "maskend": mask_maskend,
            "softnessL": mask_softnessL,
            "softnessR": mask_softnessR,
            "marination": mask_marination,
        }
    else:
        mask_args = None
        # Do the audio generation

    audio = generate_diffusion_cond(
        model,
        conditioning=conditioning,
        negative_conditioning=negative_conditioning,
        steps=steps,
        cfg_scale=cfg_scale,
        batch_size=batch_size,
        sample_size=int(sample_rate * seconds_total),
        sample_rate=sample_rate,
        seed=seed,
        device=device,
        sampler_type=sampler_type,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        init_audio=init_audio,
        init_noise_level=init_noise_level,
        mask_args=mask_args,
        callback=progress_callback if preview_every is not None else None,
        scale_phi=cfg_rescale
    )

    # Convert to WAV file
    audio = rearrange(audio, "b d n -> d (b n)")
    audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    return audio










if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run inference')
    parser.add_argument('--pretrained-name', type=str, help='Name of pretrained model', required=False)
    parser.add_argument('--base-dir', type=str, help='Video Directory', required=False)
    parser.add_argument('--infer-type', type=str, help='Inference Type', default='v2a',
                        required=False)
    parser.add_argument('--model-config', type=str, help='Path to model config', required=False)
    parser.add_argument('--ckpt-path', type=str, help='Path to model checkpoint', default=None)
    parser.add_argument('--dirname', type=str, help='output directory', required=True)
    parser.add_argument('--pretransform-ckpt-path', type=str, help='Optional to model pretransform checkpoint',
                        required=False)
    parser.add_argument('--model-half', action='store_true', help='Whether to use half precision', required=False)
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on', required=False)
    parser.add_argument('--cfg-scale', type=float, default=5)
    parser.add_argument('--mode', type=str, default="eqfov")
    args = parser.parse_args()

    device = torch.device(args.device)
    print('device:', device)

    seed=np.random.randint(42, 2**10 - 1, dtype=np.uint32)

    print("seed: ", str(seed))
    
    torch.manual_seed(seed)

    model_config_path = args.model_config
    ckpt_path = args.ckpt_path
    pretrained_name = args.pretrained_name
    pretransform_ckpt_path = args.pretransform_ckpt_path
    model_half = args.model_half
    video_path = args.base_dir
    feature_type = args.mode
    dirname = args.dirname
    infer_type = args.infer_type

    cfg_scale = args.cfg_scale


    temp_dir= os.path.join(dirname, "temp")
    extract_video_features(video_path, feature_type,temp_dir)


    if model_config_path is not None:
        # Load config from json file
        print(model_config_path)
        with open(model_config_path) as f:
            model_config = json.load(f)
    else:
        model_config = None

    _, model_config = load_model(model_config, ckpt_path, pretrained_name=pretrained_name,
                                 pretransform_ckpt_path=pretransform_ckpt_path, model_half=model_half, device=device,
                                 infer_type=infer_type)

    model_type = model_config["model_type"]
    # sample_rate = model_config["sample_rate"]
    # print(f"cfg_scale: "+str(cfg_scale))
    


    

    name=os.path.splitext(os.path.basename(video_path))[0]

    save_path= os.path.join(dirname, name+".flac")

    try:
        if infer_type == 'v2a':
            if(feature_type=="eacfov"):
                    eac = os.path.join(temp_dir,"metaclip-huge-eac", name+".npy")
                    assert os.path.exists(eac), f'{eac} must exist'
                    fov = os.path.join(temp_dir,"metaclip-huge-fov", name+".npy")
                    assert os.path.exists(fov), f'{fov} must exist'

                    videoeac = np.load(eac)
                    videoeac = torch.from_numpy(videoeac)
                    videofov = np.load(fov)
                    videofov = torch.from_numpy(videofov)


                    out_audio = generate_cond_f('video_360', videoeac,"video_fov",videofov, steps=24, seconds_total=10, cfg_scale=3.0, sigma_min=0.3,
                                                sigma_max=500, sampler_type='dpmpp-3m-sde',seed = seed)
                    sr = 44100
                    torchaudio.save(save_path, out_audio, sr)
                    print(f"Saved audio to {save_path}")

            elif(feature_type=="eqfov"):
                    eq = os.path.join(temp_dir,"metaclip-huge-eq", name+".npy")
                    assert os.path.exists(eq), f'{eq} must exist'
                    fov = os.path.join(temp_dir,"metaclip-huge-fov", name+".npy")
                    assert os.path.exists(fov), f'{fov} must exist'
   
                    videoeq= np.load(eq)
                    videoeq = torch.from_numpy(videoeq)
                    videofov = np.load(fov)
                    videofov = torch.from_numpy(videofov)

                    out_audio = generate_cond_f('video_360', videoeq,"video_fov",videofov, steps=24, seconds_total=10, cfg_scale=3.0, sigma_min=0.3,
                                                sigma_max=500, sampler_type='dpmpp-3m-sde',seed = seed)
                    
                    sr = 44100
                    torchaudio.save(save_path, out_audio, sr)
                    print(f"Saved audio to {save_path}")
            else:
                raise NotImplementedError(f"Feature type {feature_type} not supported") 


    except Exception as e:
        print(f'[LOG] Error {name} inference: {e}')


            
