
import gradio as gr
import numpy as np
from sklearn.svm import LinearSVC
from sklearn import preprocessing
import pandas as pd
import kornia
import torchvision

import random
import time

from diffusers import LCMScheduler
from diffusers.models import ImageProjection
from patch_sdxl import SDEmb
import torch


prompt_list = [p for p in list(set(
                pd.read_csv('./twitter_prompts.csv').iloc[:, 1].tolist())) if type(p) == str]


model_id = "stabilityai/stable-diffusion-xl-base-1.0"
lcm_lora_id = "latent-consistency/lcm-lora-sdxl"

pipe = SDEmb.from_pretrained(model_id, variant="fp16")
pipe.load_lora_weights(lcm_lora_id)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to(device="cuda", dtype=torch.float16)

pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")



calibrate_prompts = [
    "4k photo",
    'surrealist art',
    'a psychedelic, fractal view',
    'a beautiful collage',
    'an intricate portrait',
    'an impressionist painting',
    'abstract art',
    'an eldritch image',
    'a sketch',
    'a city full of darkness and graffiti',
    'a black & white photo',
    'a brilliant, timeless tarot card of the world',
    'a photo of a woman',
    '',
]

embs = []
ys = []

start_time = time.time()

output_hidden_state = False if isinstance(pipe.unet.encoder_hid_proj, ImageProjection) else True


transform = kornia.augmentation.RandomResizedCrop(size=(224, 224), scale=(.3, .5))
nom = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
def patch_encode_image(image):
    image = torch.tensor(torchvision.transforms.functional.pil_to_tensor(image).to(torch.float16)).repeat(16, 1, 1, 1).to('cuda')
    image = image / 255
    patches = nom(transform(image))
    output, _ = pipe.encode_image(
                patches, 'cuda', 1, output_hidden_state
            )
    return output.mean(0, keepdim=True)


glob_idx = 0

def next_image():
    global glob_idx
    glob_idx = glob_idx + 1
    with torch.no_grad():
        if len(calibrate_prompts) > 0:
            print('######### Calibrating with sample prompts #########')
            prompt = calibrate_prompts.pop(0)
            print(prompt)

            image = pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=8,
            guidance_scale=0,
            ip_adapter_emb=torch.zeros(1, 1, 1280, device='cuda', dtype=torch.float16),
            ).images


            pooled_embeds, _ = pipe.encode_image(
                image[0], 'cuda', 1, output_hidden_state
            )
            #pooled_embeds = patch_encode_image(image[0])

            embs.append(pooled_embeds)
            return image[0]
        else:
            print('######### Roaming #########')

            # sample only as many negatives as there are positives
            indices = range(len(ys))
            pos_indices = [i for i in indices if ys[i] == 1]
            neg_indices = [i for i in indices if ys[i] == 0]
            lower = min(len(pos_indices), len(neg_indices))
            neg_indices = random.sample(neg_indices, lower)
            pos_indices = random.sample(pos_indices, lower)

            cut_embs = [embs[i] for i in neg_indices] + [embs[i] for i in pos_indices]
            cut_ys = [ys[i] for i in neg_indices] + [ys[i] for i in pos_indices]

            feature_embs = torch.stack([e[0].detach().cpu() for e in cut_embs])
            scaler = preprocessing.StandardScaler().fit(feature_embs)
            feature_embs = scaler.transform(feature_embs)
            print(np.array(feature_embs).shape, np.array(ys).shape)

            lin_class = LinearSVC(max_iter=50000, dual='auto', class_weight='balanced').fit(np.array(feature_embs), np.array(cut_ys))
            lin_class.coef_ = torch.tensor(lin_class.coef_, dtype=torch.double)
            lin_class.coef_ = (lin_class.coef_.flatten() / (lin_class.coef_.flatten().norm())).unsqueeze(0)


            rng_prompt = random.choice(prompt_list)

            w = 1# if len(embs) % 2 == 0 else 0
            im_emb = w * lin_class.coef_.to(device='cuda', dtype=torch.float16)
            prompt= 'an image' if glob_idx % 2 == 0 else rng_prompt
            print(prompt)

            image = pipe(
            prompt=prompt,
            ip_adapter_emb=im_emb,
            height=1024,
            width=1024,
            num_inference_steps=8,
            guidance_scale=0,
            ).images

            im_emb, _ = pipe.encode_image(
                image[0], 'cuda', 1, output_hidden_state
            )
            #im_emb = patch_encode_image(image[0])

            embs.append(im_emb)

            torch.save(lin_class.coef_, f'./{start_time}.pt')
            return image[0]









def start(_):
    return [
            gr.Button(value='Like', interactive=True), 
            gr.Button(value='Neither', interactive=True), 
            gr.Button(value='Dislike', interactive=True),
            gr.Button(value='Start', interactive=False),
            next_image()
            ]


def choose(choice):
    if choice == 'Like':
        choice = 1
    elif choice == 'Neither':
        _ = embs.pop(-1)
        return next_image()
    else:
        choice = 0
    ys.append(choice)
    return next_image()

css = "div#output-image {height: 768px !important; width: 768px !important; margin:auto;}"
with gr.Blocks(css=css) as demo:
    with gr.Row():
        html = gr.HTML('''<div style='text-align:center; font-size:32'>You will callibrate for several prompts and then roam.</ div>''')
    with gr.Row(elem_id='output-image'):
        img = gr.Image(interactive=False, elem_id='output-image',)
    with gr.Row(equal_height=True):
        b3 = gr.Button(value='Dislike', interactive=False,)
        b2 = gr.Button(value='Neither', interactive=False,)
        b1 = gr.Button(value='Like', interactive=False,)
        b1.click(
        choose, 
        [b1],
        [img]
        )
        b2.click(
        choose, 
        [b2],
        [img]
        )
        b3.click(
        choose, 
        [b3],
        [img]
        )
    with gr.Row():
        b4 = gr.Button(value='Start')
        b4.click(start,
                 [b4],
                 [b1, b2, b3, b4, img,])

demo.launch(server_name="0.0.0.0")  # Share your demo with just 1 extra parameter 🚀

