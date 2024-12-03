from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import torch
from PIL import Image, ImageOps
import os


def generate_realistic_road(path_from, filename, path_to):
    
    # Загрузка ControlNet модели для сегментации
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-seg", torch_dtype=torch.float16
    )

    # Загрузка основной модели
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.to("cuda")

    # Загрузка и обработка маски
    mask_image = load_image(path_from + filename).resize((512, 512))
    # mask_image = ImageOps.grayscale(mask_image).point(lambda x: 255 if x > 128 else 0, mode='1')

    # Установка параметров генерации
    prompt1 = "Only one motorway on the given mask around the forest. Vertical view from above."

    generated_image = pipe(
        prompt1,
        image=mask_image,
        num_inference_steps=50,
        guidance_scale=7.5,
        control_strength=0.9 # Регулирует влияние маски
    ).images[0]

    # Сохранение результата
    generated_image.save(path_to + filename)


# TODO: FOR ILYA сделать функцию которая читает 
# названия всех файлов из папки выдает список файлов

def get_list_masks(folder='./'):
    return [f for f in os.listdir(folder) if f.endswith('.png')]

if __name__ == '__main__':
    filename = "w_20 down_376_511 right_511_178.png"
    path_from = "./masks/"
    path_to = "./roads/"
    
    masks = get_list_masks(path_from)
    print(masks)
    for mask in masks:
        generate_realistic_road(path_from, mask, path_to)
