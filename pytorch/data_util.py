
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random
import math
from typing import List

CROP_PROPORTION = 0.875  # Standard for ImageNet

def random_apply(func, p, x):
    if random.random() < p:
        return func(x)
    return x

def random_brightness(image, max_delta, impl='simclrv2'):
    if impl == 'simclrv2':
        factor = random.uniform(max(1.0 - max_delta, 0), 1.0 + max_delta)
        return image * factor
    elif impl == 'simclrv1':
        return TF.adjust_brightness(image, random.uniform(1.0 - max_delta, 1.0 + max_delta))
    else:
        raise ValueError(f"Unknown impl {impl} for random brightness.")

def to_grayscale(image, keep_channels=True):
    image = TF.to_grayscale(image, num_output_channels=3 if keep_channels else 1)
    return image

def color_jitter(image, strength, random_order=True, impl='simclrv2'):
    brightness = 0.8 * strength
    contrast = 0.8 * strength
    saturation = 0.8 * strength
    hue = 0.2 * strength
    if random_order:
        return color_jitter_rand(image, brightness, contrast, saturation, hue, impl)
    else:
        return color_jitter_nonrand(image, brightness, contrast, saturation, hue, impl)

def color_jitter_nonrand(image, brightness, contrast, saturation, hue, impl='simclrv2'):
    transforms = [
        lambda img: random_brightness(img, brightness, impl),
        lambda img: TF.adjust_contrast(img, random.uniform(1 - contrast, 1 + contrast)),
        lambda img: TF.adjust_saturation(img, random.uniform(1 - saturation, 1 + saturation)),
        lambda img: TF.adjust_hue(img, random.uniform(-hue, hue)),
    ]
    for t in transforms:
        image = t(image)
        image = torch.clamp(image, 0., 1.)
    return image

def color_jitter_rand(image, brightness, contrast, saturation, hue, impl='simclrv2'):
    transforms = [
        lambda img: random_brightness(img, brightness, impl),
        lambda img: TF.adjust_contrast(img, random.uniform(1 - contrast, 1 + contrast)),
        lambda img: TF.adjust_saturation(img, random.uniform(1 - saturation, 1 + saturation)),
        lambda img: TF.adjust_hue(img, random.uniform(-hue, hue)),
    ]
    random.shuffle(transforms)
    for t in transforms:
        image = t(image)
        image = torch.clamp(image, 0., 1.)
    return image

def _compute_crop_shape(image_height, image_width, aspect_ratio, crop_proportion):
    if aspect_ratio > image_width / image_height:
        crop_height = round(crop_proportion / aspect_ratio * image_width)
        crop_width = round(crop_proportion * image_width)
    else:
        crop_height = round(crop_proportion * image_height)
        crop_width = round(crop_proportion * aspect_ratio * image_height)
    return crop_height, crop_width

def center_crop(image, height, width, crop_proportion):
    image_height, image_width = image.shape[-2:]
    crop_height, crop_width = _compute_crop_shape(
        image_height, image_width, width / height, crop_proportion
    )
    top = (image_height - crop_height + 1) // 2
    left = (image_width - crop_width + 1) // 2
    image = TF.crop(image, top, left, crop_height, crop_width)
    image = TF.resize(image, [height, width], interpolation=TF.InterpolationMode.BICUBIC)
    image = torch.clamp(image, 0.0, 1.0)
    return image

def distorted_bounding_box_crop(image, aspect_ratio_range=(0.75, 1.33), area_range=(0.05, 1.0), max_attempts=100):
    image_height, image_width = image.shape[-2:]
    image_area = image_height * image_width
    for _ in range(max_attempts):
        area = random.uniform(*area_range) * image_area
        aspect_ratio = random.uniform(*aspect_ratio_range)
        crop_height = int(round(math.sqrt(area / aspect_ratio)))
        crop_width = int(round(crop_height * aspect_ratio))
        if crop_height <= image_height and crop_width <= image_width:
            top = random.randint(0, image_height - crop_height)
            left = random.randint(0, image_width - crop_width)
            return TF.crop(image, top, left, crop_height, crop_width)
    return image

def crop_and_resize(image, height, width):
    aspect_ratio = width / height
    image = distorted_bounding_box_crop(
        image,
        aspect_ratio_range=(3. / 4 * aspect_ratio, 4. / 3. * aspect_ratio),
        area_range=(0.08, 1.0),
        max_attempts=100
    )
    image = TF.resize(image, [height, width], interpolation=TF.InterpolationMode.BICUBIC)
    return torch.clamp(image, 0.0, 1.0)

def gaussian_blur(image, kernel_size, sigma, padding='same'):
    if kernel_size % 2 == 0:
        kernel_size += 1
    radius = kernel_size // 2
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    blur_filter = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    blur_filter = blur_filter / blur_filter.sum()
    blur_h = blur_filter.view(1, 1, 1, -1)
    blur_v = blur_filter.view(1, 1, -1, 1)
    num_channels = image.shape[0]
    blur_h = blur_h.expand(num_channels, 1, 1, -1)
    blur_v = blur_v.expand(num_channels, 1, -1, 1)
    image = image.unsqueeze(0)
    if padding == 'same':
        image = F.conv2d(image, blur_h, padding=(0, radius), groups=num_channels)
        image = F.conv2d(image, blur_v, padding=(radius, 0), groups=num_channels)
    else:
        image = F.conv2d(image, blur_h, groups=num_channels)
        image = F.conv2d(image, blur_v, groups=num_channels)
    return image.squeeze(0)

def random_crop_with_resize(image, height, width, p=1.0):
    def _transform(image):
        return crop_and_resize(image, height, width)
    return random_apply(_transform, p, image)

def random_color_jitter(image, p=1.0, strength=1.0, impl='simclrv2'):
    def _transform(image):
        image = color_jitter(image, strength=strength, impl=impl)
        return random_apply(lambda x: to_grayscale(x, keep_channels=True), 0.2, image)
    return random_apply(_transform, p, image)

def random_blur(image, height, width, p=1.0):
    def _transform(image):
        sigma = random.uniform(0.1, 2.0)
        return gaussian_blur(image, kernel_size=height//10, sigma=sigma)
    return random_apply(_transform, p, image)

def batch_random_blur(images_list, height, width, blur_probability=0.5):
    new_images_list = []
    for images in images_list:
        images_new = random_blur(images, height, width, p=1.0)
        selector = (torch.rand(images.shape[0], 1, 1, 1) < blur_probability).to(images.device)
        images = torch.where(selector, images_new, images)
        images = torch.clamp(images, 0., 1.)
        new_images_list.append(images)
    return new_images_list

def preprocess_for_train(image, height, width, color_jitter_strength=0., crop=True, flip=True, impl='simclrv2'):
    if crop:
        image = random_crop_with_resize(image, height, width)
    if flip and random.random() < 0.5:
        image = TF.hflip(image)
    if color_jitter_strength > 0:
        image = random_color_jitter(image, strength=color_jitter_strength, impl=impl)
    image = image.permute(1, 2, 0)
    return torch.clamp(image, 0., 1.)

def preprocess_for_eval(image, height, width, crop=True):
    if crop:
        image = center_crop(image, height, width, CROP_PROPORTION)
    image = image.permute(1, 2, 0)
    return torch.clamp(image, 0., 1.)

def preprocess_image(image, height, width, is_training=False, color_jitter_strength=0., test_crop=True):
    if image.dtype == torch.uint8:
        image = image.float() / 255.0
    if is_training:
        return preprocess_for_train(image, height, width, color_jitter_strength)
    else:
        return preprocess_for_eval(image, height, width, test_crop)
