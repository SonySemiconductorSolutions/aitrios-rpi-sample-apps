#
# Copyright 2024 Sony Semiconductor Solutions Corp. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import glob
import random
import shutil
import argparse
from PIL import Image
import numpy as np
import albumentations as A
import splitfolders

class ImageAugmentor:
    def __init__(self, input_dir, output_dir, class_name, images_per_class, transform_params):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.class_name = class_name
        self.images_per_class = images_per_class
        self.transform = self.create_transform(transform_params)

    @staticmethod
    def create_transform(params):
        return A.ReplayCompose([
            A.RandomBrightnessContrast(
                p=params['brightness_contrast_p'],
                brightness_limit=params['brightness_limit'],
                contrast_limit=params['contrast_limit'],
                brightness_by_max=params['brightness_by_max']
            ),
            A.HueSaturationValue(
                p=params['hue_saturation_p'],
                hue_shift_limit=params['hue_shift_limit'],
                sat_shift_limit=params['sat_shift_limit'],
                val_shift_limit=params['val_shift_limit']
            ),
            A.HorizontalFlip(p=params['horizontal_flip_p']),
        ])

    @staticmethod
    def rotate_image(image: Image.Image, angle: float) -> Image.Image:
        return image.rotate(angle, expand=True)

    def process_images(self):
        if self.class_name == 'all':
            self.process_all_classes()
        else:
            self.process_single_class()

    def process_all_classes(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "info"), exist_ok=True)

        for class_folder in os.listdir(self.input_dir):
            class_path = os.path.join(self.input_dir, class_folder, "*.bmp")
            image_list = glob.glob(class_path)

            output_image_dir = os.path.join(self.output_dir, "images", class_folder)
            output_info_dir = os.path.join(self.output_dir, "info", class_folder)

            os.makedirs(output_image_dir, exist_ok=True)
            os.makedirs(output_info_dir, exist_ok=True)

            for x in range(self.images_per_class):

                random_img = random.choice(image_list)
                self.process_single_image(random_img, output_image_dir, output_info_dir, x)

        self.split_dataset()

    def process_single_class(self):
        class_path = os.path.join(self.input_dir, self.class_name, "*.bmp")
        image_list = glob.glob(class_path)

        output_class_dir = os.path.join(self.output_dir, self.class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        for x in range(self.images_per_class):
            random_img = random.choice(image_list)
            self.process_single_image(random_img, output_class_dir, output_class_dir, x)

        shutil.make_archive(output_class_dir, 'zip', os.path.dirname(output_class_dir), os.path.basename(output_class_dir))
        shutil.rmtree(output_class_dir)

    def process_single_image(self, image_path, output_image_dir, output_info_dir, index):
        pillow_image = Image.open(image_path)
        image = np.array(pillow_image)
        transformed = self.transform(image=image)

        for rotation_angle in range(0, 360, 90):
            rotated_image = self.rotate_image(Image.fromarray(transformed["image"]), rotation_angle)
            image_name = f"{os.path.basename(image_path).replace('.bmp', '')}_deg_{rotation_angle:03d}.png"
            rotated_image.save(os.path.join(output_image_dir, f"{index:03d}_{image_name}"))

        with open(os.path.join(output_info_dir, f"{index:03d}_{os.path.basename(image_path).replace('.bmp', '.txt')}"), "w") as info_file:
            for tran in transformed['replay']['transforms']:
                info_file.write(tran['__class_fullname__'] + ": " + str(tran['params']) + "\n")

    def split_dataset(self):
        image_dir = os.path.join(self.output_dir, "images")
        splitfolders.ratio(image_dir, output=self.output_dir, seed=1337, ratio=(0.7, 0.2, 0.1))
        shutil.make_archive(self.output_dir, 'zip', self.output_dir)


def main():
    print("Dataset extender")
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-i", "--input", help="Path to images", required=True)
    arg_parser.add_argument("-o", "--output", help="Path to dataset", default="output")
    arg_parser.add_argument('-n', '--num', help='Images per class', default=100, type=int)
    arg_parser.add_argument('-c', '--class-name', help='Class to create images', default='all')
    arg_parser.add_argument('--brightness-contrast-p', type=float, default=0.8)
    arg_parser.add_argument('--brightness-limit', type=tuple, default=(-0.25, 0.25))
    arg_parser.add_argument('--contrast-limit', type=tuple, default=(-0.25, 0.25))
    arg_parser.add_argument('--brightness-by-max', type=bool, default=True)
    arg_parser.add_argument('--hue-saturation-p', type=float, default=0.3)
    arg_parser.add_argument('--hue-shift-limit', type=tuple, default=(-20, 20))
    arg_parser.add_argument('--sat-shift-limit', type=tuple, default=(-30, 30))
    arg_parser.add_argument('--val-shift-limit', type=tuple, default=(-20, 20))
    arg_parser.add_argument('--horizontal-flip-p', type=float, default=0.5)

    args = arg_parser.parse_args()

    transform_params = {
        'brightness_contrast_p': args.brightness_contrast_p,
        'brightness_limit': args.brightness_limit,
        'contrast_limit': args.contrast_limit,
        'brightness_by_max': args.brightness_by_max,
        'hue_saturation_p': args.hue_saturation_p,
        'hue_shift_limit': args.hue_shift_limit,
        'sat_shift_limit': args.sat_shift_limit,
        'val_shift_limit': args.val_shift_limit,
        'horizontal_flip_p': args.horizontal_flip_p,
    }

    augmentor = ImageAugmentor(
        input_dir=args.input,
        output_dir=args.output,
        class_name=args.class_name,
        images_per_class=args.num,
        transform_params=transform_params
    )

    augmentor.process_images()

if __name__ == "__main__":
    main()