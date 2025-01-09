# import argparse
# import glob
# import h5py
# import numpy as np
# import PIL.Image as pil_image
# from utils import convert_rgb_to_y

# def train(args):
#     h5_file = h5py.File(args.output_path, 'w')

#     # 创建组
#     lr_group = h5_file.create_group('lr')
#     hr_group = h5_file.create_group('hr')

#     patch_idx = 0

#     for image_path in sorted(glob.glob('{}/*'.format(args.images_dir))):
#         hr = pil_image.open(image_path).convert('RGB')
#         hr_width = (hr.width // args.scale) * args.scale
#         hr_height = (hr.height // args.scale) * args.scale
#         hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
#         lr = hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
#         lr = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
#         hr = np.array(hr).astype(np.float32)
#         lr = np.array(lr).astype(np.float32)
#         hr = convert_rgb_to_y(hr)
#         lr = convert_rgb_to_y(lr)

#         for i in range(0, lr.shape[0] - args.patch_size + 1, args.stride):
#             for j in range(0, lr.shape[1] - args.patch_size + 1, args.stride):
#                 lr_patch = lr[i:i + args.patch_size, j:j + args.patch_size]
#                 hr_patch = hr[i:i + args.patch_size, j:j + args.patch_size]

#                 lr_group.create_dataset(str(patch_idx), data=lr_patch)
#                 hr_group.create_dataset(str(patch_idx), data=hr_patch)
#                 patch_idx += 1

#     h5_file.close()

# def eval(args):
#     h5_file = h5py.File(args.output_path, 'w')

#     lr_group = h5_file.create_group('lr')
#     hr_group = h5_file.create_group('hr')

#     for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(args.images_dir)))):
#         hr = pil_image.open(image_path).convert('RGB')
#         hr_width = (hr.width // args.scale) * args.scale
#         hr_height = (hr.height // args.scale) * args.scale
#         hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
#         lr = hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
#         lr = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
#         hr = np.array(hr).astype(np.float32)
#         lr = np.array(lr).astype(np.float32)
#         hr = convert_rgb_to_y(hr)
#         lr = convert_rgb_to_y(lr)

#         lr_group.create_dataset(str(i), data=lr)
#         hr_group.create_dataset(str(i), data=hr)

#     h5_file.close()

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     # parser.add_argument('--images-dir', type=str, default='dataset/T91-train')
#     # parser.add_argument('--output-path', type=str, default='dataset/train_data.h5')
#     # parser.add_argument('--eval', action='store_true', default=False)

#     parser.add_argument('--images-dir', type=str, default='dataset/Set5-test')
#     parser.add_argument('--output-path', type=str, default='dataset/test_data.h5')
#     parser.add_argument('--eval', action='store_true', default=True)

#     parser.add_argument('--patch-size', type=int, default=33)
#     parser.add_argument('--stride', type=int, default=14)
#     parser.add_argument('--scale', type=int, default=4)
#     args = parser.parse_args()

#     if not args.eval:
#         train(args)
#     else:
#         eval(args)

#-----------------------------------------------------------------------------------RGB代码


import argparse
import glob
import h5py
import numpy as np
from PIL import Image
import os

def process_data(image_dir, output_path, patch_size, stride, scale, is_training=True):
    print(f"Processing {'training' if is_training else 'test'} data...")
    print(f"Input directory: {image_dir}")
    print(f"Output file: {output_path}")
    
    h5_file = h5py.File(output_path, 'w')
    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')
    
    patch_idx = 0
    image_files = sorted(glob.glob(f'{image_dir}/*'))
    print(f"Found {len(image_files)} images")
    
    for i, image_path in enumerate(image_files):
        hr = Image.open(image_path).convert('RGB')
        hr_width = (hr.width // scale) * scale
        hr_height = (hr.height // scale) * scale
        
        hr = hr.resize((hr_width, hr_height), resample=Image.BICUBIC)
        lr = hr.resize((hr_width // scale, hr_height // scale), resample=Image.BICUBIC)
        lr = lr.resize((lr.width * scale, lr.height * scale), resample=Image.BICUBIC)
        
        hr = np.array(hr).astype(np.float32) / 255.0
        lr = np.array(lr).astype(np.float32) / 255.0
        
        if is_training:
            # 对训练数据进行patch裁剪
            for i in range(0, lr.shape[0] - patch_size + 1, stride):
                for j in range(0, lr.shape[1] - patch_size + 1, stride):
                    lr_patch = lr[i:i + patch_size, j:j + patch_size, :]
                    hr_patch = hr[i:i + patch_size, j:j + patch_size, :]
                    
                    lr_group.create_dataset(str(patch_idx), data=lr_patch)
                    hr_group.create_dataset(str(patch_idx), data=hr_patch)
                    patch_idx += 1
                    
                    if patch_idx % 100 == 0:
                        print(f"Processed {patch_idx} patches")
        else:
            # 对测试数据直接存储
            lr_group.create_dataset(str(i), data=lr)
            hr_group.create_dataset(str(i), data=hr)
            print(f"Processed test image {i+1}/{len(image_files)}")
    
    h5_file.close()
    print(f"Saved {patch_idx if is_training else i+1} {'patches' if is_training else 'images'}")
    return patch_idx if is_training else i+1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', type=str, default='dataset/T91-train')
    parser.add_argument('--test-dir', type=str, default='dataset/Set5-test')
    parser.add_argument('--train-output', type=str, default='dataset/train_data.h5')
    parser.add_argument('--test-output', type=str, default='dataset/test_data.h5')
    parser.add_argument('--patch-size', type=int, default=96)
    parser.add_argument('--stride', type=int, default=48)
    parser.add_argument('--scale', type=int, default=4)
    args = parser.parse_args()
    
    # 确保目录存在
    os.makedirs(os.path.dirname(args.train_output), exist_ok=True)
    os.makedirs(os.path.dirname(args.test_output), exist_ok=True)
    
    # 处理训练数据
    print("\nProcessing training data...")
    num_train = process_data(args.train_dir, args.train_output, 
                           args.patch_size, args.stride, args.scale, 
                           is_training=True)
    
    # 处理测试数据
    print("\nProcessing test data...")
    num_test = process_data(args.test_dir, args.test_output, 
                          args.patch_size, args.stride, args.scale, 
                          is_training=False)
    
    print("\nData preparation completed!")
    print(f"Generated {num_train} training patches")
    print(f"Processed {num_test} test images")
    
    # 验证生成的数据
    print("\nVerifying generated data...")
    with h5py.File(args.train_output, 'r') as f:
        print(f"Training data - LR shape: {f['lr']['0'][:].shape}, HR shape: {f['hr']['0'][:].shape}")
    
    with h5py.File(args.test_output, 'r') as f:
        print(f"Test data - LR shape: {f['lr']['0'][:].shape}, HR shape: {f['hr']['0'][:].shape}")

if __name__ == '__main__':
    main()