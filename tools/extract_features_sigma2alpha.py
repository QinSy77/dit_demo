import os
from pathlib import Path
import sys
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
from PIL import Image
import torch
from torchvision import transforms as T
import numpy as np
import json
from tqdm import tqdm
import argparse
import threading
from queue import Queue
from torch.utils.data import DataLoader, RandomSampler
from accelerate import Accelerator
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets.folder import default_loader
from transformers import T5Tokenizer, T5EncoderModel
from diffusion.model.t5 import T5Embedder
from diffusers.models import AutoencoderKL
from diffusion.data.datasets.InternalData import InternalData
from diffusion.utils.misc import SimpleTimer
from diffusion.utils.data_sampler import AspectRatioBatchSampler
from diffusion.data.builder import DATASETS
from diffusion.data.datasets.utils import *


def get_closest_ratio(height: float, width: float, ratios: dict):
    aspect_ratio = height / width
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return ratios[closest_ratio], float(closest_ratio)


@DATASETS.register_module()
class DatasetMS(InternalData):
    def __init__(self, root,image_list_json=None, transform=None, resolution=1024, load_vae_feat=False, aspect_ratio_type=None, start_index=0, end_index=100000000, **kwargs):
        if image_list_json is None:
            image_list_json = ['data_info.json']
        #assert os.path.isabs(root), 'root must be a absolute path'
        self.root = root
        self.img_dir_name = 'InternalImgs'        # need to change to according to your data structure
        self.json_dir_name = 'InternalData'        # need to change to according to your data structure
        self.transform = transform
        self.load_vae_feat = load_vae_feat
        self.resolution = resolution
        self.meta_data_clean = []
        self.img_samples = []
        self.txt_feat_samples = []
        self.aspect_ratio = aspect_ratio_type
        assert self.aspect_ratio in [ASPECT_RATIO_1024, ASPECT_RATIO_512]
        self.ratio_index = {}
        self.ratio_nums = {}
        for k, v in self.aspect_ratio.items():
            self.ratio_index[float(k)] = []     # used for self.getitem
            self.ratio_nums[float(k)] = 0      # used for batch-sampler

        image_list_json = image_list_json if isinstance(image_list_json, list) else [image_list_json]

        for json_file in image_list_json:
            meta_data = self.load_json( json_file)
            meta_data_clean = [item for item in meta_data if item['ratio'] <= 4]
            self.meta_data_clean.extend(meta_data_clean)
            self.img_samples.extend([os.path.join(item['path']) for item in meta_data_clean])
        


        self.img_samples = self.img_samples[start_index: end_index]
        # scan the dataset for ratio static
        for i, info in enumerate(self.meta_data_clean[:len(self.meta_data_clean)//3]):
            ori_h, ori_w = info['height'], info['width']
            closest_size, closest_ratio = get_closest_ratio(ori_h, ori_w, self.aspect_ratio)
            self.ratio_nums[closest_ratio] += 1
            if len(self.ratio_index[closest_ratio]) == 0:
                self.ratio_index[closest_ratio].append(i)

        # Set loader and extensions
        if self.load_vae_feat:
            raise ValueError("No VAE loader here")
        self.loader = default_loader

    def __getitem__(self, idx):
        data_info = {}
        for _ in range(20):
            try:
                img_path = self.img_samples[idx]
                img = self.loader(img_path)
                if self.transform:
                    img = self.transform(img)
                # Calculate closest aspect ratio and resize & crop image[w, h]
                if isinstance(img, Image.Image):
                    h, w = (img.size[1], img.size[0])
                    assert h, w == (self.meta_data_clean[idx]['height'], self.meta_data_clean[idx]['width'])
                    closest_size, closest_ratio = get_closest_ratio(h, w, self.aspect_ratio)
                    closest_size = list(map(lambda x: int(x), closest_size))
                    transform = T.Compose([
                        T.Lambda(lambda img: img.convert('RGB')),
                        T.Resize(closest_size, interpolation=InterpolationMode.BICUBIC),  # Image.BICUBIC
                        T.CenterCrop(closest_size),
                        T.ToTensor(),
                        T.Normalize([.5], [.5]),
                    ])
                    img = transform(img)
                    data_info['img_hw'] = torch.tensor([h, w], dtype=torch.float32)
                    data_info['aspect_ratio'] = closest_ratio
                # change the path according to your data structure
                #return img, '_'.join(self.img_samples[idx].rsplit('/', 2)[-2:]) # change from 'serial-number-of-dir/serial-number-of-image.png' ---> 'serial-number-of-dir_serial-number-of-image.png'
                return img, self.img_samples[idx].rsplit('/', 2)[-1]
            except Exception as e:
                print(f"Error details: {str(e)}")
                idx = np.random.randint(len(self))
        raise RuntimeError('Too many bad data.')

    def get_data_info(self, idx):
        data_info = self.meta_data_clean[idx]
        return {'height': data_info['height'], 'width': data_info['width']}


def extract_caption_t5_do(q):
    while not q.empty():
        item = q.get()
        extract_caption_t5_job(item)
        q.task_done()


def extract_caption_t5_job(item):
    global mutex
    global t5
    global t5_save_dir
    global count
    global total_item

    with torch.no_grad():
        # make sure the save path is unique here
        save_path = os.path.join(t5_save_dir, f"{Path(item['path']).stem}")
        if os.path.exists(save_path + ".npz"):
            count += 1
            return

        caption = item[args.caption_label].strip()
        if isinstance(caption, str):
            caption = [caption]

        if os.path.exists(f"{save_path}.npz"):
            return

        try:
            mutex.acquire()
            
            mutex.release()
            mutex.acquire()
            caption_emb, emb_mask = t5.get_text_embeddings(caption)
            mutex.release()


            emb_dict = {
                'caption_feature': caption_emb.float().cpu().data.numpy(),
                'attention_mask': emb_mask.cpu().data.numpy(),
            }



            
            os.umask(0o000)  # file permission: 666; dir permission: 777
            np.savez_compressed(save_path, **emb_dict)
            count += 1
        except Exception as e:
            print(e)
    print(f"CUDA: {os.environ['CUDA_VISIBLE_DEVICES']}, processed: {count}/{total_item}, User Prompt = {args.caption_label}, token length: {args.max_length}, saved at: {t5_save_dir}")


def extract_caption_t5():
    global t5
    global t5_save_dir
    global count
    global total_item

    t5 = T5Embedder(device="cuda", local_cache=True, dir_or_name=f'/mnt/workspace/qinshiyang/.cache/huggingface/hub/models--PixArt-alpha--PixArt-alpha/snapshots/e5009837bc701bd72b97333e3b9417b516fd673f/t5-v1_1-xxl', model_max_length=120)
    t5_save_dir = args.t5_save_root
    

   
    count = 0

    #t5_save_dir = os.path.join(args.t5_save_root, f"{args.caption_label}_caption_features_new".replace('prompt_', ''))
    os.umask(0o000)  # file permission: 666; dir permission: 777
    os.makedirs(t5_save_dir, exist_ok=True)

    train_data_json = json.load(open(args.t5_json_path, 'r'))
    train_data = train_data_json[args.start_index: args.end_index]
    total_item = len(train_data)

    global mutex
    mutex = threading.Lock()
    jobs = Queue()

    for item in tqdm(train_data):
        jobs.put(item)

    for _ in range(20):
        worker = threading.Thread(target=extract_caption_t5_do, args=(jobs,))
        worker.start()

    jobs.join()


def extract_img_vae(bs):
    print("Starting")
    accelerator = Accelerator(mixed_precision='fp16')
    vae = AutoencoderKL.from_pretrained(f'{args.vae_models_dir}', torch_dtype=torch.float16).to(device)
    print('VAE Loaded')

    vae_save_dir = f'{args.vae_save_root}/img_sdxl_vae_features_{image_resize}resolution_new'
    os.umask(0o000)  # file permission: 666; dir permission: 777
    os.makedirs(vae_save_dir, exist_ok=True)
    interpolation = InterpolationMode.BILINEAR
    if image_resize in [2048, 2880]:
        interpolation = InterpolationMode.LANCZOS
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB')),
        T.Resize(image_resize, interpolation=interpolation),
        T.CenterCrop(image_resize),
        T.ToTensor(),
        T.Normalize([.5], [.5]),
    ])
    signature = ''
    
    aspect_ratio_type = ASPECT_RATIO_1024 if image_resize == 1024 else ASPECT_RATIO_512
    dataset = DatasetMS(args.dataset_root,image_list_json=[args.json_file], transform=transform, sample_subset=None,
                        aspect_ratio_type=aspect_ratio_type, start_index=args.start_index, end_index=args.end_index)
    
    dataloader = DataLoader(dataset, batch_size=bs, num_workers=13, pin_memory=True)
    dataloader = accelerator.prepare(dataloader, )

    inference(vae, dataloader, signature=signature, work_dir=vae_save_dir)
    accelerator.wait_for_everyone()

    return


def save_results(results, paths, signature, work_dir):
    timer = SimpleTimer(len(results), log_interval=100, desc=f"Saving at {work_dir}")
    # save to npy
    new_paths = []
    new_folder = signature
    save_folder = os.path.join(work_dir, new_folder)
    os.makedirs(save_folder, exist_ok=True)
    os.umask(0o000)  # file permission: 666; dir permission: 777
    for res, p in zip(results, paths):
        file_name = p.split('.')[0] + '.npy'
        save_path = os.path.join(save_folder, file_name)
        if os.path.exists(save_path):
            continue
        new_paths.append(os.path.join(new_folder, file_name))
        np.save(save_path, res)
        timer.log()
    # save paths
    with open(os.path.join(work_dir, f"VAE-{signature}.txt"), 'a+') as f:
        f.write('\n'.join(new_paths))


def inference(vae, dataloader, signature, work_dir):
    timer = SimpleTimer(len(dataloader), log_interval=100, desc=f"VAE-Inference")

    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                posterior = vae.encode(batch[0]).latent_dist
                results = torch.cat([posterior.mean, posterior.std], dim=1).detach().cpu().numpy()
        path = batch[1]
        save_results(results, path, signature=signature, work_dir=work_dir)
        timer.log()


def extract_img_vae_multiscale(bs=1):

    assert image_resize in [512, 1024, 2048, 2880]
    work_dir = f"{os.path.abspath(args.vae_save_root)}/img_sdxl_vae_features_{image_resize}resolution_ms_new"
    os.umask(0o000)  # file permission: 666; dir permission: 777
    os.makedirs(work_dir, exist_ok=True)
    accelerator = Accelerator(mixed_precision='fp16')
    vae = AutoencoderKL.from_pretrained(f'{args.vae_models_dir}').to(device)

    signature = ''

    aspect_ratio_type = eval(f"ASPECT_RATIO_{image_resize}")
    print(f"Aspect Ratio Here: {aspect_ratio_type}")

    # dataset = DatasetExtract(
    #     args.dataset_root, image_list_json=[args.vae_json_file], transform=None, sample_subset=None,
    #     aspect_ratio_type=aspect_ratio_type, start_index=args.start_index, end_index=args.end_index,
    #     work_dir=os.path.join(work_dir, signature)
    # )

    aspect_ratio_type = ASPECT_RATIO_1024 if image_resize == 1024 else ASPECT_RATIO_512
    dataset = DatasetMS(args.dataset_root,image_list_json=[args.vae_json_file], transform=None, sample_subset=None,
                        aspect_ratio_type=aspect_ratio_type, start_index=args.start_index, end_index=args.end_index)

    # create AspectRatioBatchSampler
    sampler = AspectRatioBatchSampler(sampler=RandomSampler(dataset), dataset=dataset, batch_size=bs, aspect_ratios=dataset.aspect_ratio, ratio_nums=dataset.ratio_nums)

    # create DataLoader
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=13, pin_memory=True)
    dataloader = accelerator.prepare(dataloader, )

    inference(vae, dataloader, signature=signature, work_dir=work_dir)
    accelerator.wait_for_everyone()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_t5_feature_extract", default=True, help="run t5 feature extracting")
    parser.add_argument("--run_vae_feature_extract", default=True, help="run VAE feature extracting")
    parser.add_argument('--start_index', default=0, type=int)
    parser.add_argument('--end_index', default=50000000, type=int)
    
    ### vae feauture extraction
    parser.add_argument("--multi_scale",default=True, help="multi-scale feature extraction")
    parser.add_argument("--img_size", default=512, type=int, help="image scale for VAE feature extraction")
    parser.add_argument('--dataset_root', default='', type=str)
    parser.add_argument('--vae_json_file', default="/mnt/workspace/qinshiyang/PixArt-alpha/data/200data_sd15_sigma/partition/200data_info.json",type=str)    # relative to args.dataset_root
    parser.add_argument(
        '--vae_models_dir', default='stabilityai/sd-vae-ft-ema', type=str
    )
    parser.add_argument(
        '--vae_save_root', default='/mnt/workspace/qinshiyang/PixArt-alpha/data/200data_sd15_sigma/img_vae_features',
        type=str
    )

    ### for t5 feature
    parser.add_argument("--max_length", default=120, type=int, help="max token length for T5")
    parser.add_argument('--t5_json_path', default="/mnt/workspace/qinshiyang/PixArt-alpha/data/200data_sd15_sigma/partition/200data_info.json",type=str)    # absolute path or relative to this project
    parser.add_argument(
        '--t5_models_dir', default='PixArt-alpha/PixArt-XL-2-1024-MS', type=str
    )
    parser.add_argument('--caption_label', default='prompt', type=str)
    parser.add_argument('--t5_save_root', default="/mnt/workspace/qinshiyang/PixArt-alpha/data/200data_sd15_sigma/caption_feature_wmask", type=str)
    return parser.parse_args()


if __name__ == '__main__':

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '4' 

    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_resize = args.img_size

    file_need_mkdir = [args.t5_save_root , args.vae_save_root]
    for dir_path in file_need_mkdir:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"创建了路径：{dir_path}")

    # prepare extracted caption t5 features for training
    if args.run_t5_feature_extract:
        print("开始t5编码")
        extract_caption_t5()
        print("t5编码结束")

    # prepare extracted image vae features for training
    if args.run_vae_feature_extract:
        print("开始vae编码")
        if args.multi_scale:
            assert args.img_size  in [512, 1024, 2048, 2880],\
                "Multi Scale VAE feature is not for 256px in PixArt-Sigma."
            print('Extracting Multi-scale Image Resolution based on %s' % image_resize)
            extract_img_vae_multiscale(bs=1)    # recommend bs = 1 for AspectRatioBatchSampler
        else:
            assert args.img_size == 256,\
                f"Single Scale VAE feature is only for 256px in PixArt-Sigma. NOT for {args.img_size}px"
            print('Extracting Single Image Resolution %s' % image_resize)
            extract_img_vae(bs=2)
        print("vae编码结束")

    print("Done")