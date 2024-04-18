_base_ = ['~/dit_demo/PixArt-alpha/configs/PixArt_xl2_internal.py']  #有点像父类的意思
data_root = '/mnt/workspace/qinshiyang/PixArt-alpha/test/200data'
##root路径放配置信息
image_list_json = ['200data_info.json',]


###data_root + root +"partition"+ 配置文件中的文件路径就是最后读取的json文件路径
##当root 为绝对路径时， data_root失效  想修改请看～/PixArt-sigma/diffusion/data/datasets/InternalData.py

data = dict(type='InternalData', root=data_root, image_list_json=image_list_json, transform='default_train', load_vae_feat=True)

##root路径放配置信息
image_size = 512

# model setting
window_block_indexes = []
window_size=0
use_rel_pos=False
model = 'PixArt_XL_2'
fp32_attention = True
#load_from = None #表示预训练
load_from = "/mnt/workspace/qinshiyang/.cache/huggingface/hub/models--PixArt-alpha--PixArt-alpha/snapshots/e5009837bc701bd72b97333e3b9417b516fd673f/PixArt-XL-2-512x512.pth"
vae_pretrained = "stabilityai--sd-vae-ft-ema"
lewei_scale = 1.0

# training setting
use_fsdp=False   # if use FSDP mode
num_workers=1
train_batch_size = 100 # 32
num_epochs = 10000 # 3
gradient_accumulation_steps = 1
grad_checkpointing = True
gradient_clip = 0.01
optimizer = dict(type='AdamW', lr=2e-5, weight_decay=3e-2, eps=1e-10)
lr_schedule_args = dict(num_warmup_steps=1000)

eval_sampling_steps = 100
log_interval = 20

save_model_epochs=1000
save_model_steps=1500

work_dir = '/mnt/workspace/qinshiyang/PixArt-alpha/testoutputs_200/debug'
