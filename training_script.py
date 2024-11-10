
import torch
import yaml

from prithvi_100m.Prithvi import MaskedAutoencoderViT
from dataset import XBDDataset
from model import DamageSegmentation
from model_compiler import ModelCompiler
from config import MODEL_ARGS, CLASS_MAPPING


# load weights
weights_path = './prithvi_100m/Prithvi_100M.pt'
checkpoint = torch.load(weights_path, map_location='cpu')

# Drop the weights associated with the three IR channels which don't exist in this dataset
checkpoint['patch_embed.proj.weight'] = checkpoint['patch_embed.proj.weight'][:,:3]

# read model config
model_cfg_path = './prithvi_100m/Prithvi_100M_config.yaml'
with open(model_cfg_path) as f:
    model_config = yaml.safe_load(f)

model_args = model_config['model_args']
train_args = model_config['train_params']

# 2 frames: pre image and post image
model_args['num_frames'] = 2

model_args['img_size'] = 1024

model_args['in_chans'] = 3

# instantiate model
encoder_model = MaskedAutoencoderViT(**model_args)
encoder_model.eval()

# load weights into model
del checkpoint['pos_embed']
del checkpoint['decoder_pos_embed']
del checkpoint['decoder_pred.weight']
del checkpoint['decoder_pred.bias']
encoder_model.load_state_dict(checkpoint, strict=False)

segmentation_model = DamageSegmentation(
    encoder_model.forward_encoder
)

compiled_model = ModelCompiler(
    segmentation_model,
    out_dir=MODEL_ARGS['out_dir'],
    num_classes=MODEL_ARGS['num_classes'],
    class_mapping=CLASS_MAPPING,
    model_init_type=MODEL_ARGS['model_init_type'],
)

train_ds = XBDDataset('data/geotiffs/hold')
print(f'Training on {len(train_ds)} images')
train_data_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True)

val_ds = XBDDataset('data/geotiffs/hold')
print(f'Validating on {len(val_ds)} images')
validation_data_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)

loss_fn = torch.nn.CrossEntropyLoss(weight=torch.Tensor(MODEL_ARGS['loss_weights']))

compiled_model.fit(
    train_data_loader,
    validation_data_loader, 
    epochs=MODEL_ARGS['epochs'],
    optimizer_name=MODEL_ARGS['optimizer'],
    lr_init=MODEL_ARGS['LR'],
    lr_policy=MODEL_ARGS['LR_policy'], 
    loss_fn=loss_fn,
    checkpoint_interval=MODEL_ARGS['checkpoint_interval'],
    resume=MODEL_ARGS['resume'],
    resume_epoch=MODEL_ARGS['resume_epoch'],
    **MODEL_ARGS['LR_kwargs']
)