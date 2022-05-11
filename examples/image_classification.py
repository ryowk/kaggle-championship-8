import random
from enum import Enum, auto
from pathlib import Path

import albumentations
import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import timm
import torch
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

wandb_logger = WandbLogger()

device = torch.device('cuda')

NUM_CLASSES = 6

pl.seed_everything(42)


class Scheduler(Enum):
    Nothing = auto()
    CosineAnnealingLR = auto()
    CosineAnnealingWarmRestarts = auto()


class CFG:
    # timm benchmark: https://github.com/rwightman/pytorch-image-models/blob/master/results/results-imagenet.csv
    model_name = "swin_base_patch4_window12_384"
    image_size = 384
    fp16 = False
    num_epochs = 50
    batch_size = 8
    num_workers = 32
    accumulate_grad_batches = 2
    lr = 1e-4
    weight_decay = 1e-6
    early_stopping_patience = 5

    scheduler_type = Scheduler.CosineAnnealingWarmRestarts

    class CosineAnnealingLRParams:
        T_max = 10
        eta_min = 1e-6

    class CosineAnnealingWarmRestartsParams:
        T_0 = 10
        eta_min = 1e-6


# https://github.com/OrKatz7/1st-place-Don-t-stop-until-you-drop
class CutoutV2(albumentations.DualTransform):
    def __init__(
        self,
        num_holes=8,
        max_h_size=8,
        max_w_size=8,
        fill_value=0,
        always_apply=False,
        p=0.5,
    ):
        super(CutoutV2, self).__init__(always_apply, p)
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.fill_value = fill_value

    def apply(self, image, fill_value=0, holes=(), **params):
        return albumentations.functional.cutout(image, holes, fill_value)

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        height, width = img.shape[:2]

        holes = []
        for _n in range(self.num_holes):
            y = random.randint(0, height)
            x = random.randint(0, width)

            y1 = np.clip(y - self.max_h_size // 2, 0, height)
            y2 = np.clip(y1 + self.max_h_size, 0, height)
            x1 = np.clip(x - self.max_w_size // 2, 0, width)
            x2 = np.clip(x1 + self.max_w_size, 0, width)
            holes.append((x1, y1, x2, y2))

        return {"holes": holes}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return ("num_holes", "max_h_size", "max_w_size")


def get_transforms(data):
    if data == 'train':
        return albumentations.Compose([
            albumentations.Resize(CFG.image_size, CFG.image_size),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=5, border_mode=0, p=0.75),
            albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
            albumentations.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=40, val_shift_limit=0, p=0.75),
            CutoutV2(
                max_h_size=int(CFG.image_size * 0.2),
                max_w_size=int(CFG.image_size * 0.2),
                num_holes=1,
                p=0.75
            ),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    elif data == 'test':
        return albumentations.Compose([
            albumentations.Resize(CFG.image_size, CFG.image_size),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    else:
        raise NotImplementedError


input_dir = Path('../input')
output_dir = Path('output/0/')
output_dir.mkdir(parents=True, exist_ok=True)


# ラベル付きのものはTrainDataset, そうでないものはTestDatasetとして、
# 渡すtransformsによってtrain, validation, testを切り替えているが、
# TrainDataset, ValidationDataset, TestDatasetに分けたほうが素直かも
class TrainDataset(Dataset):
    def __init__(self, paths, labels, transforms):
        self.paths = [str(path) for path in paths]
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        return image, self.labels[idx]


# TTAをやる場合
# 水増しの方法が決定的な場合はflipなどをやった分だけtupleで返せば良いはず
# 水増しの方法が非決定的な場合は何回も推論するだけで良い
class TestDataset(Dataset):
    def __init__(self, paths, transforms):
        self.paths = [str(path) for path in paths]
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        return image


df_train = pd.read_csv(input_dir / 'train.csv')
df_test = pd.read_csv(input_dir / 'sample_submission.csv')

# このスクリプト中でvalidationとpredictionを両方行っている
# 命名規則
# prediction: train, test
# validation: tra, va
train_image_ids = df_train['image_id'].values
train_labels = df_train['class_6'].values

tra_image_ids, val_image_ids, tra_labels, val_labels = train_test_split(
    train_image_ids, train_labels, test_size=0.33, random_state=42, stratify=train_labels)
test_image_ids = df_test['image_id'].values

train_paths = [Path(input_dir / 'images/train_images') / image_id for image_id in train_image_ids]
train_dataset = TrainDataset(train_paths, train_labels, get_transforms('train'))
train_dataloader = DataLoader(
    train_dataset,
    batch_size=CFG.batch_size,
    shuffle=True,
    num_workers=CFG.num_workers,
    pin_memory=True,
    persistent_workers=True)
test_paths = [Path('../input/images/test_images') / image_id for image_id in test_image_ids]
test_dataset = TestDataset(test_paths, get_transforms('test'))
test_dataloader = DataLoader(
    test_dataset,
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=CFG.num_workers,
    pin_memory=True,
    persistent_workers=True)

tra_paths = [Path(input_dir / 'images/train_images') / image_id for image_id in tra_image_ids]
tra_dataset = TrainDataset(tra_paths, tra_labels, get_transforms('train'))
tra_dataloader = DataLoader(
    tra_dataset,
    batch_size=CFG.batch_size,
    shuffle=True,
    num_workers=CFG.num_workers,
    pin_memory=True,
    persistent_workers=True)
val_paths = [Path(input_dir / 'images/train_images') / image_id for image_id in val_image_ids]
val_dataset = TrainDataset(val_paths, val_labels, get_transforms('test'))
val_dataloader = DataLoader(
    val_dataset,
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=CFG.num_workers,
    pin_memory=True,
    persistent_workers=True)


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(CFG.model_name, pretrained=True, num_classes=NUM_CLASSES)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log("val/loss", loss)
        return {
            "labels": labels.detach().cpu(),
            "outputs": outputs.detach().cpu(),
        }

    def validation_epoch_end(self, val_step_outputs):
        labels = np.concatenate([out['labels'] for out in val_step_outputs])
        outputs = np.concatenate([out['outputs'] for out in val_step_outputs])
        preds = np.argmax(outputs, axis=1)
        score = f1_score(labels, preds, average='micro')
        self.log("val/score", score)

    def predict_step(self, batch, batch_idx):
        inputs = batch
        outputs = self(inputs)
        return outputs.detach().cpu()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
        if CFG.scheduler_type == Scheduler.Nothing:
            return optimizer

        if CFG.scheduler_type == Scheduler.CosineAnnealingLR:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=CFG.CosineAnnealingLRParams.T_max,
                eta_min=CFG.CosineAnnealingLRParams.eta_min,
                last_epoch=-1)
        elif CFG.scheduler_type == Scheduler.CosineAnnealingWarmRestarts:
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=CFG.CosineAnnealingWarmRestartsParams.T_0,
                eta_min=CFG.CosineAnnealingWarmRestartsParams.eta_min,
                last_epoch=-1)
        else:
            raise ValueError("invalid scheduler name")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
            }
        }


trainer_params = {
    "gpus": 1,
    "precision": 16 if CFG.fp16 else 32,
    "accumulate_grad_batches": CFG.accumulate_grad_batches,
    "max_epochs": CFG.num_epochs,
}
################################
# validation
################################
model = Model()
trainer = pl.Trainer(
    **trainer_params,
    logger=wandb_logger,
    callbacks=[
        EarlyStopping(monitor="val/score", mode='max', patience=CFG.early_stopping_patience),
    ],
)
trainer.fit(
    model,
    train_dataloaders=tra_dataloader,
    val_dataloaders=val_dataloader,
)

################################
# prediction
################################
model = Model()
trainer = pl.Trainer(**trainer_params)
trainer.fit(
    model,
    train_dataloaders=train_dataloader,
)
preds = trainer.predict(model, dataloaders=test_dataloader)
preds = np.concatenate(preds)


def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


for i, pred in enumerate(preds):
    preds[i] = softmax(pred)

df_prob = df_test.copy().drop('class_6', axis=1)
for i in range(NUM_CLASSES):
    df_prob[str(i)] = preds.T[i]
df_prob.to_csv(output_dir / 'prob.csv', index=False)

pred_class = np.argmax(preds, axis=1)
df_test['class_6'] = pred_class
df_test.to_csv(output_dir / 'test.csv', index=False)
