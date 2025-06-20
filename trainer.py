import os
import pandas as pd
from fastai.vision.all import *

### DEFS
MANIFESTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'manifests')
TRAIN_CSV = os.path.join(MANIFESTS_DIR, "training_manifest.csv")
VALID_CSV = os.path.join(MANIFESTS_DIR, "validation_manifest.csv")

BATCH_SIZE = 64
IMG_CROP_SIZE = 224
NUM_WORKERS = 16
###

def build_datablock():
    training_dataframe = pd.read_csv(TRAIN_CSV)
    validation_dataframe = pd.read_csv(VALID_CSV)

    # setup for datablock splitter
    training_dataframe['is_validation'] = False
    validation_dataframe['is_validation'] = True

    dataframe = pd.concat([training_dataframe, validation_dataframe])

    print(f"Dataframe shape: {dataframe.shape}")
    print(f"Sample files exist: {dataframe['file'].head().apply(os.path.exists).all()}")
    print(f"Unique cells: {dataframe['cell'].nunique()}")

    datablock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=ColReader('file'),
        get_y=ColReader('cell'),
        splitter=ColSplitter('is_validation'),
        item_tfms=Resize(IMG_CROP_SIZE),
        batch_tfms=aug_transforms(flip_vert=False, max_rotate=10., max_zoom=1.1)
    )

    return dataframe, datablock

def build_dataloader(datablock, dataframe):
    dataloaders = datablock.dataloaders(
        dataframe,
        bs=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    # sanity check
    dataloaders.valid.show_batch(max_n=9, figsize=(6,6))
    plt.show()
    return dataloaders

def build_learner(dataloaders):
    learner = vision_learner(
        dataloaders,
        resnet50,
        metrics=[accuracy, partial(top_k_accuracy, k=3)]
    ).to_fp16()

    lr_min, lr_steep = learner.lr_find(suggest_funcs=(minimum, steep))
    return learner, lr_min, lr_steep

def train_learner(learner, lr_min, lr_steep):
    learner.fine_tune(
        8, # total epochs
        base_lr = lr_steep/2,
        freeze_epochs = 3, # train head first
    )
    return learner

def execute_pipeline():
    dataframe, datablock = build_datablock()
    dataloaders = build_dataloader(datablock, dataframe)
    model, lr_min, lr_steep = build_learner(dataloaders)
    train_learner(model, lr_min, lr_steep)

    ClassificationInterpretation.from_learner(model).plot_confusion_matrix(figsize=(10,10), dpi=100)
    plt.show()
    model_out = "resnet50_geoguessr.pkl"
    model.export(model_out)
    print(f"Model exported to {model_out}")

if __name__ == '__main__':
    execute_pipeline()