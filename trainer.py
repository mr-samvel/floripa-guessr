import os
import pandas as pd
from fastai.vision.all import *

### DEFS
MANIFESTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'manifests')
TRAIN_CSV = os.path.join(MANIFESTS_DIR, "training_manifest.csv")
VALID_CSV = os.path.join(MANIFESTS_DIR, "validation_manifest.csv")

BATCH_SIZE = 64
IMG_CROP_SIZE = 384
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
    
    class_counts = training_dataframe['cell'].value_counts()
    print(f"Most common class: {class_counts.iloc[0]} samples")
    print(f"Least common class: {class_counts.iloc[-1]} samples")
    print(f"Class imbalance ratio: {class_counts.iloc[0] / class_counts.iloc[-1]:.2f}")

    transforms = [
        *aug_transforms(
            size=IMG_CROP_SIZE,
            flip_vert=False,
            max_rotate=15.,
            max_zoom=1.15,
            max_lighting=0.3,
            max_warp=0.1,
            p_affine=0.7,
            p_lighting=0.6
        ),
        Normalize.from_stats(*imagenet_stats)
    ]

    datablock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=ColReader('file'),
        get_y=ColReader('cell'),
        splitter=ColSplitter('is_validation'),
        item_tfms=Resize(IMG_CROP_SIZE, method='squish'),
        batch_tfms=transforms
    )

    return dataframe, datablock

def build_dataloader(datablock, dataframe):
    dataloaders = datablock.dataloaders(
        dataframe,
        bs=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle_train=True
    )
    # dataloaders.to(torch_directml.device())
    # sanity check
    # dataloaders.valid.show_batch(max_n=9, figsize=(6,6))
    # plt.show()
    return dataloaders

def build_learner(dataloaders):
    learner = vision_learner(
        dataloaders,
        efficientnet_b4,
        metrics=[accuracy, partial(top_k_accuracy, k=3), partial(top_k_accuracy, k=5)],
        pretrained=True,
    )

    lr_min, lr_steep = learner.lr_find(suggest_funcs=(minimum, steep))
    return learner, lr_min, lr_steep

def train_learner(learner, lr_min, lr_steep):
    print("Starting training...")
    print("1) training classifier head...\n")
    learner.fit_one_cycle(
        3, 
        lr_max=lr_steep,
        cbs=[SaveModelCallback(monitor='valid_loss')]
    )
    
    print("2) fine-tuning entire model...\n")
    learner.unfreeze()
    learner.fit_one_cycle(
        6, 
        lr_max=slice(lr_steep/10, lr_steep/3),  # backbone gets 10x lower LR
        cbs=[
            SaveModelCallback(monitor='valid_loss'),
            EarlyStoppingCallback(monitor='valid_loss', patience=3)
        ]
    )

    return learner

def show_results(learner):
    interp = ClassificationInterpretation.from_learner(learner)
    interp.plot_confusion_matrix(figsize=(12, 10), dpi=100)
    plt.show()
    
    interp.plot_top_losses(16, nrows=4, figsize=(16, 12))
    plt.show()
    
    most_confused = interp.most_confused(min_val=3)
    print("Most Confused Classes:")
    for confusion in most_confused:
        print(f"  {confusion[0]} ↔ {confusion[1]}: {confusion[2]} misclassifications")
    
    class_accuracies = {}
    preds, targets = learner.get_preds()
    pred_classes = preds.argmax(dim=1)
    
    for i, class_name in enumerate(learner.dls.vocab):
        mask = targets == i
        if mask.sum() > 0:
            acc = (pred_classes[mask] == i).float().mean()
            class_accuracies[class_name] = acc.item()
    
    print("\nPer-Class Accuracies:")
    sorted_accs = sorted(class_accuracies.items(), key=lambda x: x[1])
    for class_name, acc in sorted_accs:
        print(f"  {class_name}: {acc:.3f}")

def execute_pipeline():
    dataframe, datablock = build_datablock()
    dataloaders = build_dataloader(datablock, dataframe)
    model, lr_min, lr_steep = build_learner(dataloaders)
    train_learner(model, lr_min, lr_steep)
    show_results(model)

    model_out = "floripa-guessr.pkl"
    model.export(model_out)
    print(f"Model exported to {model_out}")

if __name__ == '__main__':
    execute_pipeline()