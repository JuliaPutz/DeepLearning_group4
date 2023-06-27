from helpers import *
from ds import ImageSegmentationDataset

from datasets import load_dataset
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from torch.utils.data import DataLoader
from functools import partial

models = [{"type":"mask2former","trained_on":"ade","purpose":"semantic","variant":"tiny",
           "huggingface_id":"facebook/mask2former-swin-tiny-ade-semantic"}]

def load_model(name):
    # load the specified model from huggingface
    processor = AutoImageProcessor.from_pretrained(name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(name)
    return processor, model

def make_dataset(name = "segments/sidewalk-semantic"):
    dataset = load_dataset(name)
    dataset = dataset.shuffle(seed=1)
    dataset = dataset["train"].train_test_split(test_size=0.2)
    train_ds = dataset["train"]
    test_ds = dataset["test"]

    train_dataset = ImageSegmentationDataset(train_ds, transform=train_transform)
    test_dataset = ImageSegmentationDataset(test_ds, transform=test_transform)
    return train_dataset, test_dataset

def collate_fn_custom(batch, processor):
    inputs = list(zip(*batch))
    images = inputs[0]
    segmentation_maps = inputs[1]

    batch = processor(
        images,
        segmentation_maps = segmentation_maps,
        return_tensors = "pt"
    )
    batch["original_images"] = inputs[2]
    batch["original_segmentation_maps"] = inputs[3]

    return batch

def create_batch(train_ds, test_ds, batch_size, processor):
    train_data_loader = DataLoader(train_ds, batch_size = batch_size, shuffle=True, collate_fn=partial(collate_fn_custom, processor = processor))
    test_data_loader = DataLoader(test_ds, batch_size = batch_size, shuffle=False, collate_fn=partial(collate_fn_custom, processor = processor))

    return train_data_loader, test_data_loader

if __name__ == "__main__":
    # load dataset. Caches, so doesn't have to be repeatedly redownloaded. 
    processor, model = load_model("facebook/mask2former-swin-tiny-ade-semantic")
    train_ds, test_ds = make_dataset() # sidewalk-semantic is the default dataset used here
    image, map, _, _ = test_ds[0]
    pred = infer_img(processor, model, image)
    show_overlay(image, pred)

# % TODO 
# make pytorch batch system for finetuning
# do fine tuning
# evaluate models!
# write report :)