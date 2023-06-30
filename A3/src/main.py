from helpers import *
from ds import ImageSegmentationDataset

from datasets import load_dataset
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, MaskFormerForInstanceSegmentation
from torch.utils.data import DataLoader
from functools import partial
from tqdm.auto import tqdm
import evaluate
import time
import pandas as pd

models = [{"type":"mask2former","trained_on":"ade","purpose":"semantic","variant":"tiny",
           "huggingface_id":"facebook/mask2former-swin-tiny-ade-semantic"}]

def load_model(name: str, id2label = None, modelType = Mask2FormerForUniversalSegmentation):
    """
    load the specified model from huggingface

    name: Path or name of the model
    id2label: dict representing available label and its ids
    modelType: used for defining the model
    """
    processor = AutoImageProcessor.from_pretrained(name)
    if id2label is None:
        # default model
        model = modelType.from_pretrained(name)
    else:
        # replace classification head
        model = modelType.from_pretrained(name, id2label = id2label, ignore_mismatched_sizes=True)
    return processor, model


##### DATASET #####

def make_dataset(name: str):
    """
    split dataset into train and test
    and return them as ImageSegmentationDataset

    name: Path or name of the dataset
    """
    dataset = load_dataset(name)
    dataset = dataset.shuffle(seed=1)
    dataset = dataset["train"].train_test_split(test_size=0.2)
    train_ds = dataset["train"]
    test_ds = dataset["test"]

    train_dataset = ImageSegmentationDataset(train_ds, transform=train_transform)
    test_dataset = ImageSegmentationDataset(test_ds, transform=test_transform)
    return train_dataset, test_dataset

def get_label_map(repo_id: str):
    """
    return map of class ids to class labels as defined by the named dataset

    repo_id: Path or name of the dataset
    """
    filename = "id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k):v for k,v in id2label.items()}
    return id2label

def collate_fn_custom(batch, processor):
    """
    bring batches into a format the model expects
    """
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

def create_batch(train_ds: ImageSegmentationDataset, test_ds: ImageSegmentationDataset, batch_size: int, processor):
    """
    create batches for train and test set using DataLoader

    train_ds: dataset from which to load the data for training
    test_ds: dataset from which to load the data for training
    batch_size: how many samples per batch to load
    processor: processor to use for bringing data into the format the model expects
    """
    train_data_loader = DataLoader(train_ds, batch_size = batch_size, shuffle=True, collate_fn=partial(collate_fn_custom, processor = processor))
    test_data_loader = DataLoader(test_ds, batch_size = batch_size, shuffle=False, collate_fn=partial(collate_fn_custom, processor = processor))

    return train_data_loader, test_data_loader

def print_initial_loss(model, batch):
    """
    prints loss of initial pre-trained model

    model: model to compute loss for
    """
    outputs = model(batch["pixel_values"],
                    class_labels = batch["class_labels"],
                    mask_labels = batch["mask_labels"])
    
    print(f"Initial loss: {outputs.loss}")
    return outputs


##### TRAINING #####

def train_model(model, train_data_loader: DataLoader, device, epochs: int = 2):
    """
    returns the model trained for specified number of epochs

    model: model to fine-tune
    train_data_loader: data to use for training
    device: defines whether to use cuda or cpu
    epochs: how many epochs to train
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    running_loss = 0.0
    num_samples = 0

    for epoch in range(epochs):
        print(f'Epoch: {epoch}')

        #train
        model.train()
        for idx, batch in enumerate(tqdm(train_data_loader)):
            optimizer.zero_grad() # reset parameter gradients

            # forward pass
            outputs  = model(pixel_values = batch["pixel_values"].to(device),
                             mask_labels = [labels.to(device) for labels in batch["mask_labels"]],
                             class_labels = [labels.to(device) for labels in batch["class_labels"]])
            
            # propagate backwards
            loss = outputs.loss
            loss.backward()

            batch_size = batch["pixel_values"].size(0)
            running_loss += loss.item()
            num_samples += batch_size

            if idx % 100 == 0:
                print(f'\nLoss: {running_loss/num_samples}')

            # optimize
            optimizer.step()
    return model


##### EVALUATION #####

def eval_model(model, test_data_loader: DataLoader, processor, device, num_batches: int = 0, show_times: bool = False):
    """
    returns the mean_iou of the passed model

    model: model to evaluate
    test_data_loader: data to use for evaluating the model
    processor: used to perform semantic segmentation
    device: defines whether to use cuda or cpu
    num_batches: number of batches to use for evaluation; 0 if all batches should be used
    show_times: if True print information about time needed for seperate steps
    """

    # evaluate
    start_time = time.time()
    metric = evaluate.load("mean_iou")
    model = model.to(device)
    model.eval()
    setting_time = time.time()
    if show_times: print(f"model setup: {(setting_time - start_time)} seconds")
    for idx, batch in enumerate(tqdm(test_data_loader)):
        if num_batches > 0:
            if idx > num_batches:
                break
        epoch_start_time = time.time()
        pixel_values = batch["pixel_values"]

        # forward pass
        with torch.no_grad():
            outputs = model(pixel_values = pixel_values.to(device))
        forward_pass_time = time.time()
        if show_times: print(f"forward pass time: {(forward_pass_time-epoch_start_time)} seconds")
        # original images
        original_images = batch["original_images"]
        target_sizes = [(image.shape[0], image.shape[1]) for image in original_images]
        loading_time = time.time()
        if show_times: print(f"image loading time: {(loading_time - forward_pass_time)} seconds")
        # compare predict + ground truth segmentation maps
        predicted_segmentation_maps = processor.post_process_semantic_segmentation(outputs, target_sizes = target_sizes)
        ground_truth_segmentation_maps = batch["original_segmentation_maps"]
        processing_time = time.time()
        if show_times: print(f"image processing time: {(processing_time-loading_time)} seconds")
        metric.add_batch(references = torch.tensor(np.array(ground_truth_segmentation_maps)).to(device), predictions = torch.stack(predicted_segmentation_maps).to(device))
        eval_time = time.time()
        if show_times: print(f"evaluation time: {(eval_time-processing_time)} seconds")
        
    final_time = time.time()
    print(f"total time: {(final_time-start_time)} seconds")

    return metric
    
def iou(mask1, mask2):
    '''Calculate the intersection over union for the two binary masks'''
    # area = number of pixels
    mask1_area = torch.count_nonzero(mask1)
    mask2_area = torch.count_nonzero(mask2)
    # the intersection is the number of pixels where both masks are True
    intersection = torch.count_nonzero(torch.logical_and(mask1, mask2))
    # the union is computed as the sum of both areas minus the intersection, which would be counted twice otherwise
    iou = intersection / (mask1_area + mask2_area - intersection)
    # the .item() method extracts the value from this size-1 tensor
    return iou.item()

def get_ious(pred, truth):
    '''From a semantic segmentation prediction and a ground truth, compute the iou for each class (=unique value) and compute the average'''
    uniques = truth.unique()
    ious = {}
    for i in uniques:
        ious[i.item()] = iou(pred == i, truth == i)
    ious["mean"] = np.mean(list(ious.values()))
    return ious

def eval_model2(modelName, eval_dataloader, modelType = Mask2FormerForUniversalSegmentation):
    """
    custom evaluation method computing mean_iou

    modelName: model to evaluate
    eval_dataloader: data used to evaluate model
    """
    if isinstance(modelName, str):
        model = modelType.from_pretrained(modelName).to(device)
    else:
        model = modelName
    resultlist = []
    for x in tqdm(eval_dataloader):
        img = x["pixel_values"]
        with torch.no_grad():
            output = model(img.to(device))

        original_image = x["original_images"][0]
        target_size = (original_image.shape[0], original_image.shape[1])
        pred = processor.post_process_semantic_segmentation(output, target_sizes = [target_size])[0]
        truth = x["original_segmentation_maps"][0]
        ious = get_ious(pred, torch.tensor(truth).to(device))
        resultlist.append(ious)
    ioudf = pd.DataFrame(resultlist)
    ioudf = pd.concat([ioudf["mean"], ioudf.drop("mean", axis=1).sort_index(axis=1)], axis=1)
    ioudf = ioudf.rename(columns=id2label)
    return ioudf


def inference(model, test_dataloader, processor):
    """
    perform prediction using new model
    compare overlay for one image of ground truth with new prediction

    model: model used to predict segmentation
    test_dataloader: data to take example image from
    processor: used to convert images into model input and model output back into images
    """
    batch = next(iter(test_dataloader))

    # show new prediction for one image
    image = batch["original_images"][0]
    pred = infer_img(processor, model, image)
    show_overlay(image, pred)

    # ground truth
    segmentation_map = batch["original_segmentation_maps"][0]
    show_overlay(image, segmentation_map)




if __name__ == "__main__":
    # use cuda if it is available (recommended)
    if torch.cuda.is_available():
        print("using cuda!")
        device = "cuda:0"
    else:
        print("using cpu.")
        device = "cpu"
    print(device)

    # load dataset and label map
    name = "segments/sidewalk-semantic"
    train_ds, test_ds = make_dataset(name)
    id2label = get_label_map(name)
    print(id2label)

    # by passing the custom label dictionary, the classification head is replaced!
    processor, model = load_model(models[0]["huggingface_id"], id2label = id2label)

    # creating batches
    train_dataloader, test_dataloader = create_batch(train_ds, test_ds, 2, processor)

    # checking initial performance...
    metric = eval_model(model, test_dataloader, processor, device, num_batches=5)
    print(f"Pre-Finetuning Mean IoU: {metric.compute(num_labels = len(id2label), ignore_index=0)['mean_iou']}")

    # tuning! ~5 minutes per epoch...
    n_epochs = 2
    finetuned_model = train_model(model, train_dataloader, device, epochs = n_epochs)
    finetuned_model.save_pretrained(f"mask2former_finetuned_{name.replace('/','-')}_{n_epochs}epochs")

    # evaluation of finetuned model!
    metric = eval_model(finetuned_model, test_dataloader, processor, device, num_batches=5)
    metric_custom = eval_model2(finetuned_model, test_dataloader)
    print(f"Post-Finetuning Mean IoU: {metric.compute(num_labels = len(id2label), ignore_index=0)['mean_iou']}")
    print(f'Post-Finetuning custom Mean IoU: {metric_custom["mean"].mean()}')