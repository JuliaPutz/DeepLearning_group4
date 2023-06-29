from helpers import *
from ds import ImageSegmentationDataset

from datasets import load_dataset
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, MaskFormerForInstanceSegmentation
from torch.utils.data import DataLoader
from functools import partial
from tqdm.auto import tqdm
import evaluate
import time

models = [{"type":"mask2former","trained_on":"ade","purpose":"semantic","variant":"tiny",
           "huggingface_id":"facebook/mask2former-swin-tiny-ade-semantic"}]

def load_model(name, id2label=None, modelType = Mask2FormerForUniversalSegmentation):
    # load the specified model from huggingface
    processor = AutoImageProcessor.from_pretrained(name)
    if id2label is None:
        # default model
        model = modelType.from_pretrained(name)
    else:
        # replace classification head
        model = modelType.from_pretrained(name, id2label = id2label, ignore_mismatched_sizes=True)
    return processor, model

def make_dataset(name):
    dataset = load_dataset(name)
    dataset = dataset.shuffle(seed=1)
    dataset = dataset["train"].train_test_split(test_size=0.2)
    train_ds = dataset["train"]
    test_ds = dataset["test"]

    train_dataset = ImageSegmentationDataset(train_ds, transform=train_transform)
    test_dataset = ImageSegmentationDataset(test_ds, transform=test_transform)
    return train_dataset, test_dataset

def get_label_map(repo_id):
    filename = "id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k):v for k,v in id2label.items()}
    return id2label

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

def print_initial_loss(model, batch):
    outputs = model(batch["pixel_values"],
                    class_labels = batch["class_labels"],
                    mask_labels = batch["mask_labels"])
    
    print(f"Initial loss: {outputs.loss}")
    return outputs

def train_model(model, train_data_loader, test_data_loader, processor, device, epochs: int = 2):

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

def eval_model(model, test_data_loader, processor, device, metric, num_batches = 0, show_times=False):
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
    

def inference(model, batch):
    """
    perform prediction using new model
    compare overlay for one image of ground truth with new prediction
    """

    # show new prediction for one image
    image = batch["original_images"][0]
    pred = infer_img(processor, model, image)
    show_overlay(image, pred)

    # ground truth
    segmentation_map = batch["original_segmentation_maps"][0]
    show_overlay(image, segmentation_map)



if __name__ == "__main__":
    # load dataset. Caches, so doesn't have to be repeatedly redownloaded. 
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
    metric = eval_model(model, test_dataloader, processor, device, metric, num_batches=5)
    print(f"Pre-Finetuning Mean IoU: {metric.compute(num_labels = len(id2label), ignore_index=0)['mean_iou']}")

    # tuning! ~5 minutes per epoch...
    n_epochs = 20
    finetuned_model = train_model(model, train_dataloader, test_dataloader, processor, device, epochs = n_epochs)
    finetuned_model.save_pretrained(f"mask2former_finetuned_{name.replace('/','-')}_{n_epochs}epochs")

    # evaluation of finetuned model!
    metric = eval_model(finetuned_model, test_dataloader, processor, device, metric)
    print(f"Post-Finetuning Mean IoU: {metric.compute(num_labels = len(id2label), ignore_index=0)['mean_iou']}")

# % TODO 
# do fine tuning
# evaluate models!
# write report :)