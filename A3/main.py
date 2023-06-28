from helpers import *
from ds import ImageSegmentationDataset

from datasets import load_dataset
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from torch.utils.data import DataLoader
from functools import partial
from tqdm.auto import tqdm
import evaluate

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

def print_initial_loss(model, batch):
    outputs = model(batch["pixel_values"],
                    class_labels = batch["class_labels"],
                    mask_labels = batch["mask_labels"])
    
    print(f"Initial loss: {outputs.loss}")

def train_model(model, train_data_loader, test_data_loader, processor, device, epochs: int = 2):
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    metric = evaluate.load("mean_iou")
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
                print(f'Loss: {running_loss/num_samples}')

            # optimize
            optimizer.step()
        
        # evaluate
        model.eval()
        for idx, batch in enumerate(tqdm(test_data_loader)):
            if idx > 5:
                break

            pixel_values = batch["pixel_values"]

            # forward pass
            with torch.no_grad():
                outputs = model(pixel_values = pixel_values.to(device))

            # original images
            original_images = batch["original_images"]
            target_sizes = [(image.shape[0], image.shape[1]) for image in original_images]

            # compare predict + ground truth segmentation maps
            predicted_segmentation_maps = processor.post_process_semantic_segmentation(outputs, target_sizes = target_sizes)
            ground_truth_segmentation_maps = batch["original_segmentation_maps"]
            metric.add_batch(references = ground_truth_segmentation_maps, predictions = predicted_segmentation_maps)

        print(f"Mean IoU: {metric.compute(num_labels = len(id2label), ignore_index=0)['mean_iou']}")
    
    return model

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
    processor, model = load_model("facebook/mask2former-swin-tiny-ade-semantic")
    train_ds, test_ds = make_dataset() # sidewalk-semantic is the default dataset used here
    image, map, _, _ = test_ds[0]
    pred = infer_img(processor, model, image)
    show_overlay(image, pred)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create batches and train + evaluate model
    train_data_loader, test_data_loader = create_batch(train_ds, test_ds, batch_size=2, processor=processor)
    batch_train = next(iter(train_data_loader))
    batch_test = next(iter(test_data_loader))

    print_initial_loss(model, batch_train)

    model = train_model(model, train_data_loader, test_data_loader, processor, device, epochs = 100)
    inference(model, batch_test)


# % TODO 
# do fine tuning
# evaluate models!
# write report :)