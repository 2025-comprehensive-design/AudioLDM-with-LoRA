from diffusers import AudioLDMPipeline
import datasets

def load_dataset(dataset_name=None, cache_dir=None, train_data_dir=None):
    if dataset_name is not None:
        dataset = datasets.load_dataset(
            dataset_name,
            cache_dir=cache_dir,
            data_dir=train_data_dir,
        )
    else:
        data_files = {}
        if train_data_dir is not None:
            data_files["train"] = os.path.join(train_data_dir, "**")
        dataset = datasets.load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=cache_dir,
        )
    return dataset

dataset_name = "deetsadi/musiccaps_spectrograms"
dataset = load_dataset(
    dataset_name
)

filtered_dataset = dataset["train"].filter(
    lambda example: "hiphop" in example["caption"].lower()
    )
    
print(filtered_dataset[0])


base_model_id = "cvssp/audioldm-s-full-v2"
model = AudioLDMPipeline.from_pretrained(base_model_id)