import os 
from huggingface_hub import HfApi
from src.dataset_handling import sync_dataset


dataset_id = "Saving-Willy/temp_dataset"
token = os.getenv("HF_TOKEN")
dataset_filename = "data/train-00000-of-00001.parquet"

if __name__ == '__main__':
    # Initialize API client
    api = HfApi(token=token)

    # add json files to dataset
    n_new = sync_dataset(
        api, dataset_id, dataset_filename, 
        create_dataset_if_not_exists=False)

    if n_new is not None:
        if n_new == 0:
            print(f"Info: no action taken, no new observation files found")
        else:
            # print the number of new files added
            print(f"Added {n_new} new files to dataset {dataset_id}.")
    else:
        # something went wrong. - want the workflow to fail so nonzero exit  
        print(f"Failed to sync dataset {dataset_id}.")
        raise SystemExit(1) 
    
    
    
    

    