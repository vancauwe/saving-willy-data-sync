import json
import os 
import pandas as pd

from huggingface_hub import HfApi

from dataset_handling import create_blank_dataset, lookup_json_files, add_json_files_to_metadata


if __name__ == '__main__':
    # example usage, with no actions taken by the demo
    # - see also `reset_dataset_rebuild_from_json`, which wraps most of these 
    #   steps, but doesn't implement the filtering.
    token = os.getenv("HF_TOKEN")
    api = HfApi(token=token)

    data_files = "data/train-00000-of-00001.parquet"
    dataset_id = "Saving-Willy/temp_dataset"
    
    # find the uploaded observation+classification files
    all_json_files = lookup_json_files(api, dataset_id)

    # filter files for some criteria, e.g. let's exclude from a specific user 
    exclude = 'test_data@whale.org'
    ok_files = [fname for fname in all_json_files if exclude not in fname.split('/')[1]]
    
    # create an empty dataset, and add the selected file to it
    metadata = create_blank_dataset()
    add_json_files_to_metadata(ok_files, metadata, dataset_id=dataset_id)

    print(pd.DataFrame(metadata['train']))
    print("Note: to transmit data, use `metadata.push_to_hub(dataset_id)`")
    
