from typing import List
import json
import pandas as pd

import pyarrow
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, hf_hub_download

def create_blank_dataset(dataset_key:str="train"):
    """
    Creates a blank Hugging Face DatasetDict with an empty dataset for the specified field type.

    Args:
        dataset_key (str): The key for the dataset in the DatasetDict. Defaults to 'train'.

    Returns:
        DatasetDict: A dictionary containing a single key with the specified field type and an empty Dataset.
    """
    blank_dataset = DatasetDict()
    blank_dataset[dataset_key] = Dataset(pyarrow.table([]))
    return blank_dataset
  

def delete_metadata_if_exists(
    api: HfApi, dataset_id: str, 
    parquet_fname: str = "data/train-00000-of-00001.parquet"):
    """
    Deletes a specified file from a dataset repository if it exists.

    Args:
        api (HfApi): An instance of the Hugging Face API client.
        dataset_id (str): The identifier of the dataset repository.
        parquet_fname (str, optional): The name of the file to check and delete. 
            Defaults to "data/train-00000-of-00001.parquet".

    """
    # does file exist? (could lookup by wildcard, eg train-*.parquet)?
    if api.file_exists(repo_id=dataset_id, filename=parquet_fname, repo_type="dataset"):
        print(f"File {parquet_fname} in dataset {dataset_id} exists.")
        # then delete
        try:
            api.delete_file(repo_id=dataset_id, path_in_repo=parquet_fname, repo_type="dataset")
        except Exception as e:
            print(f"File {parquet_fname} failed to delete from repo {dataset_id}.")
            print(e)
    else:
        # worth looking up all files and looking for parquet extensions? why though?
        print(f"File {parquet_fname} does not exist in dataset {dataset_id}")


def lookup_json_files(api:HfApi, dataset_id:str) -> List[str]:
    """get list of json files in a dataset"""
    # get list of files and filter for json extension
    files = api.list_repo_files(dataset_id, repo_type="dataset")
    json_files = [f for f in files if f.endswith(".json")]
    return json_files


def add_json_files_to_metadata(add_files: List[str], metadata: DatasetDict, dataset_id: str) -> int:
    """
    Adds JSON files to the metadata dataset dictionary if their "image_md5" key 
    is not already present in the "train" split of the metadata.

    Args:
        add_files (List[str]): A list of file paths (relative to the dataset repository) 
            to be added to the metadata.
        metadata (DatasetDict): A dataset dictionary containing metadata, where the 
            "train" split is expected to have an "image_md5" key.
        dataset_id (str): The identifier of the dataset repository from which the JSON 
            files are being added.

    Returns:
        int: The number of new items added to the "train" split of the metadata.
    """

    # inplace
    n = 0
    for f in add_files:
        file = hf_hub_download(repo_id=dataset_id, filename=f, repo_type="dataset")
        with open(file, "r") as f:
            new = json.load(f)
        
        
        
        if ("image_md5" not in metadata["train"].column_names):
            # new dataset, no 'image_md5' dict to test for keys
            metadata["train"] = metadata["train"].add_item(new)
            n += 1
        elif new["image_md5"] not in metadata["train"]["image_md5"]:
            # dataset exists, but this image wasn't seen before
            metadata["train"] = metadata["train"].add_item(new)
            n += 1
        else:
            # this image was seen before, so we skip it
            print(f"Skipping {new['image_md5']} as it already exists in the dataset.")
            
    return n

        
def append_new_to_dataset(api:HfApi, dataset_id:str, parquet_fname:str = "data/train-00000-of-00001.parquet"):
    # procedure from https://github.com/vancauwe/saving-willy-data-sync/
    # - split into a few functions for reuse (eg in the reset mode)
    
    # steps:
    # - append new json files
    # - push metadata to hub

    # TODO: put back the exception handling

    json_files = lookup_json_files(api, dataset_id)
    metadata = load_dataset(dataset_id, data_files=parquet_fname)
    n = add_json_files_to_metadata(json_files, metadata)
    print(f"Added {n} files to metadata.")
    if n > 0:
        try:
            metadata.push_to_hub(dataset_id) # , token=token)
        except Exception as e:
            print(f"Failed to push metadata to HF hub: {e}")
            raise
        
    return metadata


def sync_dataset(
    api:HfApi, dataset_id:str, 
    dataset_filename:str = "data/train-00000-of-00001.parquet",
    create_dataset_if_not_exists:bool = False,
    ) -> int|None:
    '''docstring (basically: add any new files into dataset) '''
    
    # 0. get all the existing individual observation files (in json)
    json_files = lookup_json_files(api, dataset_id)
    # - if there are none, give up already
    if not len(json_files):
        # print a warning, and return
        print(f"No json files found in dataset {dataset_id}.")
        return 0
        
    # 1. fetch the dataset (note: doesn't need authentication)
    dataset = load_dataset(dataset_id, data_files=dataset_filename)

    # if it doesn't exist, either we give up, or we create a blank 
    if not dataset:
        if not create_dataset_if_not_exists:
            # print a warning, and return
            print(f"No dataset found in {dataset_filename}.")
            return None
        # create a blank dataset
        dataset = create_blank_dataset()
        print(f"Created new blank dataset as none existed at {dataset_filename}")

    else:
        print(f"Dataset found in {dataset_filename}. Proceeding with sync.")

    # 2. add the content of any new ones into the dataset
    n = add_json_files_to_metadata(json_files, dataset, dataset_id)
    # 3. push the updated one into HF hub
    if n > 0:
        try:
            dataset.push_to_hub(dataset_id) # , token=token)
        except Exception as e:
            print(f"Failed to push dataset to HF hub: {e}")
            raise 
            # return
                
        print(f"Updated dataset with {n} new entries pushed to HF hub: {dataset_id}.")
        
    
    return n
    


def reset_dataset_rebuild_from_json(api:HfApi, dataset_id:str) -> DatasetDict:
    """
    Resets a dataset by deleting existing metadata, rebuilding it entirely from JSON files, 
    and pushing the updated metadata to the Hugging Face Hub.

    Args:
        api (HfApi): The Hugging Face API instance used to interact with the dataset.
        dataset_id (str): The identifier of the dataset to reset and rebuild.
    Returns:
        DatasetDict: The rebuilt dataset metadata.

    Notes:
        - This function is intended for use when the data format has changed, requiring a 
            complete cleanup and rebuild of the dataset.
        - The function prints the number of JSON files added and a preview of the metadata.
    """

    # reset dataset (parquet file), rebuild entirely from json files
    # usage/context: the data format changed and we need to clean up the dataset

    # - delete meta if exists, 
    # - create blank dataset
    # - add metadata from valid json files
    # - push metea to hub
    delete_metadata_if_exists(api, dataset_id, parquet_fname="data/train-00000-of-00001.parquet")
    metadata = create_blank_dataset()
    json_files = lookup_json_files(api, dataset_id)
    n = add_json_files_to_metadata(json_files, metadata, dataset_id)
    print(f"Added {n} files to metadata.")
    print(pd.DataFrame(metadata['train']))

    if n > 0: 
        #metadata.push_to_hub(dataset_id, commit_message="refreshed dataset with new format")
        metadata.push_to_hub(dataset_id)
    
    return metadata
    
