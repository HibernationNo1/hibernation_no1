import os, os.path as osp
import json
import subprocess
import platform

def dvc_gs_credentials(remote: str, bucket_name: str, client_secrets: dict):    
    """ access google cloud with credentials

    Args:
        remote (str): name of remote of dvc  
        bucket_name (str): bucket name of google storage
        client_secrets (dict): credentials info for access google storage
    """
    if platform.system() != "Linux":
        raise OSError(f"This function only for Linux!")
    
    client_secrets_path = osp.join(os.getcwd(), "client_secrets.json")
    json.dump(client_secrets, open(client_secrets_path, "w"), indent=4)
    
    remote_bucket_command = f"dvc remote add -d -f {remote} gs://{bucket_name}"
    credentials_command = f"dvc remote modify --local {remote} credentialpath {client_secrets_path}"   
    
    subprocess.call([remote_bucket_command], shell=True)
    subprocess.call([credentials_command], shell=True)
    
    return client_secrets_path


def dvc_pull(remote: str, bucket_name: str, client_secrets: dict, dataset_name: str):
    """ run dvc pull from google cloud storage

    Args:
        remote (str): name of remote of dvc
        bucket_name (str): bucket name of google storage
        client_secrets (dict): credentials info to access google storage
        dataset_name (str): name of folder where located dataset(images)

    Returns:
        dataset_dir_path (str): path of dataset directory
    """
    if platform.system() != "Linux":
        raise OSError(f"This function only for Linux!")
    
    # check file exist (downloaded from git repo by git clone)
    dvc_path = osp.join(os.getcwd(), f'{dataset_name}.dvc')          
    assert os.path.isfile(dvc_path), f"Path: {dvc_path} is not exist!" 

    client_secrets_path = dvc_gs_credentials(remote, bucket_name, client_secrets)
    
    # download dataset from GS by dvc 
    subprocess.call(["dvc pull"], shell=True)           
    os.remove(client_secrets_path)
    
    dataset_dir_path = osp.join(os.getcwd(), dataset_name)
    assert osp.isdir(dataset_dir_path), f"Directory: {dataset_dir_path} is not exist!"\
        f"list fo dir : {os.listdir(osp.split(dataset_dir_path)[0])}"
    
    return dataset_dir_path


def dvc_push(remote: str, bucket_name: str, client_secrets: dict, target_dir: str, dvc_name: str):
    """_summary_

    Args:
        remote (str): name of remote of dvc
        bucket_name (str): bucket name of google storage
        client_secrets (dict): credentials info to access google storage
        target_dir (str): directory path where push to dvc
        dvc_name (str): name of file containing contents about dataset (`.dvc` format)
    """
    
    if platform.system() != "Linux":
        raise OSError(f"This function only for Linux!")
    
    subprocess.call([f"dvc add {target_dir}"], shell=True)
    
    recode_dir = osp.abspath(target_dir)
    dvc_file  = osp.join(recode_dir, f"{dvc_name}.dvc")
    gitignore_file = osp.join(recode_dir, ".gitignore")
    assert osp.isfile(dvc_file) and osp.isfile(gitignore_file),\
        f"dvc and .gitignore file are not exist!!" \
        f"\n files list in {recode_dir} {os.listdir(recode_dir)}"

    client_secrets_path = dvc_gs_credentials(remote, bucket_name, client_secrets)
    
    # upload dataset to GS by dvc   
    subprocess.call(["dvc push"], shell=True)          
    os.remove(client_secrets_path)
