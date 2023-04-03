import os, os.path as osp
import json
import subprocess
import platform
import re
from dvc.config import Config


def get_dvc_config(default_remote: str):
    """convert dvc 'config' file content to dict
    
    Args:
        default_remote (str): default_remote name 

    Returns:
        dict: dvc 'config' file content expressed as a dict
    """
    dvc_config_path = osp.join(Config().dvc_dir, "config")
    with open(dvc_config_path, "r") as dvc_config:
        # in here, will be erased all contents of 'dvc_config'
        assert len(list(dvc_config)) > 2,  "No remote configuration!  run 'remote add'!"
           
    
    dvc_cfg = dict()
    remotes = []
    urls = []
    with open(dvc_config_path, "r") as dvc_config:      # re open 'dvc_config'
        core_flag = False
        url_flag = False
        for i, line in enumerate(dvc_config):
            if i == 0 :
                if len(re.findall('core', line)) == 0:   # default remote not set
                    raise OSError(f"default remote is not set!!  \n"\
                                   "run:    $ dvc remote default `remote_name`")
                    
                
                if re.findall('core', line)[0] == 'core' :
                    core_flag = True
                    continue
            
            if core_flag:
                dvc_cfg['defualt_remote'] = line.split(" ")[-1].split("\n")[0]
                core_flag = False
                continue
            
            if len(line.split("[")) == 2: 
                remotes.append(line.split(" ")[-1].split("\"")[1]) 
                url_flag = True
                continue
            
            if url_flag:
                urls.append(line.split(" ")[-1].split("\n")[0])
                url_flag = False
    
    
    assert len(remotes) == len(urls)
    dvc_cfg['remotes'] = []
    
    for default, url in zip(remotes, urls):
        dvc_cfg['remotes'].append(dict(remote = default, url = url))
        
        
    if dvc_cfg['defualt_remote'] != default_remote: 
        raise OSError(f"Defualt_remote '.dvc/config: {dvc_cfg['defualt_remote']}' and "\
                      f"'cfg: {default_remote}' are not same!!")

    return dvc_cfg


def check_gs_credentials_dvc(remote):
    """_summary_

    Args:
        remote (_type_): _description_
    """
    dvc_cfg = get_dvc_config(remote)
    credentials = osp.join(os.getcwd(), ".dvc", "config.local")
    assert osp.isfile(credentials), f"\n  >> Path: {credentials} is not exist!!   "\
        f"set google storage credentials! "\
        f"\n  >> run:   $ dvc remote modify --local {dvc_cfg['defualt_remote']} credentialpath `client_secrets_path`"
    


def set_gs_credentials_dvc(remote: str, bucket_name: str, client_secrets: dict):    
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
    
    print(f"$ {remote_bucket_command}")
    subprocess.call([remote_bucket_command], shell=True)
    print(f"$ {credentials_command}")
    subprocess.call([credentials_command], shell=True)
    
    check_gs_credentials_dvc(remote)
    
    return client_secrets_path


def dvc_pull(remote: str, bucket_name: str, client_secrets: dict, data_root: str):
    """ run dvc pull from google cloud storage

    Args:
        remote (str): name of remote of dvc
        bucket_name (str): bucket name of google storage
        client_secrets (dict): credentials info to access google storage
        data_root (str): name of folder where located dataset(images)

    Returns:
        dataset_dir_path (str): path of dataset directory
    """
    if platform.system() != "Linux":
        raise OSError(f"This function only for Linux!")
    
    # check file exist (downloaded from git repo by git clone)
    dvc_path = osp.join(os.getcwd(), f'{data_root}.dvc')          
    assert osp.isfile(dvc_path), f"Path: {dvc_path} is not exist!" 

    client_secrets_path = set_gs_credentials_dvc(remote, bucket_name, client_secrets)
    
    # download dataset from GS by dvc 
    dvp_pull_srt = f"dvc pull {data_root}.dvc"
    print(f"$ {dvp_pull_srt}")
    subprocess.call([dvp_pull_srt], shell=True)           
    os.remove(client_secrets_path)
    
    if isdir(data_root):
		dataset_dir_path = data_root
	else:
		dataset_dir_path = osp.join(os.getcwd(), data_root)
    
    
    assert osp.isdir(dataset_dir_path), f"Directory: {dataset_dir_path} is not exist!\n"\
        f"list fo dir : {os.listdir(osp.split(dataset_dir_path)[0])}"
    
    return dataset_dir_path



def dvc_add(target_dir: str, dvc_name: str):
    """

    Args:
        target_dir (str): directory path where push to dvc
        dvc_name (str): name of file containing contents about dataset (`.dvc` format)

    """
    if platform.system() != "Linux":
        raise OSError(f"This function only for Linux!")
    
    subprocess.call([f"dvc add {target_dir}"], shell=True)
    
    recode_dir = osp.dirname(target_dir)
    dvc_file  = osp.join(recode_dir, f"{dvc_name}.dvc")
    gitignore_file = osp.join(recode_dir, ".gitignore")
    assert osp.isfile(dvc_file) and osp.isfile(gitignore_file),\
        f"dvc and .gitignore file are not exist!!" \
        f"\n files list in {recode_dir} {os.listdir(recode_dir)}"
        
        
        
def dvc_push(remote: str, bucket_name: str, client_secrets: dict):
    """

    Args:
        remote (str): name of remote of dvc
        bucket_name (str): bucket name of google storage
        client_secrets (dict): credentials info to access google storage
        
    """
    client_secrets_path = set_gs_credentials_dvc(remote, bucket_name, client_secrets)        
        
    # upload dataset to GS by dvc   
    subprocess.call(["dvc push"], shell=True)          
    os.remove(client_secrets_path)
