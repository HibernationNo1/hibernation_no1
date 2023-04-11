import os, os.path as osp
import json
import subprocess
import platform
import re
from dvc.config import Config


def get_dvc_config(default_remote: str, dvc_path: str):
    """convert dvc 'config' file content to dict
    
    Args:
        default_remote (str): default_remote name 

    Returns:
        dict: dvc 'config' file content expressed as a dict
    """
    if Config().dvc_dir is not None:
        dvc_config_path = osp.join(Config().dvc_dir, "config")
    else:
        dvc_config_path = osp.join(dvc_path, "config")    
    
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


def check_gs_credentials_dvc(remote, dvc_path):
    """_summary_

    Args:
        remote (_type_): _description_
    """
    dvc_cfg = get_dvc_config(remote, dvc_path)
    credentials = osp.join(dvc_path, "config.local")
    assert osp.isfile(credentials), f"\n  >> Path: {credentials} is not exist!!   "\
        f"set google storage credentials! "\
        f"\n  >> run:   $ dvc remote modify --local {dvc_cfg['defualt_remote']} credentialpath `client_secrets_path`"


def run_dvc_command(command, dvc_path):
    org_chdir = None
    if dvc_path != osp.join(os.getcwd(), ".dvc"):
        org_chdir = os.getcwd()
        os.chdir(osp.dirname(dvc_path))
    
    print(f"Run `$ {command}`")
    subprocess.call([command], shell=True)
    
    if org_chdir is not None:
        os.chdir(org_chdir)


def set_gs_credentials_dvc(remote: str, bucket_name: str, client_secrets: dict, dvc_path: str):    
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
    
    run_dvc_command(f"dvc remote add -d -f {remote} gs://{bucket_name}", dvc_path)
    run_dvc_command(f"dvc remote modify --local {remote} credentialpath {client_secrets_path}", dvc_path)
         
    check_gs_credentials_dvc(remote, dvc_path)
    
    return client_secrets_path


def dvc_pull(remote: str, bucket_name: str, client_secrets: dict, data_root: str, dvc_path = osp.join(os.getcwd(), ".dvc")):
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
    dvc_file_path = f'{data_root}.dvc'          
    assert osp.isfile(dvc_file_path), f"Path: {dvc_file_path} is not exist!" 

    client_secrets_path = set_gs_credentials_dvc(remote, bucket_name, client_secrets, dvc_path)
    
    # download dataset from GS by dvc 
    run_dvc_command(f"dvc pull {data_root}.dvc", dvc_path)     
    os.remove(client_secrets_path)
    
    if osp.isdir(data_root):
        dataset_dir_path = data_root
    else:
        dataset_dir_path = osp.join(os.getcwd(), data_root)
    
    
    assert osp.isdir(dataset_dir_path), f"Directory: {dataset_dir_path} is not exist!\n"\
        f"list fo dir : {os.listdir(osp.split(dataset_dir_path)[0])}"
    
    return dataset_dir_path



def dvc_add(target_dir: str):
    """

    Args:
        target_dir (str): directory path where push to dvc
    """
    if platform.system() != "Linux":
        raise OSError(f"This function only for Linux!")
    
    print(f"\nRun	`$ dvc add {target_dir}`")
    subprocess.call([f"dvc add {target_dir}"], shell=True)

    

        
        
        
def dvc_push(remote: str, bucket_name: str, client_secrets: dict, dvc_path = osp.join(os.getcwd(), ".dvc")):
    """

    Args:
        remote (str): name of remote of dvc
        bucket_name (str): bucket name of google storage
        client_secrets (dict): credentials info to access google storage
        
    """
    client_secrets_path = set_gs_credentials_dvc(remote, bucket_name, client_secrets, dvc_path)        
        
    # upload dataset to GS by dvc  
    print(f"Run `$ dvc push`") 
    subprocess.call(["dvc push"], shell=True)          
    os.remove(client_secrets_path)
