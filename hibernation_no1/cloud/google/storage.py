import json
import os, os.path as osp
import warnings

def set_gs_credentials(client_secrets_file_name, gs_secret):
    """
    Args:
        client_secrets_file_name (str): file name (json format)
        gs_secret (dict): google client secret dict
    """
    client_secrets_path = os.path.join(os.getcwd(), client_secrets_file_name)
        
    # save client_secrets
    json.dump(gs_secret, open(client_secrets_path, "w"), indent=4)
    
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = client_secrets_path
    
def convert_to_linebreak(environ_str):
    flag = False
    private_key = ""
    for arp in list(environ_str):
        if arp == '\\':
            
            private_key+=arp
            flag = True
            continue
        
        if arp == 'n' and flag == True:
            private_key = private_key[:-1]
            private_key +='\n'
            flag = False
            continue
        
        private_key+=arp
    return private_key
        
def get_client_secrets():
    client_secrets_dict = {'type':convert_to_linebreak(os.environ['type']),
                           'project_id': convert_to_linebreak(os.environ['project_id']),
                           'private_key_id': convert_to_linebreak(os.environ['private_key_id']),
                           'private_key': convert_to_linebreak(os.environ['private_key']),
                           'client_email': convert_to_linebreak(os.environ['client_email']),
                           'client_id': convert_to_linebreak(os.environ['client_id']),
                           'auth_uri': convert_to_linebreak(os.environ['auth_uri']),
                           'token_uri': convert_to_linebreak(os.environ['token_uri']),
                           'auth_provider_x509_cert_url': convert_to_linebreak(os.environ['auth_provider_x509_cert_url']),
                           'client_x509_cert_url': convert_to_linebreak(os.environ['client_x509_cert_url']),
                            }
    

    return client_secrets_dict


def gs_credentials(file_name = None):
    """
        save credentials file to GOOGLE_APPLICATION_CREDENTIALS
    Args:
        file_name (str, optional): name of client_secrets file. Defaults to None.
            the file will make for runs GOOGLE_APPLICATION_CREDENTIALS and delete.
    """
    if file_name is not None:
        if osp.splitext(file_name)[-1] != "json": 
            fix_path = osp.splitext(file_name)[0] + ".json"
            warnings.warn(f'The path : `{file_name}` format should be `json`.\n'\
                          f"fix the path to {fix_path}")
            path = fix_path   
            
    else:
        path = 'client_secrets.json'
        
    gs_secret = get_client_secrets()
    set_gs_credentials(path, gs_secret)
        