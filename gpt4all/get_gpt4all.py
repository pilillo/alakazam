from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import requests
from pathlib import Path
from tqdm import tqdm
import os

gpt4all_model_repo = 'http://gpt4all.io/models/'

def download(target_folder, model_name):
    local_path = os.path.join(target_folder, model_name)
    if not Path(local_path).is_file():
        url = os.path.join(gpt4all_model_repo, model_name)
        response = requests.get(url, stream=True)
        with open(local_path, 'wb') as f:
            for chunk in tqdm(response.iter_content(chunk_size=8192)):
                if chunk:
                    f.write(chunk)

def load(target_folder, model_name, n_ctx=512, n_threads=16):
    callbacks = [StreamingStdOutCallbackHandler()]
    download(target_folder, model_name)
    local_path = os.path.join(target_folder, model_name)
    return GPT4All(model=local_path, callbacks=callbacks, verbose=False, n_ctx=n_ctx, n_threads=n_threads)