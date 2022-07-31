import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

api = KaggleApi()
api.authenticate()


api.dataset_download_files('prondeau/the-car-connection-picture-dataset')


with zipfile.ZipFile('the-car-connection-picture-dataset.zip', 'r') as zipref:
    zipref.extractall('cars/')