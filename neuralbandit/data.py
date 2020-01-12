import urllib.request, pandas as pd, os, itertools, torch
from torch.utils.data import Dataset
from os.path import join
from pathlib import Path

data_path = "./data/"

def download_dataset(source_url, destination):
    Path(data_path).mkdir(parents=True, exist_ok=True)
    if not os.path.isfile(source_url):
        urllib.request.urlretrieve(source_url, destination)

class ContextualBanditDataset(Dataset):

    def __init__(self, contexts, target, optimal_accuracy = 1):
        """
        contexts: matrix of size N x D
        target: the list of size N of the classes associated to the contexts
        """
        assert len(contexts) == len(target)

        self._contexts = contexts
        self._target = target
        self._K = len(self._target.astype('category').cat.categories)
        self._D = len(self._contexts[0])
        
        self.optimal_accuracy = optimal_accuracy
            
    def __len__(self):
        return len(self._target)

    def __getitem__(self, idx):
        return self._contexts[idx], self._target.iloc[idx]
    
    @property
    def K(self): 
        return self._K
    
    @property
    def D(self): 
        return self._D        
        

def get_cov_dataset():
    """
    Forest Cover Type DataSet

    Elevation                               quantitative    meters                       Elevation in meters
    Aspect                                  quantitative    azimuth                      Aspect in degrees azimuth
    Slope                                   quantitative    degrees                      Slope in degrees
    Horizontal_Distance_To_Hydrology        quantitative    meters                       Horz Dist to nearest surface water features
    Vertical_Distance_To_Hydrology          quantitative    meters                       Vert Dist to nearest surface water features
    Horizontal_Distance_To_Roadways         quantitative    meters                       Horz Dist to nearest roadway
    Hillshade_9am                           quantitative    0 to 255 index               Hillshade index at 9am, summer solstice
    Hillshade_Noon                          quantitative    0 to 255 index               Hillshade index at noon, summer soltice
    Hillshade_3pm                           quantitative    0 to 255 index               Hillshade index at 3pm, summer solstice
    Horizontal_Distance_To_Fire_Points      quantitative    meters                       Horz Dist to nearest wildfire ignition points
    Wilderness_Area (4 binary columns)      qualitative     0 (absence) or 1 (presence)  Wilderness area designation
    Soil_Type (40 binary columns)           qualitative     0 (absence) or 1 (presence)  Soil Type designation
    Cover_Type (7 types)                    integer         1 to 7                       Forest Cover Type designation
    """

    cov_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    cov_path = join(data_path, "covtype.data.gz")

    download_dataset(cov_url, cov_path)

    names = list(itertools.chain([
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points"],
        [f"Wilderness_Area_{i}" for i in range(4)],
        [f"Soil_Type_{i}" for i in range(40)],
        ["Cover_Type"]
    ))

    df = pd.read_csv(cov_path, compression='gzip', header = None, names = names)

    """
    Following Allesiardo et al (https://arxiv.org/abs/1409.8191) we discretize each continuous features using an equal frequency binning of size 5
    """
    continuous_features = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points", "Hillshade_Noon"]
    nb_bin = 5

    for i_feature_name in continuous_features:
        tmp = pd.get_dummies(pd.qcut(df[i_feature_name], nb_bin))
        tmp.columns = [f"{i_feature_name}_{i}" for i in range(nb_bin)]

        del df[i_feature_name]
        for i_col in tmp.columns:
            df[i_col] = tmp[i_col]

        del tmp

    target_name = "Cover_Type"

    target = df[target_name].astype('category').cat.codes
    del df[target_name]

    contexts = torch.from_numpy(df.values).float()
    del df
    
    """
    The optimal policy used by Allesiardo et al (https://arxiv.org/abs/1409.8191) achieves 93% of good classification
    """
    return ContextualBanditDataset(contexts, target, 0.93)
