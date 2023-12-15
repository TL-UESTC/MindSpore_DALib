import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.dataset import GeneratorDataset
import tqdm


def collect_feature(data_loader: GeneratorDataset, feature_extractor: nn.Cell, max_num_features=None) -> mindspore.Tensor:
    """
    Fetch data from `data_loader`, and then use `feature_extractor` to collect features

    Args:
        data_loader (torch.utils.data.DataLoader): Data loader.
        feature_extractor (torch.nn.Module): A feature extractor.
        device (torch.device)
        max_num_features (int): The max number of features to return

    Returns:
        Features in shape (min(len(data_loader), max_num_features * mini-batch size), :math:`|\mathcal{F}|`).
    """
    feature_extractor.set_train(False)
    all_features = []
    for i, data in enumerate(tqdm.tqdm(data_loader)):
        if max_num_features is not None and i >= max_num_features:
            break
        inputs = data[0]
        feature = feature_extractor(inputs)
        all_features.append(feature)
    return ops.concat(all_features, axis=0)
