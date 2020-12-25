from encoder.visualizations import Visualizations
from encoder.data_objects import SpeakerVerificationDataLoader, SpeakerVerificationDataset
from encoder.params_model import *
from encoder.model import SpeakerEncoder

from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder

from pathlib import Path
from utils.profiler import Profiler
from pathlib import Path
import torch
import iter_utils
from torch.utils.data import DataLoader
from iter_dataset import ASDataset
from iter_params import get_params


def train(params):
    iter_utils.print_INFO("train", "training starts")
    logger = iter_utils.Logger(params.log_path)

    train_set = ASDataset(params, "train")
    test_set = ASDataset(params, "test")
    train_loader = DataLoader(
        train_set,
        batch_size=params.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=params.batch_size,
        shuffle=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_device = torch.device("cpu")

    synthesizer = Synthesizer(
        Path("synthesizer/saved_models/logs-pretrained/taco_pretrained/"),
        low_mem=False,
        seed=params.random_seed
    )
    vocoder.load_model("vocoder/saved_models/pretrained/pretrained.pt")

    # vis = Visualizations(run_id, vis_every, server="http://localhost", disabled=False)
    # vis.log_dataset(dataset)
    # vis.log_params()
    # device_name = str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    # vis.log_implementation({"Device": device_name})

    model = SpeakerEncoder(device, loss_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    checkpoint_path = "encoder/saved_models/pretrained.pt"
    checkpoint = torch.load(checkpoint_path)
    init_step = checkpoint["step"]
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    optimizer.param_groups[0]["lr"] = params.learning_rate

    model.train()


if __name__ == "__main__":
    params = get_params()
    train(params)
