from statistics import mean
from typing import List, Optional

from torch.utils.tensorboard import SummaryWriter

from config import SimpleArgumentParser
from net import Net

Loss = float


class Learner:

    def __init__(self, cfg: SimpleArgumentParser, device: str, writer: Optional[SummaryWriter]):
        classifier: Net = Net(cfg=cfg)
        self.cfg: SimpleArgumentParser = cfg
        self.device: str = device
        self.writer: Optional[SummaryWriter] = writer

        self.classifier = classifier.to(device)

    def smooth_end_losses(self, losses: List[float]) -> Loss:
        """average the last quarter of the losses"""
        last_few_losses = losses[-len(losses) // 4:len(losses)]
        if last_few_losses:
            return mean(last_few_losses)
        else:
            print("not enough losses to smooth")
            return losses[-1]

    def reset(self):
        """Resets to a new un-trained classifier."""
        self.classifier = Net(cfg=self.cfg)
