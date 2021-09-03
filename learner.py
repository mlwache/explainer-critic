
from config import SimpleArgumentParser
from net import Net

Loss = float


class Learner:
    def __init__(self, cfg: SimpleArgumentParser):
        classifier: Net = Net(cfg=cfg)
        self.classifier = classifier.to(cfg.DEVICE)
        self.cfg: SimpleArgumentParser = cfg

    # def _process_batch(self, loss_function: nn.Module, inputs: Tensor, labels: Tensor,
    #                    n_current_batch: int, optimizer: Optimizer, n_explainer_batch: int = 0,
    #                    explanations: List[Tensor] = None, critic_loader: DataLoader[Any] = None) -> Loss:
    #     in_critic = bool(explanations)
    #     in_explainer = bool(critic_loader)
    #     assert in_explainer != in_critic
    #
    #     inputs = inputs.to(self.cfg.DEVICE)
    #     labels = labels.to(self.cfg.DEVICE)
    #
    #     optimizer.zero_grad()
    #
    #     if in_critic:
    #         inputs *= explanations[n_current_batch]
    #     outputs = self.classifier(inputs)
    #     loss = loss_function(outputs, labels)
    #     classification_loss = loss.item()
    #
    #     if in_explainer:
    #         loss = self._add_explanation_loss(critic_loader, loss, n_current_batch)
    #
    #     loss.backward()  # or retain_graph=True ? TODO: clarify computation graph
    #     optimizer.step()
    #
    #     if in_critic:
    #         self._sanity_check(n_current_batch)
    #         self._log_results(loss, n_current_batch, n_explainer_batch)
    #     if in_explainer:
    #         self._sanity_check_batch_device(n_current_batch, outputs)
    #         self._record_losses(loss, classification_loss, n_current_batch)
    #     return loss.item()
    #
    # def _sanity_check(self, n_current_batch=0):
    #     raise NotImplementedError("should only be called in critic/explainer instance")
    #
    # def _log_results(self, loss, n_current_batch, n_explainer_batch):
    #     raise NotImplementedError("should only be called in critic/explainer instance")
    #
    # def _add_explanation_loss(self, critic_loader, loss_classification, n_current_batch):
    #     raise NotImplementedError("should only be called in critic/explainer instance")
    #
    # def _sanity_check_batch_device(self, n_current_batch, outputs):
    #     raise NotImplementedError("should only be called in critic/explainer instance")
    #
    # def _record_losses(self, loss, loss_classification, n_current_batch):
    #     raise NotImplementedError("should only be called in critic/explainer instance")
