import copy
import torch

try:  # Use tqdm if installed for a nice progress bar
    from tqdm import tqdm
except ImportError:  # Dummy tqdm polyfill
    tqdm = lambda *args, **kwargs: args[0]


def default_on_epoch_end(model, epoch, epoch_train_loss, epoch_valid_loss, best_valid_loss, epoch_train_metrics,
                         epoch_valid_metrics):
    '''
    Default callback for train_validate's on_epoch_end param. Triggered on every epoch's end.

    :param model:
    :param epoch:
    :param epoch_train_loss: Number
    :param epoch_valid_loss: Number
    :param best_valid_loss: Number
    :param epoch_train_metrics: Dict like {'Accuracy': 0.35, 'MyMetric': 5.3 }. Same keys as train_validate's metric_factories param
    :param epoch_valid_metrics: Dict like {'Accuracy': 0.35, 'MyMetric': 5.3 }. Same keys as train_validate's metric_factories param
    :return:
    '''
    print(
        f'Epoch {epoch}. Val loss: {epoch_valid_loss}. Train loss: {epoch_train_loss} Val metrics: {epoch_valid_metrics} Train metrics: {epoch_train_metrics}')
    if epoch_valid_loss > best_valid_loss:
        # With defaults here training will stop returning weights with best_valid_loss
        print(f'Validation loss increased from {best_valid_loss} to {epoch_valid_loss}.')


def train_validate(model, train_loader, valid_loader, criterion, optimizer, epochs=1, metric_factories={},
                   stop_on_loss_increase=True, use_gpu=True, on_epoch_end=default_on_epoch_end):
    '''
    :param model:
    :param train_loader:
    :param valid_loader:
    :param criterion:
    :param optimizer:
    :param epochs:
    :param metric_factories: dict with torchmetrics-like metric factories. Example: {'avg': torchmetrics.Average}
    :param stop_on_loss_increase:
    :param use_gpu: Tries to use gpu if cuda available
    :param on_epoch_end: f(model, epoch, train_loss, valid_loss, best_valid_loss, train_metrics, valid_metrics)
    :return: Best model's state_dict.
    '''
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    model.to(device)
    loaders = {'train': train_loader, 'valid': valid_loader}
    best_valid_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        # Make a train and validation pass on every epoch.
        epoch_avg_loss = {'train': 0, 'valid': 0}
        # Computed metrics. metrics ~= { 'train': { 'accuracy': 0.8, 'other': 0.5 }, 'valid': ... }
        metrics = {'train': None, 'valid': None}
        for phase in ['train', 'valid']:
            # Instantiate the metrics by calling their factories
            avg_loss, metrics[phase] = _forward_backward_pass(model, loaders[phase], criterion, optimizer,
                                                              metric_factories, phase, device)
            epoch_avg_loss[phase] += avg_loss
            if phase == 'valid':  # If epoch is completed
                if on_epoch_end:
                    on_epoch_end(model, epoch, epoch_avg_loss['train'], epoch_avg_loss['valid'], best_valid_loss,
                                 metrics['train'], metrics['valid'])
                epoch_valid_loss = epoch_avg_loss['valid']  # Alias
                # If there's a new minimum loss, save the model weights
                # If the validation loss decreases by more than 1%, save the model
                # if best_valid_loss == float('inf') or (best_valid_loss - epoch_valid_loss) / best_valid_loss > 0.01:
                if epoch_valid_loss < best_valid_loss:
                    best_valid_loss = epoch_valid_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                elif stop_on_loss_increase:
                    return best_model_state

    return best_model_state


def _forward_backward_pass(model, loader, criterion, optimizer, metric_factories, phase, device):
    '''
    @private
    :param model:
    :param loader:
    :param criterion:
    :param optimizer:
    :param metric_factories: dict with torchmetrics-like metric factories. Example: {'avg': torchmetrics.Average}
    :param phase: If 'train', it will perform backpropagation.
    :param device:
    :return: The pass avg_loss and the metric_instances (uncomputed)
    '''
    metric_instances = {key: Metric().to(device) for key, Metric in metric_factories.items()}
    model.train() if phase == 'train' else model.eval()  # Disables gradient computation and others
    avg_loss = 0
    torch.set_grad_enabled(phase == 'train') # Disable grad for validation/test
    for features, target in tqdm(loader, desc=phase):
        features, target = features.to(device), target.to(device)
        pred = model(features)
        loss = criterion(pred, target)
        if phase == 'train':
            optimizer.zero_grad()  # clear the gradients of all optimized variables
            loss.backward()  # backward pass: compute gradient of the loss with respect to model parameters
            optimizer.step()  # perform a single optimization step (parameter update)
        # Store metric data
        [metric_instance.update(pred, target) for metric_instance in metric_instances.values()]
        avg_loss += loss.item() * features.size(0)  # * To undo default loss avg
    torch.set_grad_enabled(True)    # Always enable grad
    avg_loss /= len(loader.sampler)  # loss per input (average, dividing sum by dataset size)
    # Compute metrics. metrics ~= { 'accuracy': 0.8, 'other': 0.5 }
    metrics = {key: metric_instance.compute().item() for key, metric_instance in metric_instances.items()}
    return avg_loss, metrics


def test(model, loader, criterion, optimizer, metric_factories={}, use_gpu=True):
    '''
    Performs a forward pass. Intended for user consumption
    :param model:
    :param loader:
    :param criterion:
    :param optimizer:
    :param metric_factories: dict with torchmetrics-like metric factories. Example: {'avg': torchmetrics.Average}
    :param device: None to autodetect gpu
    :return: The avg_loss and the computed metrics as a tuple
    '''
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    model.to(device)
    return _forward_backward_pass(model, loader, criterion, optimizer, metric_factories, phase='test', device=device)
