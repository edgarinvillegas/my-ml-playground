import copy
import torch

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
    :param use_gpu:
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
        metric_instances = {'train': None, 'valid': None}
        for phase in ['train', 'valid']:
            # Instantiate the metrics by calling their factories
            metric_instances[phase] = {key: Metric().to(device) for key, Metric in metric_factories.items()}
            model.train() if phase == 'train' else model.eval()  # Disables gradient computation and others
            loader = loaders[phase]
            for features, target in loader:
                features, target = features.to(device), target.to(device)
                pred = model(features)
                loss = criterion(pred, target)
                if phase == 'train':
                    optimizer.zero_grad()  # clear the gradients of all optimized variables
                    loss.backward()  # backward pass: compute gradient of the loss with respect to model parameters
                    optimizer.step()  # perform a single optimization step (parameter update)
                # Store metric data
                [metric_instance.update(pred, target) for metric_instance in metric_instances[phase].values()]
                epoch_avg_loss[phase] += loss.item() * features.size(0)  # * To undo default loss avg

            epoch_avg_loss[phase] /= len(loader.sampler)  # loss per input (average, dividing sum by dataset size)
            if phase == 'valid':  # If epoch is completed
                if on_epoch_end:
                    # Compute metrics. computed_metrics ~= { 'train': { 'accuracy': 0.8, 'other': 0.5 }, 'valid': ... }
                    computed_metrics = {phase: {key: metric_instance.compute().item() for key, metric_instance in
                                                metric_instances[phase].items()} for phase in ['train', 'valid']}
                    on_epoch_end(model, epoch, epoch_avg_loss['train'], epoch_avg_loss['valid'], best_valid_loss,
                                 computed_metrics['train'], computed_metrics['valid'])
                    # If there's a new minimum loss, save the model weights
                epoch_valid_loss = epoch_avg_loss['valid']  # Alias
                if epoch_valid_loss < best_valid_loss:
                    best_valid_loss = epoch_valid_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                elif stop_on_loss_increase:
                    return best_model_state

    return best_model_state
