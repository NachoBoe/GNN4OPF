import torch
from scripts.metric import NormalizedError

def run_epoch(model, train_loader, optimizer, criterion,epoch=None,writer=None):
    model.train()
    total_loss = 0
    total_metric = 0
    normalized_error = NormalizedError()

    for data in train_loader:
        optimizer.zero_grad()
        output = model(data[0])
        target = data[1]
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Calcular la métrica
        metric = normalized_error(output, target)
        total_metric += metric.item()

    avg_loss = total_loss / len(train_loader)
    avg_metric = total_metric / len(train_loader)

    writer.add_scalar('Train/Training Loss', avg_loss, epoch)
    writer.add_scalar('Train/Training Metric', avg_metric, epoch)

    return avg_loss, avg_metric


# Definimos la funcion de evaluacion
def evaluate(model, data_loader, criterion,epoch=None,writer=None,test=False):
    model.eval()
    total_loss = 0
    total_metric = 0
    normalized_error = NormalizedError()

    with torch.no_grad():
        for data in data_loader:
            output = model(data[0])
            target = data[1]
            loss = criterion(output, target)  # target should contain true values for nodes with missing features
            total_loss += loss.item()
            # Calcular la métrica
            metric = normalized_error(output, target)
            total_metric += metric.item()

    avg_loss = total_loss / len(data_loader)
    avg_metric = total_metric / len(data_loader)

    if not test:
        if epoch != None:
            writer.add_scalar('Val/Validation Loss', avg_loss, epoch)
            writer.add_scalar('Val/Validation Metric', avg_metric, epoch)
    else:
        writer.add_scalar('Test/Test Loss', avg_loss, epoch)
        writer.add_scalar('Test/Test Metric', avg_metric, epoch)

    return avg_loss, avg_metric
