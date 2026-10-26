"""Frozen-backbone linear probing and weighted k-NN for image classification."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


@torch.inference_mode()
def extract_features(encoder, dataset, batch_size, num_workers, device, max_batches=None):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, persistent_workers=num_workers > 0)
    encoder = encoder.to(device).eval()
    features, labels = [], []
    for index, (views, target) in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        images = views[0] if isinstance(views, (list, tuple)) else views
        features.append(encoder(images.to(device)).float().cpu())
        labels.append(target.long().cpu())
    if not features:
        raise ValueError("No images were available for feature extraction.")
    return torch.cat(features), torch.cat(labels)


@torch.inference_mode()
def weighted_knn(train_x, train_y, eval_x, eval_y, num_classes, k=20,
                 temperature=0.07, query_batch_size=256):
    if k < 1 or temperature <= 0:
        raise ValueError("k and temperature must be positive.")
    train_x = F.normalize(train_x.float(), dim=1)
    eval_x = F.normalize(eval_x.float(), dim=1)
    k = min(k, len(train_x))
    correct = 0
    for start in range(0, len(eval_x), query_batch_size):
        query = eval_x[start:start + query_batch_size]
        similarity, indices = (query @ train_x.T).topk(k, dim=1)
        neighbors = train_y[indices]
        weights = ((similarity - similarity[:, :1]) / temperature).exp()
        votes = torch.zeros(len(query), num_classes)
        votes.scatter_add_(1, neighbors, weights)
        correct += int((votes.argmax(dim=1) == eval_y[start:start + len(query)]).sum())
    return correct / len(eval_y)


def linear_probe(train_x, train_y, eval_x, eval_y, num_classes, device,
                 epochs=90, batch_size=256, lr=0.1, seed=0):
    if epochs < 1:
        raise ValueError("Linear-probe epochs must be positive.")
    torch.manual_seed(seed)
    classifier = nn.Linear(train_x.shape[1], num_classes).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr * batch_size / 256,
                                momentum=0.9)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        classifier.train()
        for indices in torch.randperm(len(train_y), generator=generator).split(batch_size):
            logits = classifier(train_x[indices].to(device))
            loss = F.cross_entropy(logits, train_y[indices].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    classifier.eval()
    correct = 0
    with torch.inference_mode():
        for start in range(0, len(eval_y), batch_size):
            predictions = classifier(eval_x[start:start + batch_size].to(device)).argmax(1)
            correct += int((predictions.cpu() == eval_y[start:start + batch_size]).sum())
    return correct / len(eval_y)


def evaluate_classification(model, data, config, smoke=False):
    """Fit probes on deterministic train features; report validation and test separately."""
    data.prepare_data()
    data.setup()
    # Pretraining uses stochastic train views; the validation transform is deterministic.
    data.train_dataset.transform = data.val_dataset.transform
    device = next(model.f.parameters()).device
    max_batches = 2 if smoke else None
    options = (config["batch_size"], config["num_workers"], device, max_batches)
    train_x, train_y = extract_features(model.f, data.train_dataset, *options)
    results = {}
    for split in ("val", "test"):
        dataset = getattr(data, f"{split}_dataset")
        if dataset is None:
            continue
        eval_x, eval_y = extract_features(model.f, dataset, *options)
        results[f"{split}/knn_top1"] = weighted_knn(
            train_x, train_y, eval_x, eval_y, data.num_classes,
            k=config.get("knn_k", 20), temperature=config.get("knn_temperature", 0.07),
        )
        results[f"{split}/linear_top1"] = linear_probe(
            train_x, train_y, eval_x, eval_y, data.num_classes, device,
            epochs=1 if smoke else config.get("linear_probe_epochs", 90),
            batch_size=config["batch_size"], lr=config.get("linear_probe_lr", 0.1),
            seed=config["seed"],
        )
    return results
