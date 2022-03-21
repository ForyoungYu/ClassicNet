import torch

def Test(model, test_dataloader, device):
    """Test"""
    correct = 0.0
    total = 0
    acclist = []
    acc = 0.0
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            if torch.cuda.is_available:
                inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            pred = outputs.argmax(dim=1)
            total += inputs.size(0)
            correct += torch.eq(pred, targets).sum().item()
            acclist.append((correct/total))
            acc = sum(acclist)/len(test_dataloader)
    return acc
