import os
import json
import torch
from s4_pt.models.s4_model import S4Model
from s4_pt.data.seq_cifar import get_seq_cifar

os.makedirs("artifacts/seq_cifar", exist_ok=True)
loss_log = []

device = "cuda" if torch.cuda.is_available() else "cpu"

model = S4Model(d_model=1, num_classes=10, depth=4).to(device)
loader = get_seq_cifar(batch_size=16)

opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(3):
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        loss_log.append(float(loss.item()))
        opt.step()

        if i % 50 == 0:
            print(f"epoch {epoch} batch {i} loss {loss.item():.3f}")

with open("artifacts/seq_cifar/loss.json", "w") as f:
    json.dump(loss_log, f)
torch.save(model.state_dict(), "artifacts/seq_cifar/model.pth")