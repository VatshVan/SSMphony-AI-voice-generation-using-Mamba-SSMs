import os
import json
import torch
from s4_pt.models.s4_model import S4Model
from s4_pt.data.speech_commands import get_speech_commands

os.makedirs("artifacts/speech", exist_ok=True)
loss_log = []

model = S4Model(d_model=1, num_classes=35)
loader = get_speech_commands()

opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(3):
    for i, (x, y) in enumerate(loader):
        # print("batch", i, x.shape)
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss_log.append(float(loss.item()))
        loss.backward()
        opt.step()
    print("epoch", epoch, "loss", loss.item())

with open("artifacts/speech/loss.json", "w") as f:
    json.dump(loss_log, f)
torch.save(model.state_dict(), "artifacts/speech/model.pth")