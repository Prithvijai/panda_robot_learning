import random

import wandb

# Start a new wandb run to track this script.
run = wandb.init(
  entity="prithvijai",         # your W&B username or team
  project="openvla-clear-cache"  # a project you have rights to
)

# Simulate training.
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset

    # Log metrics to wandb.
    run.log({"acc": acc, "loss": loss})

# Finish the run and upload any remaining data.
run.finish()