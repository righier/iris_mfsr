from tqdm.auto import tqdm 
import wandb

def test_batch(device, model, X, Y, criterion, accuracy_func):
  X, Y = X.to(device), Y.to(device)
  outputs = model(X)
  loss = float(criterion(outputs, Y))
  accuracy = float(accuracy_func(outputs, Y))
  return loss, accuracy

def test(device, model, test_loader, criterion, accuracy_func):
  model.eval()

  loss = 0
  accuracy = 0
  count = 0

  for X, Y in tqdm(test_loader, desc="Evaluation", leave=False):
    b_loss, b_accuracy = test_batch(device, model, X, Y, criterion, accuracy_func)
    b_loss, b_accuracy = float(b_loss), float(b_accuracy)
    loss += b_loss
    accuracy += b_accuracy
    count += 1
  
  loss = loss / count
  accuracy = accuracy / count
  return loss, accuracy

def train_batch(device, model, X, Y, criterion, optimizer):
  X, Y = X.to(device), Y.to(device)
  outputs = model(X)
  loss = criterion(outputs, Y)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  loss = float(loss)
  return loss

def train(cfg, device, model, train_loader, test_loader, criterion, accuracy_func, optimizer, scheduler=None, log_freq=50):

  train_loss = 0
  batch_count = 0
  elem_count = 0

  wandb.watch(model, criterion, log='all', log_freq=log_freq)
  
  for epoch in tqdm(range(cfg.epochs), desc="Epochs"):

    model.train()

    for X, Y in tqdm(train_loader, desc="Training", leave=False):
      loss_batch = train_batch(device, model, X, Y, criterion, optimizer)

      train_loss += loss_batch
      batch_count += 1
      elem_count += cfg.batch_size

      if batch_count == log_freq:
        train_loss /= batch_count
        wandb.log({"train_loss": train_loss}, step=elem_count)
        batch_count = 0
        train_loss = 0
    
    test_loss, accuracy = test(device, model, test_loader, criterion, accuracy_func)
    wandb.log({"epoch": epoch, "test_loss": test_loss, "accuracy": accuracy}, step=elem_count)

    if scheduler:
      scheduler.step()