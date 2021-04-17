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

def train(cfg, device, model, train_loader, test_loader, criterion, accuracy_func, optimizer, scheduler=None, freq_scheduler_update=False, log_freq=50, eval_freq=200):

  train_loss = 0
  batch_count = 0
  elem_count = 0

  wandb.watch(model, criterion, log='all', log_freq=log_freq)
  
  for epoch in tqdm(range(cfg.epochs), desc="Epochs"):
    for X, Y in tqdm(train_loader, desc="Training", leave=False):
      if batch_count % eval_freq == 0:
        test_loss, accuracy = test(device, model, test_loader, criterion, accuracy_func)
        wandb.log({"epoch": epoch, "test_loss": test_loss, "accuracy": accuracy, "samples"}, step=elem_count)
        model.train()

      loss_batch = train_batch(device, model, X, Y, criterion, optimizer)

      if scheduler and freq_scheduler_update:
        scheduler.step()

      train_loss += loss_batch
      batch_count += 1
      elem_count += cfg.batch_size

      if batch_count % log_freq == 0:
        train_loss /= log_freq
        learning_rate = optimizer.param_groups[0]['lr']
        wandb.log({"train_loss": train_loss, "learning_rate": learning_rate}, step=elem_count)
        batch_count = 0
        train_loss = 0
      
    
    

    if scheduler and not freq_scheduler_update:
      scheduler.step(test_loss)