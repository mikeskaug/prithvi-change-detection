


def train_one_epoch(trainData, model, loss_fn, optimizer, device, train_loss=[], verbose=False):
    '''
    Train the model.
    Arguments:
            trainData (DataLoader object) -- Batches of image chips from PyTorch custom dataset(AquacultureData)
            model (initialized model) -- Choice of segmentation Model to train.
            loss_fn -- Chosen function to calculate loss over training samples.
            optimizer -- Chosen function for optimization.
            device -- (str) Either 'cuda' or 'cpu'.
            train_loss -- (empty list) To record average training loss for each epoch.
            verbose -- (bool) Whether to print extra debug info
            
    '''
    model.train()

    epoch_loss = 0
    num_train_batches = len(trainData)

    for idx, (images, labels) in enumerate(trainData):

        if verbose:
            print(f'Batch [{idx}/{num_train_batches}]')

        img = images.to(device)
        label = labels.to(device)
        
        optimizer.zero_grad()
        model_out = model(img)

        loss = loss_fn(model_out, label)
    
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()


    print(f'train loss:{epoch_loss / num_train_batches}')

    if train_loss is not None:
        train_loss.append(float(epoch_loss / num_train_batches))