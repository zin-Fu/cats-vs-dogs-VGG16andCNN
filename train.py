from data_loader import *
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def train(model, optimizer):
    start_time = time.time()
    train_acc_lst, valid_acc_lst = [], []  # 描点
    train_loss_lst, valid_loss_lst = [], []

    for epoch in range(NUM_EPOCHS):

        model.train()

        for batch_idx, (features, targets) in enumerate(train_loader):
            # PREPARE MINIBATCH
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)

            # FORWARD AND BACK PROP
            logits, probas = model(features)
            cost = F.cross_entropy(logits, targets)
            optimizer.zero_grad()

            cost.backward()

            optimizer.step()

            # LOGGING
            if not batch_idx % 20:
                print(f'Epoch: {epoch + 1:03d}/{NUM_EPOCHS:03d} | '
                      f'Batch {batch_idx:03d}/{len(train_loader):03d} |'
                      f' Cost: {cost:.4f}')

        model.eval()
        with torch.set_grad_enabled(False):
            train_acc, train_loss = compute_accuracy_and_loss(model, train_loader, device=DEVICE)
            valid_acc, valid_loss = compute_accuracy_and_loss(model, valid_loader, device=DEVICE)
            train_acc_lst.append(train_acc)
            valid_acc_lst.append(valid_acc)
            train_loss_lst.append(train_loss)
            valid_loss_lst.append(valid_loss)
            print(f'Epoch: {epoch + 1:03d}/{NUM_EPOCHS:03d} Train Acc.: {train_acc:.2f}%'
                  f' | Validation Acc.: {valid_acc:.2f}%')

        elapsed = (time.time() - start_time)/60
        print(f'Time elapsed: {elapsed:.2f} min')

    elapsed = (time.time() - start_time)/60
    print(f'Total Training Time: {elapsed:.2f} min')

    # visualize loss
    plt.plot(range(1, NUM_EPOCHS + 1), train_loss_lst, label='Training loss')
    plt.plot(range(1, NUM_EPOCHS + 1), valid_loss_lst, label='Validation loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross entropy')
    plt.xlabel('Epoch')
    plt.show()

    # visualize accuracy
    train_acc_lst = torch.tensor(train_acc_lst)
    valid_acc_lst = torch.tensor(valid_acc_lst)

    plt.plot(range(1, NUM_EPOCHS + 1), train_acc_lst, label='Training accuracy')
    plt.plot(range(1, NUM_EPOCHS + 1), valid_acc_lst, label='Validation accuracy')
    plt.legend(loc='upper left')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()


