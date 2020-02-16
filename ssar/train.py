from data.egogest_dataset import EgoGestDataSequence
from data.data import check_and_split_data, FixedIndicesSampler, collate_fn_padd
from model.model import SSAR
from socket import gethostname
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
from utils import dfs_freeze, save_model, load_latest, set_bn_train_mode
from matplotlib.animation import FFMpegWriter

import matplotlib.pyplot as plt
import argparse
import torch
import torchvision.models.resnet as resnet
import numpy as np
import os

# import torch
# import gc

results_path = 'results'
mode = 'training'
training_mode = 'lstm-only' # Should be one of ['end-to-end', 'lstm-only'], only applies in 'training' mode
use_mask_loss = False # Should be True for end-to-end or embedding training
batch_size = 25
epochs = 1000
default_acc_bin_idx = 8
fast_forward_step = False
accuracy_bins = 10
grad_accum_steps = 4 # Effective training batch size is equal batch_size x grad_accum_steps
learning_rate = 1e-3
early_stoppping_patience = 5 # Number of epochs that validation accuracy doesn't improve before stopping
rel_poses = torch.linspace(0, 1, accuracy_bins, requires_grad=False)
rel_poses_gpu = rel_poses.cuda()
# Enable to update batch norm running means and variances (only set this if the batch size is large enough for accurate mean / var estimation)
# Only applies in 'end-to-end' training mode
enable_bn_mean_var_update = False


def main():
    global epochs

    # Config
    parser = argparse.ArgumentParser(description="To read EgoGesture Dataset and run through SSAR network")
    parser.add_argument('--path', default='', help='full path to EgoGesture Dataset')
    args = parser.parse_args()
    path = args.path

    # Setup datasets / dataloaders
    image_transform = transforms.Compose([transforms.Resize((126, 224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                          ])
    mask_transform = transforms.Compose([transforms.ToTensor()])
    hostname = gethostname() + "sequence_data"

    dataset = EgoGestDataSequence(path, hostname, image_transform, mask_transform, get_mask=use_mask_loss)
    train_indices, val_indices, test_indices = check_and_split_data(host_name=hostname,
                                                                    data_folder=path,
                                                                    dataset_len=len(dataset),
                                                                    train_fraction=0.6,
                                                                    validation_fraction=0.2)
    if mode == 'training':
        loader_indices = train_indices
    elif mode == 'validation':
        loader_indices = val_indices
    else:
        loader_indices = test_indices
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        sampler=FixedIndicesSampler(loader_indices),
        collate_fn=collate_fn_padd)
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        sampler=FixedIndicesSampler(val_indices),
        collate_fn=collate_fn_padd)
    # test_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     num_workers=8,
    #     pin_memory=True,
    #     sampler=FixedIndicesSampler(test_indices),
    #     collate_fn=collate_fn_padd)

    # Init model and load pre-trained weights
    rnet = resnet.resnet18(False)
    model = SSAR(ResNet=rnet, input_size=83, number_of_classes=83, batch_size=batch_size, dropout=0).cuda()
    model_weights = './weights/final_weights.pth'
    state = model.state_dict()
    loaded_weights = torch.load(model_weights)
    state.update(loaded_weights)
    model.load_state_dict(state)

    # Freeze parts of model we don't want to train
    if mode != 'training':
        model.eval()
    elif training_mode == 'lstm-only':
        model.eval()
        dfs_freeze(model)
        if mode == 'training':
            model.lstms.train()
            dfs_freeze(model.lstms, unfreeze=True)
    elif training_mode == 'end-to-end':
        model.train()
        dfs_freeze(model, unfreeze=True)

        # Enable / Disable running mean variance update on batchnorm layers
        set_bn_train_mode(model, train=enable_bn_mean_var_update)

    # Setup optimizer and loss
    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    if mode == 'training':
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    else:
        optimizer = None

    # Continue from previous training checkpoint
    epoch_resume, step_resume, best_val_loss = load_latest(model, results_path, training_mode, optimizer)
    if not fast_forward_step:
        step_resume = 0

    # Train / test / val setup
    if mode != 'training':
        epoch_resume = 0
        step_resume = 0
        epochs = 1

    # old_tensor_set = set()

    # Accuracy bar plot
    plt.ion()
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 7), dpi=100)
    train_acc_bars   = ax[0].bar(rel_poses, 0, 1 / accuracy_bins)
    val_acc_bars     = ax[1].bar(rel_poses, 0, 1 / accuracy_bins)
    train_loss_line, = ax[2].plot([], [], label='Train Loss')
    val_loss_line,   = ax[2].plot([], [], label='Val Loss')
    train_acc_texts  = [ax[0].text(x, y, "", horizontalalignment='center', verticalalignment='bottom') for x, y in zip(rel_poses, np.ones_like(rel_poses))]
    val_acc_texts    = [ax[1].text(x, y, "", horizontalalignment='center', verticalalignment='bottom') for x, y in zip(rel_poses, np.ones_like(rel_poses))]
    ax[0].set_ylim(0., 1.1)
    ax[0].set_title('Relative Gesture Position vs Training Accuracy')
    ax[1].set_ylim(0., 1.1)
    ax[1].set_title('Relative Gesture Position vs Validation Accuracy')
    ax[2].legend(loc='best')
    # loss_text = ax[2].text(0, -0.2, "Loss: ")
    plt.show()

    # Setup movie moviewriter for writing accuracy plot over time
    # moviewriter = FFMpegWriter(fps=1)
    # moviewriter.setup(fig, os.path.join(results_path, 'accuracy_over_time.mp4'), dpi=100)

    # Main training loop
    if mode == 'training':
        optimizer.zero_grad()
    train_history = {}
    val_history = {}
    for epoch in range(epoch_resume, epochs):
        # Display info
        print(f"Epoch: {epoch}")

        if epoch == epoch_resume and step_resume > 0:
            print(f"Fast forwarding to train step {step_resume}")
        
        # Reset epoch stats
        train_metrics = {}

        # Train
        if mode == 'training':
            print('Training:')
        for step, batch in enumerate(train_loader):
            # Advance train_loader to resume training from last checkpointed position (Note: Assumes same batch size)
            if epoch == epoch_resume and step < step_resume:
                del batch
                continue

            # Save model
            if mode == 'training' and step % 100 == 0 and (step != step_resume or epoch != epoch_resume):
                save_model(model, optimizer, training_mode, epoch, step, best_val_loss, results_path)

            # Do one training step (may not actually step optimizer if doing gradiant accumulation)
            loss, batch_correct_count_samples = process_batch(model, step, batch, criterion, optimizer, mode=mode)
            del batch
            
            # Update metrics
            update_metrics(train_metrics, epoch, loss, batch_correct_count_samples)

            if (step + 1) % 10 == 0:
                # Display metrics
                print_metrics(train_metrics, step)
                update_accuracy_plot(train_acc_bars, train_acc_texts, train_metrics['accuracy_hist'])
        
        # Update train metric history and plots for this epoch
        update_epoch_history(train_history, train_metrics)
        update_loss_plot(train_loss_line, train_history)
        
        last_step_train = step
        
        # Validation
        if mode == 'training':
            print('Validation:')
            val_metrics = {}
            for step, batch in enumerate(val_loader):
                loss, batch_correct_count_samples = process_batch(model, step, batch, criterion, optimizer, mode='validation')

                # Update metrics
                update_metrics(val_metrics, epoch, loss, batch_correct_count_samples)

                if (step + 1) % 10 == 0:
                    # Display metrics
                    print_metrics(val_metrics, step)
                    update_accuracy_plot(val_acc_bars, val_acc_texts, val_metrics['accuracy_hist'])
            
            # Update validation metric history and plots
            update_epoch_history(val_history, val_metrics)
            update_loss_plot(val_loss_line, val_history)

            # Early stoping
            if val_metrics['loss_epoch'] < best_val_loss:
                best_val_loss = val_metrics['loss_epoch']
                patience_counter = 0
                save_model(model, optimizer, training_mode, epoch, step, best_val_loss, results_path, filename_override='model_best.pth')
            else:
                patience_counter += 1
            if patience_counter >= early_stoppping_patience:
                print(f'Validation accuracy did not improve for {patience_counter} epochs, stopping')
                break

        step = last_step_train

    # Save final model
    if mode == 'training' and step != step_resume or epoch != epoch_resume:
        save_model(model, optimizer, training_mode, epoch, step, best_val_loss, results_path)
    
    print('Done!')
    
    plt.ioff()
    plt.show()

def update_loss_plot(loss_line, history):
    loss_line.set_data(history['epoch'], history['loss_epoch'])
    ax = loss_line.axes
    ax.relim()
    ax.autoscale_view()

    plt.draw()
    plt.pause(0.001)

def print_metrics(metrics, step):
    print(f"Step: {step + 1},",
          f"Processed Gestures: {metrics['gesture_count']},",
          f"Correct Count (@t={rel_poses[default_acc_bin_idx]:.2f}): {metrics['correct_count_hist'][default_acc_bin_idx]},",
          f"Accuracy (@t={rel_poses[default_acc_bin_idx]:.2f}): {metrics['accuracy_hist'][default_acc_bin_idx]:.4f},",
          f"Loss Epoch: {metrics['loss_epoch']:.5f},",
          f"Loss Last Batch: {metrics['loss_last_batch']:.5f}")

def update_accuracy_plot(rects, texts, accuracy_hist):
    # Plot accuracy histogram
    for i, rect in enumerate(rects):
        rect.set_height(accuracy_hist[i])
        texts[i].set_position([rel_poses[i], accuracy_hist[i]])
        texts[i].set_text(f'{accuracy_hist[i]:.3f}')

    plt.draw()
    plt.pause(0.001)

def update_epoch_history(history, metrics):
    for key in metrics:
        metric = np.expand_dims(metrics[key], axis=0)
        if key not in history:
            history[key] = np.zeros((0,) + metric.shape[1:])
        history[key] = np.concatenate([history[key], metric], axis=0)
    
    return history

def update_metrics(metrics, epoch, loss, batch_correct_count_samples):
    if 'gesture_count' not in metrics:
        metrics['gesture_count'] = 0
    if 'loss_sum' not in metrics:
        metrics['loss_sum'] = 0
    if 'losses_seen' not in metrics:
        metrics['losses_seen'] = 0
    if 'correct_count_hist' not in metrics:
        metrics['correct_count_hist'] = np.array([0] * accuracy_bins)

    metrics['epoch'] = epoch
    metrics['gesture_count'] += batch_size
    metrics['loss_sum'] += loss
    metrics['losses_seen'] += 1
    metrics['correct_count_hist'] += batch_correct_count_samples
    metrics['loss_last_batch'] = loss
    metrics['loss_epoch'] = metrics['loss_sum'] / metrics['losses_seen']
    metrics['accuracy_hist'] = metrics['correct_count_hist'] / metrics['gesture_count']

    return metrics

# Perfom one training step on one batch of data (may not actually step optimizer if doing gradiant accumulation)
def process_batch(model, step, batch, criterion, optimizer, mode='training'):
    images = batch['images']

    images = images.cuda()
    labels = pad_packed_sequence(batch['label'], batch_first=True)[0].cuda()
    lengths = batch['length'].cuda()
    if use_mask_loss:
        true_mask = pad_packed_sequence(batch['masks'], batch_first=True)[0].cuda()

    generated_labels = model(images, lengths=lengths, get_mask=use_mask_loss, get_lstm_state=False)
    if use_mask_loss:
        mask, generated_labels = generated_labels

    # end_indices = (lengths - 1)
    # indices = end_indices.view(-1, 1, 1).repeat(1, generated_labels.shape[1], 1)
    # generated_labels = generated_labels.gather(2, indices).squeeze(2)
    # indices = end_indices.view(-1, 1)
    # labels = labels.gather(1, indices).squeeze(1)

    loss = criterion(generated_labels, labels)
    if use_mask_loss:
        loss += criterion(mask, true_mask)
    loss /= grad_accum_steps


    if mode == 'training':
        loss.backward()
    
    if mode == 'training' and (step + 1) % grad_accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

    with torch.no_grad():
        # end_indices = (lengths // 2) # Measuring accuracy somewhere in middle
        # indices = end_indices.view(-1, 1, 1).repeat(1, generated_labels.shape[1], 1)
        # generated_label = generated_labels.gather(2, indices).squeeze(2)
        # generated_label = torch.argmax(generated_label, dim=1)
        # indices = end_indices.view(-1, 1)
        # label = labels.gather(1, indices).squeeze(1)
        # correct_count = torch.sum(label == generated_label).item()

        end_indices = (lengths - 1)
        end_indices = (end_indices.view(-1, 1) * rel_poses_gpu.view(1, -1)).long()
        indices = end_indices.view(-1, 1, accuracy_bins).repeat(1, generated_labels.shape[1], 1)
        generated_labels = generated_labels.gather(2, indices)
        generated_labels = torch.argmax(generated_labels, dim=1)
        indices = end_indices
        labels = labels.gather(1, indices)
        correct_count_hist = torch.sum(labels == generated_labels, axis=0).cpu().numpy()

        # if count == 0:
        #     im_plt = plt.imshow(masks[0, lengths[0] // 2, 1].cpu())
        # else:
        #     im_plt.set_data(masks[0, lengths[0] // 2, 1].cpu())
        # plt.draw()
        # plt.pause(0.0001)
        return loss.item() * grad_accum_steps, correct_count_hist # Return undivided loss

    # cur_tensor_set = set()
    # for obj in gc.get_objects():
    #     try:
    #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
    #             cur_tensor_set.add((type(obj), obj.size()))
    #     except:
    #         pass
    # print(cur_tensor_set - old_tensor_set)
    # print(len(cur_tensor_set))
    # old_tensor_set = cur_tensor_set

if __name__ == "__main__":
    main()