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
from spatial_transforms import MultiScaleRandomCrop, Compose, SpatialElasticDisplacement

import matplotlib.pyplot as plt
import argparse
import torch
import torchvision.models.resnet as resnet
import numpy as np
import os

# import torch
# import gc

results_path = 'results'
mode = 'training' # Should be one of ['training', 'validation', 'testing']
training_mode = 'end-to-end' # Should be one of ['end-to-end', 'lstm-only'], only applies in 'training' mode
use_mask_loss = (training_mode == 'end-to-end') # Should be True for end-to-end or embedding training
batch_size = 8
epochs = 1000
default_acc_bin_idx = 8
restore_training_variables = False # Whether to load that last epoch, training step and best validation score to resume training from
accuracy_bins = 10
grad_accum_steps = 1 # Effective training batch size is equal batch_size x grad_accum_steps
learning_rate = 1e-3
dropout = 0.0
early_stoppping_patience = 50 # Number of epochs that validation accuracy doesn't improve before stopping
# Control variables for multiscale random crop transform used during training
do_data_augmentation = True
initial_scale = 1
n_scales = 5
scale_step = 0.84089641525
# Positions at which to measure / display accuracy
rel_poses = torch.linspace(0, 1, accuracy_bins, requires_grad=False)
rel_poses_gpu = rel_poses.cuda()
label_mask_value = -100
frame_start_loss_calc = 1
num_workers = 8

# Used to quickly switch model between modes for training and validation
def set_train_mode(model, train=True):
    if train:
        # Switch to train mode while freezing parts of the model we don't want to train
        if mode != 'training':
            model.eval()
            dfs_freeze(model)
        elif training_mode == 'lstm-only':
            model.eval()
            model.lstms.train()
            dfs_freeze(model)
            dfs_freeze(model.lstms, unfreeze=True)
        elif training_mode == 'end-to-end':
            model.train()
            dfs_freeze(model, unfreeze=True)
            # Disable running mean variance update during end-to-end training (batch size too small)
            set_bn_train_mode(model, train=False)
    else:
        model.eval()
        dfs_freeze(model)

def main():
    global epochs

    # Config
    parser = argparse.ArgumentParser(description="To read EgoGesture Dataset and run through SSAR network")
    parser.add_argument('--path', default='', help='full path to EgoGesture Dataset')
    args = parser.parse_args()
    path = args.path

    # Setup multiscale random crop
    scales = [initial_scale]
    for _ in range(1, n_scales):
        scales.append(scales[-1] * scale_step)

    # Setup datasets / dataloaders
    if do_data_augmentation:
        train_spatial_transforms = Compose([MultiScaleRandomCrop(scales, (126, 224)),
                                            SpatialElasticDisplacement()])
    else:
        train_spatial_transforms = transforms.Resize((126, 224))
    image_transform_train = Compose([train_spatial_transforms,
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    image_transform_val   = Compose([transforms.Resize((126, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    image_transform_test  = Compose([transforms.Resize((126, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    mask_transform        = Compose([train_spatial_transforms, transforms.ToTensor()])
    if not do_data_augmentation:
        image_transform_train = image_transform_val


    subject_ids_train = [3,  4,  5,  6,  8,  10, 15, 16, 17, 20, 21, 22, 23,
						 25, 26, 27, 30, 32, 36, 38, 39, 40, 42, 43, 44,
						 45, 46, 48, 49, 50]
    subject_ids_val = [1, 7, 12, 13, 24, 29, 33, 34, 35, 37]
    subject_ids_test = [ 2, 9, 11, 14, 18, 19, 28, 31, 41, 47]

    if mode == 'training':
        train_dataset = EgoGestDataSequence(path, 'train_dataset', image_transform_train, mask_transform, get_mask=use_mask_loss, subject_ids=subject_ids_train)
        val_dataset   = EgoGestDataSequence(path, 'val_dataset'  , image_transform_val  , mask_transform, get_mask=use_mask_loss, subject_ids=subject_ids_val)
    # If we're not in training mode then switch the training dataset out with test or validation
    elif mode == 'validation':
        train_dataset = EgoGestDataSequence(path, 'val_dataset'  , image_transform_val  , mask_transform, get_mask=use_mask_loss, subject_ids=subject_ids_val)
    else:
        train_dataset = EgoGestDataSequence(path, 'val_dataset'  , image_transform_test , mask_transform, get_mask=use_mask_loss, subject_ids=subject_ids_test)

    # train_indices, val_indices, test_indices = check_and_split_data(host_name=hostname,
    #                                                                 data_folder=path,
    #                                                                 dataset_len=len(dataset),
    #                                                                 train_fraction=0.6,
    #                                                                 validation_fraction=0.2)

    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_fn_padd)
    if mode == 'training':
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True,
            collate_fn=collate_fn_padd)

    # Init model and load pre-trained weights
    rnet = resnet.resnet18(False)
    model = SSAR(ResNet=rnet, input_size=83, number_of_classes=83, batch_size=batch_size, dropout=dropout).cuda()
    model_weights = './weights/final_weights.pth'
    state = model.state_dict()
    loaded_weights = torch.load(model_weights)
    state.update(loaded_weights)
    model.load_state_dict(state)


    # Setup optimizer and loss
    criterion = torch.nn.CrossEntropyLoss(ignore_index=label_mask_value)
    criterion = criterion.cuda()
    if mode == 'training':
        set_train_mode(model, train=True) # Need this here so the optimizer has the correct parameters to be trained
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    else:
        optimizer = None

    # Continue from previous training checkpoint
    epoch_resume, step_resume, best_val_loss = load_latest(model, results_path, training_mode, optimizer)
    if not restore_training_variables:
        epoch_resume = 0
        step_resume = 0
        best_val_loss = np.inf

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
    patience_counter = 0
    for epoch in range(epoch_resume, epochs):
        # Display info
        print(f"Epoch: {epoch}")

        if epoch == epoch_resume and step_resume > 0:
            print(f"Fast forwarding to train step {step_resume}")
        
        # Reset epoch stats
        train_metrics = {}

        # Switch to train mode while freezing parts of the model we don't want to train
        set_train_mode(model, train=True)

        # Train
        if mode == 'training':
            print('Training:')
        for train_step, batch in enumerate(train_loader):
            # Advance train_loader to resume training from last checkpointed position (Note: Assumes same batch size)
            if epoch == epoch_resume and train_step < step_resume:
                del batch
                continue

            # Save model
            if mode == 'training' and train_step % 100 == 0 and (train_step != step_resume or epoch != epoch_resume):
                save_model(model, optimizer, training_mode, epoch, train_step, best_val_loss, results_path)

            # Do one training train_step (may not actually train_step optimizer if doing gradiant accumulation)
            loss, batch_correct_count_samples = process_batch(model, train_step, batch, criterion, optimizer, mode=mode)
            del batch
            
            # Update metrics
            update_metrics(train_metrics, epoch, loss, batch_correct_count_samples)

            if (train_step + 1) % 10 == 0:
                # Display metrics
                print_metrics(train_metrics, train_step)
                update_accuracy_plot(train_acc_bars, train_acc_texts, train_metrics['accuracy_hist'])
        
        # Update train metric history and plots for this epoch
        update_epoch_history(train_history, train_metrics)
        update_loss_plot(train_loss_line, train_history)
        
        # Validation
        if mode == 'training':
            print('Validation:')
            val_metrics = {}

            # Switch to evaluation mode for validation
            set_train_mode(model, train=False)

            for val_step, batch in enumerate(val_loader):
                loss, batch_correct_count_samples = process_batch(model, val_step, batch, criterion, optimizer, mode='validation')

                # Update metrics
                update_metrics(val_metrics, epoch, loss, batch_correct_count_samples)

                if (val_step + 1) % 10 == 0:
                    # Display metrics
                    print_metrics(val_metrics, val_step)
                    update_accuracy_plot(val_acc_bars, val_acc_texts, val_metrics['accuracy_hist'])
            
            # Update validation metric history and plots
            update_epoch_history(val_history, val_metrics)
            update_loss_plot(val_loss_line, val_history)

            # Early stoping
            if val_metrics['loss_epoch'] < best_val_loss:
                best_val_loss = val_metrics['loss_epoch']
                patience_counter = 0
                save_model(model, optimizer, training_mode, epoch, val_step, best_val_loss, results_path, filename_override='model_best.pth')
            else:
                patience_counter += 1
            if patience_counter >= early_stoppping_patience:
                print(f'Validation accuracy did not improve for {patience_counter} epochs, stopping')
                break

    # Save final model
    if mode == 'training' and (train_step != step_resume or epoch != epoch_resume):
        save_model(model, optimizer, training_mode, epoch, train_step, best_val_loss, results_path)
    
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
    labels = pad_packed_sequence(batch['label'], batch_first=True, padding_value=label_mask_value)[0].cuda()
    start_label_mask = (torch.ones([labels.shape[0], frame_start_loss_calc]).long() * label_mask_value).cuda()
    labels_masked = torch.cat([start_label_mask, labels[:, frame_start_loss_calc:]], dim=1)
    lengths = batch['length'].cuda()
    if use_mask_loss:
        true_mask = pad_packed_sequence(batch['masks'], batch_first=True, padding_value=label_mask_value)[0].cuda()

    generated_labels = model(images, lengths=lengths, get_mask=use_mask_loss, get_lstm_state=False)
    if use_mask_loss:
        mask, generated_labels = generated_labels

    # end_indices = (lengths - 1)
    # indices = end_indices.view(-1, 1, 1).repeat(1, generated_labels.shape[1], 1)
    # generated_labels = generated_labels.gather(2, indices).squeeze(2)
    # indices = end_indices.view(-1, 1)
    # labels = labels.gather(1, indices).squeeze(1)

    loss = criterion(generated_labels, labels_masked)
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

        # images, _ = pad_packed_sequence(images, batch_first=True)
        # try:
        #     im_plt.set_data(images[0, lengths[0] // 2].permute(1, 2, 0).cpu())
        # except:
        #     im_plt = plt.imshow(images[0, lengths[0] // 2].permute(1, 2, 0).cpu())
        # plt.draw()
        # plt.pause(0.5)

        # mask = torch.softmax(mask, dim=1)[:, 1, ::]
        # try:
        #     im_plt.set_data(mask[0, lengths[0] // 2].cpu())
        # except:
        #     im_plt = plt.imshow(mask[0, lengths[0] // 2].cpu())
        # plt.draw()
        # plt.pause(0.5)
        # im_plt.set_data(true_mask[0, lengths[0] // 2].cpu())
        # plt.draw()
        # plt.pause(0.5)
        # im_plt.set_data(mask[0, (lengths[0] * 0.75).long()].cpu())
        # plt.draw()
        # plt.pause(0.5)

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