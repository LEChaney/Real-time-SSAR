from data.egogest_dataset import EgoGestDataSequence
from data.data import check_and_split_data, FixedIndicesSampler, collate_fn_padd
from model.model import SSAR
from socket import gethostname
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
from utils import dfs_freeze, save_model, load_latest
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
batch_size = 25
epochs = 100
default_acc_bin_idx = 8
fast_forward_step = False
accuracy_bins = 10
grad_accum_steps = 4
rel_poses = torch.linspace(0, 1, accuracy_bins, requires_grad=False)
rel_poses_gpu = rel_poses.cuda()

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

    dataset = EgoGestDataSequence(path, hostname, image_transform, mask_transform, get_mask=False)
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
        sampler=FixedIndicesSampler(val_indices))
    # test_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     num_workers=0,
    #     pin_memory=True,
    #     sampler=FixedIndicesSampler(test_indices))

    # Init model and load pre-trained weights
    rnet = resnet.resnet18(False)
    model = SSAR(ResNet=rnet, input_size=83, number_of_classes=83, batch_size=batch_size, dropout=0).cuda()
    model_weights = './weights/final_weights.pth'
    state = model.state_dict()
    loaded_weights = torch.load(model_weights)
    state.update(loaded_weights)
    model.load_state_dict(state)

    # Freeze parts of model we don't want to train
    model.eval()
    dfs_freeze(model)
    if mode == 'training':
        model.lstms.train()
        dfs_freeze(model.lstms, unfreeze=True)

    # Setup optimizer and loss
    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    if mode == 'training':
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-3)
    else:
        optimizer = None

    # Continue from previous training checkpoint
    epoch_resume, step_resume = load_latest(model, results_path, optimizer)
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
    fig, ax = plt.subplots(nrows=2, ncols=1)
    rects = ax[0].bar(rel_poses, 1, 1 / accuracy_bins)
    texts = [ax[0].text(x, y, "1.0", horizontalalignment='center', verticalalignment='bottom') for x, y in zip(rel_poses, np.ones_like(rel_poses))]
    loss_text = ax[1].text(0, -0.2, "Loss: ")
    plt.show()

    # Setup movie moviewriter for writing accuracy plot over time
    # moviewriter = FFMpegWriter(fps=1)
    # moviewriter.setup(fig, os.path.join(results_path, 'accuracy_over_time.mp4'), dpi=100)

    # Main training loop
    if mode == 'training':
        optimizer.zero_grad()
    train_history = {}
    for epoch in range(epoch_resume, epochs):
        # Display info
        print(f"Epoch: {epoch}")

        if epoch == epoch_resume and step_resume > 0:
            print(f"Fast forwarding to train step {step_resume}")
        
        # Reset epoch stats
        metrics = {}

        # Train
        for step, batch in enumerate(train_loader):
            # Advance train_loader to resume training from last checkpointed position (Note: Assumes same batch size)
            if epoch == epoch_resume and step < step_resume:
                del batch
                continue

            # Save model
            if mode == 'training' and step % 100 == 0 and (step != step_resume or epoch != epoch_resume):
                save_model(model, optimizer, epoch, step, results_path)

            # Do one training step (may not actually step optimizer if doing gradiant accumulation)
            loss, batch_correct_count_samples = process_batch(model, step, batch, criterion, optimizer)
            del batch
            
            # Stats
            update_metrics(metrics, epoch, loss, batch_correct_count_samples)

            if (step + 1) % 10 == 0:
                print(f"Step: {step + 1},",
                    f"Processed Gestures: {metrics['gesture_count']},",
                    f"Correct Count (@t={rel_poses[default_acc_bin_idx]:.2f}): {metrics['correct_count_hist'][default_acc_bin_idx]},",
                    f"Accuracy (@t={rel_poses[default_acc_bin_idx]:.2f}): {metrics['accuracy_hist'][default_acc_bin_idx]:.4f},",
                    f"Loss Epoch: {metrics['loss_epoch']:.5f},",
                    f"Loss Batch: {loss:.5f}")

                # Plot accuracy histogram
                for i, rect in enumerate(rects):
                    rect.set_height(metrics['accuracy_hist'][i])
                    texts[i].set_position([rel_poses[i], metrics['accuracy_hist'][i]])
                    texts[i].set_text(f'{metrics["accuracy_hist"][i]:.3f}')

                loss_text.set_text(f"Epoch: {epoch} "
                    f"Step: {step + 1}, " +
                    f"Loss Epoch: {metrics['loss_epoch']:.4f}, " +
                    f"Loss Batch: {loss:.4f}")
                # moviewriter.grab_frame()
                plt.draw()
                plt.pause(0.001)
        
            update_metric_history(train_history, metrics)

        try:
            train_loss_line.set_data(train_history['epoch'], train_history['loss_epoch'])
            ax[1].relim()
            ax[1].autoscale_view()
        except NameError:
            train_loss_line, = ax[1].plot(train_history['epoch'], train_history['loss_epoch'])

        plt.draw()
        plt.pause(0.001)

        # Validation
        # for step, batch in enumerate(val_loader):
        #     loss, batch_correct_count_samples = process_batch(model, step, batch, criterion, optimizer)
            
    # Save final model
    if mode == 'training' and step != step_resume or epoch != epoch_resume:
        save_model(model, optimizer, epoch, step, results_path)
    
    print('Done!')
    
    plt.ioff()
    plt.show()

def update_metric_history(history, metrics):
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
    metrics['loss_epoch'] = metrics['loss_sum'] / metrics['losses_seen']
    metrics['accuracy_hist'] = metrics['correct_count_hist'] / metrics['gesture_count']

    return metrics

# Perfom one training step on one batch of data (may not actually step optimizer if doing gradiant accumulation)
def process_batch(model, step, batch, criterion, optimizer):
    images = batch['images']

    images = images.cuda()
    labels = pad_packed_sequence(batch['label'], batch_first=True)[0].cuda()
    lengths = batch['length'].cuda()
    # true_mask = batch['masks']

    generated_labels = model(images, lengths=lengths, get_mask=False, get_lstm_state=False)

    # end_indices = (lengths - 1)
    # indices = end_indices.view(-1, 1, 1).repeat(1, generated_labels.shape[1], 1)
    # generated_labels = generated_labels.gather(2, indices).squeeze(2)
    # indices = end_indices.view(-1, 1)
    # labels = labels.gather(1, indices).squeeze(1)

    loss = criterion(generated_labels, labels) / grad_accum_steps
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
        return loss.item() * grad_accum_steps, correct_count_hist

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