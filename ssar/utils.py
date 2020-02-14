import os
import torch
import glob
import numpy as np

# Freezes or unfreezes a model
def dfs_freeze(model, unfreeze=False):
    for _, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = unfreeze
        dfs_freeze(child, unfreeze=unfreeze)

def set_bn_train_mode(model, train=True):
    def _set_bn_train_mode(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            if train:
                m.train()
            else:
                m.eval()
    model.apply(_set_bn_train_mode)

def get_date_sorted_chkpt_files(checkpoint_path):
    old_saves = glob.glob(os.path.join(checkpoint_path, 'ssar_save_*_*.pth'))
    old_saves.sort(key=os.path.getmtime)
    return old_saves

# Save the model
def save_model(model, optimizer, training_mode, epoch, step, best_val_loss, checkpoint_path, filename_override=None):
    print('Saving model...')

    # Remove old saved models to preserve space
    if filename_override is None:
        old_saves = get_date_sorted_chkpt_files(checkpoint_path)
        while(len(old_saves) > 5):
            try:
                os.remove(old_saves[0])
            except:
                print(f'Warning: Could not remove old checkpoint file at {os.path.realpath(old_saves[0])}')
            del old_saves[0]

    # Save model
    filename = f'ssar_save_{epoch}_{step}.pth' if filename_override is None else filename_override
    save_file_path = os.path.join(checkpoint_path, filename)
    states = {
        'training_mode': training_mode,
        'epoch': epoch,
        'step': step,
        'best_val_loss': best_val_loss,
        'arch': 'ssar',
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    torch.save(states, save_file_path)

    print(f'Model saved to {save_file_path}')

# Load the latest training checkpoint from the checkpoint path
# Returns checkpointed (epoch, step, best_val_loss)
def load_latest(model, checkpoint_path, training_mode, optimizer=None):
    date_sorted_chkpts = get_date_sorted_chkpt_files(checkpoint_path)
    if date_sorted_chkpts:
        latest_chkpt_file = date_sorted_chkpts[-1]
        print(f"Loading latest checkpoint file at {os.path.realpath(latest_chkpt_file)}")
        checkpoint = torch.load(latest_chkpt_file)

        model.load_state_dict(checkpoint['state_dict'])

        # Don't restore other training state if the training mode has changed
        if 'training_mode' in checkpoint and checkpoint['training_mode'] == training_mode:
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])

            if 'best_val_loss' in checkpoint:
                best_val_loss = checkpoint['best_val_loss']
            else:
                best_val_loss = np.inf

            return checkpoint['epoch'], checkpoint['step'], best_val_loss, None
    
    print("INFO: No existing checkpoint file to load")
    return 0, 0, np.inf