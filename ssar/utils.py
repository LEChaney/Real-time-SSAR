import os
import torch
import glob

# Freezes or unfreezes a model
def dfs_freeze(model, unfreeze=False):
    for _, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = unfreeze
        dfs_freeze(child, unfreeze=unfreeze)

def get_date_sorted_chkpt_files(checkpoint_path):
    old_saves = glob.glob(os.path.join(checkpoint_path, 'ssar_save_*_*.pth'))
    old_saves.sort(key=os.path.getmtime)
    return old_saves

# Save the model
def save_model(model, optimizer, epoch, step, checkpoint_path):
    print('Saving model...')

    # Remove old saved models to preserve space
    old_saves = get_date_sorted_chkpt_files(checkpoint_path)
    while(len(old_saves) > 5):
        try:
            os.remove(old_saves[0])
        except:
            print(f'Warning: Could not remove old checkpoint file at {os.path.realpath(old_saves[0])}')
        del old_saves[0]

    # Save model
    save_file_path = os.path.join(checkpoint_path, f'ssar_save_{epoch}_{step}.pth')
    states = {
        'epoch': epoch,
        'step': step,
        'arch': 'ssar',
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    torch.save(states, save_file_path)

    print(f'Model saved to {save_file_path}')

# Load the latest training checkpoint from the checkpoint path
# Returns the checkpointed epoch and step to advance training to.
def load_latest(model, checkpoint_path, optimizer=None):
    date_sorted_chkpts = get_date_sorted_chkpt_files(checkpoint_path)
    if date_sorted_chkpts:
        latest_chkpt_file = date_sorted_chkpts[-1]
        print(f"Loading latest checkpoint file at {os.path.realpath(latest_chkpt_file)}")
        checkpoint = torch.load(latest_chkpt_file)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['epoch'], checkpoint['step']
    else:
        print("INFO: No existing checkpoint file to load")
        return 0, 0