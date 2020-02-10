from data.egogest_dataset import EgoGestDataSequence
from data.data import check_and_split_data, FixedIndicesSampler, collate_fn_padd
from model.model import SSAR
from socket import gethostname
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence

import matplotlib.pyplot as plt
import argparse
import torch
import torchvision.models.resnet as resnet
import numpy as np

# import torch
# import gc

def dfs_freeze(model, unfreeze=False):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = unfreeze
        dfs_freeze(child, unfreeze=unfreeze)

def main():
    parser = argparse.ArgumentParser(description="To read EgoGesture Dataset and run through SSAR network")
    parser.add_argument('--path', default='', help='full path to EgoGesture Dataset')
    args = parser.parse_args()
    path = args.path
    batch_size = 25
    epochs = 100


    image_transform = transforms.Compose([transforms.Resize((126, 224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                          ])
    mask_transform = transforms.Compose([transforms.ToTensor()])
    hostname = gethostname() + "sequence_data"

    dataset = EgoGestDataSequence(path, hostname, image_transform, mask_transform)
    train_indices, val_indices, test_indices = check_and_split_data(host_name=hostname,
                                                                    data_folder=path,
                                                                    dataset_len=len(dataset),
                                                                    train_fraction=0.6,
                                                                    validation_fraction=0.2)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        sampler=FixedIndicesSampler(train_indices),
        collate_fn=collate_fn_padd)
    # val_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     num_workers=0,
    #     pin_memory=True,
    #     sampler=FixedIndicesSampler(val_indices))
    # test_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     num_workers=0,
    #     pin_memory=True,
    #     sampler=FixedIndicesSampler(test_indices))

    rnet = resnet.resnet18(False)
    md = SSAR(ResNet=rnet, input_size=83, number_of_classes=83, batch_size=batch_size, dropout=0).cuda()
    md_weights = './weights/final_weights.pth'
    state = md.state_dict()
    loaded_weights = torch.load(md_weights)
    state.update(loaded_weights)
    md.load_state_dict(state)

    md.eval()
    md.lstms.train()
    dfs_freeze(md)
    dfs_freeze(md.lstms, unfreeze=True)

    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = Adam(filter(lambda p: p.requires_grad, md.parameters()), lr=2e-3)

    # old_tensor_set = set()

    correct_count = 0
    count = 0
    loss_sum = 0
    optimizer.zero_grad()
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        
        for i, sample in enumerate(train_loader):
            loss, n_correct = train_step(md, i + 1, sample, criterion, optimizer)
            del sample
            
            count += batch_size
            loss_sum += loss
            loss_mean = loss_sum / (i + 1)
            correct_count += n_correct

            if (i + 1) % 10 == 0:
                print(f"Step: {i + 1},",
                    f"Processed Gestures: {count},",
                    f"Correct Count: {correct_count},",
                    f"Accuracy: {correct_count / count},",
                    f"Loss Mean: {loss_mean},",
                    f"Loss (last batch): {loss}")

# Note: step should begin at 1
def train_step(model, step, sample, criterion, optimizer):
    images = sample['images']

    images = images.cuda()
    labels = pad_packed_sequence(sample['label'], batch_first=True)[0].cuda()
    lengths = sample['length'].cuda()
    true_mask = sample['masks']

    generated_labels = model(images, lengths=lengths, get_mask=False, get_lstm_state=False)

    # end_indices = (lengths - 1)
    # indices = end_indices.view(-1, 1, 1).repeat(1, generated_labels.shape[1], 1)
    # generated_labels = generated_labels.gather(2, indices).squeeze(2)
    # indices = end_indices.view(-1, 1)
    # labels = labels.gather(1, indices).squeeze(1)

    loss = criterion(generated_labels, labels) / 4
    loss.backward()
    
    if step % 4 == 0:
        optimizer.step()
        optimizer.zero_grad()

    with torch.no_grad():
        end_indices = (lengths // 2) # Measuring accuracy somewhere in middle
        indices = end_indices.view(-1, 1, 1).repeat(1, generated_labels.shape[1], 1)
        generated_labels = generated_labels.gather(2, indices).squeeze(2)
        generated_labels = torch.argmax(generated_labels, dim=1)
        indices = end_indices.view(-1, 1)
        labels = labels.gather(1, indices).squeeze(1)
        correct_count = torch.sum(labels == generated_labels).item()

        # if count == 0:
        #     im_plt = plt.imshow(masks[0, lengths[0] // 2, 1].cpu())
        # else:
        #     im_plt.set_data(masks[0, lengths[0] // 2, 1].cpu())
        # plt.draw()
        # plt.pause(0.0001)
        return loss.item(), correct_count

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