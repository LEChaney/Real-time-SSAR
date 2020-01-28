from data.egogest_dataset import EgoGestDataSequence
from data.data import check_and_split_data
from model.model import SSAR
from socket import gethostname
from torch.autograd import Variable
from torchvision import transforms

import argparse
import torch
import torchvision.models.resnet as resnet


def main():
    parser = argparse.ArgumentParser(description="To read EgoGesture Dataset and run through SSAR network")
    parser.add_argument('--path', default='', help='full path to EgoGesture Dataset')
    args = parser.parse_args()
    path = args.path
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
    rnet = resnet.resnet18(False)
    md_weights = './weights/final_weights.pth'
    md = SSAR(ResNet=rnet, input_size=83, number_of_classes=83, batch_size=1, dropout=0).cuda()
    state = md.state_dict()
    state.update(torch.load(md_weights))
    md.load_state_dict(state)
    md.eval()
    correct_count = 0
    for i in test_indices:
        sample = dataset[i]
        images = sample['images']

        images = Variable(images.cuda())
        label = sample['label']
        mask, generated_label = md(images, True)
        generated_label = generated_label.squeeze(dim=0)
        generated_label = torch.argmax(generated_label, dim=0).item()
        if generated_label == label:
            correct_count += 1
        if correct_count % 100 == 0:
            print("processed {}".format(correct_count))
    print(len(test_indices))
    print(correct_count)


if __name__ == "__main__":
    main()