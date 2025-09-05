import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from myutils.Blocks import *
from myutils.data import *
import time

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_folder", type=str, default="./data/input/")
    parser.add_argument("--pth_path", type=str, default="./pth/ULADNet.pth")
    args = parser.parse_args()

    output_folder_color = './data/result/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = PairedTransform(hflip_prob=0)
    test_dataset = TestDataset(root_dir=args.img_folder,transform=train_transform,)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1,)

    torch.backends.cudnn.benchmark = True
    model_path = args.pth_path
    net = ULADNet.build(n_bins=80, min_val=0.0001, max_val=1, norm="linear")
    net.load_state_dict(torch.load(model_path))

    print("Model loaded: ULADNet")
    net = net.to(device=device)
    net.eval()

    # Test
    for i,sample_batched1 in enumerate(tqdm(test_loader)):
        start_time = time.time()
        img_fn = sample_batched1['file_name'][0]
        input_img = sample_batched1['image'].to(device=device, non_blocking=True)
        x1 = input_img
        x2 = RGB_to_AIS_fast(input_img)
        _,out=net(x1,x2)
        end_time = time.time()

        img = out.detach().cpu().numpy()
        img_resized = img.reshape(240, 320)
        max_item = max(max(row) for row in img_resized)
        img_resized = img_resized / max_item * 255
        plt.imsave(os.path.join(output_folder_color, img_fn), img_resized, cmap='inferno')

    print("Total images: {0}\n".format(len(test_loader)))

