import argparse
import csv
import json
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from utils.utils import ImageFolderDataset, get_transform, InfiniteSampler, calc_mean_std, adaptive_instance_normalization
from utils.models import VGGEncoder, Decoder


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--content_dir', type=str,
                         default="D:/Mojar Project/Neural style transfer/content_data",
                         help='Location of content dataset')
    parser.add_argument('--style_dir', type=str,
                         default="D:/Mojar Project/Neural style transfer/style_data",
                         help='Location of style dataset')
    parser.add_argument('--vgg', type=str,
                         default="D:/Mojar Project/Neural style transfer/vgg_normalised.pth",
                         help='Location of pre-trained VGG')

    parser.add_argument('--experiment', type=str, default='experiment1', help='Name of experiment')

    parser.add_argument('--final_size', type=int, default=256, help='Size of final image')
    parser.add_argument('--content_size', type=int, default=512, help='Size of content image')
    parser.add_argument('--style_size', type=int, default=512, help='Size of style image')
    parser.add_argument('--crop', action='store_true', default=True, help='Crop image')

    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=5e-5, help='Learning rate decay')

    # NST training is conventionally driven by a fixed number of iterations
    # rather than epochs, since the content and style datasets are unrelated
    # in size and don't need to be "completed" together. This replaces the
    # old --epochs argument, which silently capped every epoch at
    # min(len(content_loader), len(style_loader)) and wasted most of a large
    # content dataset like COCO.
    parser.add_argument('--max_iter', type=int, default=500, help='Number of training iterations')

    parser.add_argument('--content_weight', type=float, default=1.0, help='Content weight')
    parser.add_argument('--style_weight', type=float, default=5, help='Style weight')

    parser.add_argument('--log_interval', type=int, default=20, help='Log every N iterations')
    parser.add_argument('--save_interval', type=int, default=5000, help='Save every N iterations')

    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader worker processes')
    parser.add_argument('--max_grad_norm', type=float, default=None,
                         help='If set, clips decoder gradient norm to this value')
    parser.add_argument('--amp', action='store_true', default=False,
                         help='Use automatic mixed precision (faster + less memory on CUDA)')

    parser.add_argument('--resume', action='store_true', default=False, help='Resume training')
    parser.add_argument('--decoder_path', type=str, default=None, help='Path to decoder checkpoint')
    parser.add_argument('--optimizer_path', type=str, default=None, help='Path to optimizer checkpoint')
    parser.add_argument('--start_iter', type=int, default=0, help='Iteration number to resume from')

    args = parser.parse_args()

    if args.resume and (args.decoder_path is None or args.optimizer_path is None):
        parser.error('--resume requires both --decoder_path and --optimizer_path to be set.')

    return args


def save_checkpoint(decoder, optimizer, save_dir, iteration):
    torch.save(decoder.state_dict(), save_dir / f'decoder_iter_{iteration}.pth')
    torch.save(optimizer.state_dict(), save_dir / f'optimizer_iter_{iteration}.pth')


def log_loss_row(csv_path, iteration, loss, loss_c, loss_s):
    is_new = not csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(['iteration', 'loss', 'content_loss', 'style_loss'])
        writer.writerow([iteration, loss, loss_c, loss_s])


def plot_loss_curve(csv_path, out_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed; skipping loss curve plot.')
        return

    iters, losses, closses, slosses = [], [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            iters.append(int(row['iteration']))
            losses.append(float(row['loss']))
            closses.append(float(row['content_loss']))
            slosses.append(float(row['style_loss']))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(iters, losses, label='Total loss')
    ax.plot(iters, closses, label='Content loss')
    ax.plot(iters, slosses, label='Style loss')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training loss over time')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_arguments()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_dir = Path('experiment') / args.experiment
    save_dir.mkdir(exist_ok=True, parents=True)

    with open(save_dir / 'args.txt', 'w') as args_file:
        for key, value in vars(args).items():
            args_file.write(f"{key} : {value}\n")
    with open(save_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    content_transform = get_transform(args.content_size, args.crop, args.final_size)
    style_transform = get_transform(args.style_size, args.crop, args.final_size)

    content_dataset = ImageFolderDataset(args.content_dir, content_transform)
    style_dataset = ImageFolderDataset(args.style_dir, style_transform)

    print('Content images found:', len(content_dataset))
    print('Style images found:', len(style_dataset))

    # InfiniteSampler lets each loader be pulled from forever, independently,
    # regardless of how differently sized the two datasets are - no batch
    # is ever skipped or capped by the smaller dataset.
    content_dataloader = DataLoader(
        content_dataset, batch_size=args.batch_size,
        sampler=InfiniteSampler(content_dataset),
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda'), drop_last=True,
    )
    style_dataloader = DataLoader(
        style_dataset, batch_size=args.batch_size,
        sampler=InfiniteSampler(style_dataset),
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda'), drop_last=True,
    )
    content_iter = iter(content_dataloader)
    style_iter = iter(style_dataloader)

    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)
    encoder.eval()  # frozen feature extractor, never trained

    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda it: 1.0 / (1.0 + args.lr_decay * it)
    )

    start_iter = args.start_iter
    if args.resume:
        decoder.load_state_dict(torch.load(args.decoder_path, map_location=device, weights_only=False))
        optimizer.load_state_dict(torch.load(args.optimizer_path, map_location=device, weights_only=False))
        print(f'Resumed from {args.decoder_path}, starting at iteration {start_iter}.')

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == 'cuda'))
    csv_path = save_dir / 'loss_log.csv'

    print('Training......')
    mse_loss = torch.nn.MSELoss()

    running_loss = running_closs = running_sloss = 0.0
    log_count = 0

    progress_bar = tqdm(range(start_iter, args.max_iter), initial=start_iter, total=args.max_iter)

    for iteration in progress_bar:
        content_batch = next(content_iter).to(device)
        style_batch = next(style_iter).to(device)

        with torch.autocast(device_type=device.type, enabled=(args.amp and device.type == 'cuda')):
            c_feats = encoder(content_batch)
            s_feats = encoder(style_batch)

            t = adaptive_instance_normalization(c_feats[-1], s_feats[-1])
            g = decoder(t)
            g_feats = encoder(g)

            loss_c = mse_loss(g_feats[-1], t) * args.content_weight

            loss_s = 0
            for g_f, s_f in zip(g_feats, s_feats):
                g_mean, g_std = calc_mean_std(g_f)
                s_mean, s_std = calc_mean_std(s_f)
                loss_s = loss_s + mse_loss(g_mean, s_mean) + mse_loss(g_std, s_std)
            loss_s = loss_s * args.style_weight

            loss = loss_c + loss_s

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if args.max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        progress_bar.set_description(
            f'Loss:{loss.item():.4f}, Content: {loss_c.item():.4f}, Style: {loss_s.item():.4f}'
        )

        running_loss += loss.item()
        running_closs += loss_c.item()
        running_sloss += loss_s.item()
        log_count += 1

        if (iteration + 1) % args.log_interval == 0:
            avg_loss = running_loss / log_count
            avg_closs = running_closs / log_count
            avg_sloss = running_sloss / log_count
            tqdm.write(
                f'Iter {iteration + 1}: Loss:{avg_loss:.4f}, '
                f'Content Loss: {avg_closs:.4f}, Style Loss: {avg_sloss:.4f}'
            )
            log_loss_row(csv_path, iteration + 1, avg_loss, avg_closs, avg_sloss)
            running_loss = running_closs = running_sloss = 0.0
            log_count = 0

        is_last_iter = (iteration + 1) == args.max_iter
        if (iteration + 1) % args.save_interval == 0 or is_last_iter:
            save_checkpoint(decoder, optimizer, save_dir, iteration + 1)
            with torch.no_grad():
                output = torch.cat([content_batch, style_batch, g], dim=0)
                save_image(output.clamp(0, 1), save_dir / f"output_{iteration + 1}.png", nrow=args.batch_size)

    if csv_path.exists():
        plot_loss_curve(csv_path, save_dir / 'loss_curve.png')

    print('Training complete. Checkpoints and logs saved to', save_dir)


if __name__ == '__main__':
    main()