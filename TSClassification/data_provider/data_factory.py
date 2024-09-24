from data_provider.data_loader import UEAloader
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

def data_provider(args, flag, **kwargs):
    Data = UEAloader

    if flag == 'TEST':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid

    drop_last = False
    data_set = Data(
        root_path=args.root_path,
        flag=flag,
        **kwargs
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
    )
    return data_set, data_loader

