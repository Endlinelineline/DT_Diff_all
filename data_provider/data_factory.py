import argparse
from .data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from .guangning import Dataset_Guangning

# from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    "ETTh1": Dataset_ETT_hour,
    "ETTh2": Dataset_ETT_hour,
    "ETTm1": Dataset_ETT_minute,
    "ETTm2": Dataset_ETT_minute,
    "custom": Dataset_Custom,
    "guangning": Dataset_Guangning,
    # 'm4': Dataset_M4,
    # 'PSM': PSMSegLoader,
    # 'MSL': MSLSegLoader,
    # 'SMAP': SMAPSegLoader,
    # 'SMD': SMDSegLoader,
    # 'SWAT': SWATSegLoader,
    # 'UEA': UEAloader
}


def data_provider(args, flag):
    if isinstance(args, dict):
        args = argparse.Namespace(**args)
        
    Data = data_dict[args.data]
    # timeenc = 0 if args.embed != "timeF" else 1

    shuffle_flag = flag == "train"
    drop_last = False
    batch_size = args.batch_size
    if flag in ['test', 'val'] and args.data != "guangning":
        # batch_size = 32
        batch_size = max(batch_size, 1024)
    # freq = args.freq

    extra_kwargs = {}
    if args.data == "guangning":
        extra_kwargs = {
            "start": getattr(args, "start", None) or "2026-05-01 00:00:00",
            "end": getattr(args, "end", None) or "2026-09-09 23:59:00",
            "frequency": getattr(args, "frequency", "1min"),
            "aggregation": getattr(args, "aggregation", "last"),
            "shutdown_zero_fraction": getattr(args, "shutdown_zero_fraction", 0.6),
            "shutdown_min_turbines": getattr(args, "shutdown_min_turbines", 12),
            "shutdown_min_valid_fraction": getattr(
                args, "shutdown_min_valid_fraction", 0.8
            ),
            "rebuild_cache": getattr(args, "rebuild_cache", False),
            "stride": (
                getattr(args, "train_stride", 1)
                if flag == "train"
                else getattr(args, "test_stride", 1)
            ),
        }

    data_set = Data(
        # args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        condition = args.condition,
        kernel_size=args.kernel_size,
        **extra_kwargs,
        # timeenc=timeenc,
        # freq=freq,
        # seasonal_patterns=args.seasonal_patterns,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=getattr(args, "num_workers", 0),
        drop_last=drop_last,
    )
    return data_set, data_loader
