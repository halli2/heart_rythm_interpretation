import logging
import pprint as pp
from argparse import ArgumentParser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = ArgumentParser(description="Classify cardiac rythm.")
    parser.add_argument("file_path", help="Path to .mat file.")
    # Training config
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs to run.",
        default=100,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size used to fit model (default: 32).",
        default=32,
    )
    parser.add_argument(
        "--folds",
        type=int,
        help="Number of folds used in the cross validation (stratified k fold) (default: 10).",
        default=10,
    )
    parser.add_argument(
        "--normalize_data_length",
        action="store_true",
        help="If set, normalizes data length to the same length as the lowest, if else uses weights in training.",
        default=False,
    )
    parser.add_argument(
        "--cross_validate",
        action="store_true",
        help="If set will run (folds) times cross validation.",
        default=False,
    )

    # Model config
    parser.add_argument(
        "--filters",
        metavar="N",
        type=int,
        nargs="+",
        help="List of filters (default: 32 64 128 256 256)",
        default=[32, 64, 128, 256, 256],
    )
    parser.add_argument(
        "--kernels",
        metavar="N",
        type=int,
        nargs="+",
        help="""List or int of kernel_sizes, if int all are set to the same.
            If list, have to be same length as --filters. (default: 5)""",
        default=[5],
    )
    parser.add_argument(
        "--stride",
        metavar="N",
        type=int,
        nargs="+",
        help="""List or int of strides, if int all are set to the same.
            If list, have to be same length as --filters. (default: 2)""",
        default=[2],
    )

    def pools(arg) -> list[int]:
        result = []
        for v in arg.split(";"):
            inner = [int(y) for y in v.split(",")]
            if inner[0] == 0:
                inner = None
            result.append(inner)
        return result

    parser.add_argument(
        "--pool",
        type=pools,
        help="""
        List of pool (pool_size, stride) or 0 (pooling disabled).
        Format: "3,2;3,2;0" gives: [(3, 2), (3, 2), 0]

        \n
        output_shape = (input_shape - pool_size + 1) / strides) (default: 4, 3)""",
        default=[[4, 3]],
    )
    parser.add_argument(
        "--pool_after_conv",
        type=int,
        nargs="+",
        help="0 = False, 1 = True. Int or list, if list has to be same length as filters (default: 0)",
        default=[0],
    )
    parser.add_argument(
        "--padding",
        type=str,
        help="Padding type in convolutions (default: 'same')",
        default="same",
    )
    parser.add_argument(
        "--fc_end",
        type=int,
        nargs="+",
        help="Fully connected layers at end (default: [512 1024])",
        default=[512, 1024],
    )
    args = parser.parse_args()

    pools = args.pool

    filters = args.filters
    kernels = args.kernels
    strides = args.stride
    pool_after_conv = args.pool_after_conv

    # Make sure kernels is a list of len(filters)
    def ensure_list_length(var_name: str, setting: int | list, filters: list) -> list:
        if len(setting) == 1:
            setting = setting * len(filters)
        elif not len(setting) == len(filters):
            error_msg = f"{var_name} has to be same length as filters, or int. {var_name}={setting} - {filters=}"
            raise Exception(error_msg)
        return setting

    kernels = ensure_list_length("kernels", kernels, filters)
    strides = ensure_list_length("strides", strides, filters)
    pools = ensure_list_length("pools", pools, filters)
    pool_after_conv = ensure_list_length("pool_after_conv", pool_after_conv, filters)
    pool_after_conv = [bool(x) for x in pool_after_conv]

    # Loading tf is slow, so don't do it unless we have a file.
    from models import CNNConfig
    from train import FitSettings, fit

    settings = FitSettings(
        args.epochs,
        args.batch_size,
        args.folds,
        args.normalize_data_length,
        args.cross_validate,
    )
    # PSP Model
    # model_config  = CNNConfig()
    # 5CNN Model
    model_config = CNNConfig(
        filters,  # [32, 64, 128, 256, 512, 512],
        kernels,
        strides,
        args.pool,
        args.padding,
        args.fc_end,
    )

    logging.info(f"{pp.pformat(settings)}")
    logging.info(f"{pp.pformat(model_config)}")

    fit(args.file_path, settings, model_config)


if __name__ == "__main__":
    main()
