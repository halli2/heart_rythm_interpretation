from argparse import ArgumentParser


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("file_path", help="Path to .mat file.")
    args = parser.parse_args()

    # Loading tf is slow, so don't do it unless we have a file.
    from models import CNNConfig
    from train import FitSettings, fit

    settings = FitSettings(epochs=100, batch_size=32, folds=10, normalize_data_length=False)
    # PSP Model
    # model_config  = CNNConfig()
    # 5CNN Model
    model_config = CNNConfig(
        [32, 64, 128, 256, 512, 512],
        kernel_size=[5, 5, 5, 5, 5, 5],
        stride=2,
        pool=(4, 3),
        pool_each_conv=False,
        pool_after=False,
        padding="same",
        fc1=512,
        fc2=1024,
    )
    fit(args.file_path, settings, model_config)


if __name__ == "__main__":
    main()
