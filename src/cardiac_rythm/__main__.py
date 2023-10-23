from argparse import ArgumentParser


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("file_path", help="Path to .mat file.")
    args = parser.parse_args()

    # Loading tf is slow, so don't do it unless we have a file.
    from models import CNNConfig
    from train import FitSettings, fit

    settings = FitSettings(epochs=100, batch_size=32, folds=10, normalize_data_length=True)
    model_config = CNNConfig(
        layers=4, increasing=False, kernel_size=3, pool_each_conv=True, pool_after=False, padding="same"
    )
    fit(args.file_path, settings, model_config)


if __name__ == "__main__":
    main()
