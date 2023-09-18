from argparse import ArgumentParser


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("file_path", help="Path to .mat file.")
    args = parser.parse_args()

    # Loading tf is slow, so don't do it unless we have a file.
    from train import fit

    fit(args.path)


if __name__ == "__main__":
    main()
