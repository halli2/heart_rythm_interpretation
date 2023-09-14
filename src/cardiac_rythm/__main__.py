from argparse import ArgumentParser

from train import fit


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("-t", "--test")
    args = parser.parse_args()
    if args.test:
        pass
    else:
        fit()


if __name__ == "__main__":
    main()
