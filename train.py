import argparse

from diffuse.train import TrainingConfig

def parse() -> TrainingConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", "-t", type=str, default="unconditional_unet", help="Task to perform")
    parser.add_argument("--data", "-d", type=str, default="mnist", help="Dataset to use")
    parser.add_argument("--n_epochs", "-e", type=int, default=5, help="Number of epochs to train for")

    args = parser.parse_args()

    return TrainingConfig.from_args(args)

def main(training_config: TrainingConfig):
    training_config.train()

if __name__ == "__main__":
    training_config = parse()
    main(training_config)
