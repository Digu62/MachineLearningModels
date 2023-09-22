import argparse

parser = argparse.ArgumentParser(description="To run iris model")
parser.add_argument("--epochs", type=int, help="Train epochs", default=1)
parser.add_argument("--batch_size", type=int, help="Train batch size", default=32)
parser.add_argument("--loss", type=str, help="Train loss", default='sparse_categorical_crossentropy')
parser.add_argument("--optimizer", type=str, help="Train optimizer", default='adam')

# metrics = ['accuracy'] #TODO

args = parser.parse_args()


print()

if __name__=='__main__':
    #TODO: Automate iris model run
    pass