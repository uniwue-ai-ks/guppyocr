from docopt import docopt
from guppyocr.train_calamares import ModelConfiguration
from dataclasses import asdict

USAGE="""dump.py
Dump model paramets

Usage:
    dump.py <model_file>

"""

def main():
    args = docopt(USAGE)
    mc, model = ModelConfiguration.load_model(args["<model_file>"], "cpu")
    maxlen = max(map(len, asdict(mc.mconfig).keys()))
    for k,v in asdict(mc.mconfig).items():
        print(f"{k.ljust(maxlen+2)} {v}")

    
if __name__ == "__main__":
    main()


