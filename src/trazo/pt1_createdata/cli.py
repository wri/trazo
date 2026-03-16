from .gridding import main as gridding_main
from . import plantingharvest
from .active_sample import main as active_sample_main


def main():
    gridding_main()
    plantingharvest.main()
