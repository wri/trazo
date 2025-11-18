from .gridding import main as gridding_main
from . import plantingharvest

def main():
    gridding_main()
    plantingharvest.main()

