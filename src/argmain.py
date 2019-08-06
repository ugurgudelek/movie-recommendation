__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"



import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-fit', dest='fit', action='store_true')


args = parser.parse_args()
print(args.fit)