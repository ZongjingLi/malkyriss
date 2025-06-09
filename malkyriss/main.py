
import sys
import tornado
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--type",   type=str,   default = "namo")

def main():
    sys.stdout.write("\rMalkryiss is initalized.\n")
    config = parser.parse_args()
    return 

def helper():
    sys.stdout.write("\r Helper Function Not Avaliable")
    return