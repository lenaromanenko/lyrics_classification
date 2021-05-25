""" Main file for artist-prediction-program 
"""
import argparse
from classification_model import train_model, LYRICS

parser = argparse.ArgumentParser(description = 'This program predicts the artist of a given text')  #Initialization
parser.add_argument('LYRICS', help = "Give a lyrics sample", default = "come fly with me lets fly lets fly away")
args = parser.parse_args()
train_model(args.LYRICS)