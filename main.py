import argparse
import json

import train
from ProjectConf import DefaultProjectConf

parser = argparse.ArgumentParser()

parser.add_argument('--conf', help='input configuration json path', default='default_conf.json', required=False)

args = parser.parse_args()

f = open(args.conf)
conf_dict = json.load(f)

conf = DefaultProjectConf()

conf.__dict__ = conf_dict

print(conf.net_name)

train.train(conf)
