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

for name in conf.__dict__.keys():
    if conf_dict.get(name):
        conf.__setattr__(name, conf_dict.get(name))

print(conf.net_name)

print(f'imitation: {conf.imitation}')

train.train(conf)
