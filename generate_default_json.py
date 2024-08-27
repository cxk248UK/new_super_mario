import json

from ProjectConf import DefaultProjectConf

default_conf = DefaultProjectConf()

with open(f'default_conf.json', 'w') as json_file:
    json.dump(default_conf.__dict__, json_file)
    json_file.close()



