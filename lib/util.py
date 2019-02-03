import hashlib
import json
import math
import os
import os.path
import sys
import time

import numpy as np

from alice.lib import static

# returns an int, rounded to the nearest int, 0.5 always rounds up
def round(x):
    if x - int(x) >= 0.5:
        x = x + 0.5

    return int(x)

def get_map_folder(obs, args = {}):
    bot = None
    if "bot" in args:
        bot = args["bot"]
    
    h = hashlib.md5()
    fingerprint = obs.observation['feature_minimap'][static.minimap_features["height map"]]
    h.update(fingerprint)
    map_id = h.digest().hex()

    # open index file
    if os.path.isfile("maps_index.txt") == False:
        #touch
        f = open("maps_index.txt","w")
        json.dump({}, f)
        f.close()

    # load dictionary object with mappings from hash to map folder name
    f = open("maps_index.txt","r")
    maps = {}
    try:
        maps = json.load(f)
            
    except:
        # in case there are no maps yet
        print("Error reading maps index file")
        pass

    f.close()

    map_name = map_id
    if map_id in maps:
        map_name = maps[map_id]
    else:
        # set the name of the map to its ID, we'll go through and rename the folder and index later.
        maps[map_id] = map_id
        f = open("maps_index.txt","w")
        json.dump(maps, f)
        f.close()

    return map_name

def get_finances(obs):
    minerals = obs.observation["player"][1]
    gas = obs.observation["player"][2]

    return {"minerals":minerals,"gas":gas}

