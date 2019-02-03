import hashlib
import json
import math
import os
import os.path
import sys
import time

import numpy as np

from alice.lib import static
from alice.lib import util


def load_building_plan(obs, args = {}):
    bot = None
    if "bot" in args:
        bot = args["bot"]
    
    print("hey, let's get the building plan")

    location = bot.command_home_base
    map_folder = util.get_map_folder(obs, args={"bot":bot})

    if not os.path.exists(map_folder):
        os.makedirs(map_folder)
        
    cc_x = int(location[0])
    cc_y = int(location[1])

    file_name = f"{map_folder}\\building_plan_{cc_x}_{cc_y}.txt"

    # TODO: if something goes wrong, we need to re-generate the file from scratch, do what's behind the else
    # Think about:  split each item out to be recalculated individually if needed
    if os.path.isfile(file_name) == True:

        f = open(file_name,"r")
        dict = json.load(f)
        f.close()

        if "supply depot" in dict:
            bot.building_plan["supply depot"] = dict["supply depot"]

        if "production" in dict:
            bot.building_plan["production"] = dict["production"]

        if "command" in dict:
            command = dict["command"]
            bot.building_plan["command"] = command
            
            # initialize the starting command center as already built.
            for c in command:
                if c[0] < 3 and c[1] < 3:
                    bot.buildings = [{"type":static.unit_ids["command center"],"location":c,"status":"complete","scvs":12}]

    return