# responsible for many map reading functions, mostly this file uses the height map data visible on the screen
# to chart out (record it to file) the whole map, then reference to determine where exactly on the map the bot is.
# this helps with precision for building placement and unit movement.

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

def save_map_data(obs, args = {}):
    bot = None
    if "bot" in args:
        bot = args["bot"]

    location = bot.command_home_base
    map_name = util.get_map_folder(obs, args={"bot":bot})

    if not os.path.exists(map_name):
        os.makedirs(map_name)
        
    cc_x = int(location[0])
    cc_y = int(location[1])

    file_name = f"{map_name}\\map_height_data_{cc_x}_{cc_y}.txt"

    f = open(file_name,"w")
    json.dump({"minimap offset chart":bot.minimap_offset_chart.tolist(),
               "screen height chart":bot.screen_height_chart.tolist(),
               "command home base":bot.command_home_base,
               "minimap home base":bot.minimap_home_base,
               "screen height chart offset":bot.screen_height_chart_offset,
               "minimap select area":bot.minimap_select_area
               }, f)
    f.close()

    return False

def load_map_data(obs, args = {}):
    bot = None
    if "bot" in args:
        bot = args["bot"]

    location = bot.command_home_base
    map_name = util.get_map_folder(obs, args={"bot":bot})

    if not os.path.exists(map_name):
        os.makedirs(map_name)
        
    cc_x = int(location[0])
    cc_y = int(location[1])

    file_name = f"{map_name}\map_height_data_{cc_x}_{cc_y}.txt"

    # TODO: if something goes wrong, we need to re-generate the file from scratch, do what's behind the else
    # Think about:  split each item out to be recalculated individually if needed
    if os.path.isfile(file_name) == True:
        f = open(file_name,"r")
        dict = json.load(f)
        f.close()

        if "screen height chart" in dict:
            bot.screen_height_chart = np.array(dict["screen height chart"])

        if "minimap offset chart" in dict:
            bot.minimap_offset_chart = np.array(dict["minimap offset chart"])

        if "command home base" in dict:
            bot.command_home_base = dict["command home base"]
        if "minimap home base" in dict:
            bot.minimap_home_base = dict["minimap home base"]
        if "screen height chart offset" in dict:
            bot.screen_height_chart_offset = dict["screen height chart offset"]
        if "minimap select area" in dict:
            bot.minimap_select_area = dict["minimap select area"]

        bot.priority_queue.extend([
            [8, schedule_chart_map, {"bot":bot}]
        ])

    else:
        print("height map data could not be loaded")
        # queue up the action to scan the entire map.
        bot.priority_queue.extend([
            [3, determine_minimap_boundaries, {"bot":bot}],
            [3, start_height_chart, {"bot":bot}],
            [4, center_screen_on_main, {"bot":bot}],
            [4, update_height_chart, {"relative":[1,0], "bot":bot}],
            [4, update_height_chart, {"relative":[0,1], "bot":bot}],
            [4, update_height_chart, {"relative":[-1,0], "bot":bot}],
            [4, update_height_chart, {"relative":[-1,0], "bot":bot}],
            [4, update_height_chart, {"relative":[0,-1], "bot":bot}],
            [4, update_height_chart, {"relative":[0,-1], "bot":bot}],
            [4, update_height_chart, {"relative":[1,0], "bot":bot}],
            [4, update_height_chart, {"relative":[1,0], "bot":bot}],
            [4, save_map_data, {"bot":bot}],
            [8, schedule_chart_map, {"bot":bot}]
        ])

    return None

def center_screen_on_main(obs, args):
    #print("moving home to main")
    bot = None
    if "bot" in args:
        bot = args["bot"]

    #print(f"{bot.minimap_home_base}")
    action = { 
            "id"    :   static.action_ids["move camera"], 
            "params":   [[bot.minimap_home_base[0],bot.minimap_home_base[1]]] 
        }
    return bot.try_perform_action(obs, action)


# point is relative to main base center, figure out if it's currently on the screen
def is_point_on_screen(obs, args):
    bot = None
    if "bot" in args:
        bot = args["bot"]
    else:
        return

    x, y = 0, 0 
    if "point" in args:
        x, y = args["point"]
    else:
        return
    
    width, height = bot.screen_dimensions
    
    mini_x, mini_y = get_minimap_coords(obs, args)

    # offsets based on minimap location
    off_x, off_y = get_offset(obs, args)

    screen_left = off_x
    screen_top = off_y
    screen_right = off_x + width
    screen_bottom = off_y + height

    if x <= screen_left or x >= screen_right:
        return False

    if y <= screen_top or y >= screen_bottom:
        return False

    return True


# point is relative to main base center, figure out if it's currently on the screen
def issue_move_action_on_screen(obs, args):
    bot = None
    if "bot" in args:
        bot = args["bot"]

    screen_location = get_screen_location(obs, args)

    if screen_location is None:
        print("woah, that's not on the screen")
        return None

    action = { 
        "id":       static.action_ids["move screen"],
        "params":   [[static.params["now"]], screen_location]
        }

    bot.callback_method = None
    bot.callback_parameters = {}
                         
    return bot.try_perform_action(obs, action)


# returns the screen location of an absolute location ("point")
# returns none if there is no location.
def get_screen_location(obs, args):
    bot = None
    if "bot" in args:
        bot = args["bot"]

    x, y = 0, 0
    if "point" in args:
        x, y = args["point"]

    width, height = bot.screen_dimensions
    
    # offsets based on minimap location
    off_x, off_y = get_offset(obs, args)

    screen_left = off_x
    screen_top = off_y
    screen_right = off_x + width
    screen_bottom = off_y + height

    if x <= screen_left or x >= screen_right:
        return None

    if y <= screen_top or y >= screen_bottom:
        return None

    return [x - screen_left, y - screen_top]


# returns the top left and bottom right pixel of the screen relative to the absolute coords
def get_relative_screen_location(obs, args):
    bot = None
    if "bot" in args:
        bot = args["bot"]

    tl = get_offset(obs, args)

    br = [tl[0] + bot.screen_dimensions[0],tl[1] + bot.screen_dimensions[1]]

    return [tl, br]


def get_offset(obs, args):
    bot = None
    if "bot" in args:
        bot = args["bot"]

    mini_x, mini_y = get_minimap_coords(obs, args)

    off_x = bot.minimap_offset_chart[1,mini_y,mini_x]
    off_y = bot.minimap_offset_chart[2,mini_y,mini_x]

    width, height = bot.screen_dimensions

    global_offset_x = bot.screen_height_chart_offset[0]
    global_offset_y = bot.screen_height_chart_offset[1]

    chunk_x0 = int(off_x - global_offset_x)
    chunk_y0 = int(off_y - global_offset_y)

    chunk_x1 = int(chunk_x0 + width)
    chunk_y1 = int(chunk_y0 + height)
    # TODO: need to sharpen this in case screen moved in a way other than clicking on the minimap
    screen_heights = obs.observation['feature_screen'][static.screen_features["height map"]]
    offset_heights = bot.screen_height_chart[0, chunk_y0:chunk_y1,chunk_x0:chunk_x1]

    # print(f"screen_heights shape {screen_heights.shape} offset_heights shape {offset_heights.shape}")
    # print(f"do they equal? {np.array_equal(screen_heights, offset_heights)}")
    # print(f"I need information: {chunk_x0},{chunk_y0},{chunk_y0},{chunk_y1}")

    if np.array_equal(screen_heights, offset_heights):
        return [off_x, off_y]
    else:
        # TODO: check neighboring screen_height_chart groups against the current screen
        print("Is screen aligned? FALSE!")
        print(f"I need information: {chunk_x0},{chunk_x1},{chunk_y0},{chunk_y1}")
        shift = match_screens(screen_heights, offset_heights, [0,0])
        off_x = off_x + shift[0]
        off_y = off_y + shift[1]

        return [off_x, off_y]


# how do we have to shift screen 1 to make it match screen 2?
# recursively check screen 1 vs screen 2,
#   ^^^^^
#   <^^^>
#   <<+>>
#   <vvv>
#   vvvvv
#
#   If point.x > 0 don't check left
#   If point.x < 0 don't check right
#   If point.y > 0 don't check up
#   If point.y < 0 don't check down

def match_screens(screen1, screen2, shift):
    x = shift[0]
    y = shift[1]
    if np.array_equal(screen1, screen2):
        print("successfully recursively matched screnes the shift is: "+ str(shift))
        return shift
    elif abs(x) > 20 or abs(y) > 20: # haven't thought too much about the constant here, lot's of work to search a square
        return None
    else:
        # if x < 0 don't check right
        # if abs(y) > abs(x) don't go left or right
        if(x >= 0) and abs(y) <= abs(x):
            # TODO: crop screens correctly by cutting off the right side of screen1 and the left side of screen2
            screen1_copy = np.copy(screen1[0:len(screen1),0:len(screen1[0]) - 1])
            screen2_copy = np.copy(screen2[0:len(screen2),1:len(screen2[0])])
            point = match_screens(screen1_copy, screen2_copy, [shift[0]+1,shift[1]])
            if point is not None:
                return point

        # if x > 0 don't check left
        # if abs(y) > abs(x) don't go left or right
        if(x <= 0) and abs(y) <= abs(x):
            # TODO: crop screens correctly by cutting off the left side of screen1 and the right side of screen2
            screen1_copy = np.copy(screen1[0:len(screen1),1:len(screen1[0])])
            screen2_copy = np.copy(screen2[0:len(screen2),0:len(screen2[0]) - 1])
            point = match_screens(screen1_copy, screen2_copy, [shift[0]-1,shift[1]])
            if point is not None:
                return point

        # if y < 0 don't check down
        # if abs(x) > abs(y) + 1 don't go up or down
        if(y >= 0) and abs(x) <= abs(y) + 1:
            # TODO: crop screens correctly by cutting off the bottom side of screen1 and the top side of screen2
            screen1_copy = np.copy(screen1[0:len(screen1)-1,0:len(screen1[0])])
            screen2_copy = np.copy(screen2[1:len(screen2),0:len(screen2[0])])
            point = match_screens(screen1_copy, screen2_copy, [shift[0],shift[1]+1])
            if point is not None:
                return point

        if(y <= 0) and abs(x) <= abs(y) + 1:
            # TODO: crop screens correctly by cutting off the top side of screen1 and the bottom side of screen2
            screen1_copy = np.copy(screen1[1:len(screen1),0:len(screen1[0])])
            screen2_copy = np.copy(screen2[0:len(screen2)-1,0:len(screen2[0])])
            point = match_screens(screen1_copy, screen2_copy, [shift[0],shift[1]-1])
            if point is not None:
                return point


def move_to_point(obs, args):
    bot = None
    if "bot" in args:
        bot = args["bot"]

    point = None
    if "point" in args:
        point = args["point"]

    width, height = bot.screen_dimensions
    
    center_x = int(width / 2)
    center_y = int(height / 2)

    chart = bot.minimap_offset_chart

    register = 0
    x_coord = 1
    y_coord = 2

    cmx, cmy = bot.minimap_home_base
    
    minimap_y, minimap_x = None, None

    for my in range(len(chart[0])):
        if (chart[register][my][cmx] == 1) and (chart[y_coord][my][cmx] + center_y > point[1]):
            minimap_y = my
            break
    
    for mx in range(len(chart[0][0])):
        if (chart[register][cmy][mx] == 1) and (chart[x_coord][cmy][mx] + center_x > point[0]):
            minimap_x = mx
            break

    action = { 
            "id"    :   static.action_ids["move camera"], 
            "params":   [[minimap_x,minimap_y]] 
            }

    bot.callback_method = None
    bot.callback_parameters = {}
                         
    return bot.try_perform_action(obs, action)

def get_minimap_coords(obs, args):
    bot = None
    if "bot" in args:
        bot = args["bot"]

    cam = obs.observation['feature_minimap'][static.minimap_features["camera"]]
    cam_ys, cam_xs = (cam == 1).nonzero()
    cam_x = util.round(cam_xs.mean())
    cam_y = util.round(cam_ys.mean())
       
    return [cam_x, cam_y]

# This operation must happen before the screen moves.
# 1) Make sure command center is selected.
# 2) Determine command center pixels, min and max pixels, add to self.unit_dimensions for command center.
# 3) Determine command center location center location on minimap, add to self.command_home_base
# 4) Determine how many screen pixels are in a minmap pixel
# 5) Extrapolate absolute map dimensions, absolute_home_base

def calibrate_map_data(obs, args):
    bot = None
    if "bot" in args:
        bot = args["bot"]
    #print("hey, let's calibrate our map data")

    single_select = obs.observation["single_select"]
    if single_select[0][0] == static.unit_ids["command center"]:
        # first run through with a selected cc
        # get all the cc pixels
        unit_type = obs.observation['feature_screen'][static.screen_features["unit type"]]
        ys, xs = (unit_type == static.unit_ids['command center']).nonzero()
                
        # get mins, maxes and means
        cc_mix = min(xs)
        cc_max = max(xs)
        cc_miy = min(ys)
        cc_may = max(ys)
        cc_mex = util.round(xs.mean())
        cc_mey = util.round(ys.mean())

        # save the command center height and width info into building_dimensions list
        cc_dims = next(item for item in bot.building_dimensions if item["unit id"] == static.unit_ids["command center"])
        width = cc_max - cc_mix + 1
        height = cc_may - cc_miy + 1
        cc_dims['screen pixels'] = [width, height]

        # save the tile size information
        bot.tile_size = [width/cc_dims['tiles'][0], height/cc_dims['tiles'][1]]
        #print("tile size: " + str(bot.tile_size))

        # get the selected height and width from the minimap
        sel = obs.observation['feature_minimap'][static.minimap_features["selected"]]
        mmys, mmxs = (sel == 1).nonzero()

        # get the mins, maxes and means
        mm_mix = min(mmxs)
        mm_max = max(mmxs)
        mm_miy = min(mmys)
        mm_may = max(mmys)
        mm_mex = util.round(mmxs.mean())
        mm_mey = util.round(mmys.mean())


        # set minimap home base
        bot.command_home_base = [mm_mex, mm_mey]
        #print("minimap home base: " + str(self.command_home_base))

        # save the command center minimap height and width info into building_dimensions list
        mm_width = mm_max - mm_mix + 1
        mm_height = mm_may - mm_miy + 1
        cc_dims['minimap pixels'] = [mm_width, mm_height]
        #print("CC info:" +str(cc_dims))

        # get the screen pixels on the minimap to calibrate minimap vs screen pixels
        sel = obs.observation['feature_minimap'][static.minimap_features["camera"]]
        scr_ys, scr_xs = (sel == 1).nonzero()

        scr_mix = min(scr_xs)
        scr_max = max(scr_xs)
        scr_miy = min(scr_ys)
        scr_may = max(scr_ys)

        scr_width = scr_max - scr_mix + 1
        scr_height = scr_may - scr_miy + 1

        #print("screen dimensions: " + str(self.screen_dimensions))
        #print("scr width: " + str(scr_width))
        #print("scr height: " + str(scr_height))

        bot.minimap_pixel_size = [bot.screen_dimensions[0]/scr_width, bot.screen_dimensions[1]/scr_height]
        #print("minimap pxl: " + str(self.minimap_pixel_size))

        # set the absolute map dimensions, absolute home base locations
        abs_x = bot.minimap_pixel_size[0] * bot.minimap_dimensions[0]
        abs_y = bot.minimap_pixel_size[1] * bot.minimap_dimensions[1]
        #bot.absolute_map_dimensions = [abs_x, abs_y]
        #print("abs dimensions: " + str(self.absolute_map_dimensions))
    
        #bot.absolute_home_base = [bot.minimap_pixel_size[0] * bot.command_home_base[0], bot.minimap_pixel_size[1] * bot.command_home_base[1]]
        #print("abs home base: " + str(self.absolute_home_base))

    else:
        # command center isn't selected, go ahead and select it. 
        # then, after selecting it, let's do the first part of this if statement
        action = bot.select_building(obs, {"type":static.unit_ids['command center']})

        bot.callback_method = calibrate_map_data
        bot.callback_parameters = args

        return action
            
    bot.callback_method = None
    bot.callback_parameters = {}
    return None



def determine_minimap_boundaries(obs, args={}):
    print("setting boundaries")
    bot = None
    if "bot" in args:
        bot = args["bot"]

    step = 0
    if "step" in args:
        step = args["step"]

    if step == 0:
        # move the screen to the top left most point
        action = { 
            "id"    :   static.action_ids["move camera"], 
            "params":   [[1,1]] 
        }

        bot.callback_method = determine_minimap_boundaries
        bot.callback_parameters = {
                "step"      :   1,
                "bot"       :   bot
            }

        return bot.try_perform_action(obs, action)

    elif step == 1:
        # save the center pixel of the screen
        cam = obs.observation['feature_minimap'][static.minimap_features["camera"]]
        cam_ys, cam_xs = (cam == 1).nonzero()

        cam_x = util.round(cam_xs.mean())
        cam_y = util.round(cam_ys.mean())
        bot.minimap_select_area[0] = [cam_x, cam_y]

        # move the screen to the bottom right most point
        action = { 
            "id"    :   static.action_ids["move camera"], 
            "params":   [[bot.minimap_dimensions[0] - 1,bot.minimap_dimensions[1] - 1]] 
        }

        bot.callback_method = determine_minimap_boundaries
        bot.callback_parameters = {
                "step"      :   2,
                "bot"       :   bot
            }

        return bot.try_perform_action(obs, action)

    elif step == 2:
        # save the center pixel of the screen
        cam = obs.observation['feature_minimap'][static.minimap_features["camera"]]
        cam_ys, cam_xs = (cam == 1).nonzero()

        cam_x = util.round(cam_xs.mean())
        cam_y = util.round(cam_ys.mean())

        bot.minimap_select_area[1] = [cam_x, cam_y]
        print(str(bot.minimap_select_area))
        return None


    return None

def schedule_chart_map(obs, args={}):
    #print("yo, let's scan more of the map")
    bot = None
    if "bot" in args:
        bot = args["bot"]

    # determine a chunk of 5 zones to scan
    current_time = obs.observation["game_loop"][0]
    # recurring action to print the location of the currently selected building

    if bot.charting_order == None:
        plan_chart_expansion(obs, {"bot":bot})

    #print("Remaining Tiles To Chart: " + str(len(bot.charting_order)))
    #print("Tiles: " + str(bot.charting_order[0:6]))

    if len(bot.charting_order) > 0:
        bot.priority_queue.extend([
            [8, update_height_chart, {"bot":bot,"list":True}],
            [8, update_height_chart, {"bot":bot,"list":True}],
            [8, update_height_chart, {"bot":bot,"list":True}],
            [8, update_height_chart, {"bot":bot,"list":True}],
            [8, save_map_data, {"bot":bot}]
        ])

        bot.schedule_action(obs, current_time + 50, schedule_chart_map, {"bot":bot})

    return

def plan_chart_expansion(obs, args = {}):
    print("planing chart expansion")
    bot = None
    if "bot" in args:
        bot = args["bot"]

    min_x = bot.minimap_select_area[0][0]
    min_y = bot.minimap_select_area[0][1]

    adjusted_home_x = bot.minimap_home_base[0] - min_x
    adjusted_home_y = bot.minimap_home_base[1] - min_y

    # get the minimap pixels that need charting
    area = bot.minimap_offset_chart[0, 
                                    bot.minimap_select_area[0][1]:bot.minimap_select_area[1][1]+1, # y
                                    bot.minimap_select_area[0][0]:bot.minimap_select_area[1][0]+1] # x
    
    print(f"chart dims: {len(area)},{len(area[0])}")
    print("chart area: " + str(len(area) * len(area[0])))

    list = []

    for y in range(len(area)):
        for x in range(len(area[0])):
            if area[y][x] == 0:
                dist_x = abs(adjusted_home_x - x)
                dist_y = abs(adjusted_home_y - y)


                # TODO: we need to trim out the edges of the whole offset_chart, then this next line will be obsolete
                list.append([dist_x+dist_y,x + min_x,y + min_y]) # adjust list from relative to absolute

    list.sort()
    #print("we're going to chart the following!")

    #print(list[0:10])
    #print("list size: " + str(len(list)))

    bot.charting_order = list
    return

# Returns: absolute x/y location of a pixel relative to the starting command center's starting screen
def get_absolute_location(obs, args={}):
    bot = None
    if "bot" in args:
        bot = args["bot"]
    
    point = None
    if "point" in args:
        point = args["point"]
    x, y = point
    # The minimap pixels don't line up correctly with the screen pixels, this method gets you close to the correct absolute position, it's slightly off.
        
    cam = obs.observation['feature_minimap'][static.minimap_features["camera"]]
    cam_ys, cam_xs = (cam == 1).nonzero()
    cam_x = util.round(cam_xs.mean())
    cam_y = util.round(cam_ys.mean())

    #print(f"minimap camera pos: {cam_x}:{cam_y}")

    # top left corner of the screen in the minimap, transposed from map units to screen units
    abs_x = bot.minimap_offset_chart[1,cam_y,cam_x] + x
    abs_y = bot.minimap_offset_chart[2,cam_y,cam_x] + y

    return [abs_x, abs_y]



# This only has to be done once per map per setting, if we save it for later in a map-info file.
# Returns: False
def start_height_chart(obs, args={}):
    print("starting the height chart")
    bot = None
    if "bot" in args:
        bot = args["bot"]

    step = 0

    if "step" in args:
        step = args["step"]

    if step == 0:
        # Move to minimap_home_base (the center pixel of the home command center on the minimap)
        x,y = bot.command_home_base
        print(f"home base: {x},{y}")
        
        action = { 
                "id"    :   static.action_ids["move camera"], 
                "params":   [[x,y]] 
                }

        bot.callback_method = start_height_chart
        bot.callback_parameters = {"step":1, "bot":bot}
        return bot.try_perform_action(obs, action)
        
    if step == 1:
        cam = obs.observation['feature_minimap'][static.minimap_features["camera"]]
        cam_ys, cam_xs = (cam == 1).nonzero()

        cam_x = util.round(cam_xs.mean())
        cam_y = util.round(cam_ys.mean())
            
        # we have to resave the minimap_home_base here, because after moving the screen, it is possible that the edge of the map got in the way and the center we use from here out is new
        bot.minimap_home_base = cam_x, cam_y

        print(f"minimap camera pos: {cam_x}:{cam_y}")

        height_data = obs.observation['feature_screen'][static.screen_features["height map"]]
            
        # initialize the minimap_offset_chart with zeros
        bot.minimap_offset_chart = np.zeros((3, bot.minimap_dimensions[1], bot.minimap_dimensions[0]),dtype=int)

        # initialize the minimap_screen_chart for the starting screen with zeros
        bot.screen_height_chart = np.zeros((2, bot.screen_dimensions[1], bot.screen_dimensions[0]),dtype=int)

        # copy the height data from the current screen into the screen offset chart
        bot.screen_height_chart[0, 0:bot.screen_dimensions[0], 0:bot.screen_dimensions[1]] = height_data

        # set the height data from the current screen on the minimap offset chart as plotted
        bot.screen_height_chart[1, 0:bot.screen_dimensions[0], 0:bot.screen_dimensions[1]] = 1

        #print("screen height chart: ")
        #print(self.screen_height_chart[0,20:54,20:54])

        unit_type = obs.observation['feature_screen'][static.screen_features["unit type"]]
        ys, xs = (unit_type == static.unit_ids['command center']).nonzero()

        # find the center of the pixels relating to command centers
        cc_x = util.round(xs.mean())
        cc_y = util.round(ys.mean())

        # set the offset info for this minimap pixel
        bot.minimap_offset_chart[0, cam_y, cam_x] = 1
        bot.minimap_offset_chart[1, cam_y, cam_x] = -cc_x
        bot.minimap_offset_chart[2, cam_y, cam_x] = -cc_y

        # set the offset for the screen_height_chart
        bot.screen_height_chart_offset = [-cc_x, -cc_y]

        print("screen height chart offset: ")
        #print(self.screen_height_chart_offset)

        print("minimap offset chart: ")
        print(bot.minimap_offset_chart[0:3,cam_y-3:cam_y+3, cam_x-3:cam_x+3])


    bot.callback_method = None
    bot.callback_parameters = {}
    return None

# This only has to be done once per map per setting, if we save it for later in a map-info file.

def update_height_chart(obs, args = {}):
    bot = None
    if "bot" in args:
        bot = args["bot"]

    step = 0
    point = None
    previous = None # the previous screen position if we just tried to move the screen

    cam = obs.observation['feature_minimap'][static.minimap_features["camera"]]
    cam_ys, cam_xs = (cam == 1).nonzero()

    cam_x = util.round(cam_xs.mean())
    cam_y = util.round(cam_ys.mean())

    if "step" in args:
        step = args["step"]

    # a point relative to the current position
    if "relative" in args:
        point = [args["relative"][0]+cam_x,args["relative"][1]+cam_y]
        print("point: " + str(point))

    #bot.charting_order is a list of uncharted minimap coords from closest to the command center to farthest.
    if "list" in args:
        if args["list"] == True:
            if len(bot.charting_order) > 0:
                item = bot.charting_order.pop(0)
                point = [item[1],item[2]]
            else:
                print("Seems we've charted the whole damn map!")
                bot.callback_method = None
                bot.callback_parameters = {}
                return

    if "point" in args:
        point = args["point"]

    if "previous" in args:
        previous = args["previous"]

    print(f"height-charting: {point[0]},{point[1]}")

    # don't proceed if we don't have a point
    if point is None:
        bot.callback_method = None
        bot.callback_parameters = {}
        return

    if bot.minimap_offset_chart[0,point[1],point[0]] == 1:
        print(f"we've already checked this point. It's in the map: {point[0]},{point[1]}")
        bot.callback_method = None
        bot.callback_parameters = {}
        return
                       
    # check adjacency, we can only update the chart if there's a map point filled in in an adjacent cardinal direction
    if bot.minimap_offset_chart[0,point[1],point[0]-1] == 1:
        pass
    elif bot.minimap_offset_chart[0,point[1],point[0]+1] == 1:
        pass
    elif bot.minimap_offset_chart[0,point[1]-1,point[0]] == 1:
        pass
    elif bot.minimap_offset_chart[0,point[1]+1,point[0]] == 1:
        pass
    else:
        print("not sure how to link this tile to a neighbor")
        bot.callback_method = None
        bot.callback_parameters = {}
        return

    if step == 0:
        # move the screen to the requested point and return
        # Move to minimap_home_base (the center pixel of the home command center on the minimap)
        x,y = point
        print(f"moving screen to: {x},{y}")
        
        action = { 
                "id"    :   static.action_ids["move camera"], 
                "params":   [[x,y]] 
                }

        bot.callback_method = update_height_chart
        bot.callback_parameters = {
                "step"      :   1,
                "point"     :   point,
                "previous"  :   [cam_x,cam_y],
                "bot"       :   bot
            }
                         
        return bot.try_perform_action(obs, action)

    elif step == 1:
        x,y = point
            
        # We're doing edge detection elsewhere already... this shouldn't be needed
        #px,py = previous
        #if previous == point:
        #    print("couldn't move the screen, we must be at the edge")
        #    bot.callback_method = None
        #    bot.callback_parameters = {}
        #    return

        # Get the location of the screen on the minimap for a starting point
        cam = obs.observation['feature_minimap'][static.minimap_features["camera"]]
        cam_ys, cam_xs = (cam == 1).nonzero()

        cam_x = util.round(cam_xs.mean())
        cam_y = util.round(cam_ys.mean())

        new_height_data = obs.observation['feature_screen'][static.screen_features["height map"]]

        # save a copy for after the compare
        new_height_copy = new_height_data[0:np.shape(new_height_data)[0],0:np.shape(new_height_data)[1]]
        #print("scr heights: " + str(new_height_data))

        if bot.minimap_offset_chart[0,y,x-1] == 1:
            # ok, time to scan in from the left. 
            #print("scanning from the left")
            # load the old_height_data from the screen_height_chart for the correct spot
            # the we need an 84x84 height chunk from screen_height_chart
            # the top left pixel of the chunk is minimap_offset_chart[1,point[0]-1,point[0]] - screen_height_chart_offset

            chart_offset_x = bot.minimap_offset_chart[1,y,x-1]
            chart_offset_y = bot.minimap_offset_chart[2,y,x-1]
            global_offset_x = bot.screen_height_chart_offset[0]
            global_offset_y = bot.screen_height_chart_offset[1]

            chunk_x0 = int(chart_offset_x - global_offset_x)
            chunk_y0 = int(chart_offset_y - global_offset_y)

            chunk_x1 = int(chunk_x0 + bot.screen_dimensions[0])
            chunk_y1 = int(chunk_y0 + bot.screen_dimensions[1])
            #print(str(np.shape(bot.screen_height_chart)))
            #print(f"getting data for {chunk_x0}:{chunk_x1},{chunk_y0}:{chunk_y1}")

            old_height_chunk = bot.screen_height_chart[0,chunk_y0:chunk_y1,chunk_x0:chunk_x1]
            #print("old height chunk")
            #print(str(np.shape(old_height_chunk)))
            #print(old_height_chunk[30:54,30:54])

            for i in range(30):
                # crop the right side of new_height_data and the left pixels of old_height_data
                old_height_chunk = old_height_chunk[0:np.shape(old_height_chunk)[0],0+1:np.shape(old_height_chunk)[1]]
                new_height_data = new_height_data[0:np.shape(new_height_data)[0],0:np.shape(new_height_data)[1]-1]
                # check to see if the height datas are equal.
                #print("trying i of: " + str(i) + " old shape: " + str(np.shape(old_height_chunk)) + " new shape: " + str(np.shape(new_height_data)))

                if np.array_equal(old_height_chunk, new_height_data):
                    print("old end")
                    #print(self.screen_height_chart[0,chunk_y0:chunk_y1,chunk_x1-(i+1):chunk_x1])
                        
                    # if so, add the new_height_data to the old_height_data
                    #print("new height copy: " + str(i) + " new shape: " + str(np.shape(new_height_copy)))

                    # get the new data to be added
                    new_height_copy = new_height_copy[0:np.shape(new_height_copy)[0],np.shape(new_height_copy)[1] - (i+1):np.shape(new_height_copy)[1]]
                    new_columns = np.shape(new_height_copy)[1]

                    # update it back into the screen_height_chart

                    ### Check if we have to extend the current canvas for the new data.
                    # chunk_x0 is the left pixel of the chunk we're comparing to
                    # new_columns is how much space we need left of the chunk
                    # self.screen_height_chart_offset[0] describes the left most edge of the existing height chart
                    print(f"right edge of chart: {chart_offset_x} + bot.screen_dimensions[0] {bot.screen_dimensions[0]} + new_columns + {new_columns}")
                    print(f" vs screen_height_chart {np.shape(bot.screen_height_chart)[2]}")

                    # create a new canvas big enough for both old and new data
                    ### Only need to increase this size if new columns are needed.


                    new_data_left_edge = (chart_offset_x + bot.screen_dimensions[0]) - global_offset_x
                    new_data_right_edge = new_data_left_edge + new_columns
                        
                    if new_data_right_edge > np.shape(bot.screen_height_chart)[2]:
                        print("extending right edge of canvas")

                        new_screen_height_chart = np.zeros((np.shape(bot.screen_height_chart)[0], 
                                                            np.shape(bot.screen_height_chart)[1], 
                                                            np.shape(bot.screen_height_chart)[2]+new_columns),
                                                            dtype=int)

                        # copy in old data
                        new_screen_height_chart[0:2,
                                                0:np.shape(bot.screen_height_chart)[1],
                                                0:np.shape(bot.screen_height_chart)[2]
                                                ] = bot.screen_height_chart

                        bot.screen_height_chart = new_screen_height_chart



                    #print("Screen Height Chart: "+str(np.shape(bot.screen_height_chart)))
                    print(f"new data location: {new_data_left_edge},{new_data_right_edge}")



                    # write in new data
                    bot.screen_height_chart[0,
                                            chunk_y0:chunk_y1,
                                            new_data_left_edge:new_data_right_edge] = new_height_copy


                    # set the pixels on the screen_height_chart to registered with the bool
                    bot.screen_height_chart[1,
                                            chunk_y0:chunk_y1,
                                            new_data_left_edge:new_data_right_edge] = 1

                    #print(bot.screen_height_chart[0:1,
                    #                                chunk_y0:24,
                    #                                new_data_left_edge:new_data_right_edge+new_columns])

                    # if we are scanning from the right, update the screen_height_chart_offset with the new offset info
                    # now update the minimap_offset_chart
                    bot.minimap_offset_chart[1,y,x] = chart_offset_x + new_columns
                    bot.minimap_offset_chart[2,y,x] = chart_offset_y

                    # set the pixel on the minimap_offset_chart to registered
                    bot.minimap_offset_chart[0,y,x] = 1

                    #print(str(self.minimap_offset_chart[0:3,y,x-1]))
                    #print(str(self.minimap_offset_chart[0:3,y,x]))

                    #print("minimap offset chart: ")
                    #print(bot.minimap_offset_chart[1:3,cam_y-3:cam_y+3, cam_x-3:cam_x+3])

                    break
                

        elif bot.minimap_offset_chart[0,y,x+1] == 1:
            # ok, time to scan in from the right. 
            print("scanning from the right")
            # load the old_height_data from the screen_height_chart for the correct spot
            # the we need an 84x84 height chunk from screen_height_chart
            # the top left pixel of the chunk is minimap_offset_chart[1,point[0]-1,point[0]] - screen_height_chart_offset
            chart_offset_x = bot.minimap_offset_chart[1,y,x+1]
            chart_offset_y = bot.minimap_offset_chart[2,y,x+1]
            global_offset_x = bot.screen_height_chart_offset[0]
            global_offset_y = bot.screen_height_chart_offset[1]

            chunk_x0 = int(chart_offset_x - global_offset_x)
            chunk_y0 = int(chart_offset_y - global_offset_y)

            chunk_x1 = int(chunk_x0 + bot.screen_dimensions[0])
            chunk_y1 = int(chunk_y0 + bot.screen_dimensions[1])

            print(str(np.shape(bot.screen_height_chart)))
            print(f"getting data for {chunk_x0}:{chunk_x1},{chunk_y0}:{chunk_y1}")

            old_height_chunk = bot.screen_height_chart[0,chunk_y0:chunk_y1,chunk_x0:chunk_x1]
            print("old height chunk")
            print(str(np.shape(old_height_chunk)))
            #print(old_height_chunk[30:54,30:54])

            for i in range(30):
                # crop the right side of new_height_data and the left pixels of old_height_data
                old_height_chunk = old_height_chunk[0:np.shape(old_height_chunk)[0],0:np.shape(old_height_chunk)[1]-1]
                new_height_data = new_height_data[0:np.shape(new_height_data)[0],0+1:np.shape(new_height_data)[1]]
                # check to see if the height datas are equal.
                #print("trying i of: " + str(i) + " old shape: " + str(np.shape(old_height_chunk)) + " new shape: " + str(np.shape(new_height_data)))

                if np.array_equal(old_height_chunk, new_height_data):
                    print("old end")
                    #print(self.screen_height_chart[0,chunk_y0:chunk_y1,chunk_x1-(i+1):chunk_x1])
                        
                    # if so, add the new_height_data to the old_height_data
                    print("new height copy: " + str(i) + " new shape: " + str(np.shape(new_height_copy)))

                    # get the new data to be added
                    new_height_copy = new_height_copy[0:np.shape(new_height_copy)[0],0:(i+1)]
                    new_columns = np.shape(new_height_copy)[1]

                    #print(new_height_copy)

                    # update it back into the screen_height_chart

                    ### Check if we have to extend the current canvas for the new data.
                    # chunk_x0 is the left pixel of the chunk we're comparing to
                    # new_columns is how much space we need left of the chunk
                    # self.screen_height_chart_offset[0] describes the left most edge of the existing height chart
                    #print(f"left edge of chart: {chart_offset_x} - new columns {new_columns} vs global {global_offset_x}")

                    # create a new canvas big enough for both old and new data
                    ### Only need to increase this size if new columns are needed.


                    new_data_left_edge = (chart_offset_x - new_columns) - global_offset_x
                    new_data_right_edge = new_data_left_edge + new_columns
                        
                    if new_data_left_edge < 0:
                        print("extending left edge of canvas")
                        new_screen_height_chart = np.zeros((np.shape(bot.screen_height_chart)[0], 
                                                            np.shape(bot.screen_height_chart)[1], 
                                                            new_columns+np.shape(bot.screen_height_chart)[2]),
                                                            dtype=int)

                        # copy in old data
                        new_screen_height_chart[0:2,
                                                0:np.shape(bot.screen_height_chart)[1],
                                                new_columns:np.shape(new_screen_height_chart)[2]
                                                ] = bot.screen_height_chart


                        bot.screen_height_chart = new_screen_height_chart
                        new_data_left_edge += new_columns
                        new_data_right_edge += new_columns

                        # if we are scanning from the right, update the screen_height_chart_offset with the new offset info
                        # now update the minimap_offset_chart
                        print(f"setting new global offset: {global_offset_x} - {new_columns}")
                        bot.screen_height_chart_offset[0] = global_offset_x - new_columns

                    #print("Screen Height Chart: "+str(np.shape(bot.screen_height_chart)))
                    #print(f"new data location: {new_data_left_edge},{new_data_right_edge}")

                    # write in new data
                    #new_screen_height_chart[0,
                    #                        chunk_y0:chunk_y1,
                    #                        np.shape(self.screen_height_chart)[2]:np.shape(new_screen_height_chart)[2]] = new_height_copy
                        
                    bot.screen_height_chart[0,
                                            chunk_y0:chunk_y1,
                                            new_data_left_edge:new_data_right_edge] = new_height_copy

                    #print(bot.screen_height_chart[0:1,
                    #                                chunk_y0:20,
                    #                                new_data_left_edge:new_data_right_edge+new_columns])

                    # set the pixels on the screen_height_chart to registered with the bool
                    bot.screen_height_chart[1,
                                            chunk_y0:chunk_y1,
                                            new_data_left_edge:new_data_right_edge] = 1

                    print("global offset: " + str(bot.screen_height_chart_offset))




                    bot.minimap_offset_chart[1,y,x] = chart_offset_x - new_columns
                    bot.minimap_offset_chart[2,y,x] = chart_offset_y
                        

                    # set the pixel on the minimap_offset_chart to registered
                    bot.minimap_offset_chart[0,y,x] = 1

                    #print("minimap offset chart: ")
                    #print(bot.minimap_offset_chart[1:3,cam_y-3:cam_y+3, cam_x-3:cam_x+3])


                    break
                

        elif bot.minimap_offset_chart[0,y-1,x] == 1:
            # ok, time to scan in from the top. 
            print("scanning from the top")
            # load the old_height_data from the screen_height_chart for the correct spot
            # the we need an 84x84 height chunk from screen_height_chart
            # the top left pixel of the chunk is minimap_offset_chart[1,point[1],point[0]-1] - screen_height_chart_offset
            chart_offset_x = bot.minimap_offset_chart[1,y-1,x]
            chart_offset_y = bot.minimap_offset_chart[2,y-1,x]
            global_offset_x = bot.screen_height_chart_offset[0]
            global_offset_y = bot.screen_height_chart_offset[1]

            chunk_x0 = int(chart_offset_x - global_offset_x)
            chunk_y0 = int(chart_offset_y - global_offset_y)

            chunk_x1 = int(chunk_x0 + bot.screen_dimensions[0])
            chunk_y1 = int(chunk_y0 + bot.screen_dimensions[1])

            print(str(np.shape(bot.screen_height_chart)))
            print(f"getting data for {chunk_x0}:{chunk_x1},{chunk_y0}:{chunk_y1}")

            old_height_chunk = bot.screen_height_chart[0,chunk_y0:chunk_y1,chunk_x0:chunk_x1]
            print("old height chunk")
            print(str(np.shape(old_height_chunk)))
            #print(old_height_chunk[30:54,30:54])

            for i in range(30):
                # crop the bottom side of new_height_data and the top pixels of old_height_data
                old_height_chunk = old_height_chunk[0+1:np.shape(old_height_chunk)[0],0:np.shape(old_height_chunk)[1]]
                new_height_data = new_height_data[0:np.shape(new_height_data)[0]-1,0:np.shape(new_height_data)[1]]

                # check to see if the height datas are equal.
                #print("trying i of: " + str(i) + " old shape: " + str(np.shape(old_height_chunk)) + " new shape: " + str(np.shape(new_height_data)))

                if np.array_equal(old_height_chunk, new_height_data):
                    print("old end")
                    #print(self.screen_height_chart[0,chunk_y0:chunk_y1,chunk_x1-(i+1):chunk_x1])
                        
                    # if so, add the new_height_data to the old_height_data
                    print("new height copy: " + str(i) + " new shape: " + str(np.shape(new_height_copy)))

                    # get the new data to be added
                    new_height_copy = new_height_copy[np.shape(new_height_copy)[0] - (i+1):np.shape(new_height_copy)[0],
                                                        0:np.shape(new_height_copy)[1]]
                    new_rows = np.shape(new_height_copy)[0]

                    #print(new_height_copy)

                    ## Working location

                    # update it back into the screen_height_chart

                    ### Check if we have to extend the current canvas for the new data.
                    # chart_offset_y is the top rows of the chart
                    # new_rows is how much space we need up of the chunk
                    # self.screen_height_chart_offset[0] describes the left most edge of the existing height chart

                    print(f"bottom edge of chart: {chart_offset_y} + screen_dimensions[1] {bot.screen_dimensions[1]} + new_rows {new_rows}")
                    print(f" vs screen_height_chart {np.shape(bot.screen_height_chart)[1]}")

                    # create a new canvas big enough for both old and new data
                    ### Only need to increase this size if new columns are needed.

                    # new_data_left_edge = (chart_offset_x + self.screen_dimensions[0]) - (global_offset_x)
                    # new_data_right_edge = new_data_left_edge + new_columns
                        
                    # if new_data_right_edge > np.shape(self.screen_height_chart)[2] + global_offset_x:

                    new_data_top_edge = (chart_offset_y + bot.screen_dimensions[1]) - (global_offset_y)
                    new_data_bottom_edge = new_data_top_edge + new_rows

                    if new_data_bottom_edge > np.shape(bot.screen_height_chart)[1]:
                        print("extending bottom edge of canvas")
                        new_screen_height_chart = np.zeros((np.shape(bot.screen_height_chart)[0], 
                                                            np.shape(bot.screen_height_chart)[1]+new_rows, 
                                                            np.shape(bot.screen_height_chart)[2]),
                                                            dtype=int)

                        # copy in old data
                        new_screen_height_chart[0:2,
                                                0:np.shape(bot.screen_height_chart)[1],
                                                0:np.shape(bot.screen_height_chart)[2]
                                                ] = bot.screen_height_chart

                        bot.screen_height_chart = new_screen_height_chart

                        # if we are scanning from the right, update the screen_height_chart_offset with the new offset info
                        # now update the minimap_offset_chart
                        #print(f"setting new global offset: {global_offset_y} - {new_rows}")


                    #print("Screen Height Chart: "+str(np.shape(bot.screen_height_chart)))
                    #print(f"new data location: {new_data_top_edge},{new_data_bottom_edge}")

                    # write in new data
                    #new_screen_height_chart[0,
                    #                        chunk_y0:chunk_y1,
                    #                        np.shape(self.screen_height_chart)[2]:np.shape(new_screen_height_chart)[2]] = new_height_copy
                        
                                            # write in new data




                    bot.screen_height_chart[0,
                                                new_data_top_edge:new_data_bottom_edge,
                                                chunk_x0:chunk_x1
                                            ] = new_height_copy

                    # set the pixels on the screen_height_chart to registered with the bool
                    bot.screen_height_chart[1,
                                                new_data_top_edge:new_data_bottom_edge,
                                                chunk_x0:chunk_x1
                                            ] = 1

                    #print(bot.screen_height_chart[0:1,
                    #                                new_data_top_edge:new_data_bottom_edge,
                    #                                chunk_x0:15])

                    print("global offset: " + str(bot.screen_height_chart_offset))

                    # update it back into the screen_height_chart
                    # create a new canvas big enough for both old and new data
                    new_screen_height_chart = np.zeros((np.shape(bot.screen_height_chart)[0], 
                                                        np.shape(bot.screen_height_chart)[1]+new_rows, 
                                                        np.shape(bot.screen_height_chart)[2]),
                                                        dtype=int)

                    # copy in old data
                    new_screen_height_chart[0:2,
                                            0:np.shape(bot.screen_height_chart)[1],
                                            0:np.shape(bot.screen_height_chart)[2]] = bot.screen_height_chart

                    print(str(np.shape(new_screen_height_chart)))

                    # write in new data
                    new_screen_height_chart[0,
                                            np.shape(bot.screen_height_chart)[1]:np.shape(new_screen_height_chart)[1],
                                            chunk_x0:chunk_x1] = new_height_copy

                    #print(new_screen_height_chart[0,
                    #                              np.shape(self.screen_height_chart)[2]-5:np.shape(new_screen_height_chart)[2],
                    #                              chunk_x0:chunk_x1])

                    # set the pixels on the screen_height_chart to registered with the bool
                    new_screen_height_chart[1,
                                            np.shape(bot.screen_height_chart)[1]:np.shape(new_screen_height_chart)[1],
                                            chunk_x0:chunk_x1] = 1


                    bot.screen_height_chart = new_screen_height_chart

                    # if we are scanning from the right, update the screen_height_chart_offset with the new offset info
                    # now update the minimap_offset_chart
                    bot.minimap_offset_chart[1,y,x] = chart_offset_x
                    bot.minimap_offset_chart[2,y,x] = chart_offset_y + new_rows

                    # set the pixel on the minimap_offset_chart to registered
                    bot.minimap_offset_chart[0,y,x] = 1

                    #print(str(self.minimap_offset_chart[0:3,y-1,x]))
                    #print(str(self.minimap_offset_chart[0:3,y,x]))

                    #print("minimap offset chart: ")
                    #print(bot.minimap_offset_chart[0:3,cam_y-3:cam_y+3, cam_x-3:cam_x+3])


                    break
                

        elif bot.minimap_offset_chart[0,y+1,x] == 1:
            # ok, time to scan in from the top. 
            print("scanning from the bottom")
            # load the old_height_data from the screen_height_chart for the correct spot
            # the we need an 84x84 height chunk from screen_height_chart
            # the top left pixel of the chunk is minimap_offset_chart[1,point[1],point[0]-1] - screen_height_chart_offset
            chart_offset_x = bot.minimap_offset_chart[1,y+1,x]
            chart_offset_y = bot.minimap_offset_chart[2,y+1,x]
            global_offset_x = bot.screen_height_chart_offset[0]
            global_offset_y = bot.screen_height_chart_offset[1]

            chunk_x0 = int(chart_offset_x - global_offset_x)
            chunk_y0 = int(chart_offset_y - global_offset_y)

            chunk_x1 = int(chunk_x0 + bot.screen_dimensions[0])
            chunk_y1 = int(chunk_y0 + bot.screen_dimensions[1])

            print(str(np.shape(bot.screen_height_chart)))
            print(f"getting data for {chunk_x0}:{chunk_x1},{chunk_y0}:{chunk_y1}")

            old_height_chunk = bot.screen_height_chart[0,chunk_y0:chunk_y1,chunk_x0:chunk_x1]
            print("old height chunk")
            print(str(np.shape(old_height_chunk)))
            #print(old_height_chunk[30:54,30:54])

            for i in range(30):
                # crop the bottom side of new_height_data and the top pixels of old_height_data
                old_height_chunk = old_height_chunk[0:np.shape(old_height_chunk)[0]-1,0:np.shape(old_height_chunk)[1]]
                new_height_data = new_height_data[0+1:np.shape(new_height_data)[0],0:np.shape(new_height_data)[1]]

                # check to see if the height datas are equal.
                #print("trying i of: " + str(i) + " old shape: " + str(np.shape(old_height_chunk)) + " new shape: " + str(np.shape(new_height_data)))

                if np.array_equal(old_height_chunk, new_height_data):
                    print("old end")
                    #print(self.screen_height_chart[0,chunk_y0:chunk_y1,chunk_x1-(i+1):chunk_x1])
                        
                    # if so, add the new_height_data to the old_height_data
                    print("new height copy: " + str(i) + " new shape: " + str(np.shape(new_height_copy)))

                    # get the new data to be added
                    new_height_copy = new_height_copy[0:(i+1),
                                                        0:np.shape(new_height_copy)[1]]
                    new_rows = np.shape(new_height_copy)[0]

                    #print(new_height_copy)



                    # update it back into the screen_height_chart

                    ### Check if we have to extend the current canvas for the new data.
                    # chart_offset_y is the top rows of the chart
                    # new_rows is how much space we need up of the chunk
                    # self.screen_height_chart_offset[0] describes the left most edge of the existing height chart
                    print(f"top edge of chart: {chart_offset_y} - new rows {new_rows} vs global {global_offset_y}")

                    # create a new canvas big enough for both old and new data
                    ### Only need to increase this size if new columns are needed.


                    new_data_top_edge = (chart_offset_y - new_rows) - global_offset_y
                    new_data_bottom_edge = new_data_top_edge + new_rows
                        
                    if new_data_top_edge < 0:
                        print("extending top edge of canvas")
                        new_screen_height_chart = np.zeros((np.shape(bot.screen_height_chart)[0], 
                                                            new_rows+np.shape(bot.screen_height_chart)[1], 
                                                            np.shape(bot.screen_height_chart)[2]),
                                                            dtype=int)

                        # copy in old data
                        new_screen_height_chart[0:2,
                                                new_rows:np.shape(new_screen_height_chart)[1],
                                                0:np.shape(bot.screen_height_chart)[2]
                                                ] = bot.screen_height_chart


                        bot.screen_height_chart = new_screen_height_chart
                        new_data_top_edge += new_rows
                        new_data_bottom_edge += new_rows

                        # if we are scanning from the right, update the screen_height_chart_offset with the new offset info
                        # now update the minimap_offset_chart
                        print(f"setting new global offset: {global_offset_y} - {new_rows}")
                        bot.screen_height_chart_offset[1] = global_offset_y - new_rows

                    #print("Screen Height Chart: "+str(np.shape(bot.screen_height_chart)))
                    #print(f"new data location: {new_data_top_edge},{new_data_bottom_edge}")

                    # write in new data
                    #new_screen_height_chart[0,
                    #                        chunk_y0:chunk_y1,
                    #                        np.shape(self.screen_height_chart)[2]:np.shape(new_screen_height_chart)[2]] = new_height_copy
                        
                    bot.screen_height_chart[0,
                                                new_data_top_edge:new_data_bottom_edge,
                                                chunk_x0:chunk_x1
                                            ] = new_height_copy

                    #print(bot.screen_height_chart[0:1,
                    #                                new_data_top_edge:new_data_bottom_edge+new_rows,
                    #                                chunk_x0:16
                    #                                ])

                    # set the pixels on the screen_height_chart to registered with the bool
                    bot.screen_height_chart[1,
                                                new_data_top_edge:new_data_bottom_edge,
                                                chunk_x0:chunk_x1
                                            ] = 1

                    print("global offset: " + str(bot.screen_height_chart_offset))
                        

                    bot.minimap_offset_chart[1,y,x] = chart_offset_x
                    bot.minimap_offset_chart[2,y,x] = chart_offset_y - new_rows


                    # set the pixel on the minimap_offset_chart to registered
                    bot.minimap_offset_chart[0,y,x] = 1

                    #print(str(self.minimap_offset_chart[0:3,y-1,x]))
                    #print(str(self.minimap_offset_chart[0:3,y,x]))

                    #print("minimap offset chart: ")
                    #print(bot.minimap_offset_chart[0:3,cam_y-3:cam_y+3, cam_x-3:cam_x+3])


                    break
                

    bot.callback_method = None
    bot.callback_parameters = {}
    return