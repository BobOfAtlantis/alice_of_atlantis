# Note: Thanks to Steven Brown for tutorials and Timo for quick responses when I had pysc2 questions

# Notes:
# - The bot moves the screen by clicking on the minimap through the pysc2 pixel choice interface. 
#   If the screen is moved in any other way, building locations will be off, and wall offs etc will fail. 
# - When plotting building locations in the config files, currently the way to do it is to turn off the builder ai,
#   Hand build the building, then have the game print out the building's location. SCV's or neighboring buildings will throw this off
# - The first times the bot sees a map it'll look like it's going nuts as it scans the height charts of the map and saves the map to file
#   This gives it x,y coords for each minimap location so that it can build buildings in the right spots.
# - The PPO algorithm builder_ai (also alice_the_builder) is for build timings for buildings and units... still a lot of work to be done here.

from decimal import Decimal
import hashlib
import json
import math
import os
import os.path
import sys
import time

import tensorflow as tf
import numpy as np
import pandas as pd

from alice.lib import map_reader, building_planner, static, util, builder_ai


from pysc2.agents import base_agent
from pysc2.lib import actions, features, units

np.set_printoptions(threshold=10000)

class aBot2Agent(base_agent.BaseAgent):
    def __init__(self):
        super().__init__()
        
    def setup(self, obs_spec, action_spec):
        super().setup(obs_spec, action_spec)
        self.builder = None
        self.builder_running = False
        self.reward = 0
        # tensorflow session
        #self.sess = tf.Session()
        
        # features: step, minerals, score
        # actions: build_scv, build_supply_depot, no_op

        #self.builder_actor = Actor(self.sess, n_features=3, lr=0.005, action_bound=[0, 3])
        #self.builder_critic = Critic(self.sess, n_features=3, lr=0.01)

        #self.sess.run(tf.global_variables_initializer())

        # screen dimensions in pixels (for the simple64 map it's 84x84)
        self.screen_dimensions = [obs_spec["feature_screen"][1],obs_spec["feature_screen"][2]]

        # minimap dimensions in minimap pixels for the simple64 map it's 64x64)
        self.minimap_dimensions = [obs_spec["feature_minimap"][1],obs_spec["feature_minimap"][2]]



        #print(f"obs_spec: {obs_spec}")
        #obs_spec: {'action_result': (0,), 'alerts': (0,), 'available_actions': (0,), 'build_queue': (0, 7), 'cargo': (0, 7), 'cargo_slots_available': (1,), 
        #'control_groups': (10, 2), 'game_loop': (1,), 'last_actions': (0,), 'multi_select': (0, 7), 'player': (11,), 'score_cumulative': (13,), 
        #'score_by_category': (11, 5), 'score_by_vital': (3, 3), 'single_select': (0, 7), 'feature_screen': (17, 84, 84), 'feature_minimap': (7, 64, 64), 
        #'feature_units': (0, 28), 'camera_position': (2,)} 

    def reset(self):
        super().reset()
        print(f"episode: {self.episodes}")
        print(f"reward: {self.reward}")

        # if there is a builder, we need to close out its last action
        if self.builder is not None and self.builder_running == True:
            #print("game ended, need to clear the buffer")
            r = self.reward * 1000
            self.finish_alice_the_builder(r)

        
        # location and building type for pending buildings
        # locations are relative to starting base 
        # status includes attempt number, result of last information etc
        # {"building type":bt, "builder control group": bcg, "location": [x,y], "status": state}
        self.buildings = [{"type":static.unit_ids["command center"],"location":[0,0],"status":"complete"}]
        self.building_plan = {}

        # whenever obs sends us an alert about the game. We can't handle it immediately if we have a callback to do, but we should check it out after that
        self.alert_queue = []

        # the reason not to have a callback list, is because the logic tree branches too often. If a check fails, you must try again, not move on to the next step.
        self.callback_method = None
        self.callback_parameters = {}

        # a list of last known locations of SCVs waiting to build things
        self.idle_scv_locations = []

        # a list of high priority actions that must be done before the schedule is checked
        # 1 = highest priority, 9 = lowest priority
        self.priority_queue = []

        # Functions called by the schedule should be considered higher level than actions on the priority queue.
        # time (game_loop), function, params
        # [[100,self.build_supply_depot,{}]]
        self.schedule = []

        #self.alice_thebuilder_s = None
        #self.alice_thebuilder_a = None



    def try_callback(self, obs):
        if self.callback_method is None:
            self.callback_parameters = None
            return
        else:
            return self.callback_method(obs, self.callback_parameters)

    def perform_priority_action(self, obs):
        if(len(self.priority_queue) > 0):
            # start with zero in case you really, really need something now.
            # for each priority level
            for i in range(10):
                # find the first item with that priority level
                for x in range(len(self.priority_queue)):
                    # if an element has the correct priority
                    if(self.priority_queue[x][0] == i):
                        action = self.priority_queue.pop(x)
                        func = action[1]
                        params = action[2]
                        
                        return func(obs, params)   

    # a list of scheduled actions. Some actions will re-schedule themeselves, until the ai component for schedule management is complete
    # priority queue items need to be able to fail and pass on the queue to the next item. 
    # the schedule should be responsible for dealing with the issues of a failure. On failure, a re-scheduled event should be arranged

    # add a function call to the schedule, if the game loop >= time, the function will be run
    def schedule_action(self, obs, time, function, params):
        #print("action scheduled: " + str(function))
        self.schedule.append([time, function, params])
        return

    # run any functions which are ready and on the schedule
    def check_schedule(self, obs):
        game_loop = obs.observation["game_loop"][0]

        for item in self.schedule:
            item_time = item[0]
            item_func = item[1]
            item_params = item[2]

            if game_loop >= item_time:
                #print("running scheduled action: " + str(item_func))
                # it's time to run the thing, mark it off the schedule
                self.schedule.remove(item)
                # run it
                ret = item_func(obs, item_params)
                # TODO: if ret came back with a returned action, toss it on the schedule.
                if ret is not None:
                    print("PROBLEMO: scheduled functions don't run actions! Toss that badboy on the priority queue!")

    # how many screen pixels are in a building spot tile calibrate early and after a few buildings are produced.
    tile_size = [3.8,3.8] 

    # how many screen pixels are in a minimap pixel
    minimap_pixel_size = [4.9,4.9]


    # how large is each building in pixels and in tiles
    # {"unit_id":unit_id,"tiles"[x,y],"screen pixels":[x,y],"minmap pixels":[x,y]}
    building_dimensions = [
            {"unit id":static.unit_ids["command center"], "tiles":[5,5]},
            {"unit id":static.unit_ids["barracks"], "tiles":[3,3]},
            {"unit id":static.unit_ids["supply depot"], "tiles":[2,2]}
        ]

    # how many pixels are on the map
    #absolute_map_dimensions = None

    # the center of the home command center on the minimap [int,int]
    command_home_base = None

    # should be the same as command_home_base, unless it's too close to the edge, which blocks the screen position.
    minimap_home_base = None

    # will probably be deprecated, using the screen_height_chart and minimap_offset_chart
    #absolute_home_base = None

    # The edges of the minimap can't be moved to, because of the size of the screen. 
    # Generally we care about the center of the screen, this gives us the top left pixel that can be clicked on.
    # [[x,y],[x,y]] top left and bottom right locations for the minimap selectable area
    minimap_select_area = [[-1,-1],[-1,-1]]

    # TF,x,y offsets for each minimap location to get absolute coords from screen coords
    # The first element is whether the location has been charted
    minimap_offset_chart = None

    # TF,x,y,height chart for each pixel to calibrate the minimap offset chart. (in screen pixels)
    # 0,0 is the center of the command center
    screen_height_chart = None

    # x,y of the position of the top-most pixel of screen_height_chart (in screen pixels) relative to the command center's center pixel
    screen_height_chart_offset = None

    charting_order = None

    def try_perform_action(self, obs, action):
        if action is None or action is False:
            print("You're trying to perform a False action")
            return None
        else:
            #print("Doing action: " + str(action))
            action_id = action["id"]
            action_params = action["params"]

            if action_id not in obs.observation["available_actions"]:
                #print("Warning, cannot perform action: " + str(action_id) + " We should reschedule")
                return None

            action = actions.FunctionCall(action_id, action_params)
            return action

    # the primary function for pysc2 agents. called each time the game gives access to the bot
    def step(self, obs):
        super().step(obs)
        self.reward = obs.reward
        if(self.reward != 0):
            print(f"reward: {self.reward}")
        
        # what is our absolute game turn index
        # used in qtables and scheduling
        game_loop = obs.observation["game_loop"][0]
        #print(f"step: {str(game_loop)}")

        if game_loop == 0:
            print("initializing game loop")
            self.init_and_calibrate(obs)

        # if there's a multistep action in progress, perform the required callback.
        # This prevents thrashing on the priority queue, while giving a chance to check if a previous action worked
        cb = self.try_callback(obs)
        if cb is not False and cb is not None:
            return cb 

        # add any required calls to the priority_queue
        self.check_schedule(obs)

        # if there's a priority action to do, do the most important priority action
        while len(self.priority_queue) > 0:
            action = self.perform_priority_action(obs)

            #print("action: " + str(action))
            # if this action results in an api call, do it.
            if action is not False and action is not None:
                return action

        # should be the only no_op
        return actions.FUNCTIONS.no_op()

    # Load up the priority_queue with important things to do at the beginning of the game.
    # 1) Build an SCV
    # 2) add starting command center to current buildings list
    # 3) Callibrate minimap and tile dimensions, 
    # -) Control group Command Center (10) Note, this is now linked to build an scv.
    # 4) Control group SCVs
    # 5) Plan supply depot location

    def init_and_calibrate(self, obs):
        self.priority_queue = [
                [2, self.set_up_builder_ai, {}],
                [2, map_reader.calibrate_map_data, {"bot":self}], # select command center
                [2, building_planner.load_building_plan, {"bot":self}],
                #[3, self.train_scv, {}], # control grouping command center should be linked to this. Note: build order might not need this, but if it does, do it
                [3, map_reader.load_map_data, {"bot":self}],
                [3, map_reader.center_screen_on_main, {"bot":self}], # Important step. click on the minimap where the main base is... this aligns config stuff to the screen
                [4, self.control_group_scvs, {}], # good for keeping count of current scvs. useful in build planning
                #[5, self.make_supply_depot, {}], # choose supply depot location, schedule scv move to location, schedule build supply depot
                [6, self.start_alice_the_builder, {}] # bring in Alice, THE BUILDER! Actor/Critic siloed for building buildings and units.
                #[8, self.schedule_print_data, {}] # regular printouts of what's going on.
                #[7, self.test_move_screen, {}]
            ]

        return

    # if it's already set up, don't re-set it.
    def set_up_builder_ai(self, obs, args):
        if self.builder is None:
            self.map_name = util.get_map_folder(obs, args = {"bot":self})
            self.builder = builder_ai.Builder(args = {"bot":self})
            self.builder.reset(self.get_builder_state(obs, {}))
        return

    # go to the command center, set it as the start location, record the center of the camera location on the minimap, save the height map and height map coords matrices
    def record_start_location(self, obs, args):
        return

    def test_move_screen(self, obs, args):
        print("hey, let's try moving the screen")
        x = 30.5
        y = 30.5
        action = { 
                    "id"    :   static.action_ids["move camera"], 
                    "params":   [[x,y]] 
                 }
        return self.try_perform_action(obs, action)

    # mop indicates that this action requires multiple operations
    def train_scv(self, obs, args):
        # have we been here before? are we thrashing?
        ctr = 0
        if "ctr" in args:
            ctr = args["ctr"]
        ctr = ctr + 1

        # TODO: check for available minerals, if there aren't enough set it in schedule and return False

        # if a command center is selected, or multiple are selected... let's get right to it, and train an scv
        if static.action_ids["train scv"] in obs.observation["available_actions"]:
            # if the next thing to do isn't to press the build scv button... then try to select a command center again

            to_do = { "id":static.action_ids["train scv"], 
                      "params":[ [static.params["queued"]] ] 
                    }
            action = self.try_perform_action(obs, to_do) # this is a bit of a safety measure, if it comes back false, it shouldn't run

            # LINKED ACTION: if we take this out, make sure to ctrl group the first cc in init_and_calibrate
            # while we're here, if the command center isn't in the control group, that needs to happen
            command_control_group = obs.observation["control_groups"][0]
            if len(command_control_group) == 2 and command_control_group[0] == static.unit_ids["command center"]:
                self.callback_method = None
                self.callback_parameters = {}
            else:
                self.callback_parameters = {"source":"train_scv","state":"cc selected"}
                self.callback_method = self.control_group_command_center

            self.callback_method = None
            self.callback_parameters = {}
            return action

        else: # command center not selected, let's select it
            if ctr > 5:
                #print("We're having unknown difficulty in building an scv, letting it off the queue")
                # TODO: schedule some things to check to repair the problem
                
                self.callback_parameters = {}
                self.callback_method = None
                return

            #TODO:
            # check the multi select array to see if there are command centers already selected. If so, let's single select one of them to build the scv.
            # check if there are command centers in the cc control group (0)

            to_do = self.select_building(obs, {"type":static.unit_ids['command center']})
            if to_do == None:
                # There wasn't a command center on screen, we must move to a command center.
                self.callback_parameters = {"ctr":ctr}
                self.callback_method = self.train_scv
                return map_reader.center_screen_on_main(obs, {"bot":self})

            else:
                self.callback_parameters = {"ctr":ctr}
                self.callback_method = self.train_scv
                return to_do

        self.callback_parameters = {}
        self.callback_method = None
        return

    # This action does:
    # 1) single select a command center if there is one on screen
    # 2) make sure a command center is single selected
    # 3) add that single selected command center to control group 0
    # 4) check control group 0 to make sure the command center is selected
    def control_group_command_center(self, obs, args):
        step = "unknown"
        if "step" in args:
            step = args["step"]

        if step == "verify":
            # TODO: check the selected pixels on screen, 
            # create an action to recall control group 0
            # create a callback with the selected pixels as args
            # run the action
            # on callback verify that the selected pixels are still selected
            return

        ctr = 0
        if "ctr" in args:
            ctr = args["ctr"]
        ctr = ctr + 1

        if ctr > 3:
            print("We're having unknown difficulty in control grouping a command center, letting it off the queue")
            # TODO: schedule some things to check to repair the problem

        print("hey, let's control group our command center")
        single_select = obs.observation["single_select"]
        if single_select[0][0] == static.unit_ids["command center"]:
            print("cool, command center is selected, let's control group it")
            option = [x for x, y in enumerate(actions.CONTROL_GROUP_ACT_OPTIONS) if y[0] == 'append']
            to_do = { "id":static.action_ids["control group"], "params":[option, [0]] }
            action = self.try_perform_action(obs, to_do)
            if action is not False:
                self.callback_method = self.control_group_command_center
                self.callback_parameters = {"step":"verify","ctr":ctr}
                return action

        multi_select = obs.observation["multi_select"]
        for sel in multi_select:
            if sel[0] == static.unit_ids["command center"]:
                print("There's more than just a command center selected, let's grab the first selected command center")


        self.callback_method = None
        self.callback_parameters = {}
        return

    def control_group_scvs(self, obs, args):
        print("hey, let's control group our scvs")

        self.callback_method = None
        self.callback_parameters = {}
        return

    # decide where to build the supply depot
    # add the planned depot to buildings
    # schedule to have the scv move to build the building
    # schedule to have the scv build the building
    def make_building(self, obs, args):
        building_string = None
        if "building"  in args:
            building_string = args["building"]

        building_type_string = None
        if building_string == "supply depot": building_type_string = "supply depot"
        elif building_string == "barracks": building_type_string = "production"

        #print("hey, let's prepare to build our first supply depot")

        bldg_types = list(filter(lambda building: building["type"] == static.unit_ids[building_string], self.buildings))
        bldg_type_locations = self.building_plan[building_type_string]

        #print("supply depots: " + str(supply_depots))
        #print("supply depot locations: " + str(supply_depot_locations))
        
        # find the first supply depot location without a supply depot already in buildings
        location = []
        building = None
        for l in bldg_type_locations:
            current_building = list(filter(lambda building: building["location"] == l, bldg_types))
            if len(current_building) == 0:
                # add the supply depot to buildings with status: planned
                location = l
                building = {"type":static.unit_ids[building_string],"location":l,"status":"planned"}
                self.buildings.append(building)
                break
                
            elif current_building[0]["status"] == "destroyed":
                # set the status to planned
                location = l
                current_building[0]["status"] == "planned"
                break

        if len(location) == 0:
            #print("error finding a place to build a supply depot")
            self.callback_method = None
            self.callback_parameters = {}

            return


        # schedule an scv move to the location of the building location
        current_time = obs.observation["game_loop"][0]

        # TODO: need to have a method to set this time correctly based on current income etc.
        self.schedule_action(obs, current_time, self.move_scv_to_location, {"location":location,"building":building, "schedule":True})
        

        # schedule the building of the building at location with the scv on location
        # TODO: need to have a method to set this time correctly based on distance to location from the selected SCV
        self.schedule_action(obs, current_time + 140, self.construct_building, {"building":building, "schedule":True})

        self.callback_method = None
        self.callback_parameters = {}
        return

    def move_scv_to_location(self, obs, args):
        # if we're coming from a schedule, we have to pop this item onto the queue.
        if "schedule" in args and args["schedule"]:
            args["schedule"] = False
            self.priority_queue.append([4, self.move_scv_to_location, args])
            return
        
        location = None
        if "location" in args:
            location = args["location"]

        #print("Moving SCV to location: " + str(location))
        # check if an scv is single selected
        # if not, return get_scv with self as a callback method defined in the callback args
        # add the scv to the builder control group.
        single_select = obs.observation["single_select"]
        scv = None

        if len(single_select) > 0:
            #print(single_select)
            if single_select[0][0] == static.unit_ids["scv"]:
                scv = single_select[0]
                #print("we have an SCV selected: " + str(scv))

        
        if scv is None:
            # no SCV selected, let's get one, then come back here
            self.priority_queue.append([1, self.move_scv_to_location, args])
            return self.get_scv(obs, args)

        # we should have an SCV by now, let's move him to the right place.
        # move the screen to the right place.
        if map_reader.is_point_on_screen(obs, {"bot":self, "point":location}):
            self.priority_queue.append([1, self.add_to_group, {"group":"builders"}])
            # move the scv to the right place on the screen
            return map_reader.issue_move_action_on_screen(obs, {"bot":self, "point":location})
        else:
            # move the screen to where it needs to be, then come back here
            self.priority_queue.append([1, self.move_scv_to_location, args])
            return map_reader.move_to_point(obs, {"bot":self, "point":location})

        self.callback_method = None
        self.callback_parameters = {}
        return

    # assign a builder to a control group for a building
    def assign_builder(self, obs, args):
        building = None
        if "building" in args:
            building = args["building"]

        return

    def add_to_group(self, obs, args):
        #print("Ok, let's add this to the group!")
        return

    def construct_building(self, obs, args):
        # if we're coming from a schedule, we have to pop this item onto the queue.
        if "schedule" in args and args["schedule"]:
            args["schedule"] = False
            self.priority_queue.append([4, self.construct_building, args])
            return

        building = None
        if "building" in args:
            building = args["building"]

        #print("building: " + str(building))
        # TODO: check that we have the shekels

        # move the screen to the building location
        # we should have an SCV by now, let's move him to the right place.
        # move the screen to the right place.
        
        if map_reader.is_point_on_screen(obs, {"bot":self, "point":building["location"]}):
            #self.priority_queue.append([1, self.add_to_group, {"group":"builders"}])
            
            # is an scv selected?
            single_select = obs.observation["single_select"]
            if len(single_select) > 0 and single_select[0][0] == static.unit_ids["scv"]:
                scv = single_select[0]
                # build the building
                
                build_action = None
                if building["type"] == static.unit_ids["supply depot"]:
                    build_action = static.action_ids["build supply depot"]
                elif building["type"] == static.unit_ids["barracks"]:
                    build_action = static.action_ids["build barracks"]
                elif building["type"] == static.unit_ids["command center"]:
                    build_action = static.action_ids["build command center"]

                screen_location = map_reader.get_screen_location(obs, {"bot":self, "point":building["location"]})

                if screen_location is None:
                    # the location isn't on screen, this is a problem.
                    print("error, the desired location isn't on screen something went wrong")

                action = { 
                            "id"     :   build_action, 
                            "params" :   [[static.params["queued"]], screen_location] 
                         }
                ret = self.try_perform_action(obs, action)

                # see if we were able to perform this action
                if ret is None:
                    # we were unable to perform the action, so let's re-schedule for a lower queue priority
                    current_time = obs.observation["game_loop"][0]

                    # schedule the building of the building at location with the scv on location
                    # TODO: need to have a method to set this time correctly based on distance to location from the selected SCV
                    self.schedule_action(obs, current_time + 16, self.construct_building, {"building":building, "schedule":True})
                else:
                    building["status"] = "under construction"
                    return ret



            else:
                # TODO: check if the building has a builder assigned to it with a control group
                self.priority_queue.append([1, self.construct_building, args])
                # check the building location for an scv
                return self.get_scv(obs, {"location":building["location"]})
            
            #return map_reader.issue_move_action_on_screen(obs, {"bot":self, "point":location})
            pass
        else:
            # move the screen to where it needs to be, then come back here
            location = building["location"]
            self.priority_queue.append([1, self.construct_building, args])
            return map_reader.move_to_point(obs, {"bot":self, "point":location})

        # check the builder control group for an scv
        # get an scv
        # build the building

        self.callback_method = None
        self.callback_parameters = {}
        return


    def train_marine(self, obs, args):
        # have we been here before? are we thrashing?
        ctr = 0
        if "ctr" in args:
            ctr = args["ctr"]
        ctr = ctr + 1

        # TODO: check for available minerals, if there aren't enough set it in schedule and return False
        # if a barracks is selected, or multiple are selected... let's get right to it
        if static.action_ids["train marine"] in obs.observation["available_actions"]:
            # if the next thing to do isn't to press the build scv button... then try to select a command center again

            to_do = { "id":static.action_ids["train marine"], 
                      "params":[ [static.params["queued"]] ] 
                    }
            action = self.try_perform_action(obs, to_do) # this is a bit of a safety measure, if it comes back false, it shouldn't run

            self.callback_method = None
            self.callback_parameters = {}
            return action

        else: # barracks are not selected, let's select it
            if ctr > 5:
                # this is taking too long, abandon
                self.callback_parameters = {}
                self.callback_method = None
                return

            #TODO:
            # check the multi select array to see if there are barracks already selected. If so, let's select them and start training marines
            # check if there are command centers in the production control group (5)

            to_do = self.select_building(obs, {"type":static.unit_ids['barracks'], "param":"select all"})
            if to_do == None:
                # There wasn't a barracks on screen, we must move to a barracks
                self.callback_parameters = {"ctr":ctr}
                self.callback_method = self.train_marine
                return map_reader.center_screen_on_main(obs, {"bot":self})

            else:
                self.callback_parameters = {"ctr":ctr}
                self.callback_method = self.train_marine
                return to_do

        self.callback_parameters = {}
        self.callback_method = None
        return

            
    def trigger_supply_depots(self, obs, args):
        # have we been here before? are we thrashing?
        ctr = 0
        if "ctr" in args:
            ctr = args["ctr"]
        ctr = ctr + 1

        # if supply depots are selected, trigger them
        if static.action_ids["lower supply depot"] in obs.observation["available_actions"]:
            # if the next thing to do isn't to press the build scv button... then try to select a command center again

            to_do = { "id":static.action_ids["lower supply depot"], 
                      "params":[ [static.params["now"]] ] 
                    }
            action = self.try_perform_action(obs, to_do) # this is a bit of a safety measure, if it comes back false, it shouldn't run

            self.callback_method = None
            self.callback_parameters = {}
            return action

        # if supply depots are selected, trigger them
        elif static.action_ids["raise supply depot"] in obs.observation["available_actions"]:
            # if the next thing to do isn't to press the build scv button... then try to select a command center again

            to_do = { "id":static.action_ids["raise supply depot"], 
                      "params":[ [static.params["now"]] ] 
                    }
            action = self.try_perform_action(obs, to_do) # this is a bit of a safety measure, if it comes back false, it shouldn't run

            self.callback_method = None
            self.callback_parameters = {}
            return action

        else: # supply depots not selected, select them
            if ctr > 5:
                # this is taking too long, abandon
                self.callback_parameters = {}
                self.callback_method = None
                return

            #TODO:
            # check the multi select array to see if there are barracks already selected. If so, let's select them and start training marines
            # check if there are command centers in the production control group (5)

            action = self.select_building(obs, {"type":static.unit_ids['supply depot'], "param":"select all"})
            if action == None:
                # There wasn't a barracks on screen, we must move to a barracks
                self.callback_parameters = {"ctr":ctr}
                self.callback_method = self.trigger_supply_depots
                return map_reader.center_screen_on_main(obs, {"bot":self})

            else:
                self.callback_parameters = {"ctr":ctr}
                self.callback_method = self.trigger_supply_depots
                return action

        self.callback_parameters = {}
        self.callback_method = None
        return

    # select all
    # control group
    # attack move
    def perform_zapp_brannigan_maneuver(self, obs, args):
        #print("zapp")
        my_x, my_y = self.command_home_base

        enemy_ys, enemy_xs = (obs.observation["feature_minimap"][static.screen_features["player relative"]] == static.params["player enemy"]).nonzero()

        center_x = 31
        center_y = 31

        enemy_x = center_x + center_x - my_x
        enemy_y = center_y + center_y - my_y
        
        if(len(enemy_ys) > 0):
            enemy_x = enemy_xs[0]
            enemy_y = enemy_ys[0]

        
        action = { 
            "id"    :   static.action_ids["select army"],
            "params":   [[static.params["now"]]]
            }

        # if there isn't an army to select, this will come back none... let's not send whatever SCV is selected off to die just yet.
        zapp = self.try_perform_action(obs, action)

        if zapp is None:
            return
        else:
            self.priority_queue.append([0, self.control_group_selected, {"type":"append", "group":1}])
            self.priority_queue.append([0, self.a_move, {"point":[enemy_x,enemy_y]}])
            return zapp

    def a_move(self, obs, args):
        #print("a_move")
        point = [0,0]
        if "point" in args: point = args["point"]

        action = { 
            "id"    :   static.action_ids["attack minimap"],
            "params":   [[static.params["now"]],point]
            }

        self.callback_method = None
        self.callback_parameters = {}

        return self.try_perform_action(obs, action)


    def control_group_selected(self, obs, args):
        #print("ctrl")
        type = "append"
        if "type" in args:type = args["type"]

        group = 9
        if "group" in args:group = args["group"]

        command_id = [x for x, y in enumerate(actions.CONTROL_GROUP_ACT_OPTIONS) if y[0] == type]

        action = { 
            "id"    :   static.action_ids["control group"],
            "params":   [command_id, [group]]
            }

        self.callback_method = None
        self.callback_parameters = {}

        return self.try_perform_action(obs, action)


    # whatever it takes to select an available worker scv
    # this command links back to the calling command if it's in args.
    def get_scv(self, obs, args):
        # Are we supposed to get a particular SCV?
        location = None
        if "location" in args:
            location = args["location"]

        # TODO: check if there is an SCV waiting at the location or nearby.

        # if we already have an scv single selected, then mission accomplished, return
        single_select = obs.observation["single_select"]
        scv = None
        if len(single_select) > 0:
            if single_select[0][0] == static.unit_ids["scv"]:
                scv = single_select[0]
                #print("OK, so we have an SCV selected! " + str(scv))
                self.callback_method = None
                self.callback_parameters = {}
                return


        #print("getting SCV")
        attempt = 0
        if "scv_select_attempt" in args:
            attempt = args["scv_select_attempt"]
            args["scv_select_attempt"] = attempt + 1

            if attempt >= 4:
                self.callback_method = None
                self.callback_parameters = {}
                return

        # TODO: If we have a bunch of stuff selected, snag an SCV from the selected bunch of stuff.
        # TODO: If we have a control group for SCVs, snag an SCV from the SCV control group
        
        # select SCV:
        # check for idle SCVs
        #print(str(obs.observation["player"]))
        #print("idle workers:" + str(obs.observation["player"].idle_worker_count))
        idle_workers = obs.observation["player"].idle_worker_count
        if idle_workers > 0:
            
            select_id = [x for x, y in enumerate(actions.SELECT_WORKER_OPTIONS) if y[0] == 'select']
            action = { 
                "id"    :   static.action_ids["select idle worker"],
                "params":   [select_id]
                }

            # need to get idle worker!
            #print("hey, this is broken!")
            #action = actions.FUNCTIONS.select_idle_worker('select')       
            
            # let's come back to make sure we've gotten an SCV
            self.callback_method = self.get_scv
            self.callback_parameters = args

            return self.try_perform_action(obs, action)     

        # if there are no idle SCVs check current screen
        else: 
            # TODO: Finding the center of an SCV shouldn't be too tough, will need to store the height/width of scvs somewhere and select one from that.
            unit_type = obs.observation['feature_screen'][static.screen_features["unit type"]]
            unit_y, unit_x = (unit_type == static.unit_ids["scv"]).nonzero()
              
            if len(unit_y) > 0:
                x = unit_x[0]
                y = unit_y[0]
                
                action = { 
                    "id"    :   static.action_ids["select point"], 
                    "params":   [[static.params["now"]], [x,y]] 
                }

                # let's come back to make sure we've gotten an SCV
                self.callback_method = self.get_scv
                self.callback_parameters = args

                return self.try_perform_action(obs, action) 

            else:
                # no SCVs on screen, we need to go find one!
                # if there are no SCVs on screen, go to home screen
                # TODO: there may also be SCVs working at random mineral patches, or in control groups.
                x,y = self.command_home_base

                action = { 
                    "id"    :   static.action_ids["move camera"], 
                    "params":   [[x,y]] 
                }

                self.callback_method = self.get_scv
                self.callback_parameters = args
                
                return self.try_perform_action(obs, action) 


        self.callback_method = None
        self.callback_parameters = {}
       
        return    

    def get_builder_state(self, obs, args):
        current_time = obs.observation["game_loop"][0]
        minerals = obs.observation["player"][1]
        gas = obs.observation["player"][2]
        empire_value = obs.observation["score_cumulative"][0]

        o = np.zeros([8], dtype = int)
        o[0] = current_time
        o[1] = minerals
        o[2] = gas
        o[3] = empire_value
        return o

    # the first run needs to run the get_action without the wrap_up_action command
    def start_alice_the_builder(self, obs, args):
        self.builder_running = True
        current_time = obs.observation["game_loop"][0]
        self.schedule_action(obs, current_time + 40, self.continue_alice_the_builder, {})
        self.old_empire_value = obs.observation["score_cumulative"][0]
        self.old_time = obs.observation["game_loop"][0]
        
        o = self.get_builder_state(obs, {})
        self.old_o = o

        a = self.builder.get_action(o)

        self.queue_builder_action(obs, {"action":a})

        return

    # the idea here is to have alice the builder choose the buildings and learn the build orders.
    # will need self training, and the ability to save tensorflow training data per map.
    def continue_alice_the_builder(self, obs, args):
        # check if alice has been initialized: if not, do that here.
        #print("You're calling alice the builder? That's all you had to say!")
        current_time = obs.observation["game_loop"][0]

        #obs.observation['score_cumulative']
        empire_value = obs.observation["score_cumulative"][0]
        # new observable state o
        o = self.get_builder_state(obs, {})
        r = (empire_value - self.old_empire_value) / (current_time - self.old_time)
        d = False
        _ = {}

        self.old_o = o

        trainer = self.builder.wrap_up_action(o, r, d, _)
        if trainer == 1:
            print(f"last r of epoch: {r}")
            # done training for this epoch, not sure how to quit the game, but if we don't schedule the next action, it'll just run out without any more buildings
            self.builder_running = False
            return
        elif trainer == -1:
            print(f"last r of final epoch: {r}")
            self.builder_running = False
            quit() # there has to be a pysc2 way of doing this same thing.
            return
            # done with the last epoch, close out.
        else:
            # we're still in the epoch, schedule the next decision point
            self.schedule_action(obs, current_time + 40, self.continue_alice_the_builder, {})
            a = self.builder.get_action()

            # set the old values for the next time through
            self.old_empire_value = empire_value
            self.old_time = current_time

            self.queue_builder_action(obs, {"action":a})
        return

    def finish_alice_the_builder(self, reward):
        self.builder_running = False
        #print("clearing buffer")
        o = self.old_o
        r = reward
        d = True
        _ = {}
        self.builder.wrap_up_action(o, r, d, _)
        return

    def queue_builder_action(self, obs, args):
        a = None
        if "action" in args: a=args["action"]
        
        if a == 1: # train SCV
            #print("  Alice Says: train an scv")
            self.priority_queue.append([4, self.train_scv, {}])

        elif a == 2: # build supply depot
            #print("  Alice Says: build a supply depot")
            self.priority_queue.append([4, self.make_building, {"building":"supply depot"}])

        elif a == 3: # build barracks
            #print("  Alice Says: build a supply depot")
            self.priority_queue.append([4, self.make_building, {"building":"barracks"}])

        elif a == 4: # get SCVs back to work
            pass

        elif a == 5: # build marine
            self.priority_queue.append([4, self.train_marine, {}])
            pass

        elif a == 6: # trigger supply depots
            self.priority_queue.append([4, self.trigger_supply_depots, {}])
            pass

        elif a == 7: # attack move across the map
            self.priority_queue.append([4, self.perform_zapp_brannigan_maneuver, {}])
            pass

        else:
            #print("alice says: do a no_op")
            pass

        return


    def schedule_print_data(self, obs, args):       
        current_time = obs.observation["game_loop"][0]
        # recurring action to print out some data
        self.schedule_action(obs, current_time + 80, self.schedule_print_data, {})
        self.schedule_action(obs, current_time, self.print_data, {})
        return

    # print out useful data for debugging whatever you're working on
    def print_data(self, obs, args):
        single_select = obs.observation["single_select"]
        
        #print(str(obs.observation.keys()))
        #dict_keys(['single_select', 'multi_select', 'build_queue', 'cargo', 'cargo_slots_available', 'feature_screen', 'feature_minimap', 'last_actions',
        #  'action_result', 'alerts', 'game_loop', 'score_cumulative', 'score_by_category', 'score_by_vital', 'player', 'control_groups', 'feature_units',
        #  'camera_position', 'available_actions'])

        #print("Obs Camera Position: " + str(obs.observation["camera_position"]))
        #print("Tile Size: " + str(self.tile_size))    

        if len(single_select) > 0:
            sel = obs.observation['feature_screen'][static.screen_features["selected"]]
            ys, xs = (sel == 1).nonzero()
            current_time = obs.observation["game_loop"][0]

            #print(f"score_by_category: {obs.observation['score_by_category']}")
            #if single_select[0][0] == unit_ids["command center"]:
            #print("Current Game Time: " + str(current_time))
            #print("Alerts: " + str(obs.observation["alerts"]))
            #print("Select info: " + str(single_select))
            #print("idle workers:" + str(obs.observation["player"].idle_worker_count))
            #print("player: " + str(obs.observation["player"]))
            #print(f"Score_cumulativ: " + str(obs.observation['score_cumulative']))
            #print("Last Actions: " + str(obs.observation["last_actions"]))
            #print("Action Result: " + str(obs.observation["action_result"]))
            #print("Player: " + str(obs.observation["player"]))
            #print("Feature Units: " + str(obs.observation["feature_units"]))

            if len(xs) > 0:
                x = util.round(xs.mean())
                y = util.round(ys.mean())

                ax, ay = map_reader.get_absolute_location(obs, {"point":[x, y],"bot":self})

                #print(f"Selected item at screen: x{x},y{y}")
                print(f"Selected item at absolute point: x{ax},y{ay}")
            else:
                #print("nothing is selected")
                pass

        #if single_select[0][0] == unit_ids["command center"]:
        return

    # Returns an action to select the center point of a building type
    # Returns None if that building isn't available on the screen
    # TODO: If there are multiple buildings make sure to select the center of a building based on the size of the building
    def select_building(self, obs, args):
        type = None
        if "type" in args:
            type = args["type"]

        param = "now"
        if "param" in args:
            param = args["param"]

        offset = 0
        
        for bldg in self.building_dimensions:
            if bldg["unit id"] == type:
                offset = bldg["tiles"][0] * self.tile_size[0] / 2

        #print("trying to select: " + str(type))
        # Find the pixels that relate to the unit
        unit_type = obs.observation['feature_screen'][static.screen_features["unit type"]]
        ys, xs = (unit_type == type).nonzero()

        # TODO: if after calibration, check against known building sizes to see if we have multiple on screen 
        #  - On multiple buildings, select the center point of the top left building based on known building dimensions.
        # TODO: if there is no building on screen, check the building tables and go to a minimap location with the building
            
        # find the left most pixel relating to the building
        if len(xs) > 0:
            min_x = min(xs)
            x, y = [min_x,0]
            for i in range(len(xs)):
                if xs[i] == min_x: 
                    y = ys[i]
                    x = x + offset
                    break

            if x >= self.screen_dimensions[0]: x = self.screen_dimensions[0] - 1

            # queue up an action to click on the center pixel
            self.callback_method = None
            self.callback_parameters = {}

            action = { 
                        "id"    :   static.action_ids["select point"], 
                        "params":   [[static.params[param]], [x,y]] 
                     }
            return self.try_perform_action(obs, action)

        return None


def main():
    #print(sys.path)
    os.system('python -m run_agent --map Simple64 --agent abot2.aBot2Agent --agent_race terran --max_agent_steps 0 --game_steps_per_episode 50000')

if __name__ == "__main__":
    main()

