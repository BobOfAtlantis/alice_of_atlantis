# Note: Thanks to Steven Brown for tutorials and Timo for quick responses when I had pysc2 questions

# Notes:
# - The bot moves the screen by clicking on the minimap through the pysc2 pixel choice interface. 
#   If the screen is moved in any other way, building locations will be off, and wall offs etc will fail. 
# - When plotting building locations in the config files, currently the way to do it is to turn off the builder ai,
#   Hand build the building, then have the game print out the building's location.
#   SCVs or neighboring buildings will throw this off
# - The first times the bot sees a map it'll look like it's going nuts as it scans the height charts of
#   the map and saves the map to file
#   This gives it x,y coords for each minimap location so that it can build buildings in the right spots.
# - The PPO algorithm builder_ai (also alice_the_builder) is for build timings for buildings and units...
#   still a lot of work to be done here.

import os
import os.path
# import time

import numpy as np

from alice.lib import map_reader, building_planner, static, util, builder_ai


from pysc2.agents import base_agent
from pysc2.lib import actions  # features, units

# when printing numpy arrays print out even fairly large arrays
np.set_printoptions(threshold=10000)


class aBot2Agent(base_agent.BaseAgent):
    def __init__(self):
        super().__init__()
        self.builder = None
        self.builder_running = False
        self.reward = 0

        # screen dimensions in pixels (for the simple64 map it's 84x84)
        self.screen_dimensions = None

        # minimap dimensions in minimap pixels for the simple64 map it's 64x64)
        self.minimap_dimensions = None

        self.building_plan = {}
        self.buildings = []

        self.alert_queue = []

        self.callback_method = None
        self.callback_parameters = {}

        self.idle_scv_locations = []
        self.priority_queue = []
        self.schedule = []
        self.map_name = None

        # vars to calculate reward after an action
        self.old_empire_value = None
        self.old_o = None
        self.old_time = None

        # how many screen pixels are in a building spot tile calibrate early and after a few buildings are produced.
        self.tile_size = [3.8, 3.8]

        # how many screen pixels are in a minimap pixel
        self.minimap_pixel_size = [4.9, 4.9]

        # how large is each building in pixels and in tiles
        # {"unit_id":unit_id,"tiles"[x,y],"screen pixels":[x,y],"minimap pixels":[x,y]}
        self.building_dimensions = [
            {"unit id": static.unit_ids["command center"], "tiles":[5, 5]},
            {"unit id": static.unit_ids["barracks"], "tiles":[3, 3]},
            {"unit id": static.unit_ids["supply depot"], "tiles":[2, 2]},
            {"unit id": static.unit_ids["supply depot lowered"], "tiles":[2, 2]},
            {"unit id": static.unit_ids["mineral field"], "tiles":[2, 1]},
            {"unit id": static.unit_ids["mineral field 750"], "tiles":[2, 1]}
        ]

        # how many pixels are on the map
        # absolute_map_dimensions = None

        # the center of the home command center on the minimap [int,int]
        self.command_home_base = None

        # should be the same as command_home_base, unless it's too close to the edge, which blocks the screen position.
        self.minimap_home_base = None

        # will probably be deprecated, using the screen_height_chart and minimap_offset_chart
        # absolute_home_base = None

        # The edges of the minimap can't be moved to, because of the size of the screen.
        # Generally we care about the center of the screen, this gives us the top left pixel that can be clicked on.
        # [[x,y],[x,y]] top left and bottom right locations for the minimap selectable area
        self.minimap_select_area = [[-1, -1], [-1, -1]]

        # TF,x,y offsets for each minimap location to get absolute coords from screen coords
        # The first element is whether the location has been charted
        self.map_offset_chart = None

        # TF,x,y,height chart for each pixel to calibrate the minimap offset chart. (in screen pixels)
        # 0,0 is the center of the command center
        self.screen_height_chart = None

        # x,y of the position of the top-most pixel of screen_height_chart (in screen pixels)
        # relative to the command center's center pixel
        self.screen_height_chart_offset = None

        self.charting_order = None

    # called once, after init
    def setup(self, obs_spec, action_spec):
        super().setup(obs_spec, action_spec)
        self.builder = None
        self.builder_running = False
        self.reward = 0

        # features: step, minerals, score
        # actions: build_scv, build_supply_depot, no_op

        # self.builder_actor = Actor(self.sess, n_features=3, lr=0.005, action_bound=[0, 3])
        # self.builder_critic = Critic(self.sess, n_features=3, lr=0.01)

        # self.sess.run(tf.global_variables_initializer())

        # screen dimensions in pixels (for the simple64 map it's 84x84)
        self.screen_dimensions = [obs_spec["feature_screen"][1], obs_spec["feature_screen"][2]]

        # minimap dimensions in minimap pixels for the simple64 map it's 64x64)
        self.minimap_dimensions = [obs_spec["feature_minimap"][1], obs_spec["feature_minimap"][2]]

        # print(f"obs_spec: {obs_spec}")
        # obs_spec: {
        # 'action_result': (0,),
        # 'alerts': (0,),
        # 'available_actions': (0,),
        # 'build_queue': (0, 7),
        # 'cargo': (0, 7),
        # 'cargo_slots_available': (1,),
        # 'control_groups': (10, 2),
        # 'game_loop': (1,),
        # 'last_actions': (0,),
        # 'multi_select': (0, 7),
        # 'player': (11,),
        # 'score_cumulative': (13,),
        # 'score_by_category': (11, 5),
        # 'score_by_vital': (3, 3),
        # 'single_select': (0, 7),
        # 'feature_screen': (17, 84, 84),
        # 'feature_minimap': (7, 64, 64),
        # 'feature_units': (0, 28),
        # 'camera_position': (2,)}

    # called after each game ends
    def reset(self):
        super().reset()
        print(f"episode: {self.episodes}")
        print(f"reward: {self.reward}")

        # if there is a builder, we need to close out its last action
        if self.builder is not None and self.builder_running is True:
            # print("game ended, need to clear the buffer")
            r = self.reward * 1000
            self.finish_alice_the_builder(r)

        # location and building type for pending buildings
        # locations are relative to starting base 
        # status includes attempt number, result of last information etc
        # {"building type":bt, "builder control group": bcg, "location": [x,y], "status": state}
        self.building_plan = {}
        self.buildings = []

        # whenever obs sends us an alert about the game.
        # We can't handle it immediately if we have a callback to do, but we should check it out after that
        self.alert_queue = []

        # the reason not to have a callback list, is because the logic tree branches too often.
        # If a check fails, you must try again, not move on to the next step.
        self.callback_method = None
        self.callback_parameters = {}

        # a list of last known locations of SCVs waiting to build things or having built things
        self.idle_scv_locations = []

        # a list of high priority actions that must be done before the schedule is checked
        # 1 = highest priority, 9 = lowest priority
        self.priority_queue = []

        # Functions called by the schedule should be considered higher level than actions on the priority queue.
        # time (game_loop), function, params
        # [[100,self.build_supply_depot,{}]]
        self.schedule = []

    # the primary function for pysc2 agents. called each time the game gives access to the bot
    def step(self, obs):
        super().step(obs)
        self.reward = obs.reward
        if self.reward != 0:
            print(f"reward: {self.reward}")

        # time.sleep(.5)

        alerts = obs.observation["alerts"]
        if alerts is not None and len(alerts) > 0:
            print("Alerts: " + str(alerts))

        # last_act = obs.observation["last_actions"]
        # if last_act is not None and len(last_act) > 0:
        #   print("Last Action: " + str(last_act))

        # what is our absolute game turn index
        # used in qtables and scheduling
        game_loop = obs.observation["game_loop"][0]
        # print(f"step: {str(game_loop)}")

        if game_loop == 0:
            print("initializing game loop")
            self.init_and_calibrate(obs)

        # if there's a multi step action in progress, perform the required callback.
        # This prevents thrashing on the priority queue, while giving a chance to check if a previous action worked
        cb = self.try_callback(obs)
        if cb is not False and cb is not None:
            return cb

            # add any required calls to the priority_queue
        self.check_schedule(obs)

        # if there's a priority action to do, do the most important priority action
        while len(self.priority_queue) > 0:
            action = self.perform_priority_action(obs)

            # print("action: " + str(action))
            # if this action results in an api call, do it.
            if action is not False and action is not None:
                return action

        # should be the only no_op
        return actions.FUNCTIONS.no_op()

    # if there is a callback action assigned, run through it.  make sure to unset
    #   any unwanted callback to prevent getting stuck here
    def try_callback(self, obs):
        if self.callback_method is None:
            self.callback_parameters = None
            return
        else:
            return self.callback_method(obs, self.callback_parameters)

    # if there are no callback actions, perform any priority actions in order of priority and then list order
    def perform_priority_action(self, obs):
        if len(self.priority_queue) > 0:
            # start with zero in case you really, really need something now.
            # for each priority level
            for i in range(10):
                # find the first item with that priority level
                for x in range(len(self.priority_queue)):
                    # if an element has the correct priority
                    if self.priority_queue[x][0] == i:
                        action = self.priority_queue.pop(x)
                        func = action[1]
                        params = action[2]
                        
                        return func(obs, params)   

    # a list of scheduled actions. Some actions will re-schedule themselves,

    # add a function call to the schedule, if the game loop >= time, the function will be run
    def schedule_action(self, obs, time, function, params):
        # print("action scheduled: " + str(function))
        self.schedule.append([time, function, params])
        return

    # run any functions which are ready and on the schedule based on game loop time
    def check_schedule(self, obs):
        game_loop = obs.observation["game_loop"][0]

        for item in self.schedule:
            item_time = item[0]
            item_func = item[1]
            item_params = item[2]

            if game_loop >= item_time:
                # print("running scheduled action: " + str(item_func))
                # it's time to run the thing, mark it off the schedule
                self.schedule.remove(item)
                # run it
                ret = item_func(obs, item_params)
                # TODO: if ret came back with a returned action, toss it on the schedule.
                if ret is not None:
                    print("PROBLEM: scheduled functions don't run actions! Toss that badboy on the priority queue!")

    # safety method to make sure an action is available before attempting to run it. otherwise the game crashes
    def try_perform_action(self, obs, action):
        if action is None or action is False:
            print("You're trying to perform a False action")
            return None
        else:
            # print("Doing action: " + str(action))
            action_id = action["id"]
            action_params = action["params"]

            if action_id not in obs.observation["available_actions"]:
                # print("Warning, cannot perform action: " + str(action_id) + " We should reschedule")
                return None

            action = actions.FunctionCall(action_id, action_params)
            return action

    # high level first steps for the bot to take once the game starts
    def init_and_calibrate(self, obs):
        self.priority_queue = [
                [2, self.set_up_builder_ai, {}],
                [2, map_reader.calibrate_map_data, {"bot": self}],  # select command center
                [2, building_planner.load_building_plan, {"bot": self}],
                # Note: build order might not need this, but if it does, do it
                # [3, self.train_scv, {}], # control grouping command center should be linked to this.
                [3, map_reader.load_map_data, {"bot": self}],
                # Important step. click on the minimap where the main base is... this aligns config stuff to the screen
                [3, map_reader.center_screen_on_main, {"bot": self}],
                [4, self.control_group_scvs, {}],  # good for keeping count of current scvs. useful in build planning
                # choose supply depot location, schedule scv move to location, schedule build supply depot
                # [5, self.make_supply_depot, {}],
                # bring in Alice, THE BUILDER! Actor/Critic siloed for building buildings and units.
                [6, self.start_alice_the_builder, {}],
                [8, self.schedule_print_data, {}]  # regular printouts of what's going on.
                # [7, self.test_move_screen, {}]
            ]

        return

    # PPO based RL agent for high level game management
    def set_up_builder_ai(self, obs, args):
        if self.builder is None:
            self.map_name = util.get_map_folder(obs, args={"bot": self})
            self.builder = builder_ai.Builder(args={"bot": self})
            self.builder.reset(self.get_builder_state(obs, {}))
        return

    # This is a big old complicated set of machinery:
    # The intended purpose is to check the buildings that we've ordered against what we can see on the screen.
    # If "action free" is set to True, then this is done passively with just the buildings available to be seen on
    # the screen
    # Otherwise, we go through the list of buildings that we've ordered and check up on them all, moving the screen
    # around the minimap until we've hit each building
    # Also, as we go through buildings, control group them to the appropriate control groups for quick counting and
    # checking far off buildings without moving the screen
    # Think about: using control groups to get a quick count of known buildings, will only have to check up on
    # buildings if the don't exist in the control groups
    # 0 = Command and Supply, 5 = Production and Tech
    def building_maintenance(self, obs, args):
        # print("building maintenance")
        # is this method just scanning what's available to see on screen, or doing a comprehensive check up?
        action_free = False
        if "action free" in args and args["action free"]:
            action_free = True

        # 1 copy the list of buildings to working_list
        if "working list" in args:
            # are we already in progress going through the list?
            working_list = args["working list"]
        else:
            # this must be the first step, let's get a fresh list
            working_list = self.buildings.copy()

        current_time = obs.observation["game_loop"][0]

        scr_loc = map_reader.get_relative_screen_location(obs, {"bot": self})

        # a representative sample of buildings on the screen to be added to control groups
        # control_group_me = {}

        scr_obs = obs.observation['feature_screen'][static.screen_features["unit type"]]
        scr_dmg = obs.observation['feature_screen'][static.screen_features["unit hit points ratio"]]
        # print("scr_obs type: " + str(type(scr_obs)))
        
        # 2 check the status of each building that is on the current screen, update the status to buildings
        for b in working_list:
            # is on screen?
            b_loc = b["location"]

            # print("bldg: " + str(b))
            # print("scr_loc: " + str(scr_loc))

            if b["status"] == "destroyed":
                working_list.remove(b)  # the building is already marked as destroyed, we don't need to check on it
                # print("working list: " + str(working_list))
                # print("bldg: " + str(b))
                continue

            # is b_loc between scr_loc[0] and scr_loc[1]?
            elif scr_loc[0][0] < b_loc[0] < scr_loc[1][0]:
                if scr_loc[0][1] < b_loc[1] < scr_loc[1][1]:
                    # the building is on screen, so we don't need to move the screen to check on it.
                    working_list.remove(b)
                    # building is on screen, check up on it
                    # print(str(b))

                    # current_status = b["status"]
                    b_dims = [2, 2]
                    for dim in self.building_dimensions:
                        if dim["unit id"] == b["type"]:
                            b_dims = dim["tiles"]

                    # print("dims: " + str(b_dims))
                    
                    # is the building actually there?
                    # unit_y, unit_x = (scr_obs == static.unit_ids[b["type"]]).nonzero()

                    # where should the building be? 
                    # should be the center pixel
                    b_scr = [b_loc[0] - scr_loc[0][0], b_loc[1] - scr_loc[0][1]]

                    # where the top-left of the building should be on the screen
                    b_tl_scr = [b_scr[0] - (b_dims[0] * self.tile_size[0] / 2),
                                b_scr[1] - (b_dims[1] * self.tile_size[1] / 2)]
                    # stay within the boundaries of the screen
                    b_tl_scr = [max(0, b_tl_scr[0]), max(0, b_tl_scr[1])]

                    # print("b_tl_scr: " + str(b_tl_scr))

                    # where the bottom right of the building should be on the screen
                    b_br_scr = [b_scr[0] + (b_dims[0] * self.tile_size[0] / 2),
                                b_scr[1] + (b_dims[1] * self.tile_size[1] / 2)]
                    # stay within the boundaries of the screen
                    b_br_scr = [min(self.screen_dimensions[0] - 1, b_br_scr[0]),
                                min(self.screen_dimensions[1] - 1, b_br_scr[1])]
                    
                    # next, get a numpy subset of sel for the area b_tl_scr through b_br_scr
                    # b_spot = scr_obs[util.round(b_tl_scr[0]):util.round(b_width+1),
                    #                  util.round(b_tl_scr[1]):util.round(b_height+1)]
                    # really zooming in here in order not to read neighboring buildings as the one we're looking for
                    b_spot = scr_obs[util.round(b_tl_scr[1] + 2):util.round(b_br_scr[1] - 2),
                                     util.round(b_tl_scr[0] + 2):util.round(b_br_scr[0] - 2)]
                    # print("b_spot: " + str(b_spot))

                    ys, xs = np.nonzero(b_spot == b["type"])
                    if xs is None or len(xs) == 0 and b["type"] == static.unit_ids["supply depot"]:
                        # try alternate
                        ys, xs = np.nonzero(b_spot == static.unit_ids["supply depot lowered"])

                    # print(f"Yo Yo Yo xs: {xs}, ys:{ys}")i
                    # not sure how many there should be, or how many there would be with overlap
                    if xs is not None and len(xs) > 15:
                        # print("number of pixels on screen: " + str(len(xs)))
                        # the building does seem to be there
                        pass
                    else:
                        ys, xs = b_spot.nonzero()
                        if xs is not None and len(xs) > 15:
                            # there's something in the way of the building, or something weird happening
                            # print(f"something is up with the building: {b}")
                            # time.sleep(2)
                            pass
                        else:
                            # print("that building is not there: " + str(xs))
                            # it is possible that the building has not yet been placed,
                            # if it's under construction or planned
                            # set the building with that location in the buildings list to destroyed. 
                            
                            for upd_bldg in self.buildings:
                                if upd_bldg["location"] == b["location"]:
                                    if upd_bldg["status"] == "under construction" or upd_bldg["status"] == "planned":
                                        attempt = 0
                                        if "attempt" in upd_bldg:
                                            attempt = upd_bldg["attempt"] + 1
                                            upd_bldg["attempt"] = attempt
                                        else:
                                            upd_bldg["attempt"] = 1
                                        if attempt >= 4:
                                            # something went wrong in building, this building isn't there.
                                            upd_bldg["attempt"] = 0
                                            upd_bldg["status"] = "failed"
                                    else:
                                        upd_bldg["status"] = "destroyed"
                                upd_bldg["timestamp"] = current_time

                    dmg_spot = scr_dmg[util.round(b_tl_scr[1] + 2):util.round(b_br_scr[1] - 2),
                                       util.round(b_tl_scr[0] + 2):util.round(b_br_scr[0] - 2)]

                    if dmg_spot is None or len(dmg_spot) == 0 or len(dmg_spot[0]) == 0:
                        print(f"trying to detect the damage of building: {b} but something is wrong")
                        # TODO: figure out how we get here. I think it has to do with the screen being moved
                        #  outside of a minimap click
                        pass

                    else:
                        # check the screen for hp to see how we're doing on that
                        # take a diagonal through the spot and get any non-zero numbers, what's the avg
                        # potential bug around the shape of dmg_spot
                        print("Shape of the building array" + (str(dmg_spot.shape)))
                        ctr = 0
                        total = 0
                        for i in range(len(dmg_spot)):
                            # taking a diagonal of x/y's... but if the building is hanging off the edge of the screen
                            # on the right or left, color within the lines
                            x_var = i
                            x_var = min(x_var, len(dmg_spot[i])-1)
                            x_var = max(x_var, 0)
                            if dmg_spot[i][x_var] > 0:
                                ctr += 1
                                total += dmg_spot[i][x_var]

                        health = 0
                        if ctr > 0:
                            health = total / ctr

                        # print("health: " + str(health))
                        # update the building status
                        if health == 255:
                            for upd_bldg in self.buildings:
                                if upd_bldg["location"] == b["location"]:
                                    upd_bldg["timestamp"] = current_time

                                    if upd_bldg["status"] == "under construction" or upd_bldg["status"] == "planned" \
                                            or upd_bldg["status"] == "damaged":
                                        upd_bldg["status"] = "complete"
                                        upd_bldg["attempt"] = 0

                        elif health > 0:
                            for upd_bldg in self.buildings:
                                if upd_bldg["location"] == b["location"]:
                                    upd_bldg["timestamp"] = current_time

                                    if upd_bldg["status"] == "planned":
                                        upd_bldg["status"] = "under construction"
                                        upd_bldg["attempt"] = 0
                                    if upd_bldg["status"] == "complete":
                                        upd_bldg["status"] = "damaged"

                    # 4 for each visible building that is not a command center, remove it from working_list
            pass
        
        # if we're just observing the current screen without taking any actions, leave now.
        if action_free:
            return

        # TODO: 3 control group each building type into the correct control group
        # Keep track of each building type available on screen
        # Group select each building by building type for each control group
        # Check through the list to make sure no alien buildings are in the selection.
        # Control group the selected items into the correct control group

        # 5 For the first command center on the working list, go to the building
        # Count the scvs on the screen and update the command center status to include them,
        # remove command center from working list
        # Repeat steps 1-4

        # for each remaining building on the working list go to the building and repeat steps 1-4
        if len(working_list) == 0:
            return
        for b in working_list:
            
            if "location" in b:
                loc = b["location"]

                action = map_reader.move_to_point(obs, {"bot": self, "point": loc})
                if action is None:
                    continue
                else:
                    self.callback_parameters = {"working list": working_list}
                    self.callback_method = self.building_maintenance
                    return action

            else: 
                working_list.remove(b)

    # train an scv
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

            to_do = {"id": static.action_ids["train scv"],
                     "params": [[static.params["queued"]]]}

            action = self.try_perform_action(obs, to_do)

            # LINKED ACTION: if we take this out, make sure to ctrl group the first cc in init_and_calibrate
            # while we're here, if the command center isn't in the control group, that needs to happen
            command_control_group = obs.observation["control_groups"][0]
            if len(command_control_group) == 2 and command_control_group[0] == static.unit_ids["command center"]:
                self.callback_method = None
                self.callback_parameters = {}
                return action
            else:
                self.callback_parameters = {"type": "append", "group": 0}
                self.callback_method = self.control_group_selected
                return action

        else:  # command center not selected, let's select it
            if ctr > 5:
                # print("We're having unknown difficulty in building an scv, letting it off the queue")
                # TODO: schedule some things to check to repair the problem
                
                self.callback_parameters = {}
                self.callback_method = None
                return

            # TODO:
            #   check the multi select array to see if there are command centers already selected.
            #       If so, let's single select one of them to build the scv.
            #   check if there are command centers in the cc control group (0)

            to_do = self.select_building(obs, {"type": static.unit_ids['command center']})
            if to_do is None:
                # There wasn't a command center on screen, we must move to a command center.
                self.callback_parameters = {"ctr": ctr}
                self.callback_method = self.train_scv
                return map_reader.center_screen_on_main(obs, {"bot": self})

            else:
                self.callback_parameters = {"ctr": ctr}
                self.callback_method = self.train_scv
                return to_do

    # control group on screen scvs. Useful for counting
    def control_group_scvs(self, obs, args):
        self.callback_method = None
        self.callback_parameters = {}
        return

    # decide where to build the building
    # add the planned building to buildings
    # schedule to have the scv move to build the building
    # schedule to have the scv build the building
    def make_building(self, obs, args):
        current_time = obs.observation["game_loop"][0]
        building_string = None
        if "building" in args:
            building_string = args["building"]

        building_type_string = None
        if building_string == "supply depot":
            building_type_string = "supply depot"
        elif building_string == "barracks":
            building_type_string = "production"

        # print("hey, let's prepare to build our first supply depot")

        # existing or planned buildings of the type we're trying to build
        bldg_types = list(filter(lambda bldg: bldg["type"] == static.unit_ids[building_string], self.buildings))

        # places where we can build the building we're trying to build from a config file
        bldg_type_locations = self.building_plan[building_type_string]

        # for bt in bldg_types:
        #    print("we have: "+ str(bt))
        #    pass
        if bldg_types is None or len(bldg_types) == 0:
            print("no buildings found of type: " + building_string)
            print(self.buildings)

        # print("supply depots: " + str(supply_depots))
        # print("supply depot locations: " + str(supply_depot_locations))
        
        # find the first supply depot location without a supply depot already in buildings
        location = []
        building = None
        found_location = False
        for l in bldg_type_locations:
            # check the list of building locations against existing buildings to see if there's an availability
            current_building_list = list(filter(lambda bldg: bldg["location"] == l, bldg_types))
            # print("making a bldg: " + str(current_building))
            if len(current_building_list) == 0:
                # add the building to buildings with status: planned
                location = l
                building = {"type": static.unit_ids[building_string],
                            "location": l,
                            "status": "planned",
                            "timestamp": current_time}
                self.buildings.append(building)
                # print(f"trying to build a new {building_string} at: {l}")
                found_location = True
                break
            else:
                current_building = current_building_list[0]

                if current_building["status"] == "destroyed" or current_building["status"] == "failed":
                    # print(f"trying to replace a {current_building[0]['status']} {building_string} at: {l}")

                    # set the status to planned
                    location = l
                    current_building["status"] = "planned"
                    current_building["timestamp"] = current_time
                    building = current_building
                    found_location = True
                    break

        # if there were no available places to build the building, abort
        if found_location is False:
            # print("error finding a place to build a supply depot")
            self.callback_method = None
            self.callback_parameters = {}
            return

        # schedule an scv move to the location of the building location
        current_time = obs.observation["game_loop"][0]

        finance = util.get_finances(obs)
        unit_cost = static.unit_cost[building_string]

        # only need to do this when poor.
        if finance["minerals"] < unit_cost[0] or finance["gas"] < unit_cost[1]:
            # TODO: need to have a method to set this time correctly based on distance to location from the selected SCV
            print("args check3: " + str(building))
            self.schedule_action(obs, current_time + 140,
                                 self.construct_building,
                                 {"building": building, "schedule": True})
            # schedule the building of the building at location with the scv on location
            self.callback_method = None
            self.callback_parameters = {}
            return self.move_scv_to_location(obs, {"location": location, "building": building, "schedule": False})
        else:
            print("args check4: " + str(building))
            self.callback_method = None
            self.callback_parameters = {}
            return self.construct_building(obs, {"building": building, "schedule": False})

    def move_scv_to_location(self, obs, args):
        # if we're coming from a schedule, we have to pop this item onto the queue.
        if "schedule" in args and args["schedule"]:
            args["schedule"] = False
            self.priority_queue.append([4, self.move_scv_to_location, args])
            return
        
        location = None
        if "location" in args:
            location = args["location"]

        # print("Moving SCV to location: " + str(location))
        # check if an scv is single selected
        # if not, return get_scv with self as a callback method defined in the callback args
        # add the scv to the builder control group.
        single_select = obs.observation["single_select"]
        scv = None

        if len(single_select) > 0:
            # print(single_select)
            if single_select[0][0] == static.unit_ids["scv"]:
                scv = single_select[0]
                # print("we have an SCV selected: " + str(scv))

        if scv is None:
            # no SCV selected, let's get one, then come back here
            self.priority_queue.append([1, self.move_scv_to_location, args])
            return self.get_scv(obs, args)

        # we should have an SCV by now, let's move him to the right place.
        # move the screen to the right place.
        if map_reader.is_point_on_screen(obs, {"bot": self, "point": location}):
            self.priority_queue.append([1, self.add_to_group, {"group": "builders"}])
            # move the scv to the right place on the screen
            return map_reader.issue_move_action_on_screen(obs, {"bot": self, "point": location})
        else:
            # move the screen to where it needs to be, then come back here
            self.priority_queue.append([1, self.move_scv_to_location, args])
            return map_reader.move_to_point(obs, {"bot": self, "point": location})

    # assign a builder to a control group for a building
    def assign_builder(self, obs, args):
        building = None
        if "building" in args:
            building = args["building"]

        return

    def add_to_group(self, obs, args):
        # print("Ok, let's add this to the group!")
        return

    def construct_building(self, obs, args):
        # if we're coming from a schedule, we have to pop this item onto the queue.
        if "schedule" in args and args["schedule"]:
            args["schedule"] = False
            print("checking construct_building args: " + str(args))
            self.priority_queue.append([4, self.construct_building, args])
            return

        building = None
        if "building" in args:
            building = args["building"]
        
        if building is None:
            # TODO: why are we showing up here
            print("what are you trying to build?")
            print(str(args))
            return
        # TODO: check that we have the shekels

        # move the screen to the building location
        # we should have an SCV by now, let's move him to the right place.
        # move the screen to the right place.

        print("building checkup: " + str(building))
        print("what type is location: " + str(type(building['location'])))
        if map_reader.is_point_on_screen(obs, {"bot": self, "point": building["location"]}):
            # self.priority_queue.append([1, self.add_to_group, {"group":"builders"}])
            
            # is an scv selected?
            single_select = obs.observation["single_select"]
            if len(single_select) > 0 and single_select[0][0] == static.unit_ids["scv"]:
                # scv = single_select[0]
                # build the building
                
                build_action = None
                if building["type"] == static.unit_ids["supply depot"]:
                    build_action = static.action_ids["build supply depot"]
                elif building["type"] == static.unit_ids["barracks"]:
                    build_action = static.action_ids["build barracks"]
                elif building["type"] == static.unit_ids["command center"]:
                    build_action = static.action_ids["build command center"]

                screen_location = map_reader.get_screen_location(obs, {"bot": self, "point": building["location"]})

                if screen_location is None:
                    # the location isn't on screen, this is a problem.
                    print("error, the desired location isn't on screen something went wrong")

                action = { 
                            "id":       build_action,
                            "params":   [[static.params["queued"]], screen_location]
                         }
                ret = self.try_perform_action(obs, action)

                # see if we were able to perform this action
                if ret is None:
                    # we were unable to perform the action, so let's re-schedule for a lower queue priority
                    current_time = obs.observation["game_loop"][0]

                    # schedule the building of the building at location with the scv on location
                    # TODO: need to have a method to set this time correctly based on walk distance to location
                    self.schedule_action(obs, current_time + 16,
                                         self.construct_building,
                                         {"building": building, "schedule": True})
                else:
                    building["status"] = "under construction"
                    return ret

            else:
                # TODO: check if the building has a builder assigned to it with a control group
                print("args check: " + str(args))
                self.priority_queue.append([1, self.construct_building, args])
                # check the building location for an scv
                return self.get_scv(obs, {"location": building["location"]})
            
            # return map_reader.issue_move_action_on_screen(obs, {"bot":self, "point":location})
            pass
        else:
            # move the screen to where it needs to be, then come back here
            location = building["location"]
            print("args check2: " + str(args))
            self.priority_queue.append([1, self.construct_building, args])
            return map_reader.move_to_point(obs, {"bot": self, "point": location})

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

            to_do = {"id": static.action_ids["train marine"],
                     "params": [[static.params["queued"]]]}
            action = self.try_perform_action(obs, to_do)

            self.callback_method = None
            self.callback_parameters = {}
            return action

        else:  # barracks are not selected, let's select it
            if ctr > 5:
                # this is taking too long, abandon
                self.callback_parameters = {}
                self.callback_method = None
                return

            # TODO:
            #  check the multi select array to see if there are barracks already selected.
            #    If so, let's select them and start training marines
            #  check if there are command centers in the production control group (5)

            to_do = self.select_building(obs, {"type": static.unit_ids['barracks']})
            if to_do is None:
                # There wasn't a barracks on screen, we must move to a barracks
                self.callback_parameters = {"ctr": ctr}
                self.callback_method = self.train_marine
                return map_reader.center_screen_on_main(obs, {"bot": self})

            else:
                self.callback_parameters = {"ctr": ctr}
                self.callback_method = self.train_marine
                return to_do

    def trigger_supply_depots(self, obs, args):
        # have we been here before? are we thrashing?
        # print("trying to lower or raise some supply depots")
        ctr = 0
        if "ctr" in args:
            ctr = args["ctr"]
        ctr = ctr + 1
           
        raise_em = False
        lower_em = False

        s_selected = obs.observation["single_select"]
        if s_selected is not None and len(s_selected) > 0:
            if s_selected[0][0] == static.unit_ids["supply depot"]:
                lower_em = True
            elif s_selected[0][0] == static.unit_ids["supply depot lowered"]:
                raise_em = True

        m_selected = obs.observation["multi_select"]
        if m_selected is not None and len(m_selected) > 0:
            if m_selected[0][0] == static.unit_ids["supply depot"]:
                lower_em = True
            elif m_selected[0][0] == static.unit_ids["supply depot lowered"]:
                raise_em = True

        # if supply depots are selected, trigger them
        # this is a lie -> if static.action_ids["lower supply depot"] in obs.observation["available_actions"]:
        if lower_em:
            # print("we can lower the supply depots, let's do it")
            # if the next thing to do isn't to press the build scv button... then try to select a command center again

            to_do = {
                        "id": static.action_ids["lower supply depot"],
                        "params": [[1]]
                    }
            action = self.try_perform_action(obs, to_do)
            # print(str(action))
            # print(str(selected))
            self.callback_method = None
            self.callback_parameters = {}
            return action

        # if supply depots are selected, trigger them
        # this is a lie -> elif static.action_ids["raise supply depot"] in obs.observation["available_actions"]:
        elif raise_em:
            # print("we can raise the supply depots, let's do it")
            # if the next thing to do isn't to press the build scv button... then try to select a command center again

            to_do = {
                      "id": static.action_ids["raise supply depot"],
                      "params": [[1]]
                     }

            action = self.try_perform_action(obs, to_do)
            # print(str(action))
            # print(str(selected))
            self.callback_method = None
            self.callback_parameters = {}
            return action

        else:  # supply depots not selected, select them
            if ctr > 5:
                # this is taking too long, abandon
                self.callback_parameters = {}
                self.callback_method = None
                return

            action = self.select_building(obs, {"type": static.unit_ids['supply depot']})
            if action is None:
                # print("there don't seem to be any raised supply depots to select")
                # There wasn't a supply depot on the screen... maybe they're already down
                action = self.select_building(obs, {"type": static.unit_ids['supply depot lowered']})

            if action is None:
                # print("there don't seem to be any lowered supply depots to select")

                # TODO: ok, no supply depots on screen, maybe move to the wall location, for now just go to the main
                self.callback_parameters = {"ctr": ctr}
                self.callback_method = self.trigger_supply_depots
                return map_reader.center_screen_on_main(obs, {"bot": self})

            if action is not None:
                # print("ok, selecting some supply depots")
                args["ctr"] = ctr
                self.priority_queue.append([0, self.trigger_supply_depots, args])
                self.callback_parameters = {}
                self.callback_method = None
                return action

        self.callback_parameters = {}
        self.callback_method = None
        return

    # select all
    # control group
    # attack move
    def perform_zapp_brannigan_maneuver(self, obs, args):
        # print("zapp")
        my_x, my_y = self.command_home_base

        player_positions = obs.observation["feature_minimap"][static.screen_features["player relative"]]
        enemy_ys, enemy_xs = np.nonzero(player_positions == static.params["player enemy"])

        center_x = 31
        center_y = 31

        enemy_x = center_x + center_x - my_x
        enemy_y = center_y + center_y - my_y
        
        if len(enemy_ys) > 0:
            enemy_x = enemy_xs[0]
            enemy_y = enemy_ys[0]

        action = { 
            "id":   static.action_ids["select army"],
            "params":   [[static.params["now"]]]
        }

        # TODO:
        #  if there isn't an army to select, this will come back none...
        #  let's not send whatever SCV is selected off to die just yet.

        zapp = self.try_perform_action(obs, action)

        if zapp is None:
            return
        else:
            self.priority_queue.append([0, self.control_group_selected, {"type": "append", "group": 1}])
            self.priority_queue.append([0, self.a_move, {"point": [enemy_x, enemy_y]}])
            return zapp

    def a_move(self, obs, args):
        # print("a_move")
        point = [0, 0]
        if "point" in args:
            point = args["point"]

        action = { 
            "id":   static.action_ids["attack minimap"],
            "params":   [[static.params["now"]], point]
            }

        self.callback_method = None
        self.callback_parameters = {}

        return self.try_perform_action(obs, action)

    def control_group_selected(self, obs, args):
        # print("ctrl")
        act_type = "append"
        if "type" in args:
            act_type = args["type"]

        group = 9
        if "group" in args:
            group = args["group"]

        command_id = [x for x, y in enumerate(actions.CONTROL_GROUP_ACT_OPTIONS) if y[0] == act_type]

        action = { 
            "id":       static.action_ids["control group"],
            "params":   [command_id, [group]]
            }

        self.callback_method = None
        self.callback_parameters = {}

        return self.try_perform_action(obs, action)

    def make_scv_work(self, obs, args):
        print("ya'll get back to work now, ya hear?")
        single_select = obs.observation["single_select"]
        scv = None
        if len(single_select) > 0:
            if single_select[0][0] == static.unit_ids["scv"]:
                scv = single_select[0]

        if scv is None:
            multi_select = obs.observation["multi_select"]
            if multi_select is not None and len(multi_select) > 0:
                print("haven't tested this yet, getting an scv from a multiselect")

                for i in range(len(multi_select)):
                    ms = multi_select[i]

                    if ms[0] == static.unit_ids["scv"]: 
                        command_id = [x for x, y in enumerate(actions.SELECT_UNIT_ACT_OPTIONS) if y[0] == "select"]

                        action = {
                            "id":   static.action_ids["select unit"],
                            "params":   [command_id, [i]]
                        }

                        do_action = self.try_perform_action(obs, action)

                        self.callback_method = self.make_scv_work
                        self.callback_parameters = args

                        return do_action

        if scv is not None:
            # ok, we have an scv, let's get it back to work.
            print("we have an scv, command it to work!")
            available_actions = obs.observation["available_actions"]
            print("available actions: " + str(available_actions))
            if static.action_ids["harvest return"] in available_actions:
                print("harvest return is available, do it")
                action = { 
                    "id":       static.action_ids["harvest return"],
                    "params":   [[static.params["queued"]]]
                }

                do_action = self.try_perform_action(obs, action)

                self.callback_method = None
                self.callback_parameters = {}
                return do_action

            else:
                print("harvest return wasn't available, let's find a mineral patch to go to")

                self.callback_method = None
                self.callback_parameters = {}
                return

        idle_workers = obs.observation["player"].idle_worker_count
        if idle_workers > 0:
            
            select_id = [x for x, y in enumerate(actions.SELECT_WORKER_OPTIONS) if y[0] == 'select']
            action = { 
                "id":   static.action_ids["select idle worker"],
                "params":   [select_id]
                }

            # need to get idle worker!
            # print("hey, this is broken!")
            # action = actions.FUNCTIONS.select_idle_worker('select')
            
            # let's come back to make sure we've gotten an SCV
            self.callback_method = self.make_scv_work
            self.callback_parameters = args

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
        # scv = None
        if len(single_select) > 0:
            if single_select[0][0] == static.unit_ids["scv"]:
                # scv = single_select[0]
                # print("OK, so we have an SCV selected! " + str(scv))
                self.callback_method = None
                self.callback_parameters = {}
                return

        # print("getting SCV")
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
        # print(str(obs.observation["player"]))
        # print("idle workers:" + str(obs.observation["player"].idle_worker_count))
        idle_workers = obs.observation["player"].idle_worker_count
        if idle_workers > 0:
            
            select_id = [x for x, y in enumerate(actions.SELECT_WORKER_OPTIONS) if y[0] == 'select']
            action = { 
                "id":   static.action_ids["select idle worker"],
                "params":   [select_id]
                }

            # need to get idle worker!
            # print("hey, this is broken!")
            # action = actions.FUNCTIONS.select_idle_worker('select')
            
            # let's come back to make sure we've gotten an SCV
            self.callback_method = self.get_scv
            self.callback_parameters = args

            return self.try_perform_action(obs, action)     

        # if there are no idle SCVs check current screen
        else: 
            # TODO: Finding the center of an SCV shouldn't be too tough, will need to store the height/width
            #  of scvs somewhere and select one from that.
            unit_type = obs.observation['feature_screen'][static.screen_features["unit type"]]
            unit_y, unit_x = np.nonzero(unit_type == static.unit_ids["scv"])
              
            if len(unit_y) > 0:
                x = unit_x[0]
                y = unit_y[0]
                
                action = { 
                    "id":   static.action_ids["select point"],
                    "params":   [[static.params["now"]], [x, y]]
                }

                # let's come back to make sure we've gotten an SCV
                self.callback_method = self.get_scv
                self.callback_parameters = args

                return self.try_perform_action(obs, action) 

            else:
                # no SCVs on screen, we need to go find one!
                # if there are no SCVs on screen, go to home screen
                # TODO: there may also be SCVs working at random mineral patches, or in control groups.
                x, y = self.command_home_base

                action = { 
                    "id":   static.action_ids["move camera"],
                    "params":   [[x, y]]
                }

                self.callback_method = self.get_scv
                self.callback_parameters = args
                
                return self.try_perform_action(obs, action) 

    def get_builder_state(self, obs, args):
        current_time = obs.observation["game_loop"][0]
        minerals = obs.observation["player"][1]
        gas = obs.observation["player"][2]
        empire_value = obs.observation["score_cumulative"][0]

        o = np.zeros([8], dtype=int)
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

        self.queue_builder_action(obs, {"action": a})

        return

    # the idea here is to have alice the builder choose the buildings and learn the build orders.
    # will need self training, and the ability to save tensorflow training data per map.
    def continue_alice_the_builder(self, obs, args):
        # check if alice has been initialized: if not, do that here.
        # print("You're calling alice the builder? That's all you had to say!")
        current_time = obs.observation["game_loop"][0]

        # obs.observation['score_cumulative']
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
            # done training for this epoch, not sure how to quit the game, but if we don't schedule the next action,
            # it'll just run out without any more buildings
            self.builder_running = False
            return
        elif trainer == -1:
            print(f"last r of final epoch: {r}")
            self.builder_running = False
            quit()  # there has to be a pysc2 way of doing this same thing.
            return
            # done with the last epoch, close out.
        else:
            # we're still in the epoch, schedule the next decision point
            self.schedule_action(obs, current_time + 40, self.continue_alice_the_builder, {})
            a = self.builder.get_action()

            # set the old values for the next time through
            self.old_empire_value = empire_value
            self.old_time = current_time

            self.queue_builder_action(obs, {"action": a})
        return

    def finish_alice_the_builder(self, reward):
        self.builder_running = False
        # print("clearing buffer")
        o = self.old_o
        r = reward
        d = True
        _ = {}
        self.builder.wrap_up_action(o, r, d, _)
        return

    def queue_builder_action(self, obs, args):
        a = None
        if "action" in args:
            a = args["action"]
        
        if a == 1:  # train SCV
            # print("  Alice Says: train an scv")
            self.priority_queue.append([4, self.train_scv, {}])

        elif a == 2:  # build supply depot
            # print("  Alice Says: build a supply depot")
            self.priority_queue.append([4, self.make_building, {"building": "supply depot"}])

        elif a == 3:  # build barracks
            # print("  Alice Says: build a supply depot")
            self.priority_queue.append([4, self.make_building, {"building": "barracks"}])

        elif a == 4:  # get SCVs back to work
            self.priority_queue.append([4, self.make_scv_work, {}])
            pass

        elif a == 5:  # build marine
            self.priority_queue.append([4, self.train_marine, {}])
            pass

        elif a == 6:  # trigger supply depots
            self.priority_queue.append([4, self.trigger_supply_depots, {}])
            pass

        elif a == 7:  # attack move across the map
            self.priority_queue.append([4, self.perform_zapp_brannigan_maneuver, {}])
            pass

        elif a == 8:  # perform building maintenance
            self.priority_queue.append([4, self.building_maintenance, {"action free": False}])
            pass

        elif a == 9:  # not sure yet
            
            pass

        else:  # no op
            
            pass

        return

    # Returns None if that building isn't available on the screen
    def select_building(self, obs, args):
        bldg_type = None
        if "type" in args:
            bldg_type = args["type"]

        param = "select_all_type"
        if "param" in args:
            param = args["param"]

        # TODO: if there is no building on screen, check the building tables and go to a minimap location
        #   with the building
        off_screen = False
        if "off_screen" in args:
            off_screen = args["off_screen"]

        offset = 0

        for bldg in self.building_dimensions:
            if bldg["unit id"] == bldg_type:
                offset = bldg["tiles"][0] * self.tile_size[0] / 2

        # print("trying to select: " + str(type))
        # Find the pixels that relate to the unit
        unit_type = obs.observation['feature_screen'][static.screen_features["unit type"]]
        ys, xs = np.nonzero(unit_type == bldg_type)

        # find the left most pixel relating to the building
        if len(xs) > 0:
            min_x = min(xs)
            x, y = [min_x, 0]
            for i in range(len(xs)):
                if xs[i] == min_x:
                    y = ys[i]
                    x = x + offset
                    break

            if x >= self.screen_dimensions[0]:
                x = self.screen_dimensions[0] - 1

            # queue up an action to click on the center pixel
            # self.callback_method = None
            # self.callback_parameters = {}
            select_type = [x for x, y in enumerate(actions.SELECT_POINT_ACT_OPTIONS) if y[0] == 'select_all_type']

            action = {
                "id":       static.action_ids["select point"],
                "params":   [select_type, [x, y]]
            }
            return self.try_perform_action(obs, action)

        return None

    # recurring scheduled print to console of useful game state for debugging
    def schedule_print_data(self, obs, args):
        current_time = obs.observation["game_loop"][0]
        # recurring action to print out some data
        self.schedule_action(obs, current_time + 100, self.schedule_print_data, {})
        self.print_data(obs, args)
        return

    # print out useful data for debugging whatever you're working on
    def print_data(self, obs, args):
        single_select = obs.observation["single_select"]

        # print(str(obs.observation.keys()))
        # dict_keys(['single_select', 'multi_select', 'build_queue', 'cargo', 'cargo_slots_available', 'feature_screen',
        # 'feature_minimap', 'last_actions',
        # 'action_result', 'alerts', 'game_loop', 'score_cumulative', 'score_by_category', 'score_by_vital', 'player',
        # 'control_groups', 'feature_units',
        # 'camera_position', 'available_actions'])

        # print("Obs Camera Position: " + str(obs.observation["camera_position"]))
        # print("Tile Size: " + str(self.tile_size))

        if len(single_select) > 0:
            sel = obs.observation['feature_screen'][static.screen_features["selected"]]
            ys, xs = (sel == 1).nonzero()
            # current_time = obs.observation["game_loop"][0]

            # print(str(sel))
            # buildings = self.buildings
            # for building in buildings:
            #    pass
            #    print("Building: " + str(building))

            # print(f"score_by_category: {obs.observation['score_by_category']}")
            # if single_select[0][0] == unit_ids["command center"]:
            # print("Current Game Time: " + str(current_time))
            # print("Alerts: " + str(obs.observation["alerts"]))
            # print("Select info: " + str(single_select))
            # print("idle workers:" + str(obs.observation["player"].idle_worker_count))
            # print("player: " + str(obs.observation["player"]))
            # print(f"Score_cumulative: " + str(obs.observation['score_cumulative']))
            # print("Last Actions: " + str(obs.observation["last_actions"]))
            # print("Action Result: " + str(obs.observation["action_result"]))
            # print("Player: " + str(obs.observation["player"]))
            # print("Feature Units: " + str(obs.observation["feature_units"]))

            if len(xs) > 0:
                pass
                # x = util.round(xs.mean())
                # y = util.round(ys.mean())

                # ax, ay = map_reader.get_absolute_location(obs, {"point": [x, y], "bot": self})

                # print(f"Selected item at screen: x{x},y{y}")
                # print(f"Selected item at absolute point: x{ax},y{ay}")
            else:
                # print("nothing is selected")
                pass

        # if single_select[0][0] == unit_ids["command center"]:
        return


def main():
    # print(sys.path)
    os.system('python -m run_agent '
              '--map Simple64 '
              '--agent abot2.aBot2Agent '
              '--agent_race terran '
              '--max_agent_steps 0 '
              '--game_steps_per_episode 50000')


if __name__ == "__main__":
    main()
