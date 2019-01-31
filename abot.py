# Note: Thanks to Steven Brown, Timo for tutorials and help


import os
import sys
import time

import tensorflow as tf
import numpy as np

from pysc2.agents import base_agent
from pysc2.lib import actions, features, units



#parameters
params = {
    "player self":1,
    "player enemy":4,
    "now":0,
    "queued":1,
    "minerals":1,
    "vespene":2,
    "supply used":3,
    "supply available":4,
    }

#features
features = {
    "player relative":features.SCREEN_FEATURES.player_relative.index,
    "unit type":features.SCREEN_FEATURES.unit_type.index
    }


unit_ids = {
    "scv": 45,
    "command center": 18,
    "supply depot": units.Terran.SupplyDepot,
    "barracks": 21,
    "mineral field": units.Neutral.MineralField,
    "mineral field 750": units.Neutral.MineralField750
    }

action_ids = {
    "no op": actions.FUNCTIONS.no_op.id,
    "move": actions.FUNCTIONS.Move_screen.id,
    "select point": actions.FUNCTIONS.select_point.id, 
    "control group": actions.FUNCTIONS.select_control_group.id,
    "harvest": actions.FUNCTIONS.Harvest_Gather_screen.id,
    "harvest return": actions.FUNCTIONS.Harvest_Return_quick.id,
    "train scv": actions.FUNCTIONS.Train_SCV_quick.id,
    "build supply depot": actions.FUNCTIONS.Build_SupplyDepot_screen.id,
    "train marine": actions.FUNCTIONS.Train_Marine_quick.id,
    "build barracks": actions.FUNCTIONS.Build_Barracks_screen.id,
    "rally minimap": actions.FUNCTIONS.Rally_Units_minimap.id,
    "select army": actions.FUNCTIONS.select_army.id,
    "attack minimap": actions.FUNCTIONS.Attack_minimap.id
    }

# a list of lists, with the action id from pysc2/lib/actions as the first element, and the applicable parameters as the second
action_queue = []

# a list of things that need to happen after a certain game_loop point in time, these are screen location dependent.
delayed_action_queue = []

# Ideas:
# future queue: a timing based future queue for adding future buildings to the correct control groups
#   - e.g: command an scv to build a barracks, but before the barracks is completed have an action to 
#          add the barracks to the correct ctrl group.
# multiple ai subunits to manage various tasks: train either concurrently or independently
#  building (buildings and units)
#  building location (where to build a building which has been ordered by the builder subunit).
#  scouting
#  attacking and defending

# For tensorflow learning: 
#  game time (make sure it's not a variable time step).
#  current minerals
#  current gas
#  supply available / used
#  scvs
#  known enemy units / buildings
#  friendly units / buildings

# To Do:
#  Switch Control Loop to Action Schedule.
#  Add callback function to action_queue
#  Create prioritized_task_queue - a higher level task queue with priorities
#  Create task_schedule - a schedule for adding tasks to the prioritized task queue. 
#     Replace control loop with on completion, re-schedule items in task_schedule
#  Script out a marine building and attacking bot.
#  Have trained bot decide when to build supply depots based on the variables above.
#  Add scv training, barracks building and marine training to the bot.
#  Script out expand function
#  Add expand to the bot's available actions.
#  Add building locations to the trained responsibilities.

# Ctrl Groupings:
#   0 - Command Centers
#   1-5 Attack Group 1-5
#   6 - Unit Builders
#   7 - Upgrade Buildings
#   8 - SCV Counting Group 
#   9 - Builders Group
#          - It may be nice to use working groups to area select units, ctrl group them, then single select them from the group to get a specific unit.

# Problem:
#   Actions must be available before performed. Sometimes, if an action isn't available, an alternate needs
#       to be performed to correct the inavailability.
#   EG: Training an SCV with action 490 isn't available if there's no money for an scv 
#       In this case, wait until there is available money.

class aBotAgent(base_agent.BaseAgent):
    control_loop_index = 0
    grid_height = 0
    grid_width = 0

    player_start_location = [31,31]

    def step(self, obs):
        super(aBotAgent, self).step(obs)
        # Note: Starts on step 1 
        #print(f"step: {self.steps}")


        #print ("available actions: " + str(obs.observation["available_actions"]))

        #time.sleep(.01)

        # if there are queued actions, do the next action.
        next_action = self.perform_queued_action(obs)
        if next_action is not False:
            return next_action

        # get some info we might need to determine next steps.
        game_loop = obs.observation["game_loop"][0]
        single_select = obs.observation["single_select"]
        multi_select = obs.observation["multi_select"]
        build_queue = obs.observation["build_queue"]
        control_groups = obs.observation["control_groups"]
        supply_used = obs.observation['player'][params['supply used']]
        supply_available = obs.observation['player'][params['supply available']]

        minerals = obs.observation["player"][params["minerals"]]
        gas = obs.observation["player"][params["vespene"]]


        # TODO: use game_loop instead of steps, it's more accurate of a timer.
        #build the action queue here
        if game_loop == 0:
            self.control_loop_index = 0
            self.control_group_command_centers(obs)
            self.calculate_grid_dimensions(obs)
            self.build_scv(obs)      
            self.save_player_start_location(obs)

            print("grid: " + str(self.grid_width) + "," + str(self.grid_height))

            next_action = self.perform_queued_action(obs)
            if next_action is not False:
                return next_action


        # the point of a control loop is to re-check the basics in case things go haywire. 
        # count SCVs, re-check ctrl groups. Count buildings, build supply if needed, etc. etc.
        if self.control_loop_index > 10:
            self.control_loop_index = 0

        if self.control_loop_index == 0:
            print("Control loop 0")
            # assign command centers to control group 0, select all command centers
            #print("Single Select: " + str(single_select))
            #print("Multi Select: " + str(multi_select))
            #print("Build Queue: " + str(build_queue))
            #print("Control Groups: " + str(control_groups))

            # Tomorrow: if command centers are already selected, look for new command centers to add to the group
            # Today: if the command center is already selected move on to scv building.
            # TODO: Check for mis-assignments to control group, clear them out.

            cc_control_group = control_groups[0]
            print(str(cc_control_group))
            print(str(unit_ids["command center"]))
            print("len: " + str(len(cc_control_group)))
            if len(cc_control_group) == 2 and cc_control_group[0] == unit_ids["command center"]:
                print("yep, our command center is still there")
                # nothing to do for today's 1 base strat. Tomorrow, hunt for other command centers to add to the group.
                self.control_loop_index = 1
            else:
                self.control_group_command_centers(obs)

                self.control_loop_index = self.control_loop_index + 1
                next_action = self.perform_queued_action(obs)
                if next_action is not False:
                    return next_action

        if self.control_loop_index == 1:
            print("Control loop 1")
            # select all scvs, toss them into control group 8
            #print("Single Select: " + str(single_select))
            #print("Multi Select: " + str(multi_select))
            #print("Build Queue: " + str(build_queue))
            #print("Control Groups: " + str(control_groups))

            # Tomorrow: if command centers are already selected, look for new command centers to add to the group
            # Today: if the command center is already selected move on to scv building.
            # TODO: Check for mis-assignments to control group, clear them out.
            
            self.control_group_all_scvs(obs)
            # get the command centers selected for the next step
            self.select_command_centers_control_group(obs)

            self.control_loop_index = self.control_loop_index + 1
            next_action = self.perform_queued_action(obs)
            if next_action is not False:
                return next_action
        
        if self.control_loop_index == 2:
            # build scvs if needed
            print("Control loop 2")
            scv_count = 0
            scv_control_group = control_groups[8]
            if len(scv_control_group) == 2 and scv_control_group[0] == unit_ids["scv"]:
                scv_count = scv_control_group[1]

            if supply_used == supply_available:
                print("Gonna need more supply depots")

            if minerals > 50 and scv_count < 17 and supply_used < supply_available:
                print("yo, we should build an scv")
                
                # build into a better function with support for multi_select multi command centers
                if single_select[0][0] == unit_ids["command center"]:
                    print("ok, command center is selected as expected")
                    
                    # make this better, lol. If there's no current build queue or the 2nd spot is empty, go ahead and build it.
                    if len(build_queue) < 2:
                        print("queue isn't too full, let's go ahead and build it")
                        self.build_scv(obs)

                        self.control_loop_index = self.control_loop_index + 1
                        next_action = self.perform_queued_action(obs)
                        if next_action is not False:
                            return next_action

                else:
                    # need to stay on this control loop, but select the command center if it even exists. Otherwise move to base trade loop
                    pass

            # well, no dice on scvs, let's think about building some supply
            self.control_loop_index = self.control_loop_index + 1


        if self.control_loop_index == 3:
            # build supply if needed
            # TODO: Check scvs in builders group (9) if they have availabilies, use them. Otherwise grab another SCV for builders group
            if minerals > 100 and supply_used > supply_available - 2:
                self.build_supply_depot(obs)

                self.control_loop_index = self.control_loop_index + 1
                next_action = self.perform_queued_action(obs)
                if next_action is not False:
                    return next_action

        if self.control_loop_index == 4:
            self.control_group_barracks(obs)

            self.control_loop_index = self.control_loop_index + 1
            next_action = self.perform_queued_action(obs)
            if next_action is not False:
                return next_action

        if self.control_loop_index == 5:
            # barracks should be selected, which means that train marines should be an available action.
            can_train_marines = False
            if action_ids["train marine"] in obs.observation["available_actions"]:
                can_train_marines = True

            if minerals > 50 and can_train_marines:
                self.train_marines(obs)

                self.control_loop_index = self.control_loop_index + 1
                next_action = self.perform_queued_action(obs)
                if next_action is not False:
                    return next_action
            # build marines if needed
            pass

        if self.control_loop_index == 6:
            # build barracks if needed
            if minerals > 200:
                self.build_barracks(obs)
                self.control_loop_index = self.control_loop_index + 1
                next_action = self.perform_queued_action(obs)
                if next_action is not False:
                    return next_action
            
        if self.control_loop_index == 7:
            # group up army at base, use a new ctrl group if needed
            pass

        if self.control_loop_index == 8:
            # attack move the army
            if supply_used > 50:
                self.attack_move(obs)
                self.control_loop_index = self.control_loop_index + 1
                next_action = self.perform_queued_action(obs)
                if next_action is not False:
                    return next_action

            
            


        self.control_loop_index = self.control_loop_index + 1


        #if self.steps == 45:
        #    self.build_supply_depot(obs)

        #if self.steps == 120:
        #    self.build_barracks(obs)

        #if self.steps == 220:
        #    self.control_group_barracks(obs)

        #if self.steps > 230:
        #    if obs.observation['player'][params['supply used']] < obs.observation['player'][params['supply available']]:
        #        self.train_marine(obs)
        #    else:
        #        self.attack_move(obs)


        #print ("hm: " + str(obs.observation))
        #print ("keys: " + str(obs.observation.keys())) # dict_keys(['single_select', 'multi_select', 'build_queue', 'cargo', 'cargo_slots_available', 'feature_screen', 'feature_minimap', 'last_actions', 'action_result', 'alerts', 'game_loop', 'score_cumulative', 'player', 'control_groups', 'available_actions'])
        #print("game loop: " + str(obs.observation["game_loop"]))
        #print("feature minimap: " + str(obs.observation["feature_minimap"]))
        #print("Minerals: " + str(obs.observation["player"][params["minerals"]]))
        #print(f"Supply: {obs.observation['player'][params['supply used']]} of {obs.observation['player'][params['supply available']]}")

        # if there are things to do, do them, otherwise, do nothing
        next_action = self.perform_queued_action(obs)

        if next_action is not False:
            return next_action
        else:
            return actions.FUNCTIONS.no_op()

    def perform_queued_action(self, obs):
        if len(action_queue) > 0:
            to_do = action_queue.pop(0)[0]
            print("To Do: " + str(to_do))
            action_id = to_do["id"]
            action_params = to_do["params"]

            if action_id not in obs.observation["available_actions"]:
                print("Warning, cannot perform action: " + str(action_id) + " Do you have enough money?")
                return False

            action = actions.FunctionCall(action_id, action_params)
            return action
        else:
           return False
        
    def control_group_command_centers(self, obs):
        # To Do: Differentiate between multiple command centers, add them all, rename function
        
        # Find the pixels that relate to command centers
        unit_type = obs.observation['feature_screen'][features["unit type"]]
        cc_ys, cc_xs = (unit_type == unit_ids['command center']).nonzero()
            
        # find the center of the pixels relating to command centers
        x = int(round(cc_xs.mean()))
        y = int(round(cc_ys.mean()))

        # queue up an action to click on the center pixel
        action_queue.append([{ "id":action_ids["select point"], "params":[[params["now"]], [x,y]] }])

        # assign selected to ctrl+group 0
        append_id = [x for x, y in enumerate(actions.CONTROL_GROUP_ACT_OPTIONS) if y[0] == 'append']
        action_queue.append([{ "id":action_ids["control group"], "params":[append_id, [0]] }])

    def calculate_grid_dimensions(self, obs):
        # To Do: Differentiate between multiple command centers, add them all, rename function
        
        # Find the pixels that relate to command centers
        unit_type = obs.observation['feature_screen'][features["unit type"]]
        cc_ys, cc_xs = (unit_type == unit_ids['command center']).nonzero()
            
        # find the center of the pixels relating to command centers
        mi_x = int(round(cc_xs.min()))
        ma_x = int(round(cc_xs.max()))
        mi_y = int(round(cc_ys.min()))
        ma_y = int(round(cc_ys.max()))

        self.grid_height = (ma_y - mi_y) / 5
        self.grid_width = (ma_x - mi_x) / 5

    def build_scv(self, obs):
        #load up the action_queue with the things needed to do to build an scv

        #tell it to train an scv
        action_queue.append([{ "id":action_ids["train scv"], "params":[ [params["queued"]] ] }])

    def control_group_all_scvs(self, obs):
        unit_type = obs.observation['feature_screen'][features["unit type"]]
        unit_y, unit_x = (unit_type == unit_ids["scv"]).nonzero()
              
        x = unit_x[0]
        y = unit_y[0]
                
        select_type = [x for x, y in enumerate(actions.SELECT_POINT_ACT_OPTIONS) if y[0] == 'select_all_type']
        action_queue.append([{ "id":action_ids["select point"], "params":[select_type, [x,y]] }])

        # add the scvs to the scv ctrl group for counting scvs
        control_action = [x for x, y in enumerate(actions.CONTROL_GROUP_ACT_OPTIONS) if y[0] == 'append']
        action_queue.append([{ "id":action_ids["control group"], "params":[control_action, [8]] }])


    def select_command_centers_control_group(self, obs):
        # select the second element of the tuplelist where the first element is 'recall'
        recall_id = [x for x, y in enumerate(actions.CONTROL_GROUP_ACT_OPTIONS) if y[0] == 'recall']
        # select ctrl group 0 (the command centers control group)
        action_queue.append([{ "id":action_ids["control group"], "params":[recall_id, [0]] }])


    def build_supply_depot(self, obs):
        # TODO: Select the closest scv to the point where you would like the supply depot to be.
        # TODO: Try to determine if the SCV is on the way back from a min patch or to a min patch
        #   - Command the SCV to drop of mins if on way back.
        # TODO: Create a precurser command to prep_build_supply_depot which gets an SCV to the location as budget gets close
        # TODO: Build a second supply depot in a non-shitty location
        # TODO: Find available spaces for supply depots given other buildings
        # TODO: Consider a wall
        # TODO: Assign location controls to AI

        # Select the builder SCV
        unit_type = obs.observation['feature_screen'][features["unit type"]]
        unit_y, unit_x = (unit_type == unit_ids["scv"]).nonzero()
        mineral_ys, mineral_xs = np.logical_or((unit_type == unit_ids["mineral field"]), (unit_type == unit_ids["mineral field 750"]) ).nonzero()

              
        x = unit_x[0]
        y = unit_y[0]
                
        action_queue.append([{ "id":action_ids["select point"], "params":[[params["now"]], [x,y]] }])

        # add the builder scv to the builders ctrl group
        append_id = [x for x, y in enumerate(actions.CONTROL_GROUP_ACT_OPTIONS) if y[0] == 'append']
        action_queue.append([{ "id":action_ids["control group"], "params":[append_id, [9]] }])

        location = self.get_next_supply_depot_location(obs)

        if location is not None:
            action_queue.append([{ "id":action_ids["build supply depot"], "params":[[params["queued"]], location] }])
        
        # get back to work ya darned scv
        action_queue.append([{ "id":action_ids["harvest"], "params":[[params["queued"]], [mineral_xs[15], mineral_ys[15]]] }])

        #queue up a move to where the barracks will be built (should happen first time only, but this is a test)
        #action_queue.append([{ "id":action_ids["move"], "params":[[params["queued"]], [x,y+dist_y]] }])

    def get_next_supply_depot_location(self, obs):     
        unit_type = obs.observation['feature_screen'][features["unit type"]]
        cc_ys, cc_xs = (unit_type == unit_ids["command center"]).nonzero()

        # existing supply depots
        sd_ys, sd_xs = (unit_type == unit_ids["supply depot"]).nonzero()

        #print(str(unit_type))
        mineral_ys, mineral_xs = np.logical_or((unit_type == unit_ids["mineral field"]), (unit_type == unit_ids["mineral field 750"]) ).nonzero()
        #print(str(mineral_ys))

        # center of the command center
        cc_x = int(round(cc_xs.mean()))
        cc_y = int(round(cc_ys.mean()))
        
        # center of the mineral fields
        mineral_x = int(round(mineral_xs.mean()))
        mineral_y = int(round(mineral_ys.mean()))

        # plant the new building on the opposite side of the CC from the mineral fields
        x = cc_x
        y = cc_y

        # how big is a grid tile?
        grid_x = self.grid_width
        grid_y = self.grid_height

        # which side of the command center to build on
        if mineral_x > cc_x:
            grid_x = grid_x * -1

        if mineral_y > cc_y:
            grid_y = grid_y * -1

        # build on non-mineral side, at the corner of the CC
        sd_0x = int(round(cc_x + grid_x * 3.5))
        sd_0y = int(round(cc_y - grid_y * 2.5))

        sd_x = sd_0x
        sd_y = sd_0y

        #print("xs: "+str(sd_xs))
        #print("ys: "+str(sd_ys))

        # if there aren't any supply depots on screen
        if len(sd_xs) == 0:
            return [sd_x, sd_y]

        for x in range(4):
            sd_x = int(round(sd_0x + x * 2 * grid_x))
            for y in range(3):
                sd_y = int(round(sd_0y + y * 2 * grid_y))
                #print(f"sd_x: {sd_x}, sd_y: {sd_y}")
                blocked = False

                for i in range(len(sd_xs)):
                    if sd_xs[i] == sd_x and sd_ys[i] == sd_y:
                        #there is already a supply depot here, move on.
                        #print(" but that one is blocked")
                        blocked = True
                        continue
                
                if blocked == False:
                    return [sd_x, sd_y]
                    

    def build_barracks(self, obs):
        # TODO: Select the closest scv to the point where you would like the building to be.
        # TODO: Try to determine if the SCV is on the way back from a min patch or to a min patch
        #   - Command the SCV to drop of mins if on way back.
        # TODO: Create a precurser command to prep_build_barracks which gets an SCV to the location as budget gets close
        # TODO: Build a second supply depot in a non-shitty location
        # TODO: Find available spaces for supply depots given other buildings
        # TODO: Consider a wall
        # TODO: Assign location controls to AI
        
        # select the second element of the tuplelist where the first element is 'recall'
        recall_id = [x for x, y in enumerate(actions.CONTROL_GROUP_ACT_OPTIONS) if y[0] == 'recall']
        # select ctrl group 9 (the builders control group)
        action_queue.append([{ "id":action_ids["control group"], "params":[recall_id, [9]] }])

        unit_type = obs.observation["feature_screen"][features["unit type"]]
        cc_ys, cc_xs = (unit_type == unit_ids["command center"]).nonzero()

        # get the pixels related to mineral fields
        mineral_ys, mineral_xs = np.logical_or((unit_type == unit_ids["mineral field"]), (unit_type == unit_ids["mineral field 750"]) ).nonzero()
        
        # to re-harvest after build, be careful about the mineral patches
        #mineral_ys, mineral_xs = (unit_type == unit_ids["mineral field"]).nonzero()
        

        # center of the command center
        cc_x = int(round(cc_xs.mean()))
        cc_y = int(round(cc_ys.mean()))

        # center of the mineral fields
        mineral_x = int(round(mineral_xs.mean()))
        mineral_y = int(round(mineral_ys.mean()))

        # plant the new building on the opposite side of the CC from the mineral fields
        x = cc_x
        y = cc_y

        # the distance from the CC to the center of the mineral patches
        dist_x = 20
        dist_y = 20

        if mineral_x < cc_x:
            dist_x = 20
        else:
            dist_x = -20

        if mineral_y < cc_y:
            dist_y = 20
        else:
            dist_y = -20

        #print(f"cc: {cc_x},{cc_y}  barracks: {x},{y}  minerals: {obs.observation['player']}")
        location = self.get_next_barracks_location(obs)

        # don't try to build in nullspace
        if location is not None:
            action_queue.append([{ "id":action_ids["build barracks"], "params":[[params["queued"]], location] }])
        
        # get back to work after building the barracks
        # HACKHACK - need to find the center of a mineral patch in a better way, hardcoding x,y 15 is ugly.
        action_queue.append([{ "id":action_ids["harvest"], "params":[[params["queued"]], [mineral_xs[15], mineral_ys[15]]] }])

    def get_next_barracks_location(self, obs):     
        unit_type = obs.observation['feature_screen'][features["unit type"]]
        cc_ys, cc_xs = (unit_type == unit_ids["command center"]).nonzero()

        # existing supply depots
        b_ys, b_xs = (unit_type == unit_ids["barracks"]).nonzero()

        #print(str(unit_type))
        mineral_ys, mineral_xs = np.logical_or((unit_type == unit_ids["mineral field"]), (unit_type == unit_ids["mineral field 750"]) ).nonzero()
        #print(str(mineral_ys))

        # center of the command center
        cc_x = int(round(cc_xs.mean()))
        cc_y = int(round(cc_ys.mean()))
        
        # center of the mineral fields
        mineral_x = int(round(mineral_xs.mean()))
        mineral_y = int(round(mineral_ys.mean()))

        # plant the new building on the opposite side of the CC from the mineral fields
        x = cc_x
        y = cc_y

        # how big is a grid tile?
        grid_x = self.grid_width
        grid_y = self.grid_height

        # which side of the command center to build on
        if mineral_x > cc_x:
            grid_x = grid_x * -1

        if mineral_y > cc_y:
            grid_y = grid_y * -1

        # build on non-mineral side, at the corner of the CC
        b_0x = int(round(cc_x - grid_x * 2))
        b_0y = int(round(cc_y + grid_y * 5))

        b_x = b_0x
        b_y = b_0y

        #print("xs: "+str(b_xs))
        #print("ys: "+str(b_ys))

        # if there aren't any barracks on screen
        if len(b_xs) == 0:
            return [b_x, b_y]

        for x in range(3):
            b_x = int(round(b_0x + x * 3 * grid_x))
            for y in range(2):
                b_y = int(round(b_0y + y * 3 * grid_y))
                print(f"b_x: {b_x}, b_y: {b_y}")
                blocked = False

                for i in range(len(b_xs)):
                    if b_xs[i] == b_x and b_ys[i] == b_y:
                        #there is already a barracks here, move on.
                        print(" but that one is blocked")
                        blocked = True
                        continue
                
                if blocked == False:
                    return [b_x, b_y]
                    


    def control_group_barracks(self, obs):
        # To Do: Differentiate between multiple barracks centers, select them all
        
        # Find the pixels that relate to barracks
        unit_type = obs.observation['feature_screen'][features["unit type"]]
        b_ys, b_xs = (unit_type == unit_ids['barracks']).nonzero()
            
        if len(b_xs) == 0:
            return

        # find the center of the pixels relating to barracks
        x = int(round(b_xs.mean()))
        y = int(round(b_ys.mean()))
        
        # HACKHACK to get a point that works when there are multiple barracks
        y = y - 4


        # queue up an action to click on the center pixel
        select_type = [x for x, y in enumerate(actions.SELECT_POINT_ACT_OPTIONS) if y[0] == 'select_all_type']
        action_queue.append([{ "id":action_ids["select point"], "params":[select_type, [x,y]] }])

        # assign selected to ctrl+group 6
        option = [x for x, y in enumerate(actions.CONTROL_GROUP_ACT_OPTIONS) if y[0] == 'set']
        action_queue.append([{ "id":action_ids["control group"], "params":[option, [6]] }])

        rally_point = self.get_rally_point(obs)
        action_queue.append([{ "id":action_ids["rally minimap"], "params":[[params["now"]], rally_point] }])


    def train_marines(self, obs):
        # barracks should already be selected based on control loop
        # get ctrl+group 6
        #recall_id = [x for x, y in enumerate(actions.CONTROL_GROUP_ACT_OPTIONS) if y[0] == 'recall']
        #action_queue.append([{ "id":action_ids["control group"], "params":[recall_id, [6]] }])

        action_queue.append([{ "id":action_ids["train marine"], "params":[ [params["queued"]] ] }])

    def save_player_start_location(self, obs):
        player_ys, player_xs = (obs.observation["feature_minimap"][features["player relative"]] == params["player self"]).nonzero()
        my_x = int(round(player_xs.mean()))
        my_y = int(round(player_ys.mean()))

        self.player_start_location = [my_x, my_y]


    def attack_move(self, obs):
        my_x, my_y = self.player_start_location
        print(str(obs.observation["feature_minimap"]))

        enemy_ys, enemy_xs = (obs.observation["feature_minimap"][features["player relative"]] == params["player enemy"]).nonzero()

        center_x = 31
        center_y = 31

        enemy_x = center_x + center_x - my_x
        enemy_y = center_y + center_y - my_y
        
        if(len(enemy_ys) > 0):
            print("enemy xs:" +str(enemy_xs))
            print("enemy ys:" +str(enemy_ys))

            enemy_x = enemy_xs[0]
            enemy_y = enemy_ys[0]

        #print("my x: " + str(my_x))
        #print("enemy x: " + str(enemy_x))

        action_queue.append([{ "id":action_ids["select army"], "params":[ [params["now"]]] }])
        
        # assign selected to ctrl+group 1
        append_id = [x for x, y in enumerate(actions.CONTROL_GROUP_ACT_OPTIONS) if y[0] == 'append']
        action_queue.append([{ "id":action_ids["control group"], "params":[append_id, [1]] }])

        action_queue.append([{ "id":action_ids["attack minimap"], "params":[ [params["now"]], [enemy_x, enemy_y] ] }])

    def get_rally_point(self, obs):
        player_ys, player_xs = (obs.observation["feature_minimap"][features["player relative"]] == params["player self"]).nonzero()
        my_x = int(round(player_xs.mean()))
        my_y = int(round(player_ys.mean()))

        center_x = 31
        center_y = 31

        x = int(round((my_x * 2 + center_x) / 3))
        y = int(round((my_y * 2 + center_y) / 3))

        return [x,y]

        


        


def main():
    #print(sys.path)
    os.system('python -m pysc2.bin.agent --map Simple64 --agent abot.aBotAgent --agent_race terran')

if __name__ == "__main__":
    main()

