from pysc2.lib import actions, features, units

params = {
    "player self":1,
    "player enemy":4,
    "now":0,
    "queued":1,
    "minerals":1,
    "vespene":2,
    "supply used":3,
    "supply available":4,
    "select all": actions.SelectPointAct["select_all_type"]
    }

screen_features = {
    "player relative":features.SCREEN_FEATURES.player_relative.index,
    "unit type":features.SCREEN_FEATURES.unit_type.index,
    "selected":features.SCREEN_FEATURES.selected.index,
    "height map":features.SCREEN_FEATURES.height_map.index,
    "visibility map":features.SCREEN_FEATURES.visibility_map.index,
    "creep":features.SCREEN_FEATURES.creep.index
    }

minimap_features = {
    "player relative":features.MINIMAP_FEATURES.player_relative.index,
    "selected":features.MINIMAP_FEATURES.selected.index,
    "height map":features.MINIMAP_FEATURES.height_map.index,
    "visibility map":features.MINIMAP_FEATURES.visibility_map.index,
    "creep":features.MINIMAP_FEATURES.creep.index,
    "camera":features.MINIMAP_FEATURES.camera.index
    }

unit_ids = {
    "scv": units.Terran.SCV,
    "command center": units.Terran.CommandCenter,
    "supply depot": units.Terran.SupplyDepot,
    "barracks": units.Terran.Barracks,
    "mineral field": units.Neutral.MineralField,
    "mineral field 750": units.Neutral.MineralField750
    }

unit_cost = {
    "scv": [50, 0],
    "marine": [50, 0],
    "supply depot": [100, 0],
    "barracks": [150, 0],
    "command center": [400, 0]
    }

action_ids = {
    "no op": actions.FUNCTIONS.no_op.id,
    "move camera": actions.FUNCTIONS.move_camera.id,
    "scan move minimap": actions.FUNCTIONS.Scan_Move_minimap.id,
    "scan move screen": actions.FUNCTIONS.Scan_Move_screen.id,
    "select point": actions.FUNCTIONS.select_point.id, 
    "control group": actions.FUNCTIONS.select_control_group.id,
    "harvest": actions.FUNCTIONS.Harvest_Gather_screen.id,
    "harvest return": actions.FUNCTIONS.Harvest_Return_quick.id, # a possible way to identify scvs holding minerals. Could also be a min action way to return them to work.
    "train scv": actions.FUNCTIONS.Train_SCV_quick.id,
    "build supply depot": actions.FUNCTIONS.Build_SupplyDepot_screen.id,
    "train marine": actions.FUNCTIONS.Train_Marine_quick.id,
    "build barracks": actions.FUNCTIONS.Build_Barracks_screen.id,
    "build command center": actions.FUNCTIONS.Build_CommandCenter_screen.id,
    "rally minimap": actions.FUNCTIONS.Rally_Units_minimap.id,
    "select army": actions.FUNCTIONS.select_army.id,
    "select idle worker": actions.FUNCTIONS.select_idle_worker.id,
    "attack minimap": actions.FUNCTIONS.Attack_minimap.id,
    "move screen": actions.FUNCTIONS.Move_screen.id,
    "lower supply depot": actions.FUNCTIONS.Morph_SupplyDepot_Lower_quick.id,
    "raise supply depot": actions.FUNCTIONS.Morph_SupplyDepot_Lower_quick.id
    }