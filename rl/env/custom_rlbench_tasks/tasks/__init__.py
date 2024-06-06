from .phone_on_base import PhoneOnBase
from .pick_up_cup import PickUpCup
from .put_rubbish_in_bin import PutRubbishInBin
from .stack_wine import StackWine
from .take_umbrella_out_of_umbrella_stand import TakeUmbrellaOutOfUmbrellaStand
from .take_lid_off_saucepan import TakeLidOffSaucepan
from .reach_target import ReachTarget
from .pick_and_lift import PickAndLift
from .meat_off_grill import MeatOffGrill
from .bottle_out_moving_fridge import BottleOutMovingFridge
from .barista import Barista
from .reach_gripper_and_elbow import ReachGripperAndElbow
from .slide_cup import SlideCup
from .cup_out_open_cabinet  import CupOutOpenCabinet


CUSTOM_TASKS = \
{   
    # Standard 8 tasks
    'reach_target' : ReachTarget,
    'pick_up_cup' : PickUpCup,
    'take_umbrella_out_of_umbrella_stand' : TakeUmbrellaOutOfUmbrellaStand,
    'take_lid_off_saucepan' : TakeLidOffSaucepan, 
    'pick_and_lift': PickAndLift,
    'phone_on_base' : PhoneOnBase,
    'stack_wine' : StackWine,
    'put_rubbish_in_bin' : PutRubbishInBin,
    # Abbreviations
    'umbrella_out' : TakeUmbrellaOutOfUmbrellaStand, 
    'saucepan' : TakeLidOffSaucepan, 
    # New tasks
    'reach_elbow_pose' : ReachGripperAndElbow,
    'take_bottle_out_fridge': BottleOutMovingFridge,
    'serve_coffee_obstacles' : Barista,
    'slide_cup_obstacles' : SlideCup, 
    'meat_off_grill' : MeatOffGrill,
    'take_cup_out_cabinet' : CupOutOpenCabinet,

}