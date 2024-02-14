from .bandit_task import BanditTask
from .grid_world import GridWorld

environment_library = {
    BanditTask.__name__: BanditTask,
    GridWorld.__name__: GridWorld
}
