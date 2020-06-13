import numpy as np
from crowddynamics.core.geometry import geom_to_linear_obstacles
from crowddynamics.simulation.agents import Circular, ThreeCircle, NO_TARGET, \
    Agents, AgentGroup
from crowddynamics.simulation.field import Field
from crowddynamics.simulation.logic import Reset, InsideDomain, Integrator, \
    Fluctuation, Adjusting, Navigation, ExitDetection, \
    Orientation, AgentAgentInteractions, AgentObstacleInteractions, \
    LeaderFollower, TargetReached
from crowddynamics.simulation.multiagent import MultiAgentSimulation
from shapely.geometry import Polygon, Point, LineString, MultiPolygon, MultiLineString, LinearRing
from traitlets.traitlets import Enum, Int, default

from shapely.ops import polygonize
from scipy.spatial.qhull import Delaunay
from crowddynamics.core.sampling import triangle_area_cumsum, random_sample_triangle
from crowddynamics.core.vector2D import length
from crowddynamics.core.distance import distance_circle_line, distance_circles
from crowddynamics.simulation.agents import Agents, AgentGroup, Circular


class ComplexFloorField(Field):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        left_hall_width = 40
        left_hall_height = 5 #10

        right_hall_width = left_hall_width
        right_hall_height = left_hall_height

        upper_hall_width = 5 #10
        upper_hall_height = 40 # 40

        lower_hall_width = 5
        lower_hall_height = 40

        narrow_exit_width = 1.2
        broad_exit_width = 2.5

        spawn_left_width = 5 #10
        spawn_left_height = 5 #10
        spawn_left_separation = 15

        spawn_right_width = 5 #10
        spawn_right_height = 5 #10
        spawn_right_separation = 15

        spawn_upper_width = 5 #10
        spawn_upper_height = 5 #10
        spawn_upper_separation = 15

        spawn_lower_width = 5 #10
        spawn_lower_height = 5 #10
        spawn_lower_separation = 15

        # Buffer (so that spawned agents are not intersecting with obstacles)
        buffer = 0.27

        def f(value, scale=1):
            if value:
                return tuple(map(lambda x: scale * x, value))
            else:
                return None

        # Corner points of the domain counterclockwards
        crossing = list(map(f, [
            None,
            (0, lower_hall_height),
            (left_hall_width, lower_hall_height),
            (left_hall_width, 0),
            (left_hall_width + lower_hall_width, 0),
            (left_hall_width + lower_hall_width, lower_hall_height),
            (left_hall_width + lower_hall_width + right_hall_width, lower_hall_height),
            (left_hall_width + lower_hall_width + right_hall_width, lower_hall_height + right_hall_height),
            (left_hall_width + lower_hall_width, lower_hall_height + right_hall_height),
            (left_hall_width + lower_hall_width, lower_hall_height + right_hall_height + upper_hall_height),
            (left_hall_width, lower_hall_height + left_hall_height + upper_hall_height),
            (left_hall_width, lower_hall_height + left_hall_height),
            (0, lower_hall_height + left_hall_height),
        ]))

        # Exitpoints for both exits
        exitpoints = list(map(f, [
            None,
            (0, lower_hall_height + left_hall_height / 2 + narrow_exit_width / 2),
            (0, lower_hall_height + left_hall_height / 2 - narrow_exit_width / 2),
            (left_hall_width + lower_hall_width / 2 - narrow_exit_width / 2, 0),
            (left_hall_width + lower_hall_width / 2 + narrow_exit_width / 2, 0),
            (left_hall_width + lower_hall_width + right_hall_width, lower_hall_height + right_hall_height /2 - narrow_exit_width / 2),
            (left_hall_width + lower_hall_width + right_hall_width, lower_hall_height + right_hall_height /2 + narrow_exit_width / 2),
            (left_hall_width + lower_hall_width / 2 + narrow_exit_width / 2, lower_hall_height + right_hall_height + upper_hall_height),
            (left_hall_width + lower_hall_width / 2 - narrow_exit_width / 2, lower_hall_height + right_hall_height + upper_hall_height),
        ]))

        # Spawn left area corner points
        spawn_left_points = list(map(f, [
            None,
            (spawn_left_separation + buffer, lower_hall_height + buffer),
            (spawn_left_separation + spawn_left_width - buffer, lower_hall_height + buffer),
            (spawn_left_separation + spawn_left_width - buffer, lower_hall_height + spawn_left_height - buffer),
            (spawn_left_separation + buffer, lower_hall_height + spawn_left_height - buffer),
        ]))

        # Spawn right area corner points
        spawn_right_points = list(map(f, [
            None,
            (left_hall_width + lower_hall_width + right_hall_width - spawn_right_separation - spawn_right_width + buffer, lower_hall_height + buffer),
            (left_hall_width + lower_hall_width + right_hall_width - spawn_right_separation - buffer, lower_hall_height + buffer),
            (left_hall_width + lower_hall_width + right_hall_width - spawn_right_separation - buffer, lower_hall_height + right_hall_height - buffer),
            (left_hall_width + lower_hall_width + right_hall_width - spawn_right_separation - spawn_right_width + buffer, lower_hall_height + right_hall_height - buffer),
        ]))

        # Spawn upper area corner points
        spawn_upper_points = list(map(f, [
            None,
            (left_hall_width + buffer, lower_hall_height + left_hall_height + upper_hall_height - spawn_upper_separation - buffer),
            (left_hall_width + buffer, lower_hall_height + left_hall_height + upper_hall_height - spawn_upper_separation - spawn_upper_height + buffer),
            (left_hall_width + upper_hall_width - buffer, lower_hall_height + left_hall_height + upper_hall_height - spawn_upper_separation - spawn_upper_height + buffer),
            (left_hall_width + upper_hall_width - buffer, lower_hall_height + left_hall_height + upper_hall_height - spawn_upper_separation - buffer),
        ]))

        # Spawn lower area corner points
        spawn_lower_points = list(map(f, [
            None,
            (left_hall_width + buffer, spawn_lower_separation + spawn_lower_height - buffer),
            (left_hall_width + buffer, spawn_lower_separation + buffer),
            (left_hall_width + lower_hall_width - buffer, spawn_lower_separation + buffer),
            (left_hall_width + lower_hall_width - buffer, spawn_lower_separation + spawn_lower_height - buffer),
        ]))

        # Obstacles counterclockwards
        obstacles = Polygon()

        obstacles |= LineString(
            [exitpoints[2]] + [crossing[1]] + [crossing[2]] + [crossing[3]] + [exitpoints[3]]
        )
        obstacles |= LineString(
            [exitpoints[4]] + [crossing[4]] + [crossing[5]] + [crossing[6]] + [exitpoints[5]]
        )
        obstacles |= LineString(
            [exitpoints[6]] + [crossing[7]] + [crossing[8]] + [crossing[9]] + [exitpoints[7]]
        )
        obstacles |= LineString(
            [exitpoints[8]] + [crossing[10]] + [crossing[11]] + [crossing[12]] + [exitpoints[1]]
        )

        floorplan = Polygon([
            crossing[1], crossing[2], crossing[3], crossing[4], crossing[5], crossing[6], crossing[7], crossing[8], crossing[9], crossing[10], crossing[11], crossing[12]]
        )

        # Exits from the upper right piece counterclockwards
        exit1 = LineString([exitpoints[1], exitpoints[2]])
        exit2 = LineString([exitpoints[3], exitpoints[4]])
        exit3 = LineString([exitpoints[5], exitpoints[6]])
        exit4 = LineString([exitpoints[7], exitpoints[8]])

        # Spawn areas from the upper left piece counterclockwards
        spawn_left = Polygon([
            spawn_left_points[1], spawn_left_points[2], spawn_left_points[3], spawn_left_points[4]]
        )
        spawn_lower = Polygon([
            spawn_lower_points[1], spawn_lower_points[2], spawn_lower_points[3], spawn_lower_points[4]]
        )
        spawn_right = Polygon([
            spawn_right_points[1], spawn_right_points[2], spawn_right_points[3], spawn_right_points[4]]
        )
        spawn_upper = Polygon([
            spawn_upper_points[1], spawn_upper_points[2], spawn_upper_points[3], spawn_upper_points[4]]
        )

        # Spawns
        spawns = [
            spawn_left,
            spawn_lower,
            spawn_right,
            spawn_upper
        ]

        targets = [exit1, exit2, exit3, exit4]

        self.obstacles = obstacles  # obstacles
        self.targets = targets
        self.spawns = spawns
        self.domain = floorplan



class ComplexFloor(MultiAgentSimulation):

    agent_type = Enum(
        default_value=Circular,
        values=(Circular, ThreeCircle))
    body_type = Enum(
        default_value='adult',
        values=('adult',))

    def attributes(self, familiar, has_target: bool = True, is_follower: bool = True):
        def wrapper():
            target = familiar if has_target else NO_TARGET
            orientation = np.random.uniform(-np.pi, np.pi)
            d = dict(
                target=target,
                is_leader=not is_follower,
                is_follower=is_follower,
                body_type=self.body_type,
                orientation=orientation,
                velocity=np.zeros(2),
                angular_velocity=0.0,
                target_direction=np.zeros(2),
                target_orientation=orientation,
                familiar_exit=familiar,
            )
            return d

        return wrapper

    def attributes_leader(self, target_iter, has_target: bool = True, is_follower: bool = False):
        def wrapper():
            target = next(target_iter)
            orientation = np.random.uniform(-np.pi, np.pi)
            d = dict(
                target=target,
                is_leader=not is_follower,
                is_follower=is_follower,
                body_type=self.body_type,
                orientation=orientation,
                velocity=np.zeros(2),
                angular_velocity=0.0,
                target_direction=np.zeros(2),
                target_orientation=orientation,
                familiar_exit=4,
            )
            return d

        return wrapper

    @default('logic')
    def _default_logic(self):
        return Reset(self) << \
               TargetReached(self) << (
                   Integrator(self) << (
                       #Fluctuation(self),
                       Adjusting(self) << (
                           Navigation(self) << LeaderFollower(self),
                           Orientation(self)),
                       AgentAgentInteractions(self),
                       AgentObstacleInteractions(self)))

    @default('field')
    def _default_field(self):
        return ComplexFloorField()

    @default('agents')
    def _default_agents(self):
        agents = Agents(agent_type=self.agent_type)
        return agents
