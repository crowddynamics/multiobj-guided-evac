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
    # def __init__(self, kokeilu):
    #    self.kokeilu = kokeilu

    size_spawn_left = Int(
        default_value=50, min=0, max=200, help='')
    size_spawn_lower = Int(
        default_value=50, min=0, max=200, help='')
    size_spawn_right = Int(
        default_value=50, min=0, max=200, help='')
    size_spawn_upper = Int(
        default_value=50, min=0, max=200, help='')
    size_leader = Int(
        default_value=0, min=0, max=6, help='')

    agent_type = Enum(
        default_value=Circular,
        values=(Circular, ThreeCircle))
    body_type = Enum(
        default_value='adult',
        values=('adult',))

    # CHECK THE GUIDE GENERATOR FUNCTION
    # A helper function to create spawn points for leaders out of their cell coordinates.
    # The input is the array of leader spawn cells and the number of leaders in the simulation.
    # def generate_leader_pos(cell, n_lead, seed_number):
    def generate_leader_pos(self, cell, n_lead):

        # FIRST THE DATA HAS TO BE CREATED
        # Load data of followers
        followers = np.load('complex/spawn_complex.npy')
        follower_positions = followers['position']
        follower_radii = followers['radius']

        # Minimal radius of a guide (the same value given in agents.py to the guides).
        max_r = 0.27

        # Number of times spawned leaders are allowed to overlap each other before the program is
        # terminated.
        overlaps = 10000

        # Import Complex floor field
        field = ComplexFloor().field

        # Bound box representing the room.
        width = 90
        height = 90

        # Create a grid structure over the room geometry.
        # Cell size in the grid, determines the resolution of the micro-macro converted data
        cell_size = 2
        m = np.round(width / cell_size)
        n = np.round(height / cell_size)
        m = m.astype(int)
        n = n.astype(int)
        X = np.linspace(0, width, m + 1)
        Y = np.linspace(0, height, n + 1)
        hlines = [((x1, yi), (x2, yi)) for x1, x2 in zip(X[:-1], X[1:]) for yi in Y]
        vlines = [((xi, y1), (xi, y2)) for y1, y2 in zip(Y[:-1], Y[1:]) for xi in X]
        grids = list(polygonize(MultiLineString(hlines + vlines)))

        # Leaders' spawn areas
        leader_spawns = []

        # Leader's spawn points
        spawn_points = []

        # Loop through the cells and calculate intersections with spawn areas.
        for i in range(n_lead):

            poly = field.domain.intersection(grids[cell[i]])
            if not poly.is_empty:
                leader_spawns.append(poly)

        # Import obstacles
        obstacles = field.obstacles

        # Spawn a random position from the starting area.
        # Loop through all the leaders.
        # (1) Take into account that there might be obstacles in the spawn areas, and take also
        # into account that agents have a buffer radius.
        # (2) If the spawn area is a MultiPolygon, loop through the polygons in a MultiPolygon. Create a
        # mesh grid of the spawn area with Delaunay triangulation.
        # (2.1) Spawn a random point from the mesh grid.
        # (2.2) Check that the position doesn't interfere with other agents' positions
        # (2.3) Set the Boolean value for if the leader is initially inside the Finlandiahall
        # (this is needed for the movement simulation).
        # (3) If the spawn area is not a MultiPolygon, just directly create a mesh grid of the spawn area
        # with Delaunay triangulation.
        # (3.1) Spawn a random point from the mesh grid.
        # (3.2) Check that the position doesn't interfere with other agents' positions
        # (3.3) Set the Boolean value for if the leader is initially inside the Finlandiahall (this is
        # is needed for the movement simulation).
        for i in range(n_lead):
            seed = 0
            # (1)
            n_spawnpoints = len(spawn_points)
            geom = leader_spawns[i] - obstacles.buffer(max_r)
            j = 0  # set overlaps counter to zero
            # (2)
            if isinstance(geom, MultiPolygon):
                n_polygons = len(geom)
                for j in range(n_polygons):
                    vertices = np.asarray(geom[j].convex_hull.exterior)
                    delaunay = Delaunay(vertices)
                    mesh = vertices[delaunay.simplices]
                    if j == 0:
                        meshes = mesh
                    else:
                        meshes = np.concatenate((mesh, meshes), axis=0)
                # Computes cumulative sum of the areas of the triangle mesh.
                weights = triangle_area_cumsum(meshes)
                weights /= weights[-1]

                while j < overlaps:
                    seed += 1
                    distances = []  # temporarily store distances from the spawned point to the previously spawned
                    n_overlaps = 0  # for each attempt to position the guide, set number of overlaps to zero
                    # (2.1) Spawn a random point for the guide.
                    np.random.seed(seed)
                    x = np.random.random()
                    k = np.searchsorted(weights, x)
                    a, b, c = meshes[k]
                    spawn_point = random_sample_triangle(a, b, c)
                    # spawn_point = random_sample_triangle(a, b, c, seed)
                    # (2.2)
                    if n_spawnpoints != 0:  # if there are no other spawned guides skip this step
                        for k in range(0, n_spawnpoints):
                            d = length(spawn_point - spawn_points[k])
                            h = d - 2 * max_r
                            distances.append(h)
                        distances_array = distances
                        distances_array = np.asarray(distances_array)
                        n_overlaps += len(np.where(distances_array < 0)[0])
                    for obstacle in obstacles:
                        obstacle = list(obstacle.coords)
                        n_obstacle_points = len(obstacle)
                        for k in range(0, n_obstacle_points):
                            if k == n_obstacle_points - 1:
                                h, _ = distance_circle_line(spawn_point, max_r, np.asarray(obstacle[k]),
                                                            np.asarray(obstacle[0]))
                            else:
                                h, _ = distance_circle_line(spawn_point, max_r, np.asarray(obstacle[k]),
                                                            np.asarray(obstacle[k + 1]))
                            if h < 0.0:
                                n_overlaps += 1
                    for agent in range(len(follower_radii)):
                        h, _ = distance_circles(follower_positions[agent], follower_radii[agent], spawn_point, max_r)
                        if h < 0.0:
                            n_overlaps += 1

                    if n_overlaps == 0:
                        # (2.3)
                        # Append the point to spawn points
                        spawn_points.append([spawn_point[0], spawn_point[1]])
                        # print("Guide spawned")
                        # sys.stdout.flush()
                        break
                    j += 1
                    if j == overlaps:
                        raise Exception('Leaders do not fit in the cell')
                        # (3)
            else:
                vertices = np.asarray(geom.convex_hull.exterior)
                delaunay = Delaunay(vertices)
                mesh = vertices[delaunay.simplices]
                weights = triangle_area_cumsum(mesh)
                weights /= weights[-1]

                while j < overlaps:
                    seed += 1
                    distances = []  # temporarily store distances from the spawned point to the previously spawned
                    n_overlaps = 0  # for each attempt to position the guide, set number of overlaps to zero
                    # (3.1) Spawn a random point for the guide
                    np.random.seed(seed)
                    x = np.random.random()
                    k = np.searchsorted(weights, x)
                    a, b, c = mesh[k]
                    spawn_point = random_sample_triangle(a, b, c)
                    # spawn_point = random_sample_triangle(a, b, c, seed)
                    if n_spawnpoints != 0:
                        for k in range(0, n_spawnpoints):
                            d = length(spawn_point - spawn_points[k])
                            h = d - 2 * max_r
                            distances.append(h)
                        distances_array = distances
                        distances_array = np.asarray(distances_array)
                        n_overlaps += len(np.where(distances_array < 0)[0])
                    for obstacle in obstacles:
                        obstacle = list(obstacle.coords)
                        n_obstacle_points = len(obstacle)
                        for k in range(0, n_obstacle_points):
                            if k == n_obstacle_points - 1:
                                h, _ = distance_circle_line(spawn_point, max_r, np.asarray(obstacle[k]),
                                                            np.asarray(obstacle[0]))
                            else:
                                h, _ = distance_circle_line(spawn_point, max_r, np.asarray(obstacle[k]),
                                                            np.asarray(obstacle[k + 1]))
                            if h < 0.0:
                                n_overlaps += 1
                    for agent in range(len(follower_radii)):
                        h, _ = distance_circles(follower_positions[agent], follower_radii[agent], spawn_point, max_r)
                        if h < 0.0:
                            n_overlaps += 1

                    if n_overlaps == 0:
                        # (3.3)
                        # Append the point to spawn points
                        spawn_points.append([spawn_point[0], spawn_point[1]])
                        # print("Guide spawned")
                        # sys.stdout.flush()
                        break
                    j += 1
                    if j == overlaps:
                        raise Exception('Leaders do not fit in the cell')
        return spawn_points

    # CHECK ATTRIBUTES
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

        # Generate iterators for group of leaders.
        #target_exits = [0,1,0]
        #cells = [913,695,652]
        target_exits = []
        cells = []
        n_guides = len(target_exits)

        speed_left = "fast"
        speed_lower = "fast"
        speed_right = "fast"
        speed_upper = "fast"

        # Exiting agents in left spawn
        group_follower_spawn_left = AgentGroup(
            agent_type=self.agent_type,
            size=getattr(self, 'size_spawn_left'),
            attributes=self.attributes(familiar=2, has_target=True, is_follower=True))

        agents.add_non_overlapping_group(
            speed_left,
            "spawn_left",
            group_follower_spawn_left,
            position_gen=False,
            position_iter=iter([]),
            spawn=0,
            obstacles=geom_to_linear_obstacles(self.field.obstacles))

        # Exiting agents in lower spawn
        group_follower_spawn_lower = AgentGroup(
            agent_type=self.agent_type,
            size=getattr(self, 'size_spawn_lower'),
            attributes=self.attributes(familiar=3, has_target=True, is_follower=True))

        agents.add_non_overlapping_group(
            speed_lower,
            "spawn_lower",
            group_follower_spawn_lower,
            position_gen=False,
            position_iter=iter([]),
            spawn=0,
            obstacles=geom_to_linear_obstacles(self.field.obstacles))

        # Exiting agents in right spawn
        group_follower_spawn_right = AgentGroup(
            agent_type=self.agent_type,
            size=getattr(self, 'size_spawn_right'),
            attributes=self.attributes(familiar=0, has_target=True, is_follower=True))

        agents.add_non_overlapping_group(
            speed_right,
            "spawn_right",
            group_follower_spawn_right,
            position_gen=False,
            position_iter=iter([]),
            spawn=2,
            obstacles=geom_to_linear_obstacles(self.field.obstacles))

        # Exiting agents in upper spawn
        group_follower_spawn_upper = AgentGroup(
            agent_type=self.agent_type,
            size=getattr(self, 'size_spawn_upper'),
            attributes=self.attributes(familiar=1, has_target=True, is_follower=True))

        agents.add_non_overlapping_group(
            speed_upper,
            "spawn_upper",
            group_follower_spawn_upper,
            position_gen=False,
            position_iter=iter([]),
            spawn=3,
            obstacles=geom_to_linear_obstacles(self.field.obstacles))

        if n_guides != 0:
            init_pos = self.generate_leader_pos(cells, n_guides)
            print(init_pos)
            # init_pos = [[8, 6]]
            target_exits = iter(target_exits)
            init_pos = iter(init_pos)

            # Guides in Variance Dilemma
            group_leader = AgentGroup(
                agent_type=self.agent_type,
                size=n_guides,
                attributes=self.attributes_leader(target_iter=target_exits, has_target=True, is_follower=False))

            agents.add_non_overlapping_group(
                "group_leader",
                group_leader,
                position_gen=True,
                position_iter=init_pos,
                spawn=0,
                obstacles=geom_to_linear_obstacles(self.field.obstacles))

        return agents
