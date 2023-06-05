import json
from os import path
import cityflow
import gym
import numpy as np
from gym import spaces, logger


# noinspection PyShadowingNames
class CityFlowRL(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, **kwargs):
        # super(CityFlowRL, self).__init__()
        self.config_path = path.join('configs/' + kwargs['config'] + '/config.json')
        self.cityflow = cityflow.Engine(self.config_path, thread_num=12)

        self.previousPhase = 0
        self.steps_per_action = kwargs['steps_per_action']
        self.sec_per_step = 1.0

        self.steps_per_episode = kwargs['steps_per_episode']
        self.current_step = 0
        self.is_done = False
        self.reward_range = (-float('inf'), float('inf'))

        # open cityflow config file into dict
        self.configDict = json.load(open(self.config_path))
        # open cityflow roadnet file into dict
        self.roadnetDict = json.load(open(path.join(self.configDict['dir'], self.configDict['roadnetFile'])))
        self.flowDict = json.load(open(path.join(self.configDict['dir'], self.configDict['flowFile'])))
        self.max_speed = self.flowDict[0]['vehicle']['maxSpeed']
        # create dict of controllable intersections and number of light phases
        self.intersections = {}
        for i in range(len(self.roadnetDict['intersections'])):
            # check if intersection is controllable
            if not self.roadnetDict['intersections'][i]['virtual']:
                # for each roadLink in intersection store incoming lanes, outgoing lanes and direction in lists
                incomingLanes = []
                outgoingLanes = []
                directions = []
                for j in range(len(self.roadnetDict['intersections'][i]['roadLinks'])):
                    incomingRoads = []
                    outgoingRoads = []
                    directions.append(self.roadnetDict['intersections'][i]['roadLinks'][j]['direction'])
                    for k in range(len(self.roadnetDict['intersections'][i]['roadLinks'][j]['laneLinks'])):
                        incomingRoads.append(self.roadnetDict['intersections'][i]['roadLinks'][j]['startRoad'] +
                                             '_' +
                                             str(self.roadnetDict['intersections'][i]['roadLinks'][j]['laneLinks'][k][
                                                     'startLaneIndex']))
                        outgoingRoads.append(self.roadnetDict['intersections'][i]['roadLinks'][j]['endRoad'] +
                                             '_' +
                                             str(self.roadnetDict['intersections'][i]['roadLinks'][j]['laneLinks'][k][
                                                     'endLaneIndex']))
                    incomingLanes.append(incomingRoads)
                    outgoingLanes.append(outgoingRoads)

                # add intersection to dict where key = intersection_id
                # value = no of lightPhases, incoming lane names, outgoing lane names, directions for each lane group
                self.intersections[self.roadnetDict['intersections'][i]['id']] = [
                    [len(self.roadnetDict['intersections'][i]['trafficLight']['lightphases'])],
                    incomingLanes,
                    outgoingLanes,
                    directions
                ]

        self.intersection_info = list(self.intersections.values())[0]
        self.intersection_id = list(self.intersections.keys())[0]

        x = [x[0] for x in self.intersection_info[1]]
        self.start_lane_ids = list(dict.fromkeys(x))

        self.mode = "count_waiting"
        # self.mode = 'leader_speed'

        self.action_space = spaces.Discrete(self.intersection_info[0][0])

        if self.mode == 'count_waiting':
            self.observation_space = spaces.MultiDiscrete(
                [100] * len(self.start_lane_ids))  # spaces.MultiDiscrete([[100,100,5]]*8)
        elif self.mode == 'leader_speed':
            obs_dict = {}
            for keys in self.start_lane_ids:
                obs_dict[keys] = {
                    'count': spaces.Discrete(100),
                    'speed': spaces.Box(low=0, high=self.max_speed, shape=(1,), dtype=np.float32)
                }
            self.observation_space = spaces.Dict(obs_dict)

    def step(self, action):
        # assert isinstance(action, int), "Action must be of integer type."
        self.previousPhase = action

        self.cityflow.set_tl_phase(self.intersection_id, action)
        for i in range(self.steps_per_action):
            self.cityflow.next_step()
        observation = self._get_observation()
        reward = self._get_reward()

        self.current_step += 1

        if self.is_done:
            logger.warn("You are calling 'step()' even though this environment has already returned done = True. "
                        "You should always call 'reset()' once you receive 'done = True' "
                        "-- any further steps are undefined behavior.")
            reward = 0.0

        if self.current_step + 1 == self.steps_per_episode:
            self.is_done = True

        if self.cityflow.get_vehicle_count() == 0 and self.flowDict[-1][
            'endTime'] != -1 and self.cityflow.get_current_time() > \
                self.flowDict[-1]['endTime']:
            self.is_done = True

        travel_time = self.cityflow.get_average_travel_time()
        return observation, reward, self.is_done, {'avg_travel_time': travel_time}  # false

    # {}

    def reset(self, **kwargs):
        self.cityflow.reset()
        self.is_done = False
        self.current_step = 0

        return self._get_observation()

    def render(self, mode='console'):
        print("Current time: " + str(self.cityflow.get_current_time()))

    def _get_observation(self):
        if self.mode == "count_waiting":
            lane_waiting_vehicles_dict = self.cityflow.get_lane_waiting_vehicle_count()

            observation = np.zeros((len(self.start_lane_ids)))
            for i in range(len(self.start_lane_ids)):
                observation[i] = lane_waiting_vehicles_dict[self.start_lane_ids[i]]
        elif self.mode == "leader_speed":
            lane_vehicles = self.cityflow.get_lane_vehicles()
            lane_waiting_count = self.cityflow.get_lane_waiting_vehicle_count()
            observation = {}
            for keys, values in lane_vehicles.items():
                if keys in self.start_lane_ids:

                    if values:
                        leader_speed = self.cityflow.get_vehicle_info(values[0])['speed']

                    else:
                        leader_speed = self.max_speed

                    observation[keys] = {
                        'count': lane_waiting_count[keys],
                        'leader_speed': leader_speed
                    }

        return observation

    def _get_reward(self):
        lane_waiting_vehicles_dict = self.cityflow.get_lane_waiting_vehicle_count()
        reward = 0.0

        if self.mode == "count_waiting":
            for (road_id, num_vehicles) in lane_waiting_vehicles_dict.items():
                if road_id in self.start_lane_ids:
                    reward -= self.sec_per_step * num_vehicles
        elif self.mode == "leader_speed":
            lane_vehicles = self.cityflow.get_lane_vehicles()
            lane_waiting_count = self.cityflow.get_lane_waiting_vehicle_count()
            for keys, values in lane_vehicles.items():
                if keys in self.start_lane_ids:

                    if values:
                        leader_speed = self.cityflow.get_vehicle_info(values[0])['speed']

                    else:
                        leader_speed = self.max_speed

        return reward

    def set_replay_path(self, path):
        self.cityflow.set_replay_file(path)

    def seed(self, seed=None):
        self.cityflow.set_random_seed(seed)

    def get_path_to_config(self):
        return self.config_path

    def set_save_replay(self, save_replay):
        self.cityflow.set_save_replay(save_replay)
