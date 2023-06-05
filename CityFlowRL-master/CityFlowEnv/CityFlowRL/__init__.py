from gym.envs.registration import register

# source: https://github.com/openai/gym/blob/master/gym/envs/__init__.py

register(
    id='CityFlowRL-v0', # use id to pass to gym.make(id)
    entry_point='CityFlowRL.envs:CityFlowRL'
)