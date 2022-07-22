from algorithms.tlda import TLDA
from algorithms.sac import SAC
from algorithms.drq import DrQ

algorithm = {
	'tlda': TLDA,
	'sac': SAC,
	'drq': DrQ
}


def make_agent(obs_shape, action_shape, args):
	return algorithm[args.algorithm](obs_shape, action_shape, args)
