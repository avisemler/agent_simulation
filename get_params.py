import argparse


def str2bool(s):
    if type(s) == bool:
        return s
    if s.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif s.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


def get_args(*args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--use_agent_graph', default='true', type=str2bool, help='')
    parser.add_argument('--graph_gen', default="ws", type=str, help="")
    parser.add_argument("--graph_param1", default=5, type=float, help="")
    parser.add_argument("--graph_param2", default=0.1, type=float, help="")
    parser.add_argument("--run_name", default="new run", type=str, help="")
    parser.add_argument("--total_timesteps", default=12_000, type=int, help="")
    parser.add_argument("--intervention", default="false", type=str2bool, help="")

    args = parser.parse_args(args)

    return args
