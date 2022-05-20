from .depth import DepthAgent


def get_agent(name, opt, predict_only=False):
    if name == "depth":
        agent = DepthAgent(opt, predict_only=predict_only)
    else:
        raise ValueError(f"Unknown Agent: {name}")

    return agent
