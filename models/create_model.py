from models.DiT.models import DiT_models
from models.taskrouting import TaskRouter


def create_model(model_config):
    """
    Create various architectures from model_config
    """
    if model_config.name in DiT_models.keys():
        model = DiT_models[model_config.name](
            input_size=model_config.param.latent_size,
            num_classes=model_config.param.num_classes,
            router=TaskRouter,
            sharing_ratio=model_config.routing.sharing_ratio,
            taskcount=model_config.routing.taskcount,
            init_method=model_config.routing.init_method,
        )
    else:
        raise NotImplementedError
    return model
