from pcl_pangu.context import get_context

def check_context():
    ###
    # user identify

    ###
    context = get_context()
    if not isinstance(context.backend, str):
        raise ImportError("You need to use easypangu.context.set_context first")
    BACKEND = context.backend
    return BACKEND