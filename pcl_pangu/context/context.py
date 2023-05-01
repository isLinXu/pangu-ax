from collections import namedtuple

_CONTEXT = namedtuple('CONTEXT',['backend','id','token'])


def _confirm_id_token():
    pass

def set_context(backend='pytorch',id=0,token=0):
    assert backend in ['pytorch', 'mindspore', 'onnx-cpu', 'onnx-gpu']
    global _CONTEXT
    _CONTEXT.backend = backend
    _CONTEXT.id = id
    _CONTEXT.token = token
    #id token 验证
    _confirm_id_token()

def get_context():
    return _CONTEXT