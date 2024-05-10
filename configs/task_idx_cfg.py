
_total_task_list = ['derain', 'dehaze', 'desnow', 'deraindrop']

def get_task_info(dataset: str, type: str, **kwargs):
    _task_dict = {}
    if dataset == 'allweather':
        _task_dict['list'] = ['derain', 'deraindrop', 'desnow']
        _task_dict['scale'] = '1+1+1'
        _task_dict['idx'] = {'derain': 0, 'deraindrop': 1, 'desnow': 2}
        _task_dict['task'] = {0: 'derain', 1: 'deraindrop', 2: 'desnow'}
    elif dataset == 'cityscapes':
        _task_dict['list'] = ['derain', 'dehaze']
        _task_dict['scale'] = '1+1'
        _task_dict['idx'] = {'derain': 0, 'dehaze': 1}
        _task_dict['task'] = {0: 'derain', 1: 'dehaze'}
    elif dataset == 'raindrop':
        _task_dict['list'] = ['deraindrop']
        _task_dict['scale'] = '1'
        _task_dict['idx'] = {'deraindrop': 0}
        _task_dict['task'] = {0: 'deraindrop'}
    elif dataset == 'snow100k':
        _task_dict['list'] = ['desnow']
        _task_dict['scale'] = '1'
        _task_dict['idx'] = {'desnow': 0}
        _task_dict['task'] = {0: 'desnow'}
    elif dataset in ['synthetic_rain', 'outdoor_rain']:
        _task_dict['list'] = ['derain']
        _task_dict['scale'] = '1'
        _task_dict['idx'] = {'derain': 0}
        _task_dict['task'] = {0: 'derain'}
    elif dataset == 'ots':
        _task_dict['list'] = ['dehaze']
        _task_dict['scale'] = '1'
        _task_dict['idx'] = {'dehaze': 0}
        _task_dict['task'] = {0: 'dehaze'}
    else:
        raise NotImplementedError
    
    if type == 'list':
        return _task_dict['list']

    elif type == 'dict':
        return _task_dict['dict']
    
    elif type == 'scale':
        return _task_dict['scale']

    elif type == 'idx':  # task -> idx
        assert kwargs['task'] in _task_dict['idx'].keys(), \
            "{} is not in {} _task_dict".format(kwargs['task'], dataset)
        return _task_dict['idx'][kwargs['task']]

    elif type == 'task':  # idx -> task
        assert kwargs['idx'] in _task_dict['task'].keys(), \
            "{} is not in {} _task_dict".format(kwargs['idx'], dataset)
        return _task_dict['task'][kwargs['idx']]