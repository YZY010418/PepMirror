#!/usr/bin/python
# -*- coding:utf-8 -*-
import os


CHECKPOINT_PREFIX = 'pepmirror_'
CHECKPOINT_SUFFIX = '_v1'
DEFAULT_CHECKPOINT_DIR = './checkpoints'

AXIAL_TYPE_ORDER = [
    'cross',
    'triple_projection',
    'triple_scalar',
    'commutator',
]

AXIAL_TYPE_ALIASES = {
    'cross': 'cross',
    'triple_projection': 'triple_projection',
    'triple_scalar': 'triple_scalar',
    'commutator': 'commutator',
    'threemix': 'cross_triple_projection_commutator',
    'cross_triple_projection_commutator': 'cross_triple_projection_commutator',
    'polar': 'polar',
    'none': 'polar',
}

AXIAL_POSITION_ALIASES = {
    'gnn': 'GNN',
    'ffn': 'FFN',
    'both': 'Both',
    'none': 'None',
}


def get_best_ckpt(ckpt_dir):
    with open(os.path.join(ckpt_dir, 'checkpoint', 'topk_map.txt'), 'r') as f:
        ls = f.readlines()
    ckpts = []
    for l in ls:
        k, v = l.strip().split(':')
        k = float(k)
        v = v.split('/')[-1]
        ckpts.append((k, v))

    best_ckpt = ckpts[0][1]
    return os.path.join(ckpt_dir, 'checkpoint', best_ckpt)


def _normalize_word(value):
    return str(value).strip().lower().replace('-', '_')


def _parse_axial_type_tokens(value):
    if isinstance(value, (list, tuple)):
        parts = [_normalize_single_axial_type(item) for item in value]
        return _canonicalize_type_parts(parts)

    raw_value = _normalize_word(value)
    if raw_value in AXIAL_TYPE_ALIASES:
        return AXIAL_TYPE_ALIASES[raw_value]

    for sep in ('+', ',', '|'):
        if sep in raw_value:
            parts = [_normalize_single_axial_type(item) for item in raw_value.split(sep)]
            return _canonicalize_type_parts(parts)

    tokens = raw_value.split('_')
    parts = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == 'triple' and i + 1 < len(tokens) and tokens[i + 1] in ['projection', 'scalar']:
            parts.append(f'triple_{tokens[i + 1]}')
            i += 2
            continue
        parts.append(token)
        i += 1

    parts = [_normalize_single_axial_type(item) for item in parts]
    return _canonicalize_type_parts(parts)


def _normalize_single_axial_type(value):
    key = _normalize_word(value)
    if key not in AXIAL_TYPE_ALIASES:
        supported = ', '.join(sorted(AXIAL_TYPE_ALIASES))
        raise ValueError(f'Unsupported axial_type {value!r}. Supported values: {supported}')
    return AXIAL_TYPE_ALIASES[key]


def _canonicalize_type_parts(parts):
    unique_parts = set(parts)
    if 'polar' in unique_parts:
        if len(unique_parts) > 1:
            raise ValueError('polar cannot be combined with axial feature types')
        return 'polar'

    unknown_parts = unique_parts - set(AXIAL_TYPE_ORDER)
    if unknown_parts:
        supported = ', '.join(AXIAL_TYPE_ORDER + ['polar'])
        raise ValueError(f'Unsupported axial_type combination {sorted(unknown_parts)}. Supported values: {supported}')

    ordered_parts = [part for part in AXIAL_TYPE_ORDER if part in unique_parts]
    return '_'.join(ordered_parts)


def normalize_axial_type(axial_type):
    return _parse_axial_type_tokens(axial_type)


def normalize_axial_position(axial_position):
    key = _normalize_word(axial_position)
    if key not in AXIAL_POSITION_ALIASES:
        supported = ', '.join(sorted(AXIAL_POSITION_ALIASES))
        raise ValueError(f'Unsupported axial_position {axial_position!r}. Supported values: {supported}')
    return AXIAL_POSITION_ALIASES[key]


def _release_condition_from_filename(path):
    name = os.path.basename(path)
    stem, ext = os.path.splitext(name)
    if ext != '.ckpt':
        return None
    if not stem.startswith(CHECKPOINT_PREFIX) or not stem.endswith(CHECKPOINT_SUFFIX):
        return None

    middle = stem[len(CHECKPOINT_PREFIX):-len(CHECKPOINT_SUFFIX)]
    axial_position = None
    axial_type = middle
    for position_key in AXIAL_POSITION_ALIASES:
        suffix = f'_{position_key}'
        if middle.endswith(suffix):
            axial_type = middle[:-len(suffix)]
            axial_position = normalize_axial_position(position_key)
            break

    axial_type = normalize_axial_type(axial_type)
    if axial_type == 'polar':
        axial_position = 'None'
    elif axial_position is None:
        raise ValueError(f'Checkpoint {name} is missing axial position in its filename')

    return axial_type, axial_position


def _format_condition(condition):
    axial_type, axial_position = condition
    if axial_type == 'polar':
        return 'axial_type=polar'
    return f'axial_type={axial_type}, axial_position={axial_position}'


def _requested_condition(config):
    if 'axial_type' not in config:
        raise ValueError('Generation config must define axial_type when selecting from release checkpoints.')

    axial_type = normalize_axial_type(config['axial_type'])
    if axial_type == 'polar':
        return axial_type, 'None'

    if 'axial_position' not in config:
        raise ValueError('Generation config must define axial_position when selecting from release checkpoints.')
    return axial_type, normalize_axial_position(config['axial_position'])


def _list_release_checkpoints(ckpt_dir):
    ckpt_dir = os.path.expandvars(os.path.expanduser(str(ckpt_dir)))
    if not os.path.isdir(ckpt_dir):
        return []

    entries = []
    for filename in sorted(os.listdir(ckpt_dir)):
        path = os.path.join(ckpt_dir, filename)
        condition = _release_condition_from_filename(path)
        if condition is not None:
            entries.append((condition, path))
    return entries


def _supported_conditions(entries):
    return ', '.join(_format_condition(condition) for condition, _ in entries) or 'none'


def resolve_generation_checkpoint(config, ckpt_root=None):
    ckpt_root = ckpt_root or config.get('checkpoint_dir', DEFAULT_CHECKPOINT_DIR)
    ckpt_root = os.path.expandvars(os.path.expanduser(str(ckpt_root)))

    if os.path.isfile(ckpt_root):
        condition = _release_condition_from_filename(ckpt_root)
        if 'axial_type' in config:
            requested = _requested_condition(config)
            if condition is not None and condition != requested:
                raise ValueError(
                    f'Requested {_format_condition(requested)}, but checkpoint filename '
                    f'{os.path.basename(ckpt_root)!r} declares {_format_condition(condition)}.'
                )
            condition = requested
        if condition is None:
            return ckpt_root, None
        return ckpt_root, {
            'axial_type': condition[0],
            'axial_position': condition[1],
        }

    release_checkpoints = _list_release_checkpoints(ckpt_root)
    if release_checkpoints:
        requested = _requested_condition(config)
        for condition, path in release_checkpoints:
            if condition == requested:
                return path, {
                    'axial_type': requested[0],
                    'axial_position': requested[1],
                }

        raise ValueError(
            f'Unsupported axial condition: {_format_condition(requested)}. '
            f'Supported release checkpoints in {ckpt_root}: {_supported_conditions(release_checkpoints)}'
        )

    if os.path.isdir(ckpt_root):
        return get_best_ckpt(ckpt_root), None

    raise FileNotFoundError(f'Checkpoint path does not exist: {ckpt_root}')
