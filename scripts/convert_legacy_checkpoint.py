#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import os
import sys
import types

import torch


def parse():
    parser = argparse.ArgumentParser(description='Convert legacy PepMirror checkpoints to current module names.')
    parser.add_argument('--input', required=True, help='Legacy checkpoint path')
    parser.add_argument('--output', required=True, help='Converted checkpoint path')
    parser.add_argument('--axial_type', default=None, help='Override axial type if it cannot be inferred')
    parser.add_argument('--axial_position', default=None, choices=['GNN', 'FFN', 'Both', 'None'], help='Override axial position')
    parser.add_argument('--keep_legacy_names', action='store_true', help='Keep legacy module names such as scale_after_*')
    parser.add_argument('--force', action='store_true', help='Overwrite output if it already exists')
    return parser.parse_args()


def install_legacy_checkpoint_aliases():
    """Let torch.load resolve module paths used by old full-model checkpoints."""
    from models.modules.AFIEPT import afiept as afiept_module
    from models.modules.AFIEPT import radial_basis as radial_basis_module

    legacy_names = {
        'AxialFeatureInjection': 'AxialFeatureConstructor',
        'axial_feature_injection': 'axial_feature_constructor',
    }
    for old_name, new_name in legacy_names.items():
        if not hasattr(afiept_module, old_name) and hasattr(afiept_module, new_name):
            setattr(afiept_module, old_name, getattr(afiept_module, new_name))

    legacy_methods = {
        '_inject_cross': '_construct_cross',
        '_inject_triple': '_construct_triple',
        '_inject_commutator': '_construct_commutator',
    }
    axial_cls = afiept_module.AxialFeatureConstructor
    for old_name, new_name in legacy_methods.items():
        if not hasattr(axial_cls, old_name) and hasattr(axial_cls, new_name):
            setattr(axial_cls, old_name, getattr(axial_cls, new_name))

    ept_pkg = sys.modules.get('models.modules.EPT')
    if ept_pkg is None:
        ept_pkg = types.ModuleType('models.modules.EPT')
        ept_pkg.__path__ = []
        sys.modules['models.modules.EPT'] = ept_pkg

    import models.modules as modules_pkg
    setattr(modules_pkg, 'EPT', ept_pkg)

    ept_pkg.ept = afiept_module
    ept_pkg.radial_basis = radial_basis_module
    for name in (
        'AxialFeatureConstructor',
        'EPTLayer',
        'GVPFFNLayer',
        'SelfAttnLayer',
        'SubLayerWrapper',
        'Transformer',
        'XTransEncoderAct',
    ):
        if hasattr(afiept_module, name):
            setattr(ept_pkg, name, getattr(afiept_module, name))

    sys.modules['models.modules.EPT.ept'] = afiept_module
    sys.modules['models.modules.EPT.radial_basis'] = radial_basis_module


def infer_axial_setting(path):
    name = os.path.basename(path).lower()
    if 'polar' in name:
        axial_type = 'None'
        axial_position = 'None'
        return axial_type, axial_position

    if 'cross_triple_projection_commutator' in name:
        axial_type = 'threemix'
    elif 'triple_scalar' in name:
        axial_type = 'triple_scalar'
    elif 'commutator' in name:
        axial_type = 'commutator'
    elif 'triple_projection' in name or 'triple' in name:
        axial_type = 'triple'
    else:
        axial_type = 'cross'

    axial_position = 'Both'
    if 'ffn' in name:
        axial_position = 'FFN'
    elif 'gnn' in name:
        axial_position = 'GNN'
    elif 'both' in name:
        axial_position = 'Both'

    return axial_type, axial_position


def infer_axial_type_from_module(module, fallback):
    if hasattr(module, 'scale_after_commutator'):
        return 'commutator'
    if hasattr(module, 'scale_after_cross'):
        in_features = getattr(module.scale_after_cross, 'in_features', None)
        return 'threemix' if in_features is not None and in_features % module.d_hidden == 0 and in_features // module.d_hidden == 4 else 'cross'
    if hasattr(module, 'scale_after_triple'):
        in_features = getattr(module.scale_after_triple, 'in_features', None)
        if in_features == module.d_hidden * 2:
            return fallback
    return fallback


def load_checkpoint(path):
    install_legacy_checkpoint_aliases()
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def convert_checkpoint(model, input_path, axial_type=None, axial_position=None, keep_legacy_names=False):
    from models.modules.AFIEPT.afiept import AxialFeatureConstructor

    inferred_type, inferred_position = infer_axial_setting(input_path)
    default_type = axial_type or inferred_type
    default_position = axial_position or inferred_position

    stats = {
        'transformers': 0,
        'gvp_layers': 0,
        'legacy_scale_to_mixing_axial': 0,
        'axial_constructor_added': 0,
        'axial_constructor_updated': 0,
        'use_axial_ffn_added': 0,
        'legacy_scale_removed': 0,
    }

    for module in model.modules():
        cls_name = module.__class__.__name__
        if cls_name not in ('Transformer', 'GVPFFNLayer'):
            continue

        if cls_name == 'Transformer':
            stats['transformers'] += 1
        else:
            stats['gvp_layers'] += 1

        module_axial_type = axial_type
        if module_axial_type is None:
            module_axial_type = getattr(module, 'axial_type', None)
            if module_axial_type is None and hasattr(module, 'axial_injector'):
                module_axial_type = getattr(module.axial_injector, 'axial_type', None)
            if module_axial_type is None:
                module_axial_type = default_type
            module_axial_type = infer_axial_type_from_module(module, module_axial_type)

        module.axial_type = module_axial_type
        module.axial_position = axial_position or getattr(module, 'axial_position', default_position)

        if not hasattr(module, 'axial_constructor'):
            module.axial_constructor = AxialFeatureConstructor(module.axial_type)
            stats['axial_constructor_added'] += 1
        elif getattr(module.axial_constructor, 'axial_type', None) != module.axial_type:
            module.axial_constructor = AxialFeatureConstructor(module.axial_type)
            stats['axial_constructor_updated'] += 1

        if cls_name == 'GVPFFNLayer' and not hasattr(module, 'use_axial_ffn'):
            module.use_axial_ffn = module.axial_position in ['FFN', 'Both']
            stats['use_axial_ffn_added'] += 1

        if cls_name == 'Transformer' and not hasattr(module, 'mixing_axial'):
            legacy_scale_name = next(
                (name for name in ('scale_after_cross', 'scale_after_commutator', 'scale_after_triple') if hasattr(module, name)),
                None
            )
            if legacy_scale_name is not None:
                module.mixing_axial = getattr(module, legacy_scale_name)
                stats['legacy_scale_to_mixing_axial'] += 1
                if not keep_legacy_names:
                    delattr(module, legacy_scale_name)
                    stats['legacy_scale_removed'] += 1
            elif module.axial_position != 'None':
                raise RuntimeError(
                    'Transformer is missing mixing_axial and legacy scale_after_* module. '
                    'Refusing to initialize new weights during checkpoint conversion.'
                )
        if cls_name == 'Transformer' and module.axial_position in ['GNN', 'Both']:
            expected_in_features = module.d_hidden * module.axial_constructor.out_mul
            actual_in_features = getattr(module.mixing_axial, 'in_features', None)
            if actual_in_features != expected_in_features:
                raise RuntimeError(
                    f'Transformer.mixing_axial has in_features={actual_in_features}, '
                    f'but axial_type={module.axial_type} expects {expected_in_features}.'
                )
        if cls_name == 'GVPFFNLayer':
            use_axial_ffn = module.axial_position in ['FFN', 'Both']
            if use_axial_ffn and module.axial_constructor.out_type == 'vector':
                expected_linear_v_in = module.d_hidden * module.axial_constructor.out_mul
            else:
                expected_linear_v_in = module.d_hidden
            actual_linear_v_in = getattr(module.linear_v, 'in_features', None)
            if actual_linear_v_in != expected_linear_v_in:
                raise RuntimeError(
                    f'GVPFFNLayer.linear_v has in_features={actual_linear_v_in}, '
                    f'but axial_type={module.axial_type}, axial_position={module.axial_position} '
                    f'expects {expected_linear_v_in}.'
                )

            if use_axial_ffn and module.axial_constructor.out_type == 'scalar':
                expected_ffn_in = module.d_hidden * (module.axial_constructor.out_mul + 1)
            else:
                expected_ffn_in = module.d_hidden * 2
            actual_ffn_in = getattr(module.ffn_mlp[0], 'in_features', None)
            if actual_ffn_in != expected_ffn_in:
                raise RuntimeError(
                    f'GVPFFNLayer.ffn_mlp[0] has in_features={actual_ffn_in}, '
                    f'but axial_type={module.axial_type}, axial_position={module.axial_position} '
                    f'expects {expected_ffn_in}.'
                )

    return stats


def main():
    args = parse()
    if os.path.exists(args.output) and not args.force:
        raise FileExistsError(f'{args.output} already exists. Pass --force to overwrite it.')

    model = load_checkpoint(args.input)
    stats = convert_checkpoint(
        model,
        args.input,
        axial_type=args.axial_type,
        axial_position=args.axial_position,
        keep_legacy_names=args.keep_legacy_names,
    )

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(model, args.output)

    print(f'Converted checkpoint: {args.input} -> {args.output}')
    for key, value in stats.items():
        print(f'{key}: {value}')


if __name__ == '__main__':
    main()
