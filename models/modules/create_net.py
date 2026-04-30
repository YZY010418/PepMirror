#!/usr/bin/python
# -*- coding:utf-8 -*-

from .AFIEPT.afiept import XTransEncoderAct as AFIEPT

def create_net(
    name,
    hidden_size,
    edge_size,
    opt={}
):
    if name == 'AFIEPT':
        kargs = {
            'hidden_size': hidden_size,
            'ffn_size': hidden_size,
            'edge_size': edge_size
        }
        kargs.update(opt)
        return AFIEPT(**kargs)
    else:
        raise NotImplementedError(f'{name} not implemented')