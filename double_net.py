import theano
import theano.tensor as T
from theano.tensor.signal import pool

import numpy as np

import utils

import sys, os

def randn( shape, sd, dtype=theano.config.floatX ):
    return( np.random.randn( np.prod(shape) ).astype( dtype ).reshape( shape ) * sd )

def net(original_input_size, flat_P, n_channels, filter_widths, pool_sizes, n_hidden, rotation=(3,2,1,0), learning_rate=0.002, l2reg=0.001, fixed=[], additional_channels=0):

    x=T.ftensor4('x') # samples x (4+additional channels) x 1 x sequence_length
    x_flat=T.fmatrix('x_flat')
    y=T.fmatrix('y')

    params={}

    net_output={}
    for flip in (False,True):
        input_size=original_input_size
        current_input=x[:,rotation,:,:][:,:,:,::-1] if flip else x
        
        input_channels=4+additional_channels

        for layer_index in range(len(n_channels)):

            filter_shape=(n_channels[layer_index], input_channels, 1, filter_widths[layer_index])
            w_name="f%i" % layer_index

            # input_dimension=input_size*input_channels # incorrect? 
            input_dimension=filter_widths[layer_index]*input_channels

            mu_name="g%i" % layer_index
            if not flip:
                params[ w_name ]=theano.shared( randn( filter_shape, sd=np.sqrt(2.0/input_dimension) ), borrow=True, name=w_name )
                params[ mu_name ]=theano.shared( np.zeros( n_channels[layer_index], dtype=theano.config.floatX ), borrow=True, name=mu_name )
            conv_out=T.nnet.conv2d( input=current_input, filters=params[ w_name ], filter_shape=filter_shape )

            pooled_out=pool.pool_2d(input=conv_out, ds=(1,pool_sizes[layer_index]), ignore_border=True)

            current_input=T.nnet.softplus( pooled_out + params[ mu_name ].dimshuffle('x', 0, 'x', 'x'))

            input_size=( input_size-filter_widths[layer_index]+1 ) // pool_sizes[layer_index]
            input_channels=n_channels[layer_index]
            print("ConvLayer %i size: %i channels: %s, total:%i" % (layer_index, input_size, input_channels, input_size*input_channels))

        input_dimension=input_channels * input_size
        current_input=current_input.flatten(2) # now samples x input_dimension

        current_input=T.concatenate( (current_input, x_flat), axis=1 )
        input_dimension += flat_P

        for layer_index in range(len(n_hidden)):
            w_name="w%i" % layer_index
            mu_name="mu%i" % layer_index
            if not flip:
                params[ mu_name ]=theano.shared( np.zeros( n_hidden[ layer_index ], dtype=theano.config.floatX ), borrow=True, name=mu_name )
                params[ w_name ]=theano.shared( randn( ( input_dimension, n_hidden[ layer_index ]), sd=np.sqrt(2.0/input_dimension) ), borrow=True, name=w_name )

            current_input=T.nnet.relu( T.dot( current_input, params[ w_name ] ) + params[ mu_name ] )
            input_dimension=n_hidden[ layer_index ]

        if not flip:
            params[ "b" ]=theano.shared( randn( input_dimension, sd=np.sqrt(2.0/input_dimension) ), borrow=True, name='b' )
            params[ "offset" ]=theano.shared( np.array( 0.0, dtype=theano.config.floatX), name="offset" )

        # net_output is N-vector (N is number of samples)
        net_output[flip]=T.dot( current_input, params["b"] ) + params[ "offset" ]

    net_output=T.maximum( net_output[False], net_output[True] )
        
    sml=T.nnet.sigmoid( - net_output )
    s1ml=T.nnet.sigmoid( 1.0 -net_output )
    p=T.stack( ( sml, s1ml - sml, 1.0 - s1ml ), axis=1) 

    neg_like=-( y * np.log(p + 1.0e-20) ).sum()

    cost=neg_like
    if l2reg > 0.0:
        cost = neg_like + l2reg*(params[ "b" ] ** 2).sum() # 
        for layer_index in range(len(n_channels)):
            cost += l2reg*(params[ "f%i" % layer_index ] ** 2).sum()
        for layer_index in range(len(n_hidden)):
            cost += l2reg*(params[ "w%i" % layer_index ] ** 2).sum()
    
    not_fixed_params=[ v for k,v in params.iteritems() if not k in fixed ]
    updates=utils.AdaMax( not_fixed_params, cost, alpha=learning_rate)

    train=theano.function( [x,x_flat,y], [net_output,neg_like], updates=updates  )

    test=theano.function( [x,x_flat,y], [net_output,neg_like] )

    pred=theano.function( [x,x_flat], net_output )
    
    return train,test,pred,params

