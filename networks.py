"""
tensorflow/keras networks for voxelmorph

If you use this code, please cite one of the voxelmorph papers:
https://github.com/voxelmorph/voxelmorph/blob/master/citations.bib

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in 
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or 
implied. See the License for the specific language governing permissions and limitations under 
the License.
"""

# internal python imports
import warnings
from collections.abc import Iterable

# third party imports
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow.keras.initializers as KI
from keras.layers import Lambda,add
# local imports
import neurite as ne
from .. import default_unet_features
from . import layers
from . import utils

# make directly available from vxm
ModelCheckpointParallel = ne.callbacks.ModelCheckpointParallel

###############################################################################################################
# delete redundancy
# 并行(采用一直卷积，不下采样，跳连接)
# 简化RFM
class VxmDense_train9(ne.modelio.LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @ne.modelio.store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 svf_resolution=1,
                 int_resolution=2,
                 int_downsize=None,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False,
                 input_model=None,
                 hyp_model=None,
                 fill_value=None,
                 reg_field='preintegrated',
                 name='vxm_dense',
                 training=True):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_unet_features is an integer.
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_unet_features is an
                integer. Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this
                value is 0.
            svf_resolution: Resolution (relative voxel size) of the predicted SVF.
                Default is 1.
            int_resolution: Resolution (relative voxel size) of the flow field during
                vector integration. Default is 2.
            int_downsize: Deprecated - use int_resolution instead.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Deprecated - use svf_resolution instead.
            input_model: Model to replace default input layer before concatenation. Default is None.
            hyp_model: HyperMorph hypernetwork model. Default is None.
            reg_field: Field to regularize in the loss. Options are 'svf' to return the
                SVF predicted by the Unet, 'preintegrated' to return the SVF that's been
                rescaled for vector-integration (default), 'postintegrated' to return the
                rescaled vector-integrated field, and 'warp' to return the final, full-res warp.
            name: Model name - also used as layer name prefix. Default is 'vxm_dense'.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        if input_model is None:
            # configure default input layers if an input model is not provided
            source = tf.keras.Input(shape=(*inshape, src_feats), name='%s_source_input' % name)
            target = tf.keras.Input(shape=(*inshape, trg_feats), name='%s_target_input' % name)
            input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])
        else:
            source, target = input_model.outputs[:2]

        # configure inputs
        inputs = input_model.inputs
        hyp_input = None
        hyp_tensor = None

        if int_downsize is not None:
            warnings.warn('int_downsize is deprecated, use the int_resolution parameter.')
            int_resolution = int_downsize

        # compute number of upsampling skips in the decoder (to downsize the predicted field)
        if unet_half_res:
            warnings.warn('unet_half_res is deprecated, use the svf_resolution parameter.')
            svf_resolution = 2

        nb_upsample_skips = int(np.floor(np.log(svf_resolution) / np.log(2)))

        # --------------overall branch------------
        # build core unet model and grab inputs
        unet_model1 = Unet_random(
            input_model=input_model,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            nb_upsample_skips=nb_upsample_skips,
            hyp_input=hyp_input,
            hyp_tensor=hyp_tensor,
            name='%s_globalnet' % name
        )
        # transform unet output into a flow field
        Conv = getattr(KL, 'Conv%dD' % ndims)
        flow_mean1 = Conv(ndims, kernel_size=3, padding='same',
                          kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-5),
                          name='%s_flow_globalnet' % name)(unet_model1.output)
        # optionally include probabilities
        flow1 = flow_mean1
        # rescale field to target svf resolution
        pre_svf_size1 = np.array(flow1.shape[1:-1])
        svf_size1 = np.array([np.round(dim / svf_resolution) for dim in inshape])
        if not np.array_equal(pre_svf_size1, svf_size1):
            rescale_factor = svf_size1[0] / pre_svf_size1[0]
            flow1 = layers.RescaleTransform(rescale_factor, name=f'{name}_svf_resize_globalnet')(flow1)
        # rescale field to target integration resolution
        if int_steps > 0 and int_resolution > 1:
            int_size1 = np.array([np.round(dim / int_resolution) for dim in inshape])
            if not np.array_equal(svf_size1, int_size1):
                rescale_factor1 = int_size1[0] / svf_size1[0]
                flow1 = layers.RescaleTransform(rescale_factor1, name=f'{name}_flow_resize_globalnet')(flow1)
        # optionally negate flow for bidirectional model
        pos_flow1 = flow1
        # integrate to produce diffeomorphic warp (i.e. treat flow as a stationary velocity field)
        if int_steps > 0:
            pos_flow1 = layers.VecInt(method='ss',
                                      name='%s_flow_int_globalnet' % name,
                                      int_steps=int_steps)(pos_flow1)
        # resize to final resolution
        if int_steps > 0 and int_resolution > 1:
            rescale_factor1 = inshape[0] / int_size1[0]
            pos_flow1 = layers.RescaleTransform(rescale_factor1, name='%s_diffflow_globalnet' % name)(pos_flow1)
        # warp image with flow field
        y_source1 = layers.SpatialTransformer(
            interp_method='linear',
            indexing='ij',
            fill_value=fill_value,
            name='%s_transformer_globalnet' % name)([source, pos_flow1])
        # initialize the keras model

        # --------------local branch------------
        unet_model2 = Unet_random(
            input_model=input_model,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            nb_upsample_skips=nb_upsample_skips,
            hyp_input=hyp_input,
            hyp_tensor=hyp_tensor,
            name='%s_localnet' % name
        )
        Conv = getattr(KL, 'Conv%dD' % ndims)
        flow_mean2 = Conv(ndims, kernel_size=3, padding='same',
                          kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-5),
                          name='%s_flow_localnet' % name)(unet_model2.output)
        flow2 = flow_mean2
        # rescale field to target svf resolution
        pre_svf_size2 = np.array(flow2.shape[1:-1])
        svf_size2 = np.array([np.round(dim / svf_resolution) for dim in inshape])
        if not np.array_equal(pre_svf_size2, svf_size2):
            rescale_factor2 = svf_size2[0] / pre_svf_size2[0]
            flow2 = layers.RescaleTransform(rescale_factor2, name=f'{name}_svf_resize_localnet')(flow2)
        # rescale field to target integration resolution
        if int_steps > 0 and int_resolution > 1:
            int_size2 = np.array([np.round(dim / int_resolution) for dim in inshape])
            if not np.array_equal(svf_size2, int_size2):
                rescale_factor2 = int_size2[0] / svf_size2[0]
                flow2 = layers.RescaleTransform(rescale_factor2, name=f'{name}_flow_resize_localnet')(flow2)
        # optionally negate flow for bidirectional model
        pos_flow2 = flow2
        # integrate to produce diffeomorphic warp (i.e. treat flow as a stationary velocity field)
        if int_steps > 0:
            pos_flow2 = layers.VecInt(method='ss',
                                      name='%s_flow_int_localnet' % name,
                                      int_steps=int_steps)(pos_flow2)
        # resize to final resolution
        if int_steps > 0 and int_resolution > 1:
            rescale_factor2 = inshape[0] / int_size2[0]
            pos_flow2 = layers.RescaleTransform(rescale_factor2, name='%s_diffflow_localnet' % name)(pos_flow2)
        # warp image with flow field
        y_source2 = layers.SpatialTransformer(
            interp_method='linear',
            indexing='ij',
            fill_value=fill_value,
            name='%s_transformer_localnet' % name)([source, pos_flow2])

        # -----------------RFM-------------------
        MaxPooling = getattr(KL, 'MaxPooling%dD' % ndims)
        AveragePooling = getattr(KL, 'AveragePooling%dD' % ndims)
        flow1_Max = MaxPooling(pool_size=2, strides=1, padding='SAME', name='uf_max_pooling1')(pos_flow1)
        flow2_Max = MaxPooling(pool_size=2, strides=1, padding='SAME', name='uf_max_pooling2')(pos_flow2)
        flow1_Average = AveragePooling(pool_size=2, strides=1, padding='SAME', name='uf_average_pooling1')(pos_flow1)
        flow2_Average = AveragePooling(pool_size=2, strides=1, padding='SAME', name='uf_average_pooling2')(pos_flow2)

        max_concat = KL.concatenate([flow1_Max, flow2_Max], name='uf_max_concat')
        average_concat = KL.concatenate([flow1_Average, flow2_Average], name='uf_average_concat')

        max_conv1 = _conv_block(max_concat, 2 * ndims, name='uf_max_conv1', hyp_tensor=hyp_tensor)
        max_conv2 = _conv_block(max_conv1, ndims, name='uf_max_conv2', hyp_tensor=hyp_tensor)
        average_conv1 = _conv_block(average_concat, 2 * ndims, name='uf_average_conv1', hyp_tensor=hyp_tensor)
        average_conv2 = _conv_block(average_conv1, ndims, name='uf_average_conv2', hyp_tensor=hyp_tensor)
        final = KL.concatenate([max_conv2, average_conv2], name='uf_flow_concat')
        final = Conv(ndims, kernel_size=1, padding='same',
                     kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-5),
                     name='uf_final_conv')(final)

        y_source = layers.SpatialTransformer(
            interp_method='linear',
            indexing='ij',
            fill_value=fill_value,
            name='uf_%s_transformer' % name)([source, final])
        pos_flow = final

        outputs = [y_source1, pos_flow1, y_source2, pos_flow2, y_source, pos_flow]

        super().__init__(name=name, inputs=inputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = ne.modelio.LoadableModel.ReferenceContainer()
        # self.references.unet_model = unet_model
        self.references.source = source
        self.references.target = target
        # self.references.svf = svf
        # self.references.preint_flow = preint_flow
        # self.references.postint_flow = postint_flow
        self.pos_flow1 = pos_flow1
        self.pos_flow2 = pos_flow2
        self.references.pos_flow = pos_flow
        self.references.neg_flow = None
        self.references.y_source = y_source
        self.references.y_target = None
        self.references.hyp_input = hyp_input

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def get_registration_model1(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, [self.outputs[1], self.outputs[3], self.outputs[5]])
        # return tf.keras.Model(self.inputs, [self.references.pos_flow1,self.references.pos_flow2,self.references.pos_flow])

    def register1(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model1().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        st_input = [img_input, warp_model.output]
        y_img = layers.SpatialTransformer(interp_method=interp_method)(st_input)
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])



###############################################################################################################



###############################################################################
# Utility/Core Networks
###############################################################################

class Transform(tf.keras.Model):
    """
    Simple transform model to apply dense or affine transforms.
    """

    def __init__(self,
                 inshape,
                 affine=False,
                 interp_method='linear',
                 rescale=None,
                 fill_value=None,
                 nb_feats=1):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            affine: Enable affine transform. Default is False.
            interp_method: Interpolation method. Can be 'linear' or 'nearest'. Default is 'linear'.
            rescale: Transform rescale factor. Default is None.
            fill_value: Fill value for SpatialTransformer. Default is None.
            nb_feats: Number of source image features. Default is 1.
        """

        # configure inputs
        ndims = len(inshape)
        scan_input = tf.keras.Input((*inshape, nb_feats), name='scan_input')

        if affine:
            trf_input = tf.keras.Input((ndims, ndims + 1), name='trf_input')
        else:
            trf_shape = inshape if rescale is None else [int(d / rescale) for d in inshape]
            trf_input = tf.keras.Input((*trf_shape, ndims), name='trf_input')

        trf_scaled = trf_input if rescale is None else layers.RescaleTransform(rescale)(trf_input)

        # transform and initialize the keras model
        trf_layer = layers.SpatialTransformer(interp_method=interp_method,
                                              name='transformer',
                                              fill_value=fill_value)
        y_source = trf_layer([scan_input, trf_scaled])
        super().__init__(inputs=[scan_input, trf_input], outputs=y_source)




###############################################################################################################

class Unet_random(tf.keras.Model):
    """
    A unet architecture that builds off either an input keras model or input shape. Layer features
    can be specified directly as a list of encoder and decoder features or as a single integer along
    with a number of unet levels. The default network features per layer (when no options are
    specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]

    This network specifically does not subclass LoadableModel because it's meant to be a core,
    internal model for more complex networks, and is not meant to be saved/loaded independently.
    """

    def __init__(self,
                 inshape=None,
                 input_model=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 do_res=False,
                 nb_upsample_skips=0,
                 hyp_input=None,
                 hyp_tensor=None,
                 final_activation_function=None,
                 name='unet'):
        """
        Parameters:
            inshape: Optional input tensor shape (including features). e.g. (192, 192, 192, 2).
            input_model: Optional input model that feeds directly into the unet before concatenation
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            nb_upsample_skips: Number of upsamples to skip in the decoder (to downsize the
                the output resolution). Default is 0.
            hyp_input: Hypernetwork input tensor. Enables HyperConvs if provided. Default is None.
            hyp_tensor: Hypernetwork final tensor. Enables HyperConvs if provided. Default is None.
            final_activation_function: Replace default activation function in final layer of unet.
            name: Model name - also used as layer name prefix. Default is 'unet'.
        """

        # have the option of specifying input shape or input model
        if input_model is None:
            if inshape is None:
                raise ValueError('inshape must be supplied if input_model is None')
            unet_input = KL.Input(shape=inshape, name='%s_input' % name)
            model_inputs = [unet_input]
        else:
            if len(input_model.outputs) == 1:
                unet_input = input_model.outputs[0]
            else:
                unet_input = KL.concatenate(input_model.outputs, name='%s_input_concat' % name)
            model_inputs = input_model.inputs

        '''
        # add hyp_input tensor if provided
        if hyp_input is not None:
            model_inputs = model_inputs + [hyp_input]
        '''

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        ndims = len(unet_input.get_shape()) - 2
        assert ndims in (1, 2, 3), 'ndims should be one of 1, 2, or 3. found: %d' % ndims
        MaxPooling = getattr(KL, 'MaxPooling%dD' % ndims)

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * nb_levels

        # configure encoder (down-sampling path)
        enc_layers = []
        last = unet_input
        for level in range(nb_levels - 1):
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                layer_name = '%s_enc_conv_%d_%d' % (name, level, conv)
                last = _conv_block(last, nf, name=layer_name, do_res=do_res, hyp_tensor=hyp_tensor)
            enc_layers.append(last)

            # temporarily use maxpool since downsampling doesn't exist in keras
            last = MaxPooling(max_pool[level], name='%s_enc_pooling_%d' % (name, level))(last)

        # if final_activation_function is set, we need to build a utility that checks
        # which layer is truly the last, so we know not to apply the activation there
        if final_activation_function is not None and len(final_convs) == 0:
            activate = lambda lvl, c: not (lvl == (nb_levels - 2) and c == (nb_conv_per_level - 1))
        else:
            activate = lambda lvl, c: True

        # configure decoder (up-sampling path)
        for level in range(nb_levels - 1):
            real_level = nb_levels - level - 2
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                layer_name = '%s_dec_conv_%d_%d' % (name, real_level, conv)
                last = _conv_block(last, nf, name=layer_name, do_res=do_res, hyp_tensor=hyp_tensor,
                                   include_activation=activate(level, conv))

            # upsample
            if level < (nb_levels - 1 - nb_upsample_skips):
                layer_name = '%s_dec_upsample_%d' % (name, real_level)
                last = _upsample_block(last, enc_layers.pop(), factor=max_pool[real_level],
                                       name=layer_name)

        # now build function to check which of the 'final convs' is really the last
        if final_activation_function is not None:
            activate = lambda n: n != (len(final_convs) - 1)
        else:
            activate = lambda n: True

        # now we take care of any remaining convolutions
        for num, nf in enumerate(final_convs):
            layer_name = '%s_dec_final_conv_%d' % (name, num)
            last = _conv_block(last, nf, name=layer_name, hyp_tensor=hyp_tensor,
                               include_activation=activate(num))

        # add the final activation function is set
        if final_activation_function is not None:
            last = KL.Activation(final_activation_function, name='%s_final_activation' % name)(last)

        super().__init__(inputs=model_inputs, outputs=last, name=name)


###############################################################################################################


###############################################################################
# Private functions
###############################################################################

def _conv_block(x, nfeat, strides=1, name=None, do_res=False, hyp_tensor=None,
                include_activation=True):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """
    ndims = len(x.get_shape()) - 2
    assert ndims in (1, 2, 3), 'ndims should be one of 1, 2, or 3. found: %d' % ndims

    extra_conv_params = {}
    if hyp_tensor is not None:
        Conv = getattr(ne.layers, 'HyperConv%dDFromDense' % ndims)
        conv_inputs = [x, hyp_tensor]
    else:
        Conv = getattr(KL, 'Conv%dD' % ndims)
        extra_conv_params['kernel_initializer'] = 'he_normal'
        conv_inputs = x

    convolved = Conv(nfeat, kernel_size=3, padding='same',
                     strides=strides, name=name, **extra_conv_params)(conv_inputs)

    if do_res:
        # assert nfeat == x.get_shape()[-1], 'for residual number of features should be constant'
        add_layer = x
        print('note: this is a weird thing to do, since its not really residual training anymore')
        if nfeat != x.get_shape().as_list()[-1]:
            add_layer = Conv(nfeat, kernel_size=3, padding='same',
                             name='resfix_' + name, **extra_conv_params)(conv_inputs)
        convolved = KL.Lambda(lambda x: x[0] + x[1])([add_layer, convolved])

    if include_activation:
        name = name + '_activation' if name else None
        convolved = KL.LeakyReLU(0.2, name=name)(convolved)

    return convolved



def _upsample_block(x, connection, factor=2, name=None):
    """
    Specific upsampling and concatenation layer for unet.
    """
    ndims = len(x.get_shape()) - 2
    assert ndims in (1, 2, 3), 'ndims should be one of 1, 2, or 3. found: %d' % ndims
    UpSampling = getattr(KL, 'UpSampling%dD' % ndims)

    size = (factor,) * ndims if ndims > 1 else factor
    upsampled = UpSampling(size=size, name=name)(x)
    name = name + '_concat' if name else None
    return KL.concatenate([upsampled, connection], name=name)


###############################################################################################################
