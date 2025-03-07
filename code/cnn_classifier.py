from keras.models import Sequential
from keras.layers import Dense, InputLayer, Dropout, Conv2D, MaxPooling2D, GlobalMaxPooling2D, SpatialDropout2D, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l1, l2

def create_cnn_classifier_network(image_size,
                                  nchannels,
                                  conv_layers,
                                  dense_layers,
                                  p_dropout,
                                  p_spatial_dropout,
                                  lambda_l2,
                                  lrate,
                                  n_classes,
                                  loss,
                                  metrics,
                                  padding,
                                  flatten,
                                  conv_activation,
                                  dense_activation):
    """
    Creates a CNN classifier model with specified hyperparameters.
    
    Parameters:
    - image_size: tuple, (height, width) of the input images
    - nchannels: int, number of channels in the input images
    - conv_layers: list of dictionaries with keys 'filters', 'kernel_size', 'pool_size', 'strides', 'batch_normalization'
    - dense_layers: list of dictionaries with keys 'units', 'batch_normalization'
    - p_dropout: float, dropout rate for Dense layers
    - p_spatial_dropout: float, spatial dropout rate for Conv layers
    - lambda_l2: float, L2 regularization coefficient
    - lrate: float, learning rate
    - n_classes: int, number of output classes
    - loss: loss function
    - metrics: list of metrics
    - padding: str, 'same' or 'valid' padding for convolutions
    - flatten: bool, whether to use Flatten before dense layers
    - conv_activation: str, activation function for convolutional layers
    - dense_activation: str, activation function for dense layers
    
    Returns:
    - Compiled Keras model
    """
    
    model = Sequential()
    model.add(InputLayer(input_shape=(image_size[0], image_size[1], nchannels)))
    
    # Convolutional layers with increasing filters before pooling
    for layer in conv_layers:
        # Compute factor to increase filters before pooling
        factor = layer['pool_size'][0] if layer['pool_size'] else 1 
        
        model.add(Conv2D(filters=layer['filters'] * factor,
                         kernel_size=layer['kernel_size'],
                         strides=layer['strides'] if layer['strides'] else (1, 1),
                         padding=padding,
                         activation=conv_activation,
                         kernel_regularizer=l2(lambda_l2)))
        
        if layer['batch_normalization']:
            model.add(BatchNormalization())
        
        if p_spatial_dropout > 0:
            model.add(SpatialDropout2D(p_spatial_dropout))
        
        if layer['pool_size']:
            model.add(MaxPooling2D(pool_size=layer['pool_size']))
    
    # Apply Flatten if specified, else use GlobalMaxPooling
    if flatten:
        model.add(Flatten())
    else:
        model.add(GlobalMaxPooling2D())
    
    # Dense layers
    for layer in dense_layers:
        model.add(Dense(layer['units'], activation=dense_activation, kernel_regularizer=l2(lambda_l2)))
        
        if layer['batch_normalization']:
            model.add(BatchNormalization())
        
        if p_dropout > 0:
            model.add(Dropout(p_dropout))
    
    # Output layer
    model.add(Dense(n_classes, activation='softmax'))
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=lrate),
                  loss=loss,
                  metrics=metrics)
    
    return model
