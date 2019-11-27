import pytest
from ddpg import *


def test_1():
    import numpy as np
    model = DDPG()
    model.generate_models(2,2)

    samples = 100
    
    # Dummy input, output
    input_data = np.random.uniform(size=(samples, model.input_dim))
    output_data = np.random.uniform(size=(samples, model.output_dim))

    # Default epochs value
    model.train_models(input_data, output_data)

    # Custom epochs value
    model.train_models(input_data, output_data, epochs=20)
    
