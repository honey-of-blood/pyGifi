import pytest
import numpy as np
import pandas as pd
from pygifi.morals import Morals
from pygifi.cv import cv_morals

def test_cv_morals_basic():
    # Create simple dataset
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(100, 3), columns=['X1', 'X2', 'X3'])
    y = pd.Series(np.random.rand(100))
    
    # Fit base Morals model
    model = Morals(itmax=10, xdegrees=1, ydegrees=1)
    model.fit(X, y)
    
    # Run CV
    res = cv_morals(model, k=3, random_state=42)
    
    assert hasattr(res, 'cv_error')
    assert isinstance(res.cv_error, float)
    assert res.cv_error > 0
    assert len(res.fold_errors) == 3

def test_cv_morals_unfitted():
    model = Morals()
    with pytest.raises(ValueError, match="must be fitted prior"):
        cv_morals(model, k=3)
