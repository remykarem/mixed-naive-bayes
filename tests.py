from mixed_naive_bayes import normal, _posterior_proba

def test_normal():
    assert int(normal(0,0,1) * 100) / 100 == 0.39
