from pygbe.main import main

def test_lysozyme():
    res = main(['', '../examples/lys', return_output_dict=True])

