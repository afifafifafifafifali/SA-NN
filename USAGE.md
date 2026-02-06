# SA-NN Usage

**NOTE**: SA-NN only works with PyTorch and will be it forever. Enable the ``use_progmem`` parameter if you are using an Arduino. 

SA-NN has only 1 function to use. The ``export_sa_nn`` function. 
It has only 4 parameters,where 1 is a must.
1. ``model`` : Your PyTorch Model .
2. ``vocab``: Your Vocabulary Dictionary.
3. ``filename``: Your header filename.
4. ``use_progmem``: A boolean value,set to None by default. But,if its True, then, you will get generated code for Arduino  only. 
