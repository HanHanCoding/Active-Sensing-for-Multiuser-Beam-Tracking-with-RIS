# For the BCD algorithmm
To obtain the faster convergence, we use the results of the GNN benchmark as the initial point. 
So first train the GNN benchmark within the folder before running 'BCD_with_GNN_initialization.m'

## The procedure:
1. Run 'proposed_network.py'
2. Run 'TestingPhase_Main.py'
3. Run 'generate_channel_UpperLowerbound.py'. Here, make sure a) the parameters settings are the same with the training/testing process. b) there is no file in the 'channel_data' folder.
4. Run 'BCD_with_GNN_initialization.m'. (You may find the parallel pool feature of the MATLAB extremely useful :)).
