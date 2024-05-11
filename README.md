# Active-Sensing-for-Multiuser-Beam-Tracking-with-RIS
This repository contains the code for the paper "Active Sensing for Multiuser Beam Tracking with Reconfigurable Intelligent Surface"

## [For Python codes]
Required environments: python==3.6, tensorflow==1.15.0, numpy==1.19.2 (Also valid: python==3.9, tensorflow==2.x)

## [For Training and Testing the Beam Tracking methods]
Each folder corresponds to a beam tracking method.\\
Please create two new folders named 'results_data'，'channel_data' wihtin each beam tracking method folder before training.
### For training:
For each beam tracking method, please run 'proposed_network.py' for training.\\
After training phase, you will obtain the trained neural network parameters, which are saved in the folder named 'params' whithin each beam tracking method folder.
### For testing:
To test the trained framework, please run the 'TestingPhase_Main.py' for each beam tracking method. (Notice, the testing results by running 'TestingPhase_Main.py' does not involves power allocation refinement.) \\
To obtain the results with power allocation refinment, please run the 'Power Allocation Refinment.m' after you run the 'TestingPhase_Main.py' for each beam tracking method. (You may find the parallel pool feature of the MATLAB extremely useful :)).

## [For Training and Testing the Beam Tracking methods]




## [For plotting Fig. 5]
Please first train the two approaches using the programs in plotting Fig. 6, then copy the generated 'params' file into their corresponding folder in Fig. 5.

Please only generate the 'Rician_channel' folder once and use the same generated channel statistics to test two approaches (Notice the 'isTesting' option). 

Notice, since the moving trajectory of the UE is randomly generated, there is no guarantee that Fig. 5 can be reproduced exactly the same.

## Questions
If you have any questions, please feel free to reach me at: johnny.han@mail.utoronto.ca
