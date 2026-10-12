# Open-Ended Reinforcement Learning
This project enables open-ended robotic learning by using reinforcement learning and a large language model to generate new environments from an initial base environment and learn new behaviours. Through iterative environment evolution, the robot discovers a diverse family of policies, each exhibiting distinct behaviors and skills.

<img width="480" height="480" alt="saved-agent-GeneratedHalfcheetahvelocitytrackingenvRewardEncouragesMatchingARandomTaE0Da38Ea72-v0-episode-0-ezgif com-video-to-gif-converter (1)" src="https://github.com/user-attachments/assets/536bc69a-48e3-42aa-8df1-53e28a536229" />

matching a random target velocity


<img width="480" height="480" alt="saved-agent-GeneratedHalfcheetahsinusoidalvelocityenvRewardPenalizesDeviationFromASiA15A59A1A9-v0-episode-0-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/e32bd517-1961-4b31-b905-ac600fb795c1" />

matching a sinusoidal target forward velocity function(amplitude = 2 m/s, frequency = 0.05 Hz)



<img width="480" height="480" alt="saved-agent-GeneratedHalfcheetahlanekeepingenvRewardForwardSpeedCtrlCostLanePenalty9007D4D418-v0-episode-0-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/5924d6a0-ef61-4382-9a48-79a870582b4f" />

reward = forward_speed – ctrl_cost – lane_penalty (absolute lateral displacement).


<img width="480" height="480" alt="saved-agent-GeneratedHalfcheetahdirectionenvRewardEncouragesMovingInARandomlyChosenFee74Db971-v0-episode-0-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/9eff849c-8db5-4603-9c49-03266ef5c25b" />

reward encourages moving in a randomly chosen target direction (forward or backward) each episode while penalizing control effort













