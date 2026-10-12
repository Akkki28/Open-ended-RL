# Open-Ended Reinforcement Learning


<img width="480" height="480" alt="saved-agent-GeneratedHalfcheetahvelocitytrackingenvRewardEncouragesMatchingARandomTaE0Da38Ea72-v0-episode-0-ezgif com-video-to-gif-converter (1)" src="https://github.com/user-attachments/assets/536bc69a-48e3-42aa-8df1-53e28a536229" />

matching a random target velocity

<img width="480" height="480" alt="saved-agent-GeneratedHalfcheetahsteadyspeedenvRewardEncouragesForwardSpeedPenalizesCF9Aa6A22E2-v0-episode-0-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/ee999bc0-b6fe-4ff7-abe3-0d0cac327898" />

reward encourages forward speed, penalizes control effort and penalizes rapid changes in speed, fostering smooth, steady locomotion.



<img width="480" height="480" alt="saved-agent-GeneratedHalfcheetahsmoothcontrolenvRewardForwardSpeedCtrlCostJerkCost2021B04753-v0-episode-0-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/5490ab27-2106-48a0-88d9-d38cc23bdbf5" />

reward = forward_speed – ctrl_cost – jerk_cost – pitch_penalty.



<img width="480" height="480" alt="saved-agent-GeneratedHalfcheetahsinusoidalvelocityenvRewardPenalizesDeviationFromASiA15A59A1A9-v0-episode-0-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/e32bd517-1961-4b31-b905-ac600fb795c1" />

reward penalizes deviation from a sinusoidal target forward velocity (amplitude = 2 m/s, frequency = 0.05 Hz) and control effort



<img width="480" height="480" alt="saved-agent-GeneratedHalfcheetahlanekeepingenvRewardForwardSpeedCtrlCostLanePenalty9007D4D418-v0-episode-0-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/5924d6a0-ef61-4382-9a48-79a870582b4f" />

reward = forward_speed – ctrl_cost – lane_penalty (absolute lateral displacement).



<img width="480" height="480" alt="saved-agent-GeneratedHalfcheetahenergyefficiencyenvRewardCombinesForwardSpeedPenalize629Bb230A2-v0-episode-0-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/aae32da9-7128-4485-9d07-4603b23c4d7b" />

reward combines forward speed, penalizes control effort more heavily, encourages maintaining a nominal torso height, and penalizes large pitch angles.



<img width="480" height="480" alt="saved-agent-GeneratedHalfcheetahdirectionenvRewardEncouragesMovingInARandomlyChosenFee74Db971-v0-episode-0-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/9eff849c-8db5-4603-9c49-03266ef5c25b" />

reward encourages moving in a randomly chosen target direction (forward or backward) each episode while penalizing control effort













