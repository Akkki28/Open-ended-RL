# Open-Ended Reinforcement Learning


https://github.com/user-attachments/assets/74106c7a-5589-4984-8382-ee3aeae059db
matching a random target velocity

https://github.com/user-attachments/assets/c5d4f593-a6c0-4321-9ce0-2fa2c559d562
reward encourages forward speed, penalizes control effort and penalizes rapid changes in speed, fostering smooth, steady locomotion.



https://github.com/user-attachments/assets/5c1b5d9c-2ec1-4086-94bd-20afe37cea41
reward = forward_speed – ctrl_cost – jerk_cost – pitch_penalty.



https://github.com/user-attachments/assets/9d6ac531-2272-4a86-af5a-b4918980a649
reward penalizes deviation from a sinusoidal target forward velocity (amplitude = 2 m/s, frequency = 0.05 Hz) and control effort



https://github.com/user-attachments/assets/64d2da38-b0c2-407b-8bed-82cfc96cb868
reward = forward_speed – ctrl_cost – lane_penalty (absolute lateral displacement).



https://github.com/user-attachments/assets/fc094ff7-330e-43a3-8f50-9b1631815342
reward combines forward speed, penalizes control effort more heavily, encourages maintaining a nominal torso height, and penalizes large pitch angles.



https://github.com/user-attachments/assets/65082794-b7be-4a1a-8e37-e6958c043cf1
reward encourages moving in a randomly chosen target direction (forward or backward) each episode while penalizing control effort













