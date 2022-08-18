# Co-operative Multi-agent Reinforcement Learning in Partially Observable Card Games
This project implements novel team-based variants of card games, and presents a variety of deep reinforcement learning solutions that achieve good co-operative performance against differing levels of opposition. 

The main contributions of this project lie in its focus on co-operative performance in card games. Novel reinforcement learning environments are offered for Uno and Leduc Hold'em which can be easily extended to other co-operative environments. A number of state-of-the-art reinforcement learning algorithms are also implemented. 

This project is heavily based on [RLCard](https://github.com/datamllab/rlcard/), an existing open source Python card game reinforcement learning toolkit.
## Repository Structure

Existing RLCard codebase as well as this project's extensions to it exist in `RLCard/`. The basic structure of the codebase is as follows:
* **RLCard/rlcard/models:** Reinforcement Learning models using various algorithms are stored in this directory.
* **RLCard/rlcard/agents:** Reinforcement Learning agents for various algorithms are stored in this directory.
* **RLCard/rlcard/envs:** Reinforcement Learning environments for various card games are stored in this directory.
* **RLCard/rlcard/games:** Game logic for various card games are stored in this directory.
* **RLCard/examples:** Example Python scripts to use the RLCard library.

To demonstrate contributions of this project, `examples/` contains a number of interactive Python scripts which run games against pre-trained agents using the solutions presented in this project. Example training notebooks are also available.

Finally, this project was a created as part of my BSc Computer Science Undergraduate dissertation. The accompanying research paper that outlines motivations, methodology, results and discussion is available in `paper.pdf`.
## Installation & Usage
For installation and usage of the RLCard library please refer to the [RLCard documentation](/RLCard/README.md).
