# \# ComfyUI-Anima-Enhancer

# 

# A simple ComfyUI custom node for improving \*\*Anima\*\* generations.

# 

# It adds the \*\*Anima Layer Replay Patcher\*\*, which can enhance fine detail and coherence by replaying selected internal blocks during denoising. It also includes an optional \*\*Spectrum\*\* mode for faster generation.

# 

# In tested setups, Spectrum can improve generation speed by \*\*around 35%\*\*.

# 

# \## Features

# 

# \- Enhances fine detail and structural coherence on Anima models

# \- Optional built-in Spectrum acceleration

# \- Supports custom block selection

# \- Simple workflow integration

# 

# \## Node

# 

# The node appears in ComfyUI as:

# 

# \*\*Anima Layer Replay Patcher\*\*

# 

# \## Installation

# 

# Open a terminal inside your ComfyUI `custom\_nodes` folder and run:

# 

# ```bash

# git clone https://github.com/AdamNizol/ComfyUI-Anima-Enhancer.git

# ````

# 

# Then restart ComfyUI.

# 

# Example folder layout after installation:

# 

# ```text

# ComfyUI/

# └── custom\_nodes/

# &nbsp;   └── ComfyUI-Anima-Enhancer/

# ```

# 

# \## Usage

# 

# Add the \*\*Anima Layer Replay Patcher\*\* node to your workflow and connect your model through it. Placing it directly before the sampler is advised.

# 

# \### Main inputs

# 

# \* \*\*block\_indices\*\*

# &nbsp; Comma-separated block list, for example:

# 

# &nbsp; \* `3,4,5`

# &nbsp; \* `3-5`

# &nbsp; \* `3,4,5,8`

# 

# \* \*\*denoise\_start\_pct\*\*

# &nbsp; When replay begins during denoising

# 

# \* \*\*denoise\_end\_pct\*\*

# &nbsp; When replay stops during denoising

# 

# \* \*\*enable\_spectrum\*\*

# &nbsp; Turns on optional Spectrum acceleration

# 

# \### Spectrum inputs

# 

# When Spectrum is enabled, these settings become active:

# 

# \* \*\*spectrum\_w\*\*

# \* \*\*spectrum\_m\*\*

# \* \*\*spectrum\_lam\*\*

# \* \*\*spectrum\_warmup\_steps\*\*

# 

# \## Recommended starting settings

# 

# For many Anima tests, a strong starting point is:

# 

# \* \*\*block\_indices:\*\* `3,4,5`

# \* \*\*denoise\_start\_pct:\*\* `0.50`

# \* \*\*denoise\_end\_pct:\*\* `1.00`

# 

# If using Spectrum, a good starting point is:

# 

# \* \*\*enable\_spectrum:\*\* `true`

# \* \*\*spectrum\_w:\*\* `0.2-0.3`

# \* \*\*spectrum\_m:\*\* `8-16`

# \* \*\*spectrum\_lam:\*\* `0.5`

# \* \*\*spectrum\_warmup\_steps:\*\* `6`

