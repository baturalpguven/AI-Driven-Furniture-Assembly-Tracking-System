# AI-Driven Furniture Assembly Tracking System (MAAF)

Term project for 3D data collection using Intel Realsense stereo cameras and ROS system to understand furniture assembly by analyzing human actions and mistake detection.

## Project Overview
MAAF is designed to detect furniture assembly actions and identify any mistakes in the process. This project offers two distinct solutions:

1. **ResNet + LSTM Model**: A robust approach that combines ResNet with LSTM for sequential data handling.
2. **Custom Lightweight Model**: A tailored, efficient model designed for lighter computational needs.

## Features
- **Dual Model Approach**: Choose between a high-performance or a lightweight model depending on your system's capabilities.
- **Error Detection**: Automatically detects mistakes during the furniture assembly process.
- **Integration Ready**: Incorporates foundational models like Co-Tracker and DEVA for easy integration and adaptability.
- **User-Friendly**: Designed for immediate implementation with minimal setup required.

## Demo Videos
Below are the demo videos showcasing both models in action:

<div align="center">

### Custom Lightweight Model Demo with Mistakes  
<a href="https://drive.google.com/file/d/1mTqgl9HL5pj_nOVnhX0S5uHOhNSsjuoZ/preview">
    <img src="https://github.com/user-attachments/assets/ce8dcd6d-3e6b-47d7-ae5b-57c46954d0a7" alt="Custom Lightweight Model Demo" width="400" height="300">
</a>

* Custom Lightweight Model: The yellow block represents the Efficient Channel Attention (ECA), the pink block denotes the ReductionCNN block, and the purple block illustrates the ViewAware Transformer.*

### ResNet + LSTM Model Demo with Mistakes  
<a href="https://drive.google.com/file/d/128upw8J09Fk4JG2Br4a_w1oaDDnVG-gp/preview">
    <img src="https://github.com/user-attachments/assets/93316879-4d9f-42a6-82e1-70d749f12a62" alt="ResNet + LSTM Model Demo" width="400" height="300">
</a>

* ResNet+LSTM Based Big Model: The gray block represents the ResNet component, while the blue blocks denote the LSTM units.*

</div>

## Resources
- **Model Directories**: Each model has its own directory containing all necessary files and documentation.
- **Visualizations and Analysis**: Extensive ablation analyses and demo results are available in their respective folders.

## Getting Started
To get started with MAAF, navigate to the model directories for specific installation instructions and usage examples:

- [ResNet + LSTM Model](/big_model)
- [Custom Lightweight Model](/light_weight_model)

## Acknowledgements
A huge thank you to my amazing team members for their invaluable contributions to this project. Their hard work and dedication made MAAF possible!
