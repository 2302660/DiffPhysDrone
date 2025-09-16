# DiffPhysDrone: Multi-Agent Colab Notebook Usage Guide

This guide explains how to use the provided Google Colab notebook for running multi-agent drone simulations with obstacles in a virtual environment.

## 🚀 Quick Start

### 1. Open the Notebook
- Upload `DiffPhysDrone_MultiAgent_Colab.ipynb` to Google Colab
- Or use this direct link (when available): [Open in Colab](https://colab.research.google.com/github/2302660/DiffPhysDrone/blob/main/DiffPhysDrone_MultiAgent_Colab.ipynb)

### 2. Setup Runtime
- Go to `Runtime` → `Change runtime type`
- Select `GPU` (T4 recommended)
- Save and connect

### 3. Run All Cells
- `Runtime` → `Run all` or use `Ctrl+F9`
- The notebook will automatically:
  - Install dependencies
  - Clone the repository
  - Build CUDA extensions
  - Initialize environment and model
  - Run training and visualization

## 📋 Notebook Structure

### Section 1: Environment Setup
- Installs PyTorch with CUDA support
- Installs required packages (matplotlib, tqdm, tensorboard)
- Builds custom CUDA extensions for physics simulation

### Section 2: Import Modules
- Imports all necessary libraries
- Sets up device configuration (CUDA/CPU)
- Initializes random seeds for reproducibility

### Section 3: Configuration
- Multi-agent simulation parameters
- Loss function coefficients
- Environment settings (obstacles, gates, etc.)
- Training hyperparameters

### Section 4: Environment & Model
- Initializes the simulation environment
- Creates neural network model
- Sets up optimizer and scheduler

### Section 5: Visualization Functions
- Environment state visualization
- Training progress plots
- 3D trajectory and obstacle visualization

### Section 6: Environment Demo
- Quick demonstration of the environment
- Shows initial agent positions and obstacles
- Displays depth camera view

### Section 7: Training Loop
- Complete training implementation
- Multi-agent obstacle avoidance
- Real-time loss monitoring
- Periodic visualization updates

### Section 8: Conclusion
- Summary of accomplished features
- Next steps for development
- Citation information

## 🎮 Customization Options

### Multi-Agent Configuration
```python
# Edit in Section 3
args.single = False      # Enable multi-agent (True for single agent)
args.batch_size = 64     # Number of parallel simulations
args.gate = True         # Enable gate obstacles
args.num_iters = 5000    # Training iterations
```

### Environment Parameters
```python
# Obstacle density and types
args.ground_voxels = False  # Ground obstacles
args.scaffold = False       # Scaffold structures
args.random_rotation = False # Random environment rotation
```

### Training Parameters
```python
# Loss coefficients for different objectives
args.coef_collide = 5.0        # Collision avoidance strength
args.coef_obj_avoidance = 2.0  # Obstacle avoidance strength
args.coef_v_pred = 2.0         # Velocity prediction accuracy
```

## 📊 Expected Outputs

### Training Progress
- Loss curves showing convergence
- Success rate (collision-free episodes)
- Speed statistics
- Component loss breakdown

### Visualizations
- **Depth Images**: Agent's camera view
- **3D Environment**: Top-down view with obstacles
- **Agent Velocities**: Speed of each agent
- **Obstacle Distances**: Safety margins

### Saved Files
- `diffphys_drone_multiagent_colab.pth`: Trained model weights
- `training_history.json`: Training metrics and curves

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Not Available**
   - Ensure GPU runtime is selected
   - Restart runtime if needed
   - Check with `torch.cuda.is_available()`

2. **Memory Issues**
   - Reduce `batch_size` from 64 to 32 or 16
   - Reduce `num_iters` for shorter training
   - Restart runtime to clear memory

3. **Compilation Errors**
   - Ensure all dependencies are installed
   - Check CUDA compatibility
   - Try rerunning the CUDA extension build cell

4. **Slow Training**
   - Verify GPU is being used
   - Consider reducing environment complexity
   - Monitor GPU utilization

### Performance Tips

1. **Faster Training**
   - Use larger batch sizes if memory allows
   - Reduce visualization frequency
   - Use mixed precision training (advanced)

2. **Better Results**
   - Increase training iterations
   - Fine-tune loss coefficients
   - Experiment with different environment configurations

## 🎯 Key Features Demonstrated

### Multi-Agent Capabilities
- Swarm coordination
- Inter-agent collision avoidance
- Formation maintenance
- Dynamic target assignment

### Obstacle Avoidance
- Static obstacles (spheres, boxes, cylinders)
- Gate navigation
- Dynamic obstacle generation
- Vision-based perception

### Training Features
- Differentiable physics simulation
- End-to-end learning
- Real-time visualization
- Comprehensive loss functions

## 📈 Expected Training Time

- **Quick Demo**: 5-10 minutes (1000 iterations)
- **Short Training**: 15-30 minutes (5000 iterations)
- **Full Training**: 1-2 hours (50000 iterations)

## 🔄 Loading Trained Models

To load a previously trained model:

```python
# After creating the model
model.load_state_dict(torch.load('diffphys_drone_multiagent_colab.pth'))
model.eval()
```

## 📝 Notes

- The notebook is designed for educational and research purposes
- Training times may vary based on Colab GPU availability
- For production use, consider running on dedicated hardware
- The simulation is physics-based and computationally intensive

## 🆘 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Ensure all cells are run in order
3. Restart runtime and try again
4. Check the original repository for updates

## 📚 References

- [Original Paper](https://www.nature.com/articles/s42256-025-01048-0)
- [Project Website](https://henryhuyu.github.io/DiffPhysDrone_Web/)
- [GitHub Repository](https://github.com/2302660/DiffPhysDrone)